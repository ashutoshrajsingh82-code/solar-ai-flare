import os
import time
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from typing import Optional

from app.core.config import settings
from app.services.pipeline_state import pipeline_state
from app.services.artifact_catalog import catalog_model_artifacts, verify_model_artifact_set, verify_catalog_artifact
from app.services.scientific_readiness import generate_scientific_readiness_report
from app.schemas.schemas import TrainRequest

from ml.models.baseline import MODEL_BUILDERS
from ml.models.registry import ModelRegistry, ModelRecord, new_model_id
from ml.training.train import train_deep_model, TrainConfig
from ml.training.validation import validate_classical_training_data
from ml.training.eligibility import real_training_blockers
from ml.training.dataset_eligibility import assess_dataset_eligibility
from ml.features.windowing import build_windows, WindowConfig
from ml.features.engineering import FeatureConfig, select_feature_columns
from ml.features.manifest import create_feature_manifest, save_feature_manifest
from ml.features.decomposition import (
    DecompositionConfig, HistoricalDecompositionCache, decompose_historical_windows,
    decomposition_feature_names,
)
from ml.features.manifest import data_fingerprint
from ml.evaluation.metrics import evaluate_binary_forecast, multiclass_confusion_matrix

DEEP_MODEL_TYPES = {"cnn_1d", "lstm", "cnn_lstm_fusion", "cnn_transformer", "dlstm"}
CLASSICAL_MODEL_TYPES = set(MODEL_BUILDERS.keys())  # LR, RF, SVM, gradient_boosting + climatology / flux_rule / hard_soft_ratio_rule / two_feature_logreg

_backend_root = Path(__file__).resolve().parents[2]
_model_root = Path(settings.model_dir)
settings.model_dir = str((_model_root if _model_root.is_absolute() else _backend_root / _model_root).resolve())
os.makedirs(settings.model_dir, exist_ok=True)

FORECAST_HORIZON_COLS = ["forecast_1h", "forecast_3h", "forecast_6h", "forecast_12h", "forecast_24h"]
MAX_STORED_EVALUATION_SAMPLES = 100_000

registry = ModelRegistry(settings.model_dir)


def _pipeline_feature_config() -> FeatureConfig:
    report = pipeline_state.stage_reports.get("featured", {})
    values = report.get("feature_config", {})
    return FeatureConfig(**values) if values else FeatureConfig(cadence_seconds=pipeline_state.cadence_seconds)


def _preprocessor_manifest_metadata(transformer):
    """Expose fitted train-only preprocessing parameters in readable metadata."""
    scaler = transformer.scaler
    def as_list(value):
        array = np.asarray(value)
        return array.tolist() if array.ndim else array.item()
    fitted = {}
    for attribute in ("mean_", "scale_", "var_", "center_", "n_features_in_", "n_samples_seen_"):
        if hasattr(scaler, attribute):
            fitted[attribute] = as_list(getattr(scaler, attribute))
    return {
        "scaler_type": transformer.config.scaler_type,
        "log_transform": transformer.config.log_transform,
        "fit_on": "train_split_only",
        "input_columns": list(transformer.columns),
        "reference_scales": dict(transformer.reference_scale),
        "fitted_scaler_parameters": fitted,
    }


def _period_record(frame):
    if frame is None or frame.empty or "timestamp" not in frame:
        return {"start_utc": None, "end_utc": None, "rows": 0}
    stamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce", format="mixed").dropna()
    return {"start_utc": stamps.min().isoformat() if len(stamps) else None,
            "end_utc": stamps.max().isoformat() if len(stamps) else None, "rows": int(len(stamps))}


def _save_model_feature_manifest(model_id, feature_frame, names, input_configuration,
                                 feature_config, train_df, val_df, test_df,
                                 preprocessing, random_seed, decomposition=None,
                                 architecture=None, training_configuration=None,
                                 input_window_length=None):
    manifest = create_feature_manifest(
        feature_frame, names, input_configuration, feature_config,
        source_frame=pipeline_state.raw_df,
        split_frames={"train": train_df, "validation": val_df, "test": test_df},
        preprocessing=preprocessing, random_seed=random_seed, decomposition=decomposition,
    )
    manifest["model_id"] = model_id
    manifest["architecture"] = architecture
    manifest["training_configuration"] = training_configuration or {}
    manifest["input_window_length"] = input_window_length
    path = os.path.join(settings.model_dir, f"{model_id}.feature_manifest.json")
    save_feature_manifest(path, manifest)
    return path


def _atomic_save(path, writer):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".model-artifact-", suffix=".tmp", dir=os.path.dirname(os.path.abspath(path)))
    os.close(fd)
    try:
        writer(temp_path)
        if not os.path.isfile(temp_path) or os.path.getsize(temp_path) == 0:
            raise OSError("Model artifact writer produced an empty file")
        with open(temp_path, "r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _training_status():
    if settings.data_mode == "demo" or pipeline_state.dataset_meta.get("source") == "demo":
        return "demo_only"
    return "trained" if not real_training_blockers(pipeline_state.dataset_meta) else "untrained"


def _channels_for_configuration(input_configuration: str):
    if input_configuration == "solexs_only":
        return ["solexs_flux_scaled"], 1
    if input_configuration == "helios_only":
        return ["helios_flux_scaled"], 1
    return ["solexs_flux_scaled", "helios_flux_scaled"], 2


def train_model(req: TrainRequest, progress_callback=None, stage_callback=None, job_id=None) -> dict:
    with pipeline_state.operation_lock:
        result = _train_model_locked(req, progress_callback, stage_callback, job_id=job_id)
        if stage_callback:
            stage_callback("completed")
        return result


def _train_model_locked(req: TrainRequest, progress_callback=None, stage_callback=None, job_id=None) -> dict:
    readiness = generate_scientific_readiness_report()
    dataset_source = pipeline_state.dataset_meta.get("source")
    if dataset_source != "demo" and not readiness["training_allowed"]:
        blockers = " ".join(readiness["blockers"])
        raise ValueError(
            "Real-data training is blocked by the scientific readiness gate: "
            f"{blockers} Readiness report: {readiness['report_files']['json']}"
        )
    if dataset_source != "demo":
        eligibility = assess_dataset_eligibility(pipeline_state.dataset_meta, {
            "train": pipeline_state.train_df,
            "validation": pipeline_state.val_df,
            "test": pipeline_state.test_df,
        })
        if not eligibility["training_allowed"]:
            codes = ", ".join(item["code"] for item in eligibility["rejection_reasons"])
            raise ValueError(
                "Real-data training is blocked by the dataset eligibility gate: "
                f"{eligibility['status']} ({codes})"
            )
    if not pipeline_state.is_ready_for_training():
        raise ValueError(
            "Pipeline not ready. Run /api/preprocess before training. "
            f"Scientific readiness report: {readiness['report_files']['json']}"
        )
    if _training_status() == "untrained":
        blockers = real_training_blockers(pipeline_state.dataset_meta)
        raise ValueError("Real-data training is blocked: " + " ".join(blockers))

    channel_cols, n_channels = _channels_for_configuration(req.input_configuration)
    model_id = new_model_id(req.model_type + "_" + req.input_configuration)
    if stage_callback:
        stage_callback("preparing_dataset")
        stage_callback("validating_input")

    if req.model_type in CLASSICAL_MODEL_TYPES:
        result = _train_classical(req, channel_cols, model_id, stage_callback, job_id=job_id)
    elif req.model_type in DEEP_MODEL_TYPES:
        result = _train_deep(req, channel_cols, n_channels, model_id, progress_callback, stage_callback, job_id=job_id)
    else:
        raise ValueError(f"Unknown model_type '{req.model_type}'")

    return result


def _train_classical(req: TrainRequest, channel_cols, model_id: str, stage_callback=None, job_id=None) -> dict:
    train_df = pipeline_state.train_df.copy()
    val_df = pipeline_state.val_df.copy()
    test_df = pipeline_state.test_df.copy()

    # re-attach engineered features + labels by merging on timestamp
    feat_df = pipeline_state.labeled_df
    feature_cols = pipeline_state.feature_columns
    feature_config = _pipeline_feature_config()
    use_cols = select_feature_columns(feat_df, req.input_configuration, feature_config)
    if not use_cols:
        raise ValueError(f"No registered features are available for {req.input_configuration}.")

    # Keep rows with unknown forecast labels; models fit each horizon on its known rows.
    # Features and nowcast labels still must be present.
    train_merged = train_df[["timestamp"]].merge(feat_df, on="timestamp", how="inner").dropna(subset=use_cols + ["state"])
    val_merged = val_df[["timestamp"]].merge(feat_df, on="timestamp", how="inner").dropna(subset=use_cols + ["state"])
    test_merged = test_df[["timestamp"]].merge(feat_df, on="timestamp", how="inner").dropna(subset=use_cols + ["state"])

    if len(train_merged) < 10 or len(val_merged) < 5 or len(test_merged) < 5:
        raise ValueError("Not enough clean rows in chronological train/validation/test splits after feature engineering.")

    X_train = train_merged[use_cols].values
    X_val = val_merged[use_cols].values
    X_test = test_merged[use_cols].values
    y_now_train = train_merged["state"].values
    y_now_val = val_merged["state"].values
    y_now_test = test_merged["state"].values
    y_fc_train = train_merged[FORECAST_HORIZON_COLS].values
    y_fc_val = val_merged[FORECAST_HORIZON_COLS].values
    y_fc_test = test_merged[FORECAST_HORIZON_COLS].values
    validate_classical_training_data(
        X_train, y_now_train, y_fc_train, X_val, y_now_val, y_fc_val,
    )
    y_now_train = y_now_train.astype(int)
    y_now_val = y_now_val.astype(int)
    y_now_test = y_now_test.astype(int)

    t0 = time.time()
    model = MODEL_BUILDERS[req.model_type]()
    if stage_callback:
        stage_callback("training")
    model.fit(X_train, y_now_train, y_fc_train, feature_names=use_cols)
    train_duration = time.time() - t0

    fc_prob_val = model.predict_forecast_proba(X_val)
    thresholds, threshold_report = _select_validation_thresholds(y_fc_val, fc_prob_val)

    if stage_callback:
        stage_callback("evaluating")
    # The test partition is first scored after the validation-selected threshold is frozen.
    fc_prob_test = model.predict_forecast_proba(X_test)
    now_prob_test = model.predict_nowcast_proba(X_test)
    now_pred_test = np.argmax(now_prob_test, axis=1)
    metrics = _compute_full_metrics(y_now_test, now_pred_test, y_fc_test, fc_prob_test, thresholds)

    os.makedirs(settings.model_dir, exist_ok=True)
    transformer_path = os.path.join(settings.model_dir, f"{model_id}.transformer.joblib")
    _atomic_save(transformer_path, lambda temp: joblib.dump(pipeline_state.transformer, temp))
    feature_manifest_path = _save_model_feature_manifest(
        model_id, feat_df, use_cols, req.input_configuration, feature_config, train_df, val_df, test_df,
        _preprocessor_manifest_metadata(pipeline_state.transformer), req.random_seed,
        architecture=req.model_type, training_configuration=req.model_dump(),
    )
    checkpoint_path = os.path.join(settings.model_dir, f"{model_id}.joblib")
    if stage_callback:
        stage_callback("saving artifacts")
    _atomic_save(checkpoint_path, lambda temp: joblib.dump(
        {"model": model, "feature_cols": use_cols, "thresholds": thresholds,
         "threshold_selection": threshold_report,
         "feature_manifest_path": feature_manifest_path,
         "target_definition": "pipeline_forecast_targets_v1",
         "forecast_horizons": ["1h", "3h", "6h", "12h", "24h"]}, temp))
    report_path = _persist_evaluation_report(
        model_id, metrics, use_cols, threshold_report,
        test_timestamps=test_merged["timestamp"], y_test=y_fc_test, p_test=fc_prob_test,
        validation_timestamps=val_merged["timestamp"], y_validation=y_fc_val, p_validation=fc_prob_val,
    )
    compatibility = {"target_definition": "pipeline_forecast_targets_v1",
                     "forecast_horizons": ["1h", "3h", "6h", "12h", "24h"],
                     "feature_schema_version": "causal-time-aware-v2"}
    artifact_refs = catalog_model_artifacts(model_id, settings.model_dir,
        [("model", checkpoint_path, "joblib"), ("preprocessor", transformer_path, "joblib"),
         ("evaluation", report_path, "json"), ("feature_manifest", feature_manifest_path, "json")], compatibility, job_id=job_id)

    record = ModelRecord(
        model_id=model_id, name=f"{req.model_type} ({req.input_configuration})",
        architecture=req.model_type, version="v1",
        created_at=str(pd.Timestamp.utcnow()),
        training_dataset=pipeline_state.dataset_meta.get("filename", "demo"),
        input_configuration=req.input_configuration,
        metrics=metrics, checkpoint_path=checkpoint_path,
        scaler_path=transformer_path, transformer_path=transformer_path, active=False, model_kind="classical",
        status=_training_status(),
        transformer_required=True,
        params={"n_features": len(use_cols), "train_duration_sec": round(train_duration, 3),
                "random_seed": req.random_seed, "training_configuration": req.model_dump(),
                "training_samples": int(len(train_merged)), "validation_samples": int(len(val_merged)),
                "test_samples": int(len(test_merged)), "evaluation_report_path": report_path,
                "thresholds": thresholds, "threshold_selection": threshold_report,
                "target_definition": "pipeline_forecast_targets_v1", "forecast_horizons": ["1h", "3h", "6h", "12h", "24h"],
                "feature_schema_version": "causal-time-aware-v2", "feature_manifest_path": feature_manifest_path,
                "source_data_fingerprint": data_fingerprint(pipeline_state.raw_df),
                "dataset_id": pipeline_state.dataset_meta.get("dataset_id"),
                "chronological_periods": {
                    "train": _period_record(train_df),
                    "validation": _period_record(val_df),
                    "test": _period_record(test_df),
                },
                "feature_manifest_artifact_id": f"{model_id}:feature_manifest", "artifact_ids": [a["artifact_id"] for a in artifact_refs]},
    )
    registry.register(record)

    return {"model_id": model_id, "status": _training_status(), "metrics": metrics, "history": [],
            "input_configuration": req.input_configuration, "artifact_refs": artifact_refs}


def _train_deep(req: TrainRequest, channel_cols, n_channels, model_id: str, progress_callback=None, stage_callback=None, job_id=None) -> dict:
    training_started = time.perf_counter()
    is_dlstm = req.model_type == "dlstm"
    raw_channels = [c.removesuffix("_scaled") for c in channel_cols] if is_dlstm else []
    window_channels = raw_channels if is_dlstm else channel_cols
    win_cfg = WindowConfig(sequence_length=req.sequence_length, channel_cols=window_channels)
    decomposition_cfg = DecompositionConfig(
        period_samples=req.decomposition_period_samples,
        robust=req.decomposition_robust,
    ) if is_dlstm else None
    if is_dlstm and req.sequence_length < decomposition_cfg.min_history:
        raise ValueError(
            f"DLSTM sequence_length={req.sequence_length} is shorter than the required "
            f"{decomposition_cfg.min_history} samples for two STL cycles."
        )
    source_fingerprint = data_fingerprint(pipeline_state.raw_df) if is_dlstm else ""
    decomposition_cache = HistoricalDecompositionCache() if is_dlstm else None
    training_scales = {
        channel: pipeline_state.transformer.reference_scale[channel]
        for channel in raw_channels
    } if is_dlstm else {}

    # Forecast origins belong to their partition, while input windows may use
    # genuinely available history from preceding partitions. Targets remain
    # split-scoped and purged at each partition's horizon tail.
    labeled = pipeline_state.labeled_df
    required = ["timestamp"] + window_channels + ["state"] + FORECAST_HORIZON_COLS
    source = labeled[required].copy()

    def _windows_for(partition):
        if partition.empty:
            return build_windows(partition.assign(**{c: pd.Series(dtype=float) for c in required if c != "timestamp"}), win_cfg)
        start = pd.to_datetime(partition.timestamp, utc=True).min()
        timestamps = pd.to_datetime(source.timestamp, utc=True)
        current = source.loc[timestamps.isin(pd.to_datetime(partition.timestamp, utc=True))].copy()
        context = source.loc[timestamps < start].tail(max(0, req.sequence_length - 1)).copy()
        X, y_now, y_fc, origins = build_windows(current, win_cfg, context_df=context)
        if is_dlstm:
            X, _ = decompose_historical_windows(
                X, origins, raw_channels, training_scales, decomposition_cfg,
                source_fingerprint=source_fingerprint, cache=decomposition_cache,
            )
        return X, y_now, y_fc, origins

    X_train, y_now_train, y_fc_train, _ = _windows_for(pipeline_state.train_df)
    X_val, y_now_val, y_fc_val, val_origins = _windows_for(pipeline_state.val_df)
    X_test, y_now_test, y_fc_test, test_origins = _windows_for(pipeline_state.test_df)

    if len(X_train) < 10 or len(X_test) < 5:
        raise ValueError("Not enough windows to train (need more demo data or a shorter sequence_length).")
    if len(X_val) < 5:
        raise ValueError("Not enough chronological validation windows; training cannot reuse training samples as validation data.")

    model_channels = n_channels * 3 if is_dlstm else n_channels
    if req.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of 'auto', 'cpu', or 'cuda'.")
    if req.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was explicitly requested but is unavailable.")
    training_device = ("cuda" if torch.cuda.is_available() else "cpu") if req.device == "auto" else req.device
    train_cfg = TrainConfig(
        model_type=req.model_type, sequence_length=req.sequence_length, batch_size=req.batch_size,
        epochs=req.epochs, learning_rate=req.learning_rate, dropout=req.dropout,
        alpha=req.alpha, beta=req.beta, class_weighting=req.class_weighting,
        n_channels=model_channels, device=training_device,
        random_seed=req.random_seed,
        early_stopping_patience=req.early_stopping_patience,
        minimum_improvement=req.minimum_improvement,
        scheduler_patience=req.scheduler_patience,
        deterministic=req.deterministic,
        dlstm_hidden_size=req.dlstm_hidden_size,
        dlstm_num_layers=req.dlstm_num_layers,
    )

    history = []
    if stage_callback:
        stage_callback("training")
    model, history = train_deep_model(
        X_train, y_now_train, y_fc_train, X_val, y_now_val, y_fc_val,
        config=train_cfg, progress_callback=progress_callback,
    )
    train_duration = time.perf_counter() - training_started

    fc_prob_val = _predict_forecast_probabilities(model, X_val, train_cfg.device)
    thresholds, threshold_report = _select_validation_thresholds(y_fc_val, fc_prob_val)
    # Freeze all validation-selected settings before touching the holdout predictions.
    if stage_callback:
        stage_callback("evaluating")
    model.eval()
    with torch.no_grad():
        now_logits, fc_logits = model(torch.tensor(X_test, dtype=torch.float32, device=train_cfg.device))
        now_prob_test = torch.softmax(now_logits, dim=1).detach().cpu().numpy()
        fc_prob_test = torch.sigmoid(fc_logits).detach().cpu().numpy()
    now_pred_test = np.argmax(now_prob_test, axis=1)
    metrics = _compute_full_metrics(y_now_test, now_pred_test, y_fc_test, fc_prob_test, thresholds)

    os.makedirs(settings.model_dir, exist_ok=True)
    transformer_path = os.path.join(settings.model_dir, f"{model_id}.transformer.joblib")
    _atomic_save(transformer_path, lambda temp: joblib.dump(pipeline_state.transformer, temp))
    manifest_channels = channel_cols
    decomposition_manifest = None
    if is_dlstm:
        manifest_channels = decomposition_feature_names(raw_channels)
        decomposition_manifest = {
            **decomposition_cfg.__dict__, "raw_channels": raw_channels,
            "instrument_names": {"solexs_flux": "SoLEXS", "helios_flux": "HEL1OS"},
            "lookback_steps": req.sequence_length,
            "minimum_history_samples": decomposition_cfg.min_history,
            "training_reference_scales": training_scales,
            "source_data_fingerprint": source_fingerprint,
            "cache_entries_precomputed": len(decomposition_cache),
            "endpoint_policy": "STL fitted independently on each complete historical window ending at forecast origin; endpoint component used; no future rows",
        }
    feature_manifest_path = _save_model_feature_manifest(
        model_id, labeled, manifest_channels, req.input_configuration, _pipeline_feature_config(),
        pipeline_state.train_df, pipeline_state.val_df, pipeline_state.test_df,
        _preprocessor_manifest_metadata(pipeline_state.transformer), train_cfg.random_seed,
        decomposition_manifest, architecture=req.model_type, training_configuration=req.model_dump(),
        input_window_length=req.sequence_length,
    )
    checkpoint_path = os.path.join(settings.model_dir, f"{model_id}.pt")
    if stage_callback:
        stage_callback("saving artifacts")
    training_metadata = getattr(model, "_training_metadata", {})
    training_summary = {key: value for key, value in training_metadata.items()
                        if key not in {"optimizer_state_dict", "scheduler_state_dict"}}
    def period_bounds(frame):
        stamps = pd.to_datetime(frame.timestamp, utc=True, errors="coerce").dropna()
        return {"start_utc": stamps.min().isoformat() if len(stamps) else None,
                "end_utc": stamps.max().isoformat() if len(stamps) else None}
    _atomic_save(checkpoint_path, lambda temp: torch.save(
        {"state_dict": model.state_dict(), "model_type": req.model_type, "n_channels": model_channels,
         "architecture_version": 2 if req.model_type in {"cnn_transformer", "dlstm"} else 1,
         "training_metadata": training_metadata,
         "model_config": {"sequence_length": req.sequence_length, "dropout": req.dropout,
                          "n_raw_channels": n_channels, "n_model_channels": model_channels,
                          "hidden_size": req.dlstm_hidden_size if is_dlstm else None,
                          "num_layers": req.dlstm_num_layers if is_dlstm else None,
                          "max_sequence_length": req.sequence_length if req.model_type == "cnn_transformer" else None},
         "chronological_periods": {"train": period_bounds(pipeline_state.train_df),
                                    "validation": period_bounds(pipeline_state.val_df),
                                    "test": period_bounds(pipeline_state.test_df)},
         "feature_manifest_path": feature_manifest_path,
         "source_data_fingerprint": data_fingerprint(pipeline_state.raw_df),
         "target_definition": "pipeline_forecast_targets_v1",
         "forecast_horizons": ["1h", "3h", "6h", "12h", "24h"],
         "thresholds": thresholds, "threshold_selection": threshold_report,
         "decomposition": decomposition_manifest}, temp))
    report_path = _persist_evaluation_report(
        model_id, metrics, manifest_channels, threshold_report,
        test_timestamps=test_origins, y_test=y_fc_test, p_test=fc_prob_test,
        validation_timestamps=val_origins, y_validation=y_fc_val, p_validation=fc_prob_val,
    )
    compatibility = {"target_definition": "pipeline_forecast_targets_v1",
                     "forecast_horizons": ["1h", "3h", "6h", "12h", "24h"],
                     "feature_schema_version": "causal-time-aware-v2"}
    artifact_refs = catalog_model_artifacts(model_id, settings.model_dir,
        [("model", checkpoint_path, "torch"), ("preprocessor", transformer_path, "joblib"),
         ("evaluation", report_path, "json"), ("feature_manifest", feature_manifest_path, "json")], compatibility, job_id=job_id)

    record = ModelRecord(
        model_id=model_id, name=f"{req.model_type} ({req.input_configuration})",
        architecture=req.model_type, version="v2" if req.model_type in {"cnn_transformer", "dlstm"} else "v1",
        created_at=str(pd.Timestamp.utcnow()),
        training_dataset=pipeline_state.dataset_meta.get("filename", "demo"),
        input_configuration=req.input_configuration,
        metrics=metrics, checkpoint_path=checkpoint_path,
        scaler_path=transformer_path, transformer_path=transformer_path, active=False, model_kind="deep",
        status=_training_status(),
        transformer_required=True,
        params={"sequence_length": req.sequence_length, "epochs": req.epochs,
                "training_configuration": req.model_dump(),
                "n_channels": model_channels, "n_raw_channels": n_channels, "random_seed": train_cfg.random_seed,
                "training_samples": int(len(X_train)), "validation_samples": int(len(X_val)),
                "test_samples": int(len(X_test)), "device": training_device,
                "evaluation_report_path": report_path,
                "train_duration_sec": round(train_duration, 3),
                "training_metadata": training_summary,
                "thresholds": thresholds, "threshold_selection": threshold_report,
                "decomposition": decomposition_manifest,
                "target_definition": "pipeline_forecast_targets_v1", "forecast_horizons": ["1h", "3h", "6h", "12h", "24h"],
                "feature_schema_version": "causal-time-aware-v2", "feature_manifest_path": feature_manifest_path,
                "source_data_fingerprint": data_fingerprint(pipeline_state.raw_df),
                "dataset_id": pipeline_state.dataset_meta.get("dataset_id"),
                "chronological_periods": {
                    "train": _period_record(pipeline_state.train_df),
                    "validation": _period_record(pipeline_state.val_df),
                    "test": _period_record(pipeline_state.test_df),
                },
                "feature_manifest_artifact_id": f"{model_id}:feature_manifest", "artifact_ids": [a["artifact_id"] for a in artifact_refs]},
    )
    registry.register(record)

    return {"model_id": model_id, "status": _training_status(), "metrics": metrics, "history": history,
            "input_configuration": req.input_configuration, "artifact_refs": artifact_refs}


def _predict_forecast_probabilities(model, X, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float32, device=device))[1]
        return torch.sigmoid(logits).detach().cpu().numpy()


def _select_validation_thresholds(y_true, probabilities, min_positive_support=2):
    """Select per-horizon max-TSS thresholds from validation only."""
    labels = np.asarray(y_true, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    thresholds, report = [], {}
    for i, horizon in enumerate(["1h", "3h", "6h", "12h", "24h"][:labels.shape[1]]):
        known = np.isfinite(labels[:, i]) & np.isfinite(probs[:, i])
        y, p = labels[known, i].astype(int), probs[known, i]
        positives, negatives = int((y == 1).sum()), int((y == 0).sum())
        if positives < min_positive_support or negatives == 0:
            threshold = 0.5
            reason = f"uncertain: validation support is {positives} positive and {negatives} negative windows"
            method = "uncertain_fallback_0.5"
        else:
            from ml.evaluation.metrics import binary_confusion_counts, true_skill_statistic
            candidates = np.unique(np.r_[0.0, p, 1.0])
            scored = []
            for candidate in candidates:
                counts = binary_confusion_counts(y, p >= candidate)
                scored.append((true_skill_statistic(**counts), -abs(float(candidate) - 0.5), float(candidate)))
            threshold = max(scored)[2]
            reason, method = None, "validation_max_tss"
        thresholds.append(float(threshold))
        validation_frame = pipeline_state.val_df
        validation_timestamps = (
            pd.to_datetime(validation_frame.timestamp, utc=True, errors="coerce").dropna()
            if validation_frame is not None and not validation_frame.empty else pd.Series(dtype="datetime64[ns, UTC]")
        )
        report[horizon] = {
            "threshold": float(threshold), "method": method, "reason": reason,
            "validation_positive_windows": positives, "validation_negative_windows": negatives,
            "selection_period": {
                "start_utc": validation_timestamps.min().isoformat() if len(validation_timestamps) else None,
                "end_utc": validation_timestamps.max().isoformat() if len(validation_timestamps) else None,
            },
        }
    return thresholds, report


def _compute_full_metrics(y_now_true, y_now_pred, y_fc_true, y_fc_prob, thresholds=None) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    now_acc = float(accuracy_score(y_now_true, y_now_pred))
    now_f1_macro = float(f1_score(y_now_true, y_now_pred, average="macro", zero_division=0))
    cm = multiclass_confusion_matrix(y_now_true, y_now_pred, n_classes=4)

    per_horizon = {}
    y_fc_true = np.asarray(y_fc_true, dtype=float)
    y_fc_prob = np.asarray(y_fc_prob, dtype=float)
    if y_fc_true.shape != y_fc_prob.shape or y_fc_true.ndim != 2:
        raise ValueError("Forecast targets and probabilities must be aligned 2D arrays.")
    for i, h in enumerate(["1h", "3h", "6h", "12h", "24h"][:y_fc_true.shape[1]]):
        known = np.isfinite(y_fc_true[:, i])
        if not known.any():
            per_horizon[h] = {
                "sample_count": 0, "positive_count": 0, "negative_count": 0,
                "positive_prevalence": None, "threshold": 0.5,
                "tp": 0, "fp": 0, "tn": 0, "fn": 0,
                "false_alarms": 0, "missed_positive_windows": 0,
                "accuracy": None, "precision": None, "recall": None,
                "pod": None, "f1": None, "false_alarm_rate": None, "far": None,
                "tss": None, "hss": None, "roc_auc": None, "pr_auc": None,
                "brier_score": None,
                "undefined_metric_reasons": {
                    "all": "No known labels in the chronological test split for this horizon."
                },
            }
        else:
            threshold = thresholds[i] if thresholds and i < len(thresholds) else 0.5
            per_horizon[h] = evaluate_binary_forecast(y_fc_true[known, i], y_fc_prob[known, i], threshold=threshold)

    tss_values = [m["tss"] for m in per_horizon.values() if m["tss"] is not None]
    hss_values = [m["hss"] for m in per_horizon.values() if m["hss"] is not None]
    precision_values = [m["precision"] for m in per_horizon.values() if m["precision"] is not None]
    recall_values = [m["recall"] for m in per_horizon.values() if m["recall"] is not None]
    f1_values = [m["f1"] for m in per_horizon.values() if m["f1"] is not None]
    far_values = [m["far"] for m in per_horizon.values() if m["far"] is not None]

    return {
        "evaluation_split": "chronological_test",
        "evaluation_samples": int(len(y_now_true)),
        "evaluation_note": "Synthetic/demo results are demonstrations, not observational validation." if _training_status() == "demo_only" else "Non-demo dataset; observational provenance has not been independently validated.",
        "nowcast_accuracy": now_acc,
        "nowcast_f1_macro": now_f1_macro,
        "nowcast_confusion_matrix": cm,
        "forecast_per_horizon": per_horizon,
        "tss_mean": float(np.mean(tss_values)) if tss_values else None,
        "hss_mean": float(np.mean(hss_values)) if hss_values else None,
        "tss": float(np.mean(tss_values)) if tss_values else None,
        "hss": float(np.mean(hss_values)) if hss_values else None,
        "precision": float(np.mean(precision_values)) if precision_values else None,
        "recall": float(np.mean(recall_values)) if recall_values else None,
        "f1": float(np.mean(f1_values)) if f1_values else None,
        "false_alarm_rate": float(np.mean(far_values)) if far_values else None,
        "pr_auc": float(np.mean([m["pr_auc"] for m in per_horizon.values() if m["pr_auc"] is not None])) if any(m["pr_auc"] is not None for m in per_horizon.values()) else None,
    }


def _persist_evaluation_report(model_id: str, metrics: dict, feature_schema: list[str], threshold_report=None,
                               test_timestamps=None, y_test=None, p_test=None,
                               validation_timestamps=None, y_validation=None, p_validation=None) -> str:
    """Persist the actual test-split evaluation alongside registered model artifacts."""
    def period(frame):
        if frame is None or frame.empty or "timestamp" not in frame:
            return {"start": None, "end": None, "samples": 0}
        stamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
        return {"start": stamps.min().isoformat() if len(stamps) else None,
                "end": stamps.max().isoformat() if len(stamps) else None,
                "samples": int(len(frame))}

    meta = pipeline_state.dataset_meta or {}
    prediction_data = {"available": False, "reason": "Out-of-sample prediction pairs were not supplied by this training path."}
    if test_timestamps is not None and y_test is not None and p_test is not None:
        labels = np.asarray(y_test, dtype=float)
        probabilities = np.asarray(p_test, dtype=float)
        stamps = pd.to_datetime(test_timestamps, utc=True, errors="coerce", format="mixed")
        if labels.ndim == 2 and probabilities.shape == labels.shape and len(stamps) == len(labels):
            if len(labels) > MAX_STORED_EVALUATION_SAMPLES:
                prediction_data = {
                    "available": False,
                    "reason": f"Test partition has {len(labels)} samples, above the bounded storage limit of {MAX_STORED_EVALUATION_SAMPLES}.",
                    "max_samples": MAX_STORED_EVALUATION_SAMPLES,
                }
            elif stamps.isna().any() or stamps.duplicated().any():
                prediction_data = {"available": False, "reason": "Test prediction timestamps are invalid or duplicated."}
            else:
                prediction_horizons = {}
                for index, horizon in enumerate(("1h", "3h", "6h", "12h", "24h")[:labels.shape[1]]):
                    known = np.isfinite(labels[:, index])
                    if np.any(~np.isin(labels[known, index], [0, 1])):
                        raise ValueError(f"Test labels for {horizon} must be binary or unknown.")
                    if np.any(~np.isfinite(probabilities[:, index])) or np.any((probabilities[:, index] < 0) | (probabilities[:, index] > 1)):
                        raise ValueError(f"Test probabilities for {horizon} must be finite and between zero and one.")
                    prediction_horizons[horizon] = {
                        "timestamps_utc": [stamp.isoformat() for stamp in stamps[known]],
                        "labels": labels[known, index].astype(int).tolist(),
                        "probabilities": probabilities[known, index].astype(float).tolist(),
                        "unknown_label_count": int((~known).sum()),
                    }
                prediction_data = {"available": True, "sample_count": int(len(labels)),
                                   "evaluation_horizons": prediction_horizons}
        else:
            prediction_data = {"available": False, "reason": "Test prediction arrays do not have matching row and horizon shapes."}
    validation_data = {"available": False, "reason": "Out-of-sample validation prediction pairs were not supplied."}
    if validation_timestamps is not None and y_validation is not None and p_validation is not None:
        labels = np.asarray(y_validation, dtype=float)
        probabilities = np.asarray(p_validation, dtype=float)
        stamps = pd.to_datetime(validation_timestamps, utc=True, errors="coerce", format="mixed")
        if labels.ndim == 2 and probabilities.shape == labels.shape and len(stamps) == len(labels) and len(labels) <= MAX_STORED_EVALUATION_SAMPLES and not stamps.isna().any() and not stamps.duplicated().any():
            horizons = {}
            for index, horizon in enumerate(("1h", "3h", "6h", "12h", "24h")[:labels.shape[1]]):
                known = np.isfinite(labels[:, index])
                if np.any(~np.isin(labels[known, index], [0, 1])) or np.any(~np.isfinite(probabilities[:, index])) or np.any((probabilities[:, index] < 0) | (probabilities[:, index] > 1)):
                    raise ValueError(f"Validation pairs for {horizon} contain invalid labels or probabilities.")
                horizons[horizon] = {"timestamps_utc": [stamp.isoformat() for stamp in stamps[known]],
                                     "labels": labels[known, index].astype(int).tolist(),
                                     "probabilities": probabilities[known, index].astype(float).tolist(),
                                     "unknown_label_count": int((~known).sum())}
            validation_data = {"available": True, "sample_count": int(len(labels)), "evaluation_horizons": horizons}
        else:
            validation_data = {"available": False, "reason": "Validation prediction arrays are invalid, unaligned, or exceed the storage bound."}
    report = {
        "report_version": 3,
        "model_id": model_id,
        "model_status": _training_status(),
        "evaluation_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "evaluation_split": "chronological_test",
        "decision_thresholds": {k: v.get("threshold") for k, v in (threshold_report or {}).items()},
        "threshold_selection": threshold_report or "fixed a priori; no holdout tuning",
        "label_definition": "Existing coverage-aware pipeline forecast labels at (issue_time, issue_time + horizon]; unknown labels are excluded.",
        "feature_schema": list(feature_schema),
        "data_provenance": meta,
        "periods": {"train": period(pipeline_state.train_df),
                    "validation": period(pipeline_state.val_df),
                    "holdout": period(pipeline_state.test_df)},
        "metrics": metrics,
        "test_predictions": prediction_data,
        "validation_predictions": validation_data,
        "scientific_limitations": [
            "Demo results are not observational validation." if _training_status() == "demo_only" else
            "Observational provenance evidence must be reviewed with source files and product documentation."
        ],
    }
    os.makedirs(settings.model_dir, exist_ok=True)
    report_path = os.path.join(settings.model_dir, f"{model_id}.evaluation.json")
    tmp_path = report_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, report_path)
    return report_path
_BASELINE_SEEDS = [
    ("climatology", "Climatology No-Skill Reference", {"predicts": "train-split class priors and flare rates"}),
    ("flux_rule", "Flux Threshold Rule (Level + Trend)", {"max_depth": 3, "features": ["flux_level", "flux_slope_5m"]}),
    ("hard_soft_ratio_rule", "Hard/Soft Ratio Rule (Neupert Test)", {"max_depth": 3, "features": ["log_hard_soft_ratio", "hard_soft_ratio_slope_5m"]}),
    ("two_feature_logreg", "Flux + Trend Logistic Regression", {"features": ["flux_level", "flux_slope_5m"]}),
    ("gradient_boosting", "Gradient Boosting (Histogram) Baseline", {"max_depth": 6, "max_iter": 200}),
]


def seed_baseline_models():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "ml", "baseline_seed_metrics.json")
    if not os.path.exists(path):
        print(f"[seed_baseline_models] metrics file not found: {path}")
        return
    with open(path) as f:
        seed = json.load(f)
    meta = seed.get("_meta", {})
    dataset = f"{meta.get('dataset', 'benchmark')} (seed {meta.get('seed')}, {meta.get('days')} days)"
    for arch, name, params in _BASELINE_SEEDS:
        model_id = f"{arch}_v1"
        if arch not in seed or registry.get(model_id):
            continue
        registry.register(ModelRecord(
            model_id=model_id, name=name, architecture=arch, version="v1.0",
            created_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            training_dataset=dataset, input_configuration="fusion",
            metrics=seed[arch], checkpoint_path="", scaler_path="", active=False,
            model_kind="classical",
            params={**params, "reference_only": True, "benchmark": meta},
        ))
def seed_initial_models():
    """Compatibility shim: architectures are not seeded as trained records."""
    return


def list_models():
    public = []
    for rec in registry.list_all():
        metadata = {key: rec.get(key) for key in (
            "model_id", "name", "architecture", "version", "created_at", "training_dataset",
            "input_configuration", "metrics", "active", "model_kind", "status",
        )}
        params = rec.get("params") or {}
        metadata["params"] = {key: params[key] for key in (
            "reference_only", "target_definition", "forecast_horizons", "feature_schema_version",
            "thresholds", "threshold_selection", "n_features", "sequence_length", "random_seed",
        ) if key in params}
        metadata["supported_forecast_horizons"] = params.get("forecast_horizons", [])
        try:
            if rec.get("status") not in {"trained", "demo_only"}:
                raise ValueError("Model has not been trained.")
            verify_model_artifact_set(rec, settings.model_dir)
            registry.validate_artifacts(rec["model_id"], rec)
            metadata["model_ready"] = True
            metadata["availability_reason"] = None
        except Exception:
            metadata["model_ready"] = False
            metadata["availability_reason"] = "Model is not trained or its required artifacts are unavailable or incompatible."
        public.append(metadata)
    return public


def get_model_metrics(model_id: str):
    rec = registry.get(model_id)
    if not rec:
        raise KeyError(f"Model {model_id} not found")
    return {
        "model_id": model_id,
        "version": rec.get("version"),
        "status": rec.get("status"),
        "target_definition": (rec.get("params") or {}).get("target_definition"),
        "forecast_horizons": (rec.get("params") or {}).get("forecast_horizons", []),
        "thresholds": (rec.get("params") or {}).get("thresholds", []),
        "threshold_selection": (rec.get("params") or {}).get("threshold_selection", {}),
        "evaluation_report_available": bool((rec.get("params") or {}).get("evaluation_report_path")),
        "metrics": rec.get("metrics", {}),
    }


def get_model_evaluation(model_id: str, horizon: str, threshold=None):
    """Recompute holdout metrics from persisted out-of-sample pairs only."""
    horizons = ("1h", "3h", "6h", "12h", "24h")
    if horizon not in horizons:
        raise ValueError("Unsupported forecast horizon.")
    rec = registry.get(model_id)
    if not rec:
        raise KeyError(f"Model {model_id} not found")
    params = rec.get("params") or {}
    report_path = params.get("evaluation_report_path")
    if not report_path:
        raise ValueError("Stored test prediction pairs are unavailable for this legacy model.")
    try:
        artifact = verify_catalog_artifact(f"{model_id}:evaluation", settings.model_dir)
        safe_path = os.path.join(settings.model_dir, artifact.storage_key)
        with open(safe_path, "r", encoding="utf-8") as stream:
            report = json.load(stream)
    except Exception as exc:
        raise ValueError("The model's evaluation artifact is missing or failed integrity verification.") from exc
    pairs = report.get("test_predictions") or {}
    if not pairs.get("available"):
        raise ValueError(pairs.get("reason") or "Stored test prediction pairs are unavailable.")
    payload = (pairs.get("evaluation_horizons") or {}).get(horizon)
    if not payload:
        raise ValueError(f"No stored test prediction pairs are available for horizon {horizon}.")
    labels, probs, stamps = payload.get("labels"), payload.get("probabilities"), payload.get("timestamps_utc")
    if not isinstance(labels, list) or not isinstance(probs, list) or not isinstance(stamps, list) or not (len(labels) == len(probs) == len(stamps)) or not labels or len(labels) > MAX_STORED_EVALUATION_SAMPLES:
        raise ValueError("Stored test prediction pairs failed validation.")
    selected = threshold
    thresholds = params.get("thresholds") or report.get("decision_thresholds") or {}
    validation_threshold = thresholds.get(horizon, 0.5) if isinstance(thresholds, dict) else thresholds[horizons.index(horizon)]
    if selected is None:
        selected = validation_threshold
    selected = float(selected)
    if not np.isfinite(selected) or not 0 <= selected <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    metrics = evaluate_binary_forecast(labels, probs, selected)
    from ml.evaluation.metrics import reliability_diagram
    reliability = reliability_diagram(labels, probs)
    times = pd.to_datetime(stamps, utc=True, errors="coerce", format="mixed")
    if times.isna().any():
        raise ValueError("Stored evaluation timestamps failed validation.")
    return {
        "model_id": model_id, "model_name": rec.get("name"), "version": rec.get("version"),
        "model_status": rec.get("status"), "dataset": report.get("data_provenance", {}),
        "evaluation_split": report.get("evaluation_split", "chronological_test"),
        "horizon": horizon, "period_start_utc": times.min().isoformat(), "period_end_utc": times.max().isoformat(),
        "observation_count": int(pairs.get("sample_count", len(labels))),
        "known_label_count": len(labels), "unknown_label_count": int(payload.get("unknown_label_count", 0)),
        "threshold_used_for_test": selected,
        "validation_selected_threshold": float(validation_threshold),
        "threshold_selection_provenance": (report.get("threshold_selection") or {}).get(horizon) if isinstance(report.get("threshold_selection"), dict) else report.get("threshold_selection"),
        "threshold_exploratory": threshold is not None,
        "metrics": metrics, "reliability": reliability,
        "scientific_limitations": report.get("scientific_limitations", []),
    }


def get_model_readiness(model_id: str):
    rec = registry.get(model_id)
    if not rec:
        raise KeyError(f"Model {model_id} not found")
    try:
        if rec.get("status") not in {"trained", "demo_only"}:
            raise ValueError("Model has not been trained.")
        verify_model_artifact_set(rec, settings.model_dir)
        registry.validate_artifacts(model_id, rec)
        return {"model_id": model_id, "version": rec.get("version"), "ready": True,
                "status": "ready", "supported_forecast_horizons": (rec.get("params") or {}).get("forecast_horizons", [])}
    except Exception:
        return {"model_id": model_id, "version": rec.get("version"), "ready": False,
                "status": "unavailable", "message": "Model is not trained or its artifacts are unavailable or incompatible."}


def set_active_model(model_id: str):
    rec = registry.get(model_id)
    if rec and rec.get("status") in {"trained", "demo_only"}:
        verify_model_artifact_set(rec, settings.model_dir)
    registry.set_active(model_id)
    return registry.get(model_id)


def get_active_model():
    return registry.get_active()


def run_fusion_experiment(base_req: TrainRequest) -> dict:
    """Report readiness; never train or compare without a valid paired cohort."""
    meta = pipeline_state.dataset_meta or {}
    source = str(meta.get("source", "unknown")).lower()
    blockers = ["No verified paired real-data cohort with independent coverage-aware reference labels is available."]
    if source == "demo" or settings.data_mode == "demo":
        blockers.append("Synthetic/demo observations cannot support an observational fusion conclusion.")
    blockers.extend([
        "The current SoLEXS adapter exposes native counts/s rather than calibrated flux comparable to the HEL1OS channel.",
        "The HEL1OS product adapter fails closed until its product manual and representative files are verified.",
        "No five-seed shared chronological evaluation has been run on identical paired test examples.",
    ])
    return {
        "status": "blocked", "completed": False,
        "scientific_conclusion": "cannot_yet_be_determined", "blockers": blockers,
        "requested_seed_count": 5, "completed_seed_count": 0,
        "configurations": ["solexs_only", "helios_only", "fusion", "persistence", "climatology"],
        "results": {}, "evaluation_split": None,
    }
