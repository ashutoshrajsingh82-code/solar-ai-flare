import joblib
import torch
import numpy as np
import pandas as pd
import time
import uuid
from typing import Optional

from app.core.config import settings
from app.services.pipeline_state import pipeline_state
from app.services.artifact_catalog import verify_model_artifact_set
from app.services.model_service import registry
from ml.models.deep_models import DEEP_MODEL_BUILDERS
from ml.inference.predictor import _format_prediction, risk_level_from_probability
from ml.preprocessing.transform import FluxTransformer
from ml.features.engineering import add_rolling_features, FeatureConfig
from ml.features.manifest import load_feature_manifest, validate_feature_manifest
from ml.features.decomposition import DecompositionConfig, decompose_historical_window
from ml.inference.input_validation import normalize_prediction_rows

_loaded_cache = {}


class ModelUnavailableError(RuntimeError):
    """A requested or default model cannot safely serve predictions."""


class ModelNotFoundError(LookupError):
    """The requested model identifier is not registered."""


def _load_classical(model_id: str, checkpoint_path: str):
    if model_id in _loaded_cache:
        return _loaded_cache[model_id]
    obj = joblib.load(checkpoint_path)
    _loaded_cache[model_id] = obj
    return obj


import os

def _load_deep(model_id: str, checkpoint_path: str, arch: str = "cnn_lstm_fusion", n_channels: int = 2):
    if model_id in _loaded_cache:
        return _loaded_cache[model_id]
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise ValueError("The selected model checkpoint is missing. Train and activate a compatible model.")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_type = ckpt.get("model_type", arch)
    n_chan = ckpt.get("n_channels", n_channels)
    if model_type != arch or model_type not in DEEP_MODEL_BUILDERS:
        raise ValueError("The selected model checkpoint metadata is incompatible with its registry entry.")
    if arch in {"cnn_transformer", "dlstm"} and ckpt.get("architecture_version") != 2:
        raise ValueError(f"Legacy {arch} checkpoint has no compatible architecture version; retraining is required.")
    model_config = ckpt.get("model_config", {})
    builder_kwargs = {key: model_config[key] for key in ("hidden_size", "num_layers", "max_sequence_length")
                      if model_config.get(key) is not None}
    model = DEEP_MODEL_BUILDERS[model_type](n_channels=n_chan, **builder_kwargs)
    if "state_dict" not in ckpt:
        raise ValueError("The selected model checkpoint has no trained weights.")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _loaded_cache[model_id] = model
    return model


def resolve_model(model_id: Optional[str] = None) -> dict:
    if model_id:
        rec = registry.get(model_id)
        if not rec:
            raise ModelNotFoundError(f"Model {model_id} is not available.")
        if rec.get("status") == "untrained":
            raise ModelUnavailableError(f"Model {model_id} is registered but has not been trained.")
        try:
            if rec.get("status") in {"trained", "demo_only"}:
                verify_model_artifact_set(rec, settings.model_dir)
            registry.validate_artifacts(model_id, rec)
        except Exception as exc:
            raise ModelUnavailableError(f"Model {model_id} has missing, corrupted, or incompatible artifacts.") from exc
        return rec
    rec = registry.get_active()
    if not rec:
        raise ModelUnavailableError("No active trained model is available.")
    try:
        if rec.get("status") in {"trained", "demo_only"}:
            verify_model_artifact_set(rec, settings.model_dir)
        registry.validate_artifacts(rec["model_id"], rec)
    except Exception as exc:
        raise ModelUnavailableError(f"Active model {rec['model_id']} has missing, corrupted, or incompatible artifacts.") from exc
    return rec


def predict_from_raw_window(rows: list, model_id: Optional[str] = None) -> dict:
    """
    Accepts a list of raw row dicts (timestamp, solexs_flux, helios_flux),
    runs them through the FITTED transformer + feature engineering, and
    produces a prediction using the resolved model.
    """
    rec = resolve_model(model_id)
    try:
        feature_manifest = _load_record_feature_manifest(rec)
    except Exception as exc:
        raise ModelUnavailableError("The selected model feature metadata is missing, corrupted, or incompatible.") from exc
    cadence = feature_manifest.get("feature_config", {}).get("cadence_seconds", 60)
    df = normalize_prediction_rows(rows, expected_cadence_seconds=cadence)
    transformer_path = rec.get("transformer_path") or rec.get("scaler_path")
    if rec.get("transformer_required", True):
        if not transformer_path or not os.path.isfile(transformer_path):
            raise ModelUnavailableError("The selected model preprocessing transformer is missing.")
        try:
            transformer: FluxTransformer = joblib.load(transformer_path)
        except Exception as exc:
            raise ModelUnavailableError("The selected model preprocessing transformer could not be loaded.") from exc
    else:
        transformer = None
    if transformer is None:
        raise ValueError("The selected model has no compatible preprocessing transformer.")
    transformed = transformer.transform(df)
    manifest = feature_manifest
    feature_config = FeatureConfig(**manifest.get("feature_config", {}))
    featured = add_rolling_features(transformed, feature_config)
    selected = validate_feature_manifest(manifest, featured, rec.get("input_configuration"))
    if rec.get("model_kind") == "classical":
        checkpoint = _load_classical(rec["model_id"], rec["checkpoint_path"])
        if checkpoint.get("feature_cols") != manifest["ordered_feature_names"]:
            raise ValueError("Checkpoint feature order does not match its feature manifest.")
        required = selected.iloc[-1].to_numpy(dtype=float)
        if not np.isfinite(required).all():
            raise ValueError("Latest observation lacks sufficient valid history for the model's required features.")
    elif not manifest.get("decomposition"):
        channel_cols = manifest["ordered_feature_names"]
        latest_window = featured[channel_cols].tail(rec.get("params", {}).get("sequence_length", 30)).to_numpy(dtype=float)
        if not np.isfinite(latest_window).all():
            raise ValueError("Prediction window contains missing required measurements; no values were imputed.")

    started = time.perf_counter()
    try:
        result = _predict_with_record(rec, featured)
    except (ValueError, KeyError):
        raise
    except Exception as exc:
        raise ModelUnavailableError("The selected model failed to produce a prediction.") from exc
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    prediction_id = str(uuid.uuid4())
    origin_time = df["timestamp"].iloc[-1]
    horizons = ["1h", "3h", "6h", "12h", "24h"]
    horizon_minutes = {"1h": 60, "3h": 180, "6h": 360, "12h": 720, "24h": 1440}
    result["prediction_id"] = prediction_id
    result["model_id"] = rec["model_id"]
    result["model_version"] = rec.get("version", "unknown")
    result["score_semantics"] = "Uncalibrated model classification score; calibration has not been validated."
    result["target_definition"] = rec.get("params", {}).get("target_definition", "pipeline_forecast_targets_v1")
    result["forecast_horizons"] = horizons
    result["prediction_timestamp_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    result["forecast_origin_timestamp_utc"] = origin_time.isoformat()
    result["forecast_periods_utc"] = {
        horizon: {"start_exclusive_utc": origin_time.isoformat(),
                  "end_inclusive_utc": (origin_time + pd.Timedelta(minutes=minutes)).isoformat()}
        for horizon, minutes in horizon_minutes.items()
    }
    result["input_window"] = {
        "row_count": int(len(df)),
        "start_timestamp_utc": df["timestamp"].iloc[0].isoformat(),
        "end_timestamp_utc": origin_time.isoformat(),
    }
    result["inference_latency_ms"] = round(elapsed_ms, 3)
    result["model"] = rec["architecture"]
    result["model_status"] = rec.get("status")
    selected_channels = ["solexs_flux"] if rec["input_configuration"] == "solexs_only" else (
        ["helios_flux"] if rec["input_configuration"] == "helios_only" else ["solexs_flux", "helios_flux"])
    result["data_quality"] = {
        "completeness": float(1.0 - df[selected_channels].isna().mean().mean()),
        "warnings": list(df.attrs.get("data_quality_warnings", [])),
        "missing_measurements_by_channel": {c: int(df[c].isna().sum()) for c in selected_channels},
        "cadence_seconds_expected": cadence,
    }
    result["input"] = {
        "solexs_flux": _finite_or_none(df["solexs_flux"].iloc[-1]),
        "helios_flux": _finite_or_none(df["helios_flux"].iloc[-1]),
        "hard_soft_ratio": _finite_or_none(featured["hard_soft_ratio"].iloc[-1]) if "hard_soft_ratio" in featured.columns else None,
    }
    result["timestamp"] = str(df["timestamp"].iloc[-1])
    return result


def _finite_or_none(value):
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _predict_with_record(rec: dict, featured_window: pd.DataFrame) -> dict:
    checkpoint = rec.get("checkpoint_path")
    if not checkpoint or not os.path.isfile(checkpoint):
        raise ValueError("The selected model checkpoint is missing.")
    if rec.get("status") == "untrained":
        raise ValueError("The selected model is untrained and unavailable for prediction.")
    manifest = _load_record_feature_manifest(rec)
    selected = validate_feature_manifest(manifest, featured_window, rec.get("input_configuration"))
    if rec["model_kind"] == "classical":
        obj = _load_classical(rec["model_id"], rec["checkpoint_path"])
        model, feature_cols = obj["model"], obj["feature_cols"]
        if feature_cols != manifest["ordered_feature_names"]:
            raise ValueError("Checkpoint feature order does not match its feature manifest.")
        row = selected.iloc[[-1]].to_numpy(dtype=float)
        if not np.isfinite(row).all():
            raise ValueError("Latest observation lacks valid history for required model features.")
        now_prob = model.predict_nowcast_proba(row)[0]
        fc_prob = model.predict_forecast_proba(row)[0]
    else:
        seq_len = rec["params"].get("sequence_length", 30)
        n_channels = rec["params"].get("n_channels", 2)
        model = _load_deep(rec["model_id"], rec.get("checkpoint_path", ""), arch=rec.get("architecture", "cnn_lstm_fusion"), n_channels=n_channels)
        decomposition = manifest.get("decomposition")
        if decomposition:
            raw_channels = decomposition["raw_channels"]
            history = featured_window[raw_channels].tail(seq_len).to_numpy(dtype=float)
            if len(history) < seq_len:
                raise ValueError(f"DLSTM requires {seq_len} historical observations; received {len(history)}.")
            dcfg = DecompositionConfig(
                period_samples=decomposition["period_samples"],
                min_cycles=decomposition["min_cycles"],
                robust=decomposition["robust"],
                log_transform=decomposition["log_transform"],
            )
            window = decompose_historical_window(
                history, raw_channels, decomposition["training_reference_scales"], dcfg
            )
        else:
            channel_cols = manifest["ordered_feature_names"]
            window = featured_window[channel_cols].tail(seq_len).values.astype(np.float32)
        if len(window) < seq_len:
            raise ValueError(f"Prediction requires {seq_len} observations; received {len(window)}.")
        if window.shape[1] != n_channels or not np.isfinite(window).all():
            raise ValueError("Prediction window is missing required finite model inputs.")
        x = torch.tensor(window[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            now_logits, fc_logits = model(x)
            now_prob = torch.softmax(now_logits, dim=1).numpy()[0]
            fc_prob = torch.sigmoid(fc_logits).numpy()[0]

    result = _format_prediction(now_prob, fc_prob)
    thresholds = rec.get("params", {}).get("thresholds")
    if thresholds:
        horizons = ["1h", "3h", "6h", "12h", "24h"]
        result["forecast_thresholds"] = dict(zip(horizons, thresholds))
        result["forecast_decisions"] = {
            horizon: bool(result["forecast"].get(horizon, 0.0) >= threshold)
            for horizon, threshold in zip(horizons, thresholds)
        }
    return result


def make_simulator_predict_fn(model_id: Optional[str] = None):
    """Returns a callable(window_df) -> prediction dict, bound to a resolved model, for the live simulator."""
    rec = resolve_model(model_id)
    transformer_path = rec.get("transformer_path") or rec.get("scaler_path")
    if not transformer_path or not os.path.isfile(transformer_path):
        raise ValueError("The selected model preprocessing transformer is missing.")
    transformer = joblib.load(transformer_path)
    manifest = _load_record_feature_manifest(rec)

    def _fn(window_df: pd.DataFrame) -> dict:
        transformed = transformer.transform(window_df)
        featured = add_rolling_features(transformed, FeatureConfig(**manifest.get("feature_config", {})))
        validate_feature_manifest(manifest, featured, rec.get("input_configuration"))
        result = _predict_with_record(rec, featured)
        result["model"] = rec["architecture"]
        return result

    return _fn


def _load_record_feature_manifest(rec):
    path = rec.get("params", {}).get("feature_manifest_path")
    artifact_id = rec.get("params", {}).get("feature_manifest_artifact_id")
    if not path or not os.path.isfile(path) or not artifact_id:
        raise ValueError("The model has no compatible feature manifest; retrain it with the current feature schema.")
    from app.services.artifact_catalog import verify_catalog_artifact
    catalogued = verify_catalog_artifact(artifact_id, settings.model_dir)
    if os.path.basename(path) != catalogued.storage_key:
        raise ValueError("Feature manifest path does not match its integrity-catalog entry.")
    manifest = load_feature_manifest(path)
    if manifest.get("input_configuration") != rec.get("input_configuration"):
        raise ValueError("Model and feature manifest instrument configurations do not match.")
    return manifest
