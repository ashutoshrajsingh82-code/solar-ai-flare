import os
import time
import json
import numpy as np
import pandas as pd
import joblib
import torch
from typing import Optional

from app.core.config import settings
from app.services.pipeline_state import pipeline_state
from app.schemas.schemas import TrainRequest

from ml.models.baseline import MODEL_BUILDERS
from ml.models.registry import ModelRegistry, ModelRecord, new_model_id
from ml.training.train import train_deep_model, TrainConfig
from ml.features.windowing import build_windows, WindowConfig
from ml.evaluation.metrics import evaluate_binary_forecast, multiclass_confusion_matrix

DEEP_MODEL_TYPES = {"cnn_1d", "lstm", "cnn_lstm_fusion", "cnn_transformer"}
CLASSICAL_MODEL_TYPES = set(MODEL_BUILDERS.keys())  # LR, RF, SVM, gradient_boosting + climatology / flux_rule / hard_soft_ratio_rule / two_feature_logreg

FORECAST_HORIZON_COLS = ["forecast_1h", "forecast_3h", "forecast_6h", "forecast_12h", "forecast_24h"]

registry = ModelRegistry(settings.model_dir)


def _training_status():
    return "demo_only" if settings.data_mode == "demo" or pipeline_state.dataset_meta.get("source") == "demo" else "trained"


def _channels_for_configuration(input_configuration: str):
    if input_configuration == "solexs_only":
        return ["solexs_flux_scaled"], 1
    if input_configuration == "helios_only":
        return ["helios_flux_scaled"], 1
    return ["solexs_flux_scaled", "helios_flux_scaled"], 2


def train_model(req: TrainRequest, progress_callback=None, stage_callback=None) -> dict:
    if not pipeline_state.is_ready_for_training():
        raise ValueError("Pipeline not ready. Run /api/preprocess before training.")

    channel_cols, n_channels = _channels_for_configuration(req.input_configuration)
    model_id = new_model_id(req.model_type + "_" + req.input_configuration)
    if stage_callback:
        stage_callback("validating input")

    if req.model_type in CLASSICAL_MODEL_TYPES:
        result = _train_classical(req, channel_cols, model_id, stage_callback)
    elif req.model_type in DEEP_MODEL_TYPES:
        result = _train_deep(req, channel_cols, n_channels, model_id, progress_callback, stage_callback)
    else:
        raise ValueError(f"Unknown model_type '{req.model_type}'")

    return result


def _train_classical(req: TrainRequest, channel_cols, model_id: str, stage_callback=None) -> dict:
    train_df = pipeline_state.train_df.copy()
    val_df = pipeline_state.val_df.copy()
    test_df = pipeline_state.test_df.copy()

    # re-attach engineered features + labels by merging on timestamp
    feat_df = pipeline_state.labeled_df
    feature_cols = pipeline_state.feature_columns

    train_merged = train_df[["timestamp"]].merge(feat_df, on="timestamp", how="inner").dropna(subset=feature_cols + ["state"] + FORECAST_HORIZON_COLS)
    val_merged = val_df[["timestamp"]].merge(feat_df, on="timestamp", how="inner").dropna(subset=feature_cols + ["state"] + FORECAST_HORIZON_COLS)
    test_merged = test_df[["timestamp"]].merge(feat_df, on="timestamp", how="inner").dropna(subset=feature_cols + ["state"] + FORECAST_HORIZON_COLS)

    if len(train_merged) < 10 or len(test_merged) < 5:
        raise ValueError("Not enough clean rows after feature engineering to train. Try a shorter cadence or more demo data.")

    use_cols = [c for c in feature_cols if any(ch.split("_")[0] in c for ch in channel_cols) or "ratio" not in c]
    # For single-instrument experiments, drop features derived purely from the excluded channel
    if req.input_configuration == "solexs_only":
        use_cols = [c for c in feature_cols if not c.startswith("helios") and "hard_soft_ratio" not in c]
    elif req.input_configuration == "helios_only":
        use_cols = [c for c in feature_cols if not c.startswith("solexs") and "hard_soft_ratio" not in c]
    else:
        use_cols = feature_cols

    X_train = train_merged[use_cols].values
    X_test = test_merged[use_cols].values
    y_now_train = train_merged["state"].values.astype(int)
    y_now_test = test_merged["state"].values.astype(int)
    y_fc_train = train_merged[FORECAST_HORIZON_COLS].values
    y_fc_test = test_merged[FORECAST_HORIZON_COLS].values

    t0 = time.time()
    model = MODEL_BUILDERS[req.model_type]()
    if stage_callback:
        stage_callback("training")
    model.fit(X_train, y_now_train, y_fc_train, feature_names=use_cols)
    train_duration = time.time() - t0

    fc_prob_test = model.predict_forecast_proba(X_test)
    now_prob_test = model.predict_nowcast_proba(X_test)
    now_pred_test = np.argmax(now_prob_test, axis=1)

    if stage_callback:
        stage_callback("evaluating")
    metrics = _compute_full_metrics(y_now_test, now_pred_test, y_fc_test, fc_prob_test)

    os.makedirs(settings.model_dir, exist_ok=True)
    transformer_path = os.path.join(settings.model_dir, f"{model_id}.transformer.joblib")
    joblib.dump(pipeline_state.transformer, transformer_path)
    checkpoint_path = os.path.join(settings.model_dir, f"{model_id}.joblib")
    if stage_callback:
        stage_callback("saving artifacts")
    joblib.dump({"model": model, "feature_cols": use_cols}, checkpoint_path)

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
        params={"n_features": len(use_cols), "train_duration_sec": round(train_duration, 3)},
    )
    registry.register(record)

    return {"model_id": model_id, "status": _training_status(), "metrics": metrics, "history": [], "input_configuration": req.input_configuration}


def _train_deep(req: TrainRequest, channel_cols, n_channels, model_id: str, progress_callback=None, stage_callback=None) -> dict:
    win_cfg = WindowConfig(sequence_length=req.sequence_length, channel_cols=channel_cols)

    X_train, y_now_train, y_fc_train, _ = build_windows(pipeline_state.train_df.merge(
        pipeline_state.labeled_df[["timestamp"] + channel_cols + ["state"] + FORECAST_HORIZON_COLS],
        on="timestamp", how="inner"
    ), win_cfg)
    X_val, y_now_val, y_fc_val, _ = build_windows(pipeline_state.val_df.merge(
        pipeline_state.labeled_df[["timestamp"] + channel_cols + ["state"] + FORECAST_HORIZON_COLS],
        on="timestamp", how="inner"
    ), win_cfg)
    X_test, y_now_test, y_fc_test, _ = build_windows(pipeline_state.test_df.merge(
        pipeline_state.labeled_df[["timestamp"] + channel_cols + ["state"] + FORECAST_HORIZON_COLS],
        on="timestamp", how="inner"
    ), win_cfg)

    if len(X_train) < 10 or len(X_test) < 5:
        raise ValueError("Not enough windows to train (need more demo data or a shorter sequence_length).")
    if len(X_val) < 5:
        raise ValueError("Not enough chronological validation windows; training cannot reuse training samples as validation data.")

    train_cfg = TrainConfig(
        model_type=req.model_type, sequence_length=req.sequence_length, batch_size=req.batch_size,
        epochs=req.epochs, learning_rate=req.learning_rate, dropout=req.dropout,
        alpha=req.alpha, beta=req.beta, class_weighting=req.class_weighting,
        n_channels=n_channels, device="cuda" if torch.cuda.is_available() else "cpu",
    )

    history = []
    if stage_callback:
        stage_callback("training")
    model, history = train_deep_model(
        X_train, y_now_train, y_fc_train, X_val, y_now_val, y_fc_val,
        config=train_cfg, progress_callback=progress_callback,
    )

    model.eval()
    with torch.no_grad():
        now_logits, fc_logits = model(torch.tensor(X_test, dtype=torch.float32))
        now_prob_test = torch.softmax(now_logits, dim=1).numpy()
        fc_prob_test = torch.sigmoid(fc_logits).numpy()
    now_pred_test = np.argmax(now_prob_test, axis=1)

    if stage_callback:
        stage_callback("evaluating")
    metrics = _compute_full_metrics(y_now_test, now_pred_test, y_fc_test, fc_prob_test)

    os.makedirs(settings.model_dir, exist_ok=True)
    transformer_path = os.path.join(settings.model_dir, f"{model_id}.transformer.joblib")
    joblib.dump(pipeline_state.transformer, transformer_path)
    checkpoint_path = os.path.join(settings.model_dir, f"{model_id}.pt")
    if stage_callback:
        stage_callback("saving artifacts")
    torch.save({"state_dict": model.state_dict(), "model_type": req.model_type, "n_channels": n_channels}, checkpoint_path)

    record = ModelRecord(
        model_id=model_id, name=f"{req.model_type} ({req.input_configuration})",
        architecture=req.model_type, version="v1",
        created_at=str(pd.Timestamp.utcnow()),
        training_dataset=pipeline_state.dataset_meta.get("filename", "demo"),
        input_configuration=req.input_configuration,
        metrics=metrics, checkpoint_path=checkpoint_path,
        scaler_path=transformer_path, transformer_path=transformer_path, active=False, model_kind="deep",
        status=_training_status(),
        transformer_required=True,
        params={"sequence_length": req.sequence_length, "epochs": req.epochs, "n_channels": n_channels},
    )
    registry.register(record)

    return {"model_id": model_id, "status": _training_status(), "metrics": metrics, "history": history, "input_configuration": req.input_configuration}


def _compute_full_metrics(y_now_true, y_now_pred, y_fc_true, y_fc_prob) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    now_acc = float(accuracy_score(y_now_true, y_now_pred))
    now_f1_macro = float(f1_score(y_now_true, y_now_pred, average="macro", zero_division=0))
    cm = multiclass_confusion_matrix(y_now_true, y_now_pred, n_classes=4)

    per_horizon = {}
    for i, h in enumerate(["1h", "3h", "6h", "12h", "24h"]):
        per_horizon[h] = evaluate_binary_forecast(y_fc_true[:, i], y_fc_prob[:, i])

    tss_values = [m["tss"] for m in per_horizon.values()]
    hss_values = [m["hss"] for m in per_horizon.values()]

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
        "precision": float(np.mean([m["precision"] for m in per_horizon.values()])),
        "recall": float(np.mean([m["recall"] for m in per_horizon.values()])),
        "f1": float(np.mean([m["f1"] for m in per_horizon.values()])),
        "false_alarm_rate": float(np.mean([m["false_alarm_rate"] for m in per_horizon.values()])),
        "pr_auc": float(np.mean([m["pr_auc"] for m in per_horizon.values() if m["pr_auc"] is not None])) if any(m["pr_auc"] is not None for m in per_horizon.values()) else None,
    }
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
    return registry.list_all()


def get_model_metrics(model_id: str):
    rec = registry.get(model_id)
    if not rec:
        raise KeyError(f"Model {model_id} not found")
    return rec


def set_active_model(model_id: str):
    registry.set_active(model_id)
    return registry.get(model_id)


def get_active_model():
    return registry.get_active()


def run_fusion_experiment(base_req: TrainRequest) -> dict:
    """Trains the SAME model architecture under 3 input configurations and compares metrics."""
    results = {}
    for config in ["solexs_only", "helios_only", "fusion"]:
        req = TrainRequest(**{**base_req.dict(), "input_configuration": config})
        try:
            res = train_model(req)
            results[config] = res["metrics"]
        except Exception as e:
            results[config] = {"error": str(e)}

    valid = {k: v for k, v in results.items() if "error" not in v and v.get("tss_mean") is not None}
    best_config = max(valid, key=lambda k: valid[k]["tss_mean"]) if valid else None

    return {
        "results": results,
        "best_configuration": best_config,
        "label": "DEMONSTRATION RESULTS — trained on synthetic demo data. Replace with trained experimental results on real Aditya-L1 data for scientific conclusions.",
    }
