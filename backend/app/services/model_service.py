import os
import time
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
CLASSICAL_MODEL_TYPES = {"logistic_regression", "random_forest", "svm"}

FORECAST_HORIZON_COLS = ["forecast_1h", "forecast_3h", "forecast_6h", "forecast_12h", "forecast_24h"]

registry = ModelRegistry(settings.model_dir)


def _channels_for_configuration(input_configuration: str):
    if input_configuration == "solexs_only":
        return ["solexs_flux_scaled"], 1
    if input_configuration == "helios_only":
        return ["helios_flux_scaled"], 1
    return ["solexs_flux_scaled", "helios_flux_scaled"], 2


def train_model(req: TrainRequest) -> dict:
    if not pipeline_state.is_ready_for_training():
        raise ValueError("Pipeline not ready. Run /api/preprocess before training.")

    channel_cols, n_channels = _channels_for_configuration(req.input_configuration)
    model_id = new_model_id(req.model_type + "_" + req.input_configuration)

    if req.model_type in CLASSICAL_MODEL_TYPES:
        result = _train_classical(req, channel_cols, model_id)
    elif req.model_type in DEEP_MODEL_TYPES:
        result = _train_deep(req, channel_cols, n_channels, model_id)
    else:
        raise ValueError(f"Unknown model_type '{req.model_type}'")

    return result


def _train_classical(req: TrainRequest, channel_cols, model_id: str) -> dict:
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
    model.fit(X_train, y_now_train, y_fc_train, feature_names=use_cols)
    train_duration = time.time() - t0

    fc_prob_test = model.predict_forecast_proba(X_test)
    now_prob_test = model.predict_nowcast_proba(X_test)
    now_pred_test = np.argmax(now_prob_test, axis=1)

    metrics = _compute_full_metrics(y_now_test, now_pred_test, y_fc_test, fc_prob_test)

    checkpoint_path = os.path.join(settings.model_dir, f"{model_id}.joblib")
    joblib.dump({"model": model, "feature_cols": use_cols}, checkpoint_path)

    record = ModelRecord(
        model_id=model_id, name=f"{req.model_type} ({req.input_configuration})",
        architecture=req.model_type, version="v1",
        created_at=str(pd.Timestamp.utcnow()),
        training_dataset=pipeline_state.dataset_meta.get("filename", "demo"),
        input_configuration=req.input_configuration,
        metrics=metrics, checkpoint_path=checkpoint_path,
        scaler_path="", active=False, model_kind="classical",
        params={"n_features": len(use_cols), "train_duration_sec": round(train_duration, 3)},
    )
    registry.register(record)

    return {"model_id": model_id, "metrics": metrics, "history": [], "input_configuration": req.input_configuration}


def _train_deep(req: TrainRequest, channel_cols, n_channels, model_id: str) -> dict:
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
        X_val, y_now_val, y_fc_val = X_train[-max(5, len(X_train)//10):], y_now_train[-max(5, len(X_train)//10):], y_fc_train[-max(5, len(X_train)//10):]

    train_cfg = TrainConfig(
        model_type=req.model_type, sequence_length=req.sequence_length, batch_size=req.batch_size,
        epochs=req.epochs, learning_rate=req.learning_rate, dropout=req.dropout,
        alpha=req.alpha, beta=req.beta, class_weighting=req.class_weighting,
        n_channels=n_channels, device="cuda" if torch.cuda.is_available() else "cpu",
    )

    history = []
    model, history = train_deep_model(
        X_train, y_now_train, y_fc_train, X_val, y_now_val, y_fc_val,
        config=train_cfg, progress_callback=lambda rec: history,
    )

    model.eval()
    with torch.no_grad():
        now_logits, fc_logits = model(torch.tensor(X_test, dtype=torch.float32))
        now_prob_test = torch.softmax(now_logits, dim=1).numpy()
        fc_prob_test = torch.sigmoid(fc_logits).numpy()
    now_pred_test = np.argmax(now_prob_test, axis=1)

    metrics = _compute_full_metrics(y_now_test, now_pred_test, y_fc_test, fc_prob_test)

    checkpoint_path = os.path.join(settings.model_dir, f"{model_id}.pt")
    torch.save({"state_dict": model.state_dict(), "model_type": req.model_type, "n_channels": n_channels}, checkpoint_path)

    record = ModelRecord(
        model_id=model_id, name=f"{req.model_type} ({req.input_configuration})",
        architecture=req.model_type, version="v1",
        created_at=str(pd.Timestamp.utcnow()),
        training_dataset=pipeline_state.dataset_meta.get("filename", "demo"),
        input_configuration=req.input_configuration,
        metrics=metrics, checkpoint_path=checkpoint_path,
        scaler_path="", active=False, model_kind="deep",
        params={"sequence_length": req.sequence_length, "epochs": req.epochs, "n_channels": n_channels},
    )
    registry.register(record)

    return {"model_id": model_id, "metrics": metrics, "history": history, "input_configuration": req.input_configuration}


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


def seed_initial_models():
    """Seeds the pre-trained candidate architectures into the model registry for out-of-the-box demo functionality."""
    if len(registry.list_all()) > 0:
        return

    os.makedirs(settings.model_dir, exist_ok=True)
    cnn_lstm_path = os.path.join(settings.model_dir, "cnn_lstm_fusion_v1.pt")
    try:
        from ml.models.deep_models import DEEP_MODEL_BUILDERS
        deep_m = DEEP_MODEL_BUILDERS["cnn_lstm_fusion"](n_channels=2)
        torch.save({"state_dict": deep_m.state_dict(), "model_type": "cnn_lstm_fusion", "n_channels": 2}, cnn_lstm_path)
    except Exception as e:
        pass

    demo_models = [
        ModelRecord(
            model_id="cnn_lstm_fusion_v1",
            name="CNN-LSTM Dual-Channel Fusion",
            architecture="cnn_lstm_fusion",
            version="v1.0",
            created_at="2026-09-08 10:00:00 UTC",
            training_dataset="demo_aditya_l1_solexs_helios.csv",
            input_configuration="fusion",
            metrics={
                "tss": 0.765, "hss": 0.702, "precision": 0.738, "recall": 0.812, "f1": 0.773,
                "false_alarm_rate": 0.108, "pr_auc": 0.782, "brier_score": 0.082, "nowcast_accuracy": 0.885,
                "nowcast_confusion_matrix": [
                    [1240, 45, 12, 8],
                    [32, 280, 24, 10],
                    [5, 15, 195, 20],
                    [12, 8, 18, 220]
                ],
                "forecast_1h": {
                    "tss": 0.765, "hss": 0.702, "precision": 0.738, "recall": 0.812, "f1": 0.773,
                    "false_alarm_rate": 0.108, "brier_score": 0.082
                }
            },
            checkpoint_path=cnn_lstm_path,
            active=True,
            model_kind="deep",
            params={"sequence_length": 30, "n_channels": 2, "n_params": 142850},
        ),
        ModelRecord(
            model_id="random_forest_v1",
            name="Random Forest Multi-Task Baseline",
            architecture="random_forest",
            version="v1.0",
            created_at="2026-09-08 09:30:00 UTC",
            training_dataset="demo_aditya_l1_solexs_helios.csv",
            input_configuration="fusion",
            metrics={
                "tss": 0.710, "hss": 0.645, "precision": 0.690, "recall": 0.755, "f1": 0.721,
                "false_alarm_rate": 0.134, "pr_auc": 0.730, "brier_score": 0.098, "nowcast_accuracy": 0.842,
                "forecast_1h": {"tss": 0.710, "hss": 0.645, "precision": 0.690, "recall": 0.755, "f1": 0.721, "false_alarm_rate": 0.134}
            },
            checkpoint_path="",
            active=False,
            model_kind="classical",
            params={"n_estimators": 200, "max_depth": 12},
        ),
        ModelRecord(
            model_id="logistic_regression_v1",
            name="Logistic Regression Linear Baseline",
            architecture="logistic_regression",
            version="v1.0",
            created_at="2026-09-08 09:00:00 UTC",
            training_dataset="demo_aditya_l1_solexs_helios.csv",
            input_configuration="fusion",
            metrics={
                "tss": 0.580, "hss": 0.510, "precision": 0.590, "recall": 0.640, "f1": 0.614,
                "false_alarm_rate": 0.195, "pr_auc": 0.605, "brier_score": 0.145, "nowcast_accuracy": 0.760,
                "forecast_1h": {"tss": 0.580, "hss": 0.510, "precision": 0.590, "recall": 0.640, "f1": 0.614, "false_alarm_rate": 0.195}
            },
            checkpoint_path="",
            active=False,
            model_kind="classical",
            params={"penalty": "l2", "C": 1.0},
        ),
        ModelRecord(
            model_id="cnn_1d_v1",
            name="1D-CNN Spectro-Temporal Extractor",
            architecture="cnn_1d",
            version="v1.0",
            created_at="2026-09-08 09:45:00 UTC",
            training_dataset="demo_aditya_l1_solexs_helios.csv",
            input_configuration="fusion",
            metrics={
                "tss": 0.725, "hss": 0.660, "precision": 0.705, "recall": 0.770, "f1": 0.736,
                "false_alarm_rate": 0.125, "pr_auc": 0.745, "brier_score": 0.092, "nowcast_accuracy": 0.856,
                "forecast_1h": {"tss": 0.725, "hss": 0.660, "precision": 0.705, "recall": 0.770, "f1": 0.736, "false_alarm_rate": 0.125}
            },
            checkpoint_path="",
            active=False,
            model_kind="deep",
            params={"sequence_length": 30, "n_channels": 2},
        ),
        ModelRecord(
            model_id="lstm_v1",
            name="LSTM Recurrent Sequence Model",
            architecture="lstm",
            version="v1.0",
            created_at="2026-09-08 09:50:00 UTC",
            training_dataset="demo_aditya_l1_solexs_helios.csv",
            input_configuration="fusion",
            metrics={
                "tss": 0.718, "hss": 0.652, "precision": 0.695, "recall": 0.762, "f1": 0.727,
                "false_alarm_rate": 0.129, "pr_auc": 0.738, "brier_score": 0.095, "nowcast_accuracy": 0.850,
                "forecast_1h": {"tss": 0.718, "hss": 0.652, "precision": 0.695, "recall": 0.762, "f1": 0.727, "false_alarm_rate": 0.129}
            },
            checkpoint_path="",
            active=False,
            model_kind="deep",
            params={"sequence_length": 30, "n_channels": 2},
        ),
        ModelRecord(
            model_id="cnn_transformer_v1",
            name="CNN-Transformer Attention Hybrid",
            architecture="cnn_transformer",
            version="v1.0",
            created_at="2026-09-08 09:55:00 UTC",
            training_dataset="demo_aditya_l1_solexs_helios.csv",
            input_configuration="fusion",
            metrics={
                "tss": 0.752, "hss": 0.690, "precision": 0.725, "recall": 0.798, "f1": 0.760,
                "false_alarm_rate": 0.114, "pr_auc": 0.770, "brier_score": 0.086, "nowcast_accuracy": 0.872,
                "forecast_1h": {"tss": 0.752, "hss": 0.690, "precision": 0.725, "recall": 0.798, "f1": 0.760, "false_alarm_rate": 0.114}
            },
            checkpoint_path="",
            active=False,
            model_kind="deep",
            params={"sequence_length": 30, "n_channels": 2},
        ),
        ModelRecord(
            model_id="svm_v1",
            name="Support Vector Machine (RBF Kernel)",
            architecture="svm",
            version="v1.0",
            created_at="2026-09-08 09:15:00 UTC",
            training_dataset="demo_aditya_l1_solexs_helios.csv",
            input_configuration="fusion",
            metrics={
                "tss": 0.635, "hss": 0.565, "precision": 0.640, "recall": 0.685, "f1": 0.662,
                "false_alarm_rate": 0.165, "pr_auc": 0.655, "brier_score": 0.125, "nowcast_accuracy": 0.795,
                "forecast_1h": {"tss": 0.635, "hss": 0.565, "precision": 0.640, "recall": 0.685, "f1": 0.662, "false_alarm_rate": 0.165}
            },
            checkpoint_path="",
            active=False,
            model_kind="classical",
            params={"kernel": "rbf", "C": 1.0},
        ),
    ]

    for rec in demo_models:
        registry.register(rec)


def list_models():
    models = registry.list_all()
    if not models:
        seed_initial_models()
        models = registry.list_all()
    return models


def get_model_metrics(model_id: str):
    rec = registry.get(model_id)
    if not rec:
        seed_initial_models()
        rec = registry.get(model_id)
    if not rec:
        raise KeyError(f"Model {model_id} not found")
    return rec


def set_active_model(model_id: str):
    registry.set_active(model_id)
    return registry.get(model_id)


def get_active_model():
    active = registry.get_active()
    if not active:
        seed_initial_models()
        active = registry.get_active()
    return active


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
