import joblib
import torch
import numpy as np
import pandas as pd
from typing import Optional

from app.services.pipeline_state import pipeline_state
from app.services.model_service import registry
from ml.models.deep_models import DEEP_MODEL_BUILDERS
from ml.inference.predictor import _format_prediction, risk_level_from_probability
from ml.preprocessing.transform import FluxTransformer
from ml.features.engineering import add_rolling_features, FeatureConfig

_loaded_cache = {}


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
    model = DEEP_MODEL_BUILDERS[model_type](n_channels=n_chan)
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
            raise KeyError(f"Model {model_id} not found in registry.")
        if rec.get("status") == "untrained":
            raise ValueError(f"Model {model_id} is untrained and unavailable for prediction.")
        registry.validate_artifacts(model_id, rec)
        return rec
    rec = registry.get_active()
    if not rec:
        raise ValueError("No active trained model is available. Train a compatible model and activate it first.")
    registry.validate_artifacts(rec["model_id"], rec)
    return rec


def predict_from_raw_window(rows: list, model_id: Optional[str] = None) -> dict:
    """
    Accepts a list of raw row dicts (timestamp, solexs_flux, helios_flux),
    runs them through the FITTED transformer + feature engineering, and
    produces a prediction using the resolved model.
    """
    df = pd.DataFrame(rows)
    required = {"timestamp", "solexs_flux", "helios_flux"}
    if not required.issubset(df.columns) or len(df) == 0:
        raise ValueError("Prediction rows must include timestamp, solexs_flux, and helios_flux.")
    df["solexs_flux"] = pd.to_numeric(df["solexs_flux"], errors="coerce")
    df["helios_flux"] = pd.to_numeric(df["helios_flux"], errors="coerce")
    if not np.isfinite(df[["solexs_flux", "helios_flux"]].to_numpy()).all():
        raise ValueError("Prediction flux values must be finite numbers.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Prediction timestamps must be valid.")
    df = df.sort_values("timestamp").reset_index(drop=True)
    rec = resolve_model(model_id)
    transformer_path = rec.get("transformer_path") or rec.get("scaler_path")
    if rec.get("transformer_required", True):
        if not transformer_path or not os.path.isfile(transformer_path):
            raise ValueError("The selected model preprocessing transformer is missing.")
        transformer: FluxTransformer = joblib.load(transformer_path)
    else:
        transformer = None
    if transformer is None:
        raise ValueError("The selected model has no compatible preprocessing transformer.")
    transformed = transformer.transform(df)
    featured = add_rolling_features(transformed, FeatureConfig(cadence_seconds=pipeline_state.cadence_seconds))
    featured = featured.bfill().ffill().fillna(0)

    result = _predict_with_record(rec, featured)
    result["model"] = rec["architecture"]
    result["data_quality"] = float(1.0 - df[["solexs_flux", "helios_flux"]].isna().mean().mean())
    result["input"] = {
        "solexs_flux": float(df["solexs_flux"].iloc[-1]),
        "helios_flux": float(df["helios_flux"].iloc[-1]),
        "hard_soft_ratio": float(featured["hard_soft_ratio"].iloc[-1]) if "hard_soft_ratio" in featured.columns else None,
    }
    result["timestamp"] = str(df["timestamp"].iloc[-1])
    return result


def _predict_with_record(rec: dict, featured_window: pd.DataFrame) -> dict:
    checkpoint = rec.get("checkpoint_path")
    if not checkpoint or not os.path.isfile(checkpoint):
        raise ValueError("The selected model checkpoint is missing.")
    if rec.get("status") == "untrained":
        raise ValueError("The selected model is untrained and unavailable for prediction.")
    if rec["model_kind"] == "classical":
        obj = _load_classical(rec["model_id"], rec["checkpoint_path"])
        model, feature_cols = obj["model"], obj["feature_cols"]
        row = featured_window[feature_cols].iloc[[-1]].values
        now_prob = model.predict_nowcast_proba(row)[0]
        fc_prob = model.predict_forecast_proba(row)[0]
    else:
        seq_len = rec["params"].get("sequence_length", 30)
        n_channels = rec["params"].get("n_channels", 2)
        model = _load_deep(rec["model_id"], rec.get("checkpoint_path", ""), arch=rec.get("architecture", "cnn_lstm_fusion"), n_channels=n_channels)
        channel_cols = ["solexs_flux_scaled", "helios_flux_scaled"][:n_channels] if n_channels == 2 else \
            (["solexs_flux_scaled"] if rec["input_configuration"] == "solexs_only" else ["helios_flux_scaled"])
        window = featured_window[channel_cols].tail(seq_len).values.astype(np.float32)
        if len(window) < seq_len:
            pad = np.repeat(window[:1], seq_len - len(window), axis=0) if len(window) else np.zeros((seq_len, len(channel_cols)))
            window = np.vstack([pad, window])
        x = torch.tensor(window[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            now_logits, fc_logits = model(x)
            now_prob = torch.softmax(now_logits, dim=1).numpy()[0]
            fc_prob = torch.sigmoid(fc_logits).numpy()[0]

    return _format_prediction(now_prob, fc_prob)


def make_simulator_predict_fn(model_id: Optional[str] = None):
    """Returns a callable(window_df) -> prediction dict, bound to a resolved model, for the live simulator."""
    rec = resolve_model(model_id)
    transformer_path = rec.get("transformer_path") or rec.get("scaler_path")
    if not transformer_path or not os.path.isfile(transformer_path):
        raise ValueError("The selected model preprocessing transformer is missing.")
    transformer = joblib.load(transformer_path)

    def _fn(window_df: pd.DataFrame) -> dict:
        transformed = transformer.transform(window_df)
        featured = add_rolling_features(transformed, FeatureConfig(cadence_seconds=pipeline_state.cadence_seconds))
        featured = featured.bfill().ffill().fillna(0)
        result = _predict_with_record(rec, featured)
        result["model"] = rec["architecture"]
        return result

    return _fn
