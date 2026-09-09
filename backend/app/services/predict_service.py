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
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model_type = ckpt.get("model_type", arch)
        n_chan = ckpt.get("n_channels", n_channels)
        model_fn = DEEP_MODEL_BUILDERS.get(model_type, DEEP_MODEL_BUILDERS["cnn_lstm_fusion"])
        model = model_fn(n_channels=n_chan)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
    else:
        model_fn = DEEP_MODEL_BUILDERS.get(arch, DEEP_MODEL_BUILDERS["cnn_lstm_fusion"])
        model = model_fn(n_channels=n_channels)
    model.eval()
    _loaded_cache[model_id] = model
    return model


def resolve_model(model_id: Optional[str] = None) -> dict:
    if model_id:
        rec = registry.get(model_id)
        if not rec:
            from app.services.model_service import seed_initial_models
            seed_initial_models()
            rec = registry.get(model_id)
        if not rec:
            raise KeyError(f"Model {model_id} not found in registry.")
        return rec
    rec = registry.get_active()
    if not rec:
        from app.services.model_service import seed_initial_models
        seed_initial_models()
        rec = registry.get_active()
    if not rec:
        all_models = registry.list_all()
        if not all_models:
            raise ValueError("No trained models available. Train a model first (POST /api/models/train).")
        rec = all_models[-1]
    return rec


def predict_from_raw_window(rows: list, model_id: Optional[str] = None) -> dict:
    """
    Accepts a list of raw row dicts (timestamp, solexs_flux, helios_flux),
    runs them through the FITTED transformer + feature engineering, and
    produces a prediction using the resolved model.
    """
    if pipeline_state.transformer is None:
        df_temp = pd.DataFrame(rows)
        pipeline_state.transformer = FluxTransformer().fit(df_temp)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    transformer: FluxTransformer = pipeline_state.transformer
    transformed = transformer.transform(df)
    featured = add_rolling_features(transformed, FeatureConfig(cadence_seconds=pipeline_state.cadence_seconds))
    featured = featured.bfill().ffill().fillna(0)

    rec = resolve_model(model_id)
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

    def _fn(window_df: pd.DataFrame) -> dict:
        featured = add_rolling_features(window_df, FeatureConfig(cadence_seconds=pipeline_state.cadence_seconds))
        featured = featured.bfill().ffill().fillna(0)
        result = _predict_with_record(rec, featured)
        result["model"] = rec["architecture"]
        return result

    return _fn
