"""
Unified prediction interface. Wraps either a classical (sklearn) or deep
(PyTorch) model + its fitted preprocessing objects into a single
`.predict(window_df)` call, returning the structured response contract used
across the whole application (see API docs / README for schema).
"""
import numpy as np
import joblib
import torch
import os
from typing import Dict, Any, Optional

NOWCAST_LABELS = ["quiet", "pre_flare", "flare", "decay"]
FORECAST_HORIZONS = ["1h", "3h", "6h", "12h", "24h"]

RISK_THRESHOLDS = [
    (0.85, "CRITICAL"),
    (0.70, "HIGH"),
    (0.50, "MODERATE"),
    (0.30, "WATCH"),
    (0.0, "LOW"),
]


def risk_level_from_probability(p: float, thresholds=None) -> str:
    thr = thresholds or RISK_THRESHOLDS
    for cutoff, label in thr:
        if p >= cutoff:
            return label
    return "LOW"


class DeepModelPredictor:
    def __init__(self, model: torch.nn.Module, transformer, device="cpu"):
        self.model = model
        self.transformer = transformer
        self.device = device
        self.model.eval()

    def predict_window(self, window_df, channel_cols=("solexs_flux_scaled", "helios_flux_scaled")) -> Dict[str, Any]:
        x = window_df[list(channel_cols)].values.astype(np.float32)
        x_tensor = torch.tensor(x[None, :, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            now_logits, fc_logits = self.model(x_tensor)
            now_prob = torch.softmax(now_logits, dim=1).cpu().numpy()[0]
            fc_prob = torch.sigmoid(fc_logits).cpu().numpy()[0]
        return _format_prediction(now_prob, fc_prob)


class ClassicalModelPredictor:
    def __init__(self, model, transformer):
        self.model = model
        self.transformer = transformer

    def predict_features(self, feature_row: np.ndarray) -> Dict[str, Any]:
        now_prob = self.model.predict_nowcast_proba(feature_row.reshape(1, -1))[0]
        fc_prob = self.model.predict_forecast_proba(feature_row.reshape(1, -1))[0]
        return _format_prediction(now_prob, fc_prob)


def _format_prediction(now_prob: np.ndarray, fc_prob: np.ndarray) -> Dict[str, Any]:
    now_idx = int(np.argmax(now_prob))
    forecast = {h: float(p) for h, p in zip(FORECAST_HORIZONS, fc_prob)}
    max_prob_1h = forecast.get("1h", max(forecast.values()) if forecast else 0.0)
    risk = risk_level_from_probability(max_prob_1h)
    return {
        "nowcast": {
            "state": NOWCAST_LABELS[now_idx],
            "confidence": float(now_prob[now_idx]),
            "probabilities": {label: float(p) for label, p in zip(NOWCAST_LABELS, now_prob)},
        },
        "forecast": forecast,
        "risk": risk,
    }
