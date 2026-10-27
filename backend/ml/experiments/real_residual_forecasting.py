"""Phase 4O: Residual and Trend-Aware Real-Data SoLEXS Forecasting Models.

This module implements controlled model architectures for self-supervised count-rate
forecasting designed to overcome prediction lag and high-flux degradation:
1. ResidualCNNLSTM: Predicts a residual correction delta relative to the forecast origin (r_T).
2. TrendAwareResidualCNNLSTM: Augments input sequences with strictly causal past trend,
   velocity, and volatility features before residual forecasting.

All models operate on native counts/s, never fit scalers on evaluation data,
and maintain complete production registry isolation.
"""
from __future__ import annotations

import io
import gzip
import json
import zipfile
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from astropy.io import fits

from ml.experiments.real_self_supervised import (
    CountRateScaler,
    SelfSupervisedConfig,
    SOLEXS_ARCHIVE_SHA256,
    MARKER_EXPERIMENTAL,
    validate_solexs_fits_table,
    extract_contiguous_segments,
    create_sliding_windows,
    persistence_forecast,
    rolling_mean_forecast,
    evaluate_forecast,
)


class ResidualCNNLSTM(nn.Module):
    """Residual CNN-LSTM sequence model for autoregressive native count-rate forecasting.
    
    Instead of predicting absolute future counts from scratch, this network predicts
    the forward residual increment delta relative to the last observed rate (r_T):
        y_{T+k} = r_T + delta_{T+k}
    
    This guarantees that at initialization, the model behaves identically to persistence,
    eliminating baseline level drift during high-flux and stable periods.
    """

    def __init__(self, in_features: int = 1, horizon: int = 10, cnn_channels: int = 32,
                 lstm_hidden: int = 64, lstm_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.horizon = horizon
        self.in_features = in_features
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, horizon)
        )
        # Initialize output projection near zero so initial prediction matches persistence
        nn.init.zeros_(self.fc[2].bias)
        nn.init.normal_(self.fc[2].weight, std=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, in_features)
        x_conv = x.transpose(1, 2)  # (batch, in_features, seq_len)
        feat = self.conv(x_conv)   # (batch, cnn_channels, seq_len // 2)
        feat_lstm = feat.transpose(1, 2)  # (batch, seq_len // 2, cnn_channels)
        out, _ = self.lstm(feat_lstm)
        last_hidden = out[:, -1, :]  # (batch, lstm_hidden)
        delta_preds = self.fc(last_hidden)  # (batch, horizon)
        
        # Residual addition: delta added to the last observed value in channel 0
        x_last = x[:, -1:, 0]  # (batch, 1)
        pred_scaled = x_last + delta_preds  # (batch, horizon)
        return pred_scaled


def compute_causal_trend_features(X_raw: np.ndarray) -> np.ndarray:
    """Compute strictly past-only causal trend and volatility features.
    
    Inputs:
        X_raw: shape (N, 60, 1) of raw count rates.
    Outputs:
        X_features: shape (N, 60, 4) where:
            Channel 0: Raw count rates (normalized by scaler externally)
            Channel 1: 5-second relative velocity: (r_t - r_{t-5}) / max(1, r_{t-5})
            Channel 2: 15-second relative momentum: (r_t - r_{t-15}) / max(1, r_{t-15})
            Channel 3: 15-second relative rolling volatility: std(r_{t-14..t}) / max(1, mean)
    """
    N, seq_len, _ = X_raw.shape
    features = np.zeros((N, seq_len, 4), dtype=np.float32)
    features[:, :, 0] = X_raw[:, :, 0]
    
    counts = X_raw[:, :, 0]
    
    # 5-second relative velocity (for t >= 5, else 0)
    for t in range(5, seq_len):
        diff5 = counts[:, t] - counts[:, t - 5]
        denom = np.maximum(counts[:, t - 5], 1.0)
        features[:, t, 1] = np.clip(diff5 / denom, -5.0, 5.0)
    
    # 15-second relative momentum (for t >= 15, else 0)
    for t in range(15, seq_len):
        diff15 = counts[:, t] - counts[:, t - 15]
        denom = np.maximum(counts[:, t - 15], 1.0)
        features[:, t, 2] = np.clip(diff15 / denom, -5.0, 5.0)
    
    # 15-second relative volatility
    for t in range(14, seq_len):
        window = counts[:, t - 14 : t + 1]
        std15 = np.std(window, axis=1)
        mean15 = np.maximum(np.mean(window, axis=1), 1.0)
        features[:, t, 3] = np.clip(std15 / mean15, 0.0, 5.0)
        
    return features


class TrendAwareResidualCNNLSTM(nn.Module):
    """Trend-Aware Residual CNN-LSTM taking multi-channel causal trend inputs.
    
    Features: (batch, 60, 4) -> CNN -> LSTM -> Residual Head relative to r_T.
    """

    def __init__(self, in_features: int = 4, horizon: int = 10, cnn_channels: int = 32,
                 lstm_hidden: int = 64, lstm_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.horizon = horizon
        self.in_features = in_features
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, horizon)
        )
        nn.init.zeros_(self.fc[2].bias)
        nn.init.normal_(self.fc[2].weight, std=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 4)
        x_conv = x.transpose(1, 2)
        feat = self.conv(x_conv)
        feat_lstm = feat.transpose(1, 2)
        out, _ = self.lstm(feat_lstm)
        last_hidden = out[:, -1, :]
        delta_preds = self.fc(last_hidden)
        
        # Residual addition relative to Channel 0 (the count rate) at the last time step
        x_last = x[:, -1:, 0]  # (batch, 1)
        pred_scaled = x_last + delta_preds
        return pred_scaled
