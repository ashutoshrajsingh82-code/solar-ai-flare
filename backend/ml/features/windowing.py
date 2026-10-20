"""
Sliding sequence window generator for the deep-learning models.

Produces:
  X_seq: (n_windows, sequence_length, n_channels)   raw/scaled SoLEXS+HEL1OS (+ optional engineered channels)
  y_nowcast: (n_windows,)                            int class label at the END of each window
  y_forecast: (n_windows, n_horizons)                binary forecast targets at the END of each window

Data leakage prevention:
  - Windows are generated strictly WITHIN a single split (train/val/test are
    windowed separately after chronological splitting, so no window ever
    spans the train/test boundary).
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

DEFAULT_HORIZON_COLS = ["forecast_1h", "forecast_3h", "forecast_6h", "forecast_12h", "forecast_24h"]


@dataclass
class WindowConfig:
    sequence_length: int = 60         # number of timesteps per window
    stride: int = 1                   # step between consecutive windows
    channel_cols: List[str] = field(default_factory=lambda: ["solexs_flux_scaled", "helios_flux_scaled"])
    extra_feature_cols: Optional[List[str]] = None
    horizon_cols: List[str] = field(default_factory=lambda: DEFAULT_HORIZON_COLS)


def build_windows(df: pd.DataFrame, config: WindowConfig = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows from a single contiguous (already split) dataframe.

    Returns
    -------
    X : np.ndarray, shape (n_windows, seq_len, n_channels)
    y_nowcast : np.ndarray, shape (n_windows,)
    y_forecast : np.ndarray, shape (n_windows, n_horizons)
    end_timestamps : np.ndarray of the timestamp at each window's final step
    """
    cfg = config or WindowConfig()
    if cfg.sequence_length <= 0 or cfg.stride <= 0:
        raise ValueError("Window sequence_length and stride must be positive.")
    cols = list(cfg.channel_cols)
    if cfg.extra_feature_cols:
        cols += cfg.extra_feature_cols

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for windowing: {missing}")
    missing_targets = [c for c in ["timestamp", "state", *cfg.horizon_cols] if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing required targets for windowing: {missing_targets}")
    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", format="mixed")
    if timestamps.isna().any():
        raise ValueError("Window timestamps must be valid UTC-compatible values.")
    if timestamps.duplicated().any():
        raise ValueError("Window timestamps must not contain duplicates.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Window observations must be ordered by timestamp.")

    data = df[cols].values.astype(np.float32)
    n = len(df)
    seq_len = cfg.sequence_length

    if n < seq_len:
        return (np.empty((0, seq_len, len(cols)), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0, len(cfg.horizon_cols)), dtype=np.float32),
                np.empty((0,), dtype="datetime64[ns]"))

    starts = np.arange(0, n - seq_len + 1, cfg.stride)
    X = np.stack([data[s:s + seq_len] for s in starts])

    end_idx = starts + seq_len - 1
    nowcast_rows = pd.to_numeric(df["state"], errors="coerce").to_numpy(dtype=float)[end_idx]
    forecast_rows = df[cfg.horizon_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[end_idx]
    end_timestamps = np.asarray(timestamps.iloc[end_idx].tolist(), dtype=object)

    # drop windows containing NaNs (from unfilled large telemetry gaps)
    # Unknown forecast labels are retained as NaN; they are masked during loss/evaluation.
    # Only invalid features or nowcast targets make a sample unusable.
    valid_mask = np.isfinite(X).all(axis=(1, 2)) & np.isfinite(nowcast_rows)
    valid_nowcast = nowcast_rows[valid_mask]
    valid_forecast = forecast_rows[valid_mask]
    if not np.equal(valid_nowcast, np.floor(valid_nowcast)).all() or ((valid_nowcast < 0) | (valid_nowcast > 3)).any():
        raise ValueError("Window nowcast targets must be class indices from 0 to 3.")
    known_forecast = np.isfinite(valid_forecast)
    if not np.isin(valid_forecast[known_forecast], (0, 1)).all():
        raise ValueError("Known window forecast targets must be binary; unknown values must remain missing.")
    return (X[valid_mask], valid_nowcast.astype(np.int64), valid_forecast.astype(np.float32),
            end_timestamps[valid_mask])


def chronological_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    CRITICAL: split time series chronologically (never shuffle) so that no
    future information leaks into training. Earliest 70% -> train,
    next 15% -> validation, final 15% (most recent) -> test.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)
    return train, val, test
