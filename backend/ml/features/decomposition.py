"""Origin-scoped STL inputs for DLSTM.

Each sample is decomposed independently from exactly its historical window.
No endpoint uses observations after the sample's forecast origin. The STL
endpoint is still a two-sided estimate within that historical window and is
therefore less reliable than interior values; this limitation is part of the
model contract, not interpreted as a real-time measurement.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


COMPONENTS = ("trend", "seasonal", "residual")


@dataclass(frozen=True)
class DecompositionConfig:
    period_samples: int = 12
    min_cycles: int = 2
    robust: bool = False
    log_transform: str = "log1p(value / training_median_positive)"

    def validate(self):
        if self.period_samples < 2 or self.min_cycles < 2:
            raise ValueError("STL period must be >= 2 and minimum history must cover at least two cycles.")
        if self.log_transform != "log1p(value / training_median_positive)":
            raise ValueError("Unsupported DLSTM transform; retrain with the documented training-scale transform.")
        return self

    @property
    def min_history(self):
        return self.period_samples * self.min_cycles


class HistoricalDecompositionCache:
    """In-memory cache keyed by source fingerprint, settings, and forecast origin."""

    def __init__(self):
        self._values = {}

    def get_or_compute(self, frame: np.ndarray, origin, channels: Sequence[str],
                       training_scales: dict, config: DecompositionConfig,
                       source_fingerprint: str):
        config.validate()
        window = _validate_window(frame, channels, training_scales, config)
        stamp = str(pd.to_datetime(origin, utc=True, format="mixed"))
        data_hash = hashlib.sha256(window.tobytes()).hexdigest()
        cfg_hash = hashlib.sha256(json.dumps({
            **asdict(config), "lookback_steps": len(window),
            "channels": list(channels), "training_scales": training_scales,
        }, sort_keys=True).encode("utf-8")).hexdigest()
        key = (source_fingerprint, cfg_hash, stamp, data_hash)
        if key not in self._values:
            self._values[key] = _decompose_validated(window, channels, training_scales, config)
        return self._values[key].copy(), {
            "forecast_origin_utc": stamp,
            "source_data_fingerprint": source_fingerprint,
            "decomposition_config_fingerprint": cfg_hash,
        }

    def __len__(self):
        return len(self._values)


def _validate_window(values, channels, training_scales, config):
    config.validate()
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(channels):
        raise ValueError("DLSTM window must have shape [lookback, selected raw channels].")
    if raw.shape[0] < config.min_history:
        raise ValueError(
            f"STL requires at least {config.min_history} samples; received {raw.shape[0]}."
        )
    if not np.isfinite(raw).all():
        raise ValueError("STL window has missing or invalid measurements; sample must be excluded.")
    if (raw < 0).any():
        raise ValueError("STL window contains negative flux; invalid values are not transformed or imputed.")
    for channel in channels:
        scale = training_scales.get(channel)
        if scale is None or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"No valid training-only positive reference scale is recorded for {channel}.")
    return raw


def _decompose_validated(raw, channels, training_scales, config):
    output = np.empty((len(raw), len(channels) * len(COMPONENTS)), dtype=np.float32)
    for channel_index, channel in enumerate(channels):
        values = np.log1p(raw[:, channel_index] / float(training_scales[channel]))
        components = STL(values, period=config.period_samples, robust=config.robust).fit()
        start = channel_index * len(COMPONENTS)
        output[:, start:start + 3] = np.column_stack(
            (components.trend, components.seasonal, components.resid)
        ).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("STL returned non-finite endpoint components; sample must be excluded.")
    return output


def decompose_historical_window(values, channels, training_scales, config=None):
    """Return [time, channel-major (trend, seasonal, residual)] for one origin."""
    cfg = (config or DecompositionConfig()).validate()
    raw = _validate_window(values, channels, training_scales, cfg)
    return _decompose_validated(raw, channels, training_scales, cfg)


def decompose_historical_windows(windows, origins, channels, training_scales,
                                config=None, source_fingerprint="", cache=None):
    """Precompute origin-scoped decompositions; invalid windows fail explicitly."""
    cfg = (config or DecompositionConfig()).validate()
    windows = np.asarray(windows)
    if windows.ndim != 3 or len(windows) != len(origins):
        raise ValueError("Windows and forecast origins must be aligned [samples, time, channels].")
    cache = cache or HistoricalDecompositionCache()
    result = []
    metadata = []
    for window, origin in zip(windows, origins):
        values, info = cache.get_or_compute(window, origin, channels, training_scales, cfg, source_fingerprint)
        result.append(values)
        metadata.append(info)
    if not result:
        return np.empty((0, windows.shape[1], len(channels) * 3), dtype=np.float32), metadata
    return np.stack(result), metadata


def decomposition_feature_names(channels):
    return [f"{channel}::{component}" for channel in channels for component in COMPONENTS]
