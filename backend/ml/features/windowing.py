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


def build_windows(df: pd.DataFrame, config: WindowConfig = None,
                  context_df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    target = df.copy().reset_index(drop=True)
    context = context_df.copy().reset_index(drop=True) if context_df is not None else target.iloc[:0].copy()
    if len(context):
        missing_context = [c for c in cols + ["timestamp"] if c not in context.columns]
        if missing_context:
            raise ValueError(f"Missing required historical context columns: {missing_context}")
        context = context.tail(max(0, cfg.sequence_length - 1)).copy()
    timestamps = pd.to_datetime(target["timestamp"], utc=True, errors="coerce", format="mixed")
    if timestamps.isna().any():
        raise ValueError("Window timestamps must be valid UTC-compatible values.")
    if timestamps.duplicated().any():
        raise ValueError("Window timestamps must not contain duplicates.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Window observations must be ordered by timestamp.")

    context_timestamps = pd.to_datetime(context["timestamp"], utc=True, errors="coerce", format="mixed")
    if context_timestamps.isna().any() or context_timestamps.duplicated().any() or not context_timestamps.is_monotonic_increasing:
        raise ValueError("Historical context timestamps must be valid, unique, and ordered.")
    if len(context_timestamps) and len(timestamps) and context_timestamps.iloc[-1] >= timestamps.iloc[0]:
        raise ValueError("Historical context must end strictly before the target partition.")
    combined = pd.concat([context, target], ignore_index=True)
    combined_timestamps = pd.concat([context_timestamps, timestamps], ignore_index=True)
    data = combined[cols].values.astype(np.float32)
    n = len(combined)
    target_offset = len(context)
    seq_len = cfg.sequence_length

    if n < seq_len or len(target) == 0:
        return (np.empty((0, seq_len, len(cols)), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0, len(cfg.horizon_cols)), dtype=np.float32),
                np.empty((0,), dtype="datetime64[ns]"))

    starts = np.arange(0, n - seq_len + 1, cfg.stride)
    end_idx = starts + seq_len - 1
    # Every returned forecast origin belongs to df; context is input-only.
    keep = end_idx >= target_offset
    starts, end_idx = starts[keep], end_idx[keep]
    if not len(starts):
        return (np.empty((0, seq_len, len(cols)), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0, len(cfg.horizon_cols)), dtype=np.float32),
                np.empty((0,), dtype="datetime64[ns]"))
    X = np.stack([data[s:s + seq_len] for s in starts])
    target_end_idx = end_idx - target_offset
    nowcast_rows = pd.to_numeric(target["state"], errors="coerce").to_numpy(dtype=float)[target_end_idx]
    forecast_rows = target[cfg.horizon_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[target_end_idx]
    end_timestamps = np.asarray(combined_timestamps.iloc[end_idx].tolist(), dtype=object)

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


def chronological_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15,
                        train_end=None, val_end=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    CRITICAL: split time series chronologically (never shuffle) so that no
    future information leaks into training. Earliest 70% -> train,
    next 15% -> validation, final 15% (most recent) -> test.
    """
    if "timestamp" not in df:
        raise ValueError("Chronological split requires a timestamp column.")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", format="mixed")
    if df["timestamp"].isna().any():
        raise ValueError("Chronological split timestamps must be valid UTC-compatible values.")
    if df["timestamp"].duplicated().any():
        raise ValueError("Chronological split timestamps must not contain duplicates.")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    n = len(df)
    if n < 3:
        raise ValueError("Chronological split requires at least three observations.")
    if (train_end is None) != (val_end is None):
        raise ValueError("Provide both train_end and val_end, or neither.")
    if train_end is not None:
        train_boundary = _split_timestamp(train_end, "train_end")
        val_boundary = _split_timestamp(val_end, "val_end")
        if train_boundary >= val_boundary:
            raise ValueError("Split boundaries must be strictly ordered: train_end < val_end.")
        train = df.loc[df.timestamp < train_boundary].reset_index(drop=True)
        val = df.loc[(df.timestamp >= train_boundary) & (df.timestamp < val_boundary)].reset_index(drop=True)
        test = df.loc[df.timestamp >= val_boundary].reset_index(drop=True)
    else:
        if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
            raise ValueError("Split fractions must be positive and sum to less than one.")
        train_idx = int(n * train_frac)
        val_idx = int(n * (train_frac + val_frac))
        if train_idx == 0 or val_idx <= train_idx or val_idx >= n:
            raise ValueError("Split fractions create an empty partition; provide more observations or adjust fractions.")
        train = df.iloc[:train_idx].reset_index(drop=True)
        val = df.iloc[train_idx:val_idx].reset_index(drop=True)
        test = df.iloc[val_idx:].reset_index(drop=True)
    if any(part.empty for part in (train, val, test)):
        raise ValueError("Split boundaries must produce three non-empty chronological partitions.")
    assert train.timestamp.max() < val.timestamp.min() <= val.timestamp.max() < test.timestamp.min()
    return train, val, test


def _split_timestamp(value, name):
    timestamp = pd.to_datetime(value, utc=True, errors="coerce", format="mixed")
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid UTC-compatible timestamp.")
    return timestamp


def build_split_audit(partitions, horizon_cols=None, split_config=None, provenance=None,
                      minimum_mx_events=5, sequence_length=60,
                      channel_cols=None, event_catalog=None, context_frames=None):
    """Return reproducible per-partition audit rows; synthetic provenance is explicit."""
    horizons = list(horizon_cols or ["forecast_1h"])
    rows = []
    provenance = provenance or {}
    synthetic = provenance.get("source") == "demo" or provenance.get("synthetic") is True
    for name, frame in partitions.items():
        timestamps = pd.to_datetime(frame.get("timestamp", pd.Series(dtype=object)), utc=True, errors="coerce", format="mixed")
        target_name = horizons[0] if horizons else None
        labels = pd.to_numeric(frame[target_name], errors="coerce") if target_name in frame else pd.Series(np.nan, index=frame.index)
        known = labels.isin([0, 1])
        events = frame.get("flare_class", pd.Series("-", index=frame.index)).fillna("-").astype(str).str.upper().str.extract(r"^\s*([BCMX])", expand=False)
        state = pd.to_numeric(frame.get("state", pd.Series(np.nan, index=frame.index)), errors="coerce")
        observed_event = state.eq(2) & events.isin(["C", "M", "X"])
        event_starts = observed_event & ~observed_event.shift(1, fill_value=False)
        if event_catalog is not None:
            partition_start = timestamps.min()
            partition_end = timestamps.max()
            catalog = pd.DataFrame(event_catalog, columns=["timestamp", "flare_class_group"])
            if len(catalog):
                catalog["timestamp"] = pd.to_datetime(catalog["timestamp"], utc=True, errors="coerce", format="mixed")
                event_groups = catalog.loc[
                    catalog.timestamp.between(partition_start, partition_end, inclusive="both"),
                    "flare_class_group",
                ]
            else:
                event_groups = pd.Series(dtype=object)
            event_count = int(len(event_groups))
        else:
            event_groups = events[event_starts]
            event_count = int(event_starts.sum())
        channels = list(channel_cols or ["solexs_flux_scaled", "helios_flux_scaled"])
        context = (context_frames or {}).get(name)
        forecast_window_counts = {}
        valid_windows = 0
        if channels and sequence_length > 0 and set(channels).issubset(frame.columns) and all(c in frame for c in horizons):
            window_config = WindowConfig(sequence_length=sequence_length, channel_cols=channels,
                                         horizon_cols=horizons)
            X, _, window_targets, origins = build_windows(frame, window_config, context_df=context)
            valid_windows = int(len(X))
            origin_labels = frame.set_index(pd.to_datetime(frame.timestamp, utc=True, format="mixed"))
            origin_index = pd.to_datetime(origins, utc=True, format="mixed")
            aligned_labels = origin_labels.reindex(origin_index)[horizons].apply(pd.to_numeric, errors="coerce")
            for column in horizons:
                target = aligned_labels[column].to_numpy(dtype=float)
                known_target = np.isfinite(target)
                forecast_window_counts[column] = {
                    "known": int(known_target.sum()),
                    "positive": int(np.sum(target[known_target] == 1)),
                    "negative": int(np.sum(target[known_target] == 0)),
                    "unknown": int((~known_target).sum()),
                }
        else:
            forecast_window_counts = {column: {"known": 0, "positive": 0, "negative": 0,
                                                "unknown": 0} for column in horizons}
        rows.append({
            "partition": name,
            "start_timestamp_utc": timestamps.min().isoformat() if timestamps.notna().any() else None,
            "end_timestamp_utc": timestamps.max().isoformat() if timestamps.notna().any() else None,
            "raw_observations": int(len(frame)), "valid_supervised_windows": valid_windows,
            "audit_sequence_length": int(sequence_length),
            "historical_context_observations": int(len(context)) if context is not None else 0,
            "forecast_window_counts": forecast_window_counts,
            "positive_forecast_examples": int(labels.eq(1).sum()), "negative_forecast_examples": int(labels.eq(0).sum()),
            "excluded_or_ambiguous_examples": int((~known).sum()),
            "distinct_observed_flare_events": event_count,
            "distinct_observed_mx_events": int(event_groups.isin(["M", "X"]).sum()),
            "c_class_events": int(event_groups.eq("C").sum()), "m_class_events": int(event_groups.eq("M").sum()),
            "x_class_events": int(event_groups.eq("X").sum()),
            "positive_prevalence": float(labels[known].eq(1).mean()) if known.any() else None,
            "missing_timestamp_count": int(timestamps.isna().sum()), "missing_label_count": int((~known).sum()),
            "windows_excluded_by_purge": int(provenance.get("purged", {}).get(name, {}).get(target_name, 0)),
            "exclusion_reasons": "unknown/missing target" if (~known).any() else "",
            "event_observation_status": "synthetic_fixture" if synthetic else ("verified" if provenance.get("event_metadata_verified") else "unverified_source"),
            "positive_horizon": target_name,
            "rare_event_evaluation_status": (
                "synthetic_fixture_not_scientific_evaluation" if synthetic else
                "unverified_event_metadata" if not provenance.get("event_metadata_verified") else
                "inconclusive_insufficient_distinct_mx_events"
                if int(event_groups.isin(["M", "X"]).sum()) < minimum_mx_events else "event_count_threshold_met"
            ),
            "minimum_mx_events_required": int(minimum_mx_events),
        })
    return {"split_config": split_config or {}, "provenance": provenance, "partitions": rows}
