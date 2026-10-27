"""Causal data validation and quality reporting; no interpolation or zero fill."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class CleaningConfig:
    drop_duplicate_timestamps: bool = True  # explicit policy: stable keep-first
    enforce_monotonic: bool = True
    negative_flux_strategy: str = "nan"  # negative flux is invalid, retained as missing
    spike_zscore_threshold: float = 6.0
    spike_strategy: str = "flag_only"  # observations are retained; flags are features
    spike_lookback_minutes: int = 10
    spike_min_samples: int = 5
    max_gap_interpolate_steps: int = 5  # legacy config field; interpolation is disabled
    min_valid_flux: float = 1e-12  # legacy config field; no epsilon/clipping is applied


def clean_dataframe(df: pd.DataFrame, config: CleaningConfig = None) -> tuple[pd.DataFrame, dict]:
    cfg = config or CleaningConfig()
    if cfg.negative_flux_strategy != "nan":
        raise ValueError("Negative flux clipping is disabled; invalid negative measurements are marked missing.")
    if cfg.spike_strategy != "flag_only":
        raise ValueError("Spike interpolation/correction is disabled; use causal flag_only detection.")
    if cfg.spike_lookback_minutes <= 0 or cfg.spike_min_samples < 2 or cfg.spike_zscore_threshold <= 0:
        raise ValueError("Spike lookback, minimum sample count, and threshold must be positive.")
    absent = [c for c in ("timestamp", "solexs_flux", "helios_flux") if c not in df]
    if absent:
        raise ValueError(f"Cleaning requires columns: {absent}")

    out = df.copy()
    raw_ts = pd.to_datetime(out.timestamp, utc=True, errors="coerce", format="mixed")
    report = {
        "input_rows": len(out), "output_rows": 0,
        "invalid_timestamps_removed": int(raw_ts.isna().sum()),
        "duplicates_removed": 0, "out_of_order_adjacent_pairs": int(raw_ts.dropna().diff().lt(pd.Timedelta(0)).sum()),
        "negative_values_marked_missing": 0, "spikes_detected": 0, "spikes_corrected": 0,
        "nan_count_before": int(out[["solexs_flux", "helios_flux"]].isna().sum().sum()),
        "nan_count_after_flagging": 0, "nan_count_final": 0,
        "large_gaps_left_as_nan": 0, "interpolation": "none",
        "duplicate_policy": "stable keep-first" if cfg.drop_duplicate_timestamps else "reject",
        "spike_policy": "prior-only elapsed-time median/MAD flags; measurements are retained",
        "spike_lookback_minutes": cfg.spike_lookback_minutes,
        "spike_min_samples": cfg.spike_min_samples,
        "spike_threshold": cfg.spike_zscore_threshold,
        "warnings": [],
    }
    out["timestamp"] = raw_ts
    out = out.dropna(subset=["timestamp"])
    if cfg.drop_duplicate_timestamps:
        report["duplicates_removed"] = int(out.timestamp.duplicated().sum())
        out = out.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="first")
    elif out.timestamp.duplicated().any():
        raise ValueError("Duplicate timestamps are not allowed by the selected cleaning policy.")
    if cfg.enforce_monotonic:
        out = out.sort_values("timestamp", kind="stable")
    elif not out.timestamp.is_monotonic_increasing:
        raise ValueError("Input observations are out of order and enforce_monotonic is disabled.")
    out = out.reset_index(drop=True)

    times = pd.DatetimeIndex(out.timestamp)
    for column in ("solexs_flux", "helios_flux"):
        numeric = pd.to_numeric(out[column], errors="coerce").astype(float)
        malformed = out[column].notna() & numeric.isna()
        if malformed.any():
            report["warnings"].append(f"{int(malformed.sum())} non-numeric values in {column} were marked missing.")
        negative = numeric < 0
        report["negative_values_marked_missing"] += int(negative.sum())
        numeric.loc[negative] = np.nan
        # Causal robust spike count is diagnostic only; no peak is replaced,
        # since real flare peaks can be scientifically important.
        values = numeric.to_numpy(dtype=float)
        spike_count = 0
        span = pd.Timedelta(minutes=cfg.spike_lookback_minutes).value
        time_values = times.asi8
        for i, stamp in enumerate(time_values):
            start = np.searchsorted(time_values, stamp - span, side="left")
            history = values[start:i]
            history = history[np.isfinite(history)]
            if len(history) < cfg.spike_min_samples or not np.isfinite(values[i]):
                continue
            median = float(np.median(history))
            mad = float(np.median(np.abs(history - median)))
            if mad > 0 and abs(values[i] - median) / (1.4826 * mad) >= cfg.spike_zscore_threshold:
                spike_count += 1
        report["spikes_detected"] += spike_count
        out[column] = numeric

    report["nan_count_after_flagging"] = int(out[["solexs_flux", "helios_flux"]].isna().sum().sum())
    report["nan_count_final"] = report["nan_count_after_flagging"]
    deltas = out.timestamp.diff().dt.total_seconds()
    report["observation_gaps_over_3x_median_cadence"] = int(
        (deltas > deltas.dropna().median() * 3).sum()
    ) if deltas.notna().any() else 0
    report["large_gaps_left_as_nan"] = report["observation_gaps_over_3x_median_cadence"]
    report["output_rows"] = len(out)
    return out, report
