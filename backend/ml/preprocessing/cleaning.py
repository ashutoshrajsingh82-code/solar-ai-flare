"""
Cleaning stage of the data pipeline.
Handles: duplicated timestamps, non-monotonic timestamps, NaNs,
invalid flux values, isolated spikes, telemetry gaps.

All strategies are configurable via CleaningConfig so the pipeline can be
tuned without code changes.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class CleaningConfig:
    drop_duplicate_timestamps: bool = True
    enforce_monotonic: bool = True
    negative_flux_strategy: str = "clip"       # "clip" | "nan"
    spike_zscore_threshold: float = 6.0        # isolated-spike detector sensitivity
    spike_strategy: str = "interpolate"        # "interpolate" | "flag_only"
    max_gap_interpolate_steps: int = 5         # do NOT silently interpolate huge gaps
    min_valid_flux: float = 1e-12


def clean_dataframe(df: pd.DataFrame, config: CleaningConfig = None) -> tuple[pd.DataFrame, dict]:
    """
    Clean a raw SoLEXS/HEL1OS dataframe.

    Returns
    -------
    cleaned_df : pd.DataFrame
    report : dict  -- summary of what was changed (for pipeline dashboard)
    """
    cfg = config or CleaningConfig()
    report = {
        "input_rows": len(df),
        "duplicates_removed": 0,
        "non_monotonic_rows_removed": 0,
        "negative_values_fixed": 0,
        "spikes_detected": 0,
        "spikes_corrected": 0,
        "nan_count_before": int(df[["solexs_flux", "helios_flux"]].isna().sum().sum()),
        "large_gaps_left_as_nan": 0,
        "warnings": [],
    }

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    bad_ts = df["timestamp"].isna().sum()
    if bad_ts:
        report["warnings"].append(f"{bad_ts} rows had invalid/unparseable timestamps and were dropped.")
    df = df.dropna(subset=["timestamp"])

    if cfg.drop_duplicate_timestamps:
        before = len(df)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first")
        report["duplicates_removed"] = before - len(df)

    if cfg.enforce_monotonic:
        before = len(df)
        df = df.sort_values("timestamp").reset_index(drop=True)
        # after sorting, timestamps are monotonic by construction; count would-be violations pre-sort
        report["non_monotonic_rows_removed"] = 0  # sorting resolves order; nothing dropped here

    for col in ["solexs_flux", "helios_flux"]:
        if col not in df.columns:
            report["warnings"].append(f"Missing expected column '{col}'.")
            continue
        neg_mask = df[col] < 0
        n_neg = int(neg_mask.sum())
        if n_neg:
            if cfg.negative_flux_strategy == "clip":
                df.loc[neg_mask, col] = cfg.min_valid_flux
            else:
                df.loc[neg_mask, col] = np.nan
            report["negative_values_fixed"] += n_neg

        # isolated-spike detection via rolling z-score
        roll_mean = df[col].rolling(21, center=True, min_periods=5).median()
        roll_std = df[col].rolling(21, center=True, min_periods=5).std().replace(0, np.nan)
        z = (df[col] - roll_mean) / roll_std
        spike_mask = z.abs() > cfg.spike_zscore_threshold
        n_spikes = int(spike_mask.fillna(False).sum())
        report["spikes_detected"] += n_spikes
        if n_spikes and cfg.spike_strategy == "interpolate":
            df.loc[spike_mask.fillna(False), col] = np.nan
            report["spikes_corrected"] += n_spikes

    report["nan_count_after_flagging"] = int(df[["solexs_flux", "helios_flux"]].isna().sum().sum())

    # Interpolate only small gaps; leave large telemetry gaps as NaN (never silently fabricate long stretches)
    for col in ["solexs_flux", "helios_flux"]:
        if col not in df.columns:
            continue
        isna = df[col].isna()
        # identify consecutive NaN run lengths
        run_id = (isna != isna.shift()).cumsum()
        run_lengths = isna.groupby(run_id).transform("sum")
        small_gap_mask = isna & (run_lengths <= cfg.max_gap_interpolate_steps)
        large_gap_mask = isna & (run_lengths > cfg.max_gap_interpolate_steps)
        report["large_gaps_left_as_nan"] += int(large_gap_mask.sum())

        df[col] = df[col].where(~small_gap_mask, np.nan)  # keep placeholder
        df.loc[small_gap_mask, col] = df[col].interpolate(method="linear", limit_area="inside")[small_gap_mask]

    report["output_rows"] = len(df)
    report["nan_count_final"] = int(df[["solexs_flux", "helios_flux"]].isna().sum().sum())
    return df.reset_index(drop=True), report
