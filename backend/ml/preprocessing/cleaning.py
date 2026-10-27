
"""
Cleaning stage of the data pipeline.

Purpose:
Clean raw SoLEXS and HEL1OS data before using it for ML.

This stage handles:
1. Invalid timestamps
2. Duplicate timestamps
3. Unsorted timestamps
4. Negative flux values
5. Unusual spikes in flux readings
6. Missing values (NaNs)
7. Small and large telemetry gaps

All cleaning strategies can be configured through CleaningConfig.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class CleaningConfig:
    # Remove rows that have the same timestamp.
    drop_duplicate_timestamps: bool = True

    # Sort the data chronologically by timestamp.
    enforce_monotonic: bool = True

    # How to handle negative flux values:
    # "clip" replaces them with a small positive value.
    # "nan" replaces them with NaN (missing value).
    negative_flux_strategy: str = "clip"

    # Sensitivity of the spike detector.
    # A higher threshold detects fewer spikes.
    spike_zscore_threshold: float = 6.0

    # "interpolate" replaces detected spikes with estimated values.
    # "flag_only" detects spikes but does not change them.
    spike_strategy: str = "interpolate"

    # Fill only gaps containing at most 5 consecutive missing readings.
    # Larger gaps are left as NaN to avoid inventing too much data.
    max_gap_interpolate_steps: int = 5

    # Minimum positive flux value used when clipping negative values.
    min_valid_flux: float = 1e-12


def clean_dataframe(
    df: pd.DataFrame,
    config: CleaningConfig = None
) -> tuple[pd.DataFrame, dict]:

    # Use default cleaning settings if no configuration is provided.
    cfg = config or CleaningConfig()

    # Keep a report of the changes made during cleaning.
    # This helps us understand data quality and display statistics
    # on the pipeline dashboard.
    report = {
        "input_rows": len(df),
        "duplicates_removed": 0,
        "non_monotonic_rows_removed": 0,
        "negative_values_fixed": 0,
        "spikes_detected": 0,
        "spikes_corrected": 0,
        "nan_count_before": int(
            df[["solexs_flux", "helios_flux"]].isna().sum().sum()
        ),
        "large_gaps_left_as_nan": 0,
        "warnings": [],
    }

    # Work on a copy so that the original dataframe is not modified.
    df = df.copy()

    # Convert timestamps into a consistent UTC datetime format.
    # Invalid timestamps become NaN (NaT) and are removed below.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    )

    # Remove rows whose timestamps could not be interpreted.
    bad_ts = df["timestamp"].isna().sum()
    if bad_ts:
        report["warnings"].append(
            f"{bad_ts} rows had invalid/unparseable timestamps and were dropped."
        )

    df = df.dropna(subset=["timestamp"])

    # Remove duplicate timestamps, keeping the first occurrence.
    # This avoids multiple observations with the same timestamp.
    if cfg.drop_duplicate_timestamps:
        before = len(df)

        df = (
            df.sort_values("timestamp")
              .drop_duplicates(subset=["timestamp"], keep="first")
        )

        report["duplicates_removed"] = before - len(df)

    # Sort observations from earliest to latest.
    # Sorting makes timestamps chronological; rows are not removed here.
    if cfg.enforce_monotonic:
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Sorting resolves the ordering issue, so no rows are removed.
        report["non_monotonic_rows_removed"] = 0

    # Clean each instrument's flux column separately.
    for col in ["solexs_flux", "helios_flux"]:

        # Warn if an expected flux column is missing.
        if col not in df.columns:
            report["warnings"].append(
                f"Missing expected column '{col}'."
            )
            continue

        # Flux values should not be negative under this pipeline's
        # assumed data convention.
        neg_mask = df[col] < 0
        n_neg = int(neg_mask.sum())

        if n_neg:
            if cfg.negative_flux_strategy == "clip":
                # Replace negative values with a small positive number.
                df.loc[neg_mask, col] = cfg.min_valid_flux
            else:
                # Mark negative values as missing so they can be handled later.
                df.loc[neg_mask, col] = np.nan

            report["negative_values_fixed"] += n_neg

        # Detect unusual spikes using a rolling window of 21 readings.
        # The rolling median estimates the typical local flux level.
        # The rolling standard deviation measures local variation.
        # A large z-score means a reading is far from the local median.
        roll_mean = df[col].rolling(
            21, center=True, min_periods=5
        ).median()

        roll_std = (
            df[col].rolling(21, center=True, min_periods=5)
            .std()
            .replace(0, np.nan)
        )

        z = (df[col] - roll_mean) / roll_std

        # Flag readings whose absolute z-score exceeds the threshold.
        spike_mask = z.abs() > cfg.spike_zscore_threshold
        n_spikes = int(spike_mask.fillna(False).sum())

        report["spikes_detected"] += n_spikes

        if n_spikes and cfg.spike_strategy == "interpolate":
            # Mark detected spikes as missing.
            # They will be interpolated later if they belong to a small gap.
            # WARNING: genuine solar flare peaks could also be flagged.
            df.loc[spike_mask.fillna(False), col] = np.nan
            report["spikes_corrected"] += n_spikes

    # Count missing flux values after negative-value and spike handling.
    report["nan_count_after_flagging"] = int(
        df[["solexs_flux", "helios_flux"]].isna().sum().sum()
    )

    # Handle missing readings and telemetry gaps.
    for col in ["solexs_flux", "helios_flux"]:

        if col not in df.columns:
            continue

        # Identify which rows contain missing flux values.
        isna = df[col].isna()

        # Give each consecutive run of missing/non-missing values
        # a group ID, allowing us to calculate the length of each gap.
        run_id = (isna != isna.shift()).cumsum()
        run_lengths = isna.groupby(run_id).transform("sum")

        # A small gap contains at most the configured number of
        # consecutive missing readings.
        small_gap_mask = isna & (
            run_lengths <= cfg.max_gap_interpolate_steps
        )

        # Larger gaps are preserved because estimating long stretches
        # could fabricate unrealistic instrument measurements.
        large_gap_mask = isna & (
            run_lengths > cfg.max_gap_interpolate_steps
        )

        report["large_gaps_left_as_nan"] += int(
            large_gap_mask.sum()
        )

        # Keep missing values as placeholders before interpolation.
        df[col] = df[col].where(~small_gap_mask, np.nan)

        # Estimate small internal gaps using linear interpolation.
        # This fills missing values between known readings.
        # Gaps at the beginning/end and large gaps remain missing.
        df.loc[small_gap_mask, col] = (
            df[col]
            .interpolate(method="linear", limit_area="inside")[
                small_gap_mask
            ]
        )

    # Record final statistics for the cleaning report.
    report["output_rows"] = len(df)

    report["nan_count_final"] = int(
        df[["solexs_flux", "helios_flux"]].isna().sum().sum()
    )

    # Reset the row index and return both the cleaned data and report.
    return df.reset_index(drop=True), report
