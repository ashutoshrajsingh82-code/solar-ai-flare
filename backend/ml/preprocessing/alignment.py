
"""
Time alignment / resampling stage.
Aligns observations onto a common regular time grid at a configurable cadence.
Forecast targets are preserved conservatively: a bin receives a known target
only when every target in that bin is known and all targets agree.
"""
import pandas as pd
from dataclasses import dataclass

VALID_CADENCES = {"1s": 1, "10s": 10, "1min": 60, "5min": 300}


@dataclass
class AlignmentConfig:
    target_cadence: str = "1min"
    agg: str = "mean"


def _aggregate_forecast_target(series: pd.Series):
    """Never average binary labels or turn unknown labels into negatives."""
    values = pd.to_numeric(series, errors="coerce")

    if values.empty or values.isna().any():
        return float("nan")

    if not values.isin([0, 1]).all():
        return float("nan")

    unique_values = values.unique()
    if len(unique_values) != 1:
        return float("nan")

    return float(unique_values[0])


def align_and_resample(
    df: pd.DataFrame,
    config: AlignmentConfig = None,
) -> tuple[pd.DataFrame, dict]:
    cfg = config or AlignmentConfig()
    if cfg.target_cadence not in VALID_CADENCES:
        raise ValueError(
            f"Unsupported cadence '{cfg.target_cadence}'. "
            f"Choose from {list(VALID_CADENCES)}"
        )

    if "timestamp" not in df.columns:
        raise ValueError("Missing required timestamp column.")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    if df["timestamp"].isna().any():
        raise ValueError("Timestamps must be valid UTC-compatible values.")

    df = df.set_index("timestamp").sort_index()
    rule = {
        "1s": "1s",
        "10s": "10s",
        "1min": "1min",
        "5min": "5min",
    }[cfg.target_cadence]

    numeric_cols = ["solexs_flux", "helios_flux"]
    agg_dict = {c: cfg.agg for c in numeric_cols if c in df.columns}

    if agg_dict:
        resampled = df[list(agg_dict)].resample(rule).agg(agg_dict)
    else:
        resampled = pd.DataFrame(
            index=df.resample(rule).size().index
        )

    # Preserve independent forecast targets without averaging them.
    forecast_cols = [
        c for c in df.columns
        if c.startswith("forecast_")
    ]
    for col in forecast_cols:
        resampled[col] = df[col].resample(rule).agg(
            _aggregate_forecast_target
        )

    # Retain existing categorical behavior.
    for cat_col in ["state", "flare_class"]:
        if cat_col in df.columns:
            resampled[cat_col] = df[cat_col].resample(rule).agg(
                lambda s: (
                    s.mode().iloc[0]
                    if not s.mode().empty
                    else (s.iloc[-1] if len(s) else None)
                )
            )

    report = {
        "input_rows": len(df),
        "output_rows": len(resampled),
        "target_cadence": cfg.target_cadence,
        "date_range_start": (
            str(resampled.index.min()) if len(resampled) else None
        ),
        "date_range_end": (
            str(resampled.index.max()) if len(resampled) else None
        ),
        "missing_after_resample": (
            int(resampled[list(agg_dict)].isna().sum().sum())
            if agg_dict else 0
        ),
        "forecast_target_columns_preserved": forecast_cols,
    }

    return resampled.reset_index().rename(
        columns={"index": "timestamp"}
    ), report
