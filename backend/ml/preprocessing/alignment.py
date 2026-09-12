"""
Time alignment / resampling stage.
Aligns SoLEXS + HEL1OS onto a common regular time grid at a configurable cadence.
"""
import pandas as pd
from dataclasses import dataclass

VALID_CADENCES = {"1s": 1, "10s": 10, "1min": 60, "5min": 300}


@dataclass
class AlignmentConfig:
    target_cadence: str = "1min"   # one of VALID_CADENCES
    agg: str = "mean"              # aggregation when downsampling


def align_and_resample(df: pd.DataFrame, config: AlignmentConfig = None) -> tuple[pd.DataFrame, dict]:
    cfg = config or AlignmentConfig()
    if cfg.target_cadence not in VALID_CADENCES:
        raise ValueError(f"Unsupported cadence '{cfg.target_cadence}'. Choose from {list(VALID_CADENCES)}")

    df = df.copy()
    df = df.set_index("timestamp").sort_index()

    freq_map = {"1s": "1s", "10s": "10s", "1min": "1min", "5min": "5min"}
    rule = freq_map[cfg.target_cadence]

    numeric_cols = ["solexs_flux", "helios_flux"]
    agg_dict = {c: cfg.agg for c in numeric_cols if c in df.columns}

    resampled = df[list(agg_dict.keys())].resample(rule).agg(agg_dict)

    # Carry forward categorical labels (state / flare_class) using the mode/last value in bucket
    for cat_col in ["state", "flare_class"]:
        if cat_col in df.columns:
            resampled[cat_col] = df[cat_col].resample(rule).agg(
                lambda s: s.mode().iloc[0] if not s.mode().empty else (s.iloc[-1] if len(s) else None)
            )

    report = {
        "input_rows": len(df),
        "output_rows": len(resampled),
        "target_cadence": cfg.target_cadence,
        "date_range_start": str(resampled.index.min()) if len(resampled) else None,
        "date_range_end": str(resampled.index.max()) if len(resampled) else None,
        "missing_after_resample": int(resampled[list(agg_dict.keys())].isna().sum().sum()),
    }

    resampled = resampled.reset_index().rename(columns={"index": "timestamp"})
    return resampled, report
