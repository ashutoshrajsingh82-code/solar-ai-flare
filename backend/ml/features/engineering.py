"""
Feature engineering module.

Produces rolling-window statistical features for SoLEXS + HEL1OS, plus the
scientifically-central Hard-to-Soft Ratio (HEL1OS / SoLEXS), and optional
STL decomposition (trend/seasonal/residual) for regularly-sampled sequences.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List

EPS = 1e-12
ROLL_WINDOWS_MIN = [5, 10, 30, 60]  # minutes


@dataclass
class FeatureConfig:
    cadence_seconds: int = 60
    windows_minutes: List[int] = field(default_factory=lambda: ROLL_WINDOWS_MIN)
    compute_stl: bool = False
    stl_period_minutes: int = 60


def _steps_for_minutes(minutes: int, cadence_seconds: int) -> int:
    return max(2, int(minutes * 60 / cadence_seconds))


def add_rolling_features(df: pd.DataFrame, config: FeatureConfig = None) -> pd.DataFrame:
    cfg = config or FeatureConfig()
    out = df.copy()

    for channel in ["solexs_flux", "helios_flux"]:
        if channel not in out.columns:
            continue
        prefix = channel.split("_")[0]  # "solexs" / "helios"
        series = out[channel]

        # simple 1-step rate of change / gradient
        out[f"{prefix}_diff1"] = series.diff()
        out[f"{prefix}_pct_change"] = series.pct_change().replace([np.inf, -np.inf], np.nan)

        for w_min in cfg.windows_minutes:
            w = _steps_for_minutes(w_min, cfg.cadence_seconds)
            roll = series.rolling(window=w, min_periods=max(2, w // 3))
            out[f"{prefix}_mean_{w_min}m"] = roll.mean()
            out[f"{prefix}_median_{w_min}m"] = roll.median()
            out[f"{prefix}_std_{w_min}m"] = roll.std()
            out[f"{prefix}_min_{w_min}m"] = roll.min()
            out[f"{prefix}_max_{w_min}m"] = roll.max()
            out[f"{prefix}_var_{w_min}m"] = roll.var()
            out[f"{prefix}_p90_{w_min}m"] = roll.quantile(0.90)
            out[f"{prefix}_recent_peak_{w_min}m"] = roll.max()
            out[f"{prefix}_flux_diff_{w_min}m"] = series - series.shift(w)
            # slope via simple linear regression over the window
            out[f"{prefix}_slope_{w_min}m"] = series.rolling(w).apply(
                lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] if np.all(np.isfinite(y)) else np.nan,
                raw=True,
            )

    # ---- CRITICAL DOMAIN FEATURE: Hard-to-Soft Ratio ----
    if "solexs_flux" in out.columns and "helios_flux" in out.columns:
        out["hard_soft_ratio"] = out["helios_flux"] / (out["solexs_flux"] + EPS)
        out["hard_soft_ratio_change"] = out["hard_soft_ratio"].diff()
        for w_min in [5, 10, 30]:
            w = _steps_for_minutes(w_min, cfg.cadence_seconds)
            roll = out["hard_soft_ratio"].rolling(window=w, min_periods=max(2, w // 3))
            out[f"hard_soft_ratio_mean_{w_min}m"] = roll.mean()
            out[f"hard_soft_ratio_var_{w_min}m"] = roll.var()
            out[f"hard_soft_ratio_slope_{w_min}m"] = out["hard_soft_ratio"].rolling(w).apply(
                lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] if np.all(np.isfinite(y)) else np.nan,
                raw=True,
            )

    if cfg.compute_stl:
        out = add_stl_features(out, cfg)

    return out


def add_stl_features(df: pd.DataFrame, config: FeatureConfig = None) -> pd.DataFrame:
    """Optional STL decomposition. Skipped gracefully if sequence too short."""
    from statsmodels.tsa.seasonal import STL

    cfg = config or FeatureConfig()
    out = df.copy()
    period_steps = _steps_for_minutes(cfg.stl_period_minutes, cfg.cadence_seconds)

    for channel in ["solexs_flux", "helios_flux"]:
        if channel not in out.columns:
            continue
        series = out[channel].interpolate(limit_direction="both")
        if len(series) < period_steps * 2 or series.isna().all():
            out[f"{channel}_stl_trend"] = np.nan
            out[f"{channel}_stl_seasonal"] = np.nan
            out[f"{channel}_stl_resid"] = np.nan
            continue
        try:
            log_series = np.log10(series + EPS)
            stl_result = STL(log_series, period=period_steps, robust=True).fit()
            out[f"{channel}_stl_trend"] = stl_result.trend
            out[f"{channel}_stl_seasonal"] = stl_result.seasonal
            out[f"{channel}_stl_resid"] = stl_result.resid
        except Exception:
            out[f"{channel}_stl_trend"] = np.nan
            out[f"{channel}_stl_seasonal"] = np.nan
            out[f"{channel}_stl_resid"] = np.nan
    return out


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"timestamp", "state", "flare_class", "solexs_flux", "helios_flux"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
