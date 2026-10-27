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
    _validate_feature_input(df, cfg)
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce", format="mixed")

    for channel in ["solexs_flux", "helios_flux"]:
        if channel not in out.columns:
            continue
        out[channel] = pd.to_numeric(out[channel], errors="coerce")
        prefix = channel.split("_")[0]  # "solexs" / "helios"
        series = out[channel]

        # simple 1-step rate of change / gradient
        out[f"{prefix}_diff1"] = series.diff()
        out[f"{prefix}_pct_change"] = series.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

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
    """Add causal STL estimates from trailing windows only.

    Each row's decomposition is fitted using observations ending at that row.
    This is more expensive than one full-series decomposition, but prevents
    future measurements from changing features at an earlier forecast time.
    Missing values in a trailing window leave its STL values unknown.
    """
    from statsmodels.tsa.seasonal import STL

    cfg = config or FeatureConfig()
    _validate_feature_input(df, cfg)
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce", format="mixed")
    period_steps = _steps_for_minutes(cfg.stl_period_minutes, cfg.cadence_seconds)
    window_steps = period_steps * 2

    for channel in ["solexs_flux", "helios_flux"]:
        if channel not in out.columns:
            continue
        values = pd.to_numeric(out[channel], errors="coerce").to_numpy(dtype=float)
        out[channel] = values
        components = np.full((len(values), 3), np.nan, dtype=float)
        for end in range(window_steps - 1, len(values)):
            start = end - window_steps + 1
            window = values[start:end + 1]
            if not np.isfinite(window).all() or np.any(window + EPS <= 0):
                continue
            result = STL(np.log10(window + EPS), period=period_steps, robust=True).fit()
            components[end] = (result.trend[-1], result.seasonal[-1], result.resid[-1])
        out[f"{channel}_stl_trend"] = components[:, 0]
        out[f"{channel}_stl_seasonal"] = components[:, 1]
        out[f"{channel}_stl_resid"] = components[:, 2]
    return out


def _validate_feature_input(df: pd.DataFrame, config: FeatureConfig) -> None:
    if config.cadence_seconds <= 0 or config.stl_period_minutes <= 0:
        raise ValueError("Feature cadence and STL period must be positive.")
    if any(not isinstance(w, int) or w <= 0 for w in config.windows_minutes):
        raise ValueError("Rolling feature windows must be positive integer minutes.")
    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", format="mixed")
        if timestamps.isna().any():
            raise ValueError("Feature timestamps must be valid UTC-compatible values.")
        if timestamps.duplicated().any():
            raise ValueError("Feature timestamps must not contain duplicates.")
        if not timestamps.is_monotonic_increasing:
            raise ValueError("Feature observations must be ordered by timestamp.")
    for channel in ("solexs_flux", "helios_flux"):
        if channel in df.columns:
            values = pd.to_numeric(df[channel], errors="coerce").to_numpy(dtype=float)
            malformed = df[channel].notna().to_numpy() & np.isnan(values)
            if malformed.any():
                raise ValueError(f"Feature input {channel} contains non-numeric values.")
            if np.isinf(values).any():
                raise ValueError(f"Feature input {channel} contains infinite values.")


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"timestamp", "state", "flare_class", "solexs_flux", "helios_flux"}
    target_prefixes = ("forecast_", "target_", "label_", "nowcast_")
    return [
        c for c in df.columns
        if c not in exclude
        and not c.startswith(target_prefixes)
        and pd.api.types.is_numeric_dtype(df[c])
    ]
