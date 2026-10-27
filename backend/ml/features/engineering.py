"""Causal, timestamp-aware feature generation for the normalized project channels.

The available normalized schema names are ``solexs_flux`` and ``helios_flux``.
Their physical units/calibration and matched energy bands are not established
for the bundled synthetic data, so the paired ratio is disabled unless callers
provide explicit, compatible metadata. No feature routine fills missing data.
"""
from dataclasses import asdict, dataclass, field
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

FEATURE_REGISTRY_VERSION = "feature-registry-v2"
FEATURE_SCHEMA_VERSION = "causal-time-aware-v2"
ROLL_WINDOWS_MIN = [5, 10, 30, 60]
CHANNELS = {"solexs_flux": "SoLEXS", "helios_flux": "HEL1OS"}
CHANNEL_PREFIX = {"solexs": "solexs_flux", "helios": "helios_flux"}


@dataclass
class FeatureConfig:
    cadence_seconds: int = 60
    windows_minutes: List[int] = field(default_factory=lambda: list(ROLL_WINDOWS_MIN))
    compute_stl: bool = False
    stl_period_minutes: int = 60
    min_rolling_samples: int = 2
    min_rolling_fraction: float = 0.33
    spike_lookback_minutes: int = 10
    spike_min_samples: int = 5
    spike_threshold: float = 6.0
    irregular_tolerance_fraction: float = 0.25
    long_gap_multiplier: float = 3.0
    enable_hard_soft_ratio: bool = False
    channel_metadata: Optional[dict] = None
    quality_columns: Optional[Dict[str, str]] = None


def _config(config=None):
    cfg = config or FeatureConfig()
    if cfg.cadence_seconds <= 0 or cfg.stl_period_minutes <= 0:
        raise ValueError("Feature cadence and STL period must be positive.")
    if any(not isinstance(w, int) or w <= 0 for w in cfg.windows_minutes):
        raise ValueError("Rolling-window durations must be positive integer minutes.")
    if cfg.min_rolling_samples < 2 or not 0 < cfg.min_rolling_fraction <= 1:
        raise ValueError("Rolling statistics require at least two samples and a fraction in (0, 1].")
    if cfg.spike_lookback_minutes <= 0 or cfg.spike_min_samples < 2 or cfg.spike_threshold <= 0:
        raise ValueError("Spike lookback, minimum sample count, and threshold must be positive.")
    return cfg


def add_rolling_features(df: pd.DataFrame, config: FeatureConfig = None) -> pd.DataFrame:
    """Create trailing elapsed-time features; all values at t use data at/before t."""
    cfg = _config(config)
    _validate_feature_input(df, cfg)
    out = df.copy()
    timestamps = pd.to_datetime(out["timestamp"], utc=True, errors="coerce", format="mixed")
    out["timestamp"] = timestamps
    index = pd.DatetimeIndex(timestamps)
    out.index = index
    cadence = pd.Timedelta(seconds=cfg.cadence_seconds)
    deltas = pd.Series(timestamps.diff().dt.total_seconds().to_numpy(), index=index)
    out["sampling_interval_seconds"] = deltas
    deviation = (deltas - cfg.cadence_seconds).abs() / cfg.cadence_seconds
    out["cadence_deviation_fraction"] = deviation
    out["irregular_sampling"] = (deviation > cfg.irregular_tolerance_fraction).astype(float).where(deltas.notna())
    out["long_observation_gap"] = (deltas > cfg.cadence_seconds * cfg.long_gap_multiplier).astype(float).where(deltas.notna())

    for channel, instrument in CHANNELS.items():
        if channel not in out:
            continue
        series = pd.to_numeric(out[channel], errors="coerce").astype(float)
        out[channel] = series
        prefix = channel.removesuffix("_flux")
        valid = series.notna() & np.isfinite(series)
        out[f"{prefix}_missing"] = (~valid).astype(float)
        last_valid_at = pd.Series(index=index, dtype="datetime64[ns, UTC]")
        last_valid_at.loc[valid.to_numpy()] = index[valid.to_numpy()]
        out[f"{prefix}_time_since_valid_min"] = (pd.Series(index, index=index) - last_valid_at.ffill()).dt.total_seconds() / 60.0

        # Adjacent-observation changes retain their original names and remain causal.
        out[f"{prefix}_diff1"] = series.diff()
        previous = series.shift(1)
        out[f"{prefix}_pct_change"] = (series / previous - 1.0).where(previous.ne(0) & previous.notna())

        for minutes in cfg.windows_minutes:
            duration = pd.Timedelta(minutes=minutes)
            min_samples = max(cfg.min_rolling_samples, int(np.ceil(minutes * 60 / cfg.cadence_seconds * cfg.min_rolling_fraction)))
            rolling = series.rolling(f"{minutes}min", min_periods=min_samples, closed="both")
            out[f"{prefix}_mean_{minutes}m"] = rolling.mean()
            out[f"{prefix}_median_{minutes}m"] = rolling.median()
            out[f"{prefix}_std_{minutes}m"] = rolling.std()
            out[f"{prefix}_min_{minutes}m"] = rolling.min()
            out[f"{prefix}_max_{minutes}m"] = rolling.max()
            out[f"{prefix}_var_{minutes}m"] = rolling.var()
            out[f"{prefix}_p90_{minutes}m"] = rolling.quantile(0.90)
            out[f"{prefix}_recent_peak_{minutes}m"] = rolling.max()
            out[f"{prefix}_flux_diff_{minutes}m"] = _elapsed_difference(series, index, duration)
            out[f"{prefix}_slope_{minutes}m"] = _elapsed_slope(series, index, duration, min_samples, cfg.cadence_seconds)

            valid_count = valid.astype(float).rolling(f"{minutes}min", min_periods=1, closed="both").sum()
            elapsed_ns = index.as_unit("ns").asi8
            expected_count = np.minimum(
                int(np.ceil(minutes * 60 / cfg.cadence_seconds)) + 1,
                np.floor((elapsed_ns - elapsed_ns[0]) / cadence.value).astype(int) + 1,
            )
            missing_fraction = 1.0 - valid_count.to_numpy() / np.maximum(expected_count, 1)
            out[f"{prefix}_missing_fraction_{minutes}m"] = np.clip(missing_fraction, 0.0, 1.0)

        score, flag = _causal_spike_features(series, index, cfg)
        out[f"{prefix}_spike_score"] = score
        out[f"{prefix}_spike_flag"] = flag

    # Include only explicitly named, source-associated quality flags.
    for channel, quality_col in (cfg.quality_columns or {}).items():
        if channel not in CHANNELS or quality_col not in out:
            continue
        out[f"{channel.removesuffix('_flux')}_quality_flag"] = pd.to_numeric(out[quality_col], errors="coerce")

    ratio_report = _add_validated_ratio(out, cfg, index)
    if cfg.compute_stl:
        out = add_stl_features(out.reset_index(drop=True), cfg).set_index("timestamp", drop=False)
    out.attrs["feature_quality_report"] = ratio_report
    out.attrs["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    out.attrs["feature_config"] = asdict(cfg)
    return out.reset_index(drop=True)


def _elapsed_difference(series, timestamps, duration):
    values = series.to_numpy(dtype=float)
    times = timestamps.as_unit("ns").asi8
    target_delta = duration.value
    result = np.full(len(values), np.nan, dtype=float)
    for i, stamp in enumerate(times):
        j = np.searchsorted(times, stamp - target_delta, side="right") - 1
        while j >= 0 and not np.isfinite(values[j]):
            j -= 1
        if j >= 0 and np.isfinite(values[i]):
            result[i] = values[i] - values[j]
    return result


def _elapsed_slope(series, timestamps, duration, min_samples, cadence_seconds):
    values = series.to_numpy(dtype=float)
    times = timestamps.as_unit("ns").asi8
    duration_ns = duration.value
    result = np.full(len(values), np.nan, dtype=float)
    for i, stamp in enumerate(times):
        start = np.searchsorted(times, stamp - duration_ns, side="right")
        x = (times[start:i + 1] - stamp) / 60e9
        y = values[start:i + 1]
        valid = np.isfinite(y)
        if valid.sum() < min_samples:
            continue
        x, y = x[valid], y[valid]
        centered_x = x - x.mean()
        denominator = np.dot(centered_x, centered_x)
        if denominator > 0:
            result[i] = np.dot(centered_x, y - y.mean()) / denominator
    return result


def _causal_spike_features(series, timestamps, config):
    """Score x(t) against the prior lookback's median/MAD; score=delta/(1.4826*MAD)."""
    values = series.to_numpy(dtype=float)
    times = timestamps.as_unit("ns").asi8
    span = pd.Timedelta(minutes=config.spike_lookback_minutes).value
    scores = np.full(len(values), np.nan, dtype=float)
    flags = np.full(len(values), np.nan, dtype=float)
    for i, stamp in enumerate(times):
        start = np.searchsorted(times, stamp - span, side="right")
        history = values[start:i]
        history = history[np.isfinite(history)]
        if len(history) < config.spike_min_samples or not np.isfinite(values[i]):
            continue
        median = float(np.median(history))
        mad = float(np.median(np.abs(history - median)))
        scale = 1.4826 * mad
        if scale <= 0:
            continue
        scores[i] = (values[i] - median) / scale
        flags[i] = float(abs(scores[i]) >= config.spike_threshold)
    return scores, flags


def _ratio_metadata_valid(config):
    metadata = config.channel_metadata or {}
    numerator = metadata.get("helios_flux", {})
    denominator = metadata.get("solexs_flux", {})
    required = ("units", "energy_band_kev", "physical_quantity", "measurement_convention", "ratio_compatible")
    if not config.enable_hard_soft_ratio:
        return False, "disabled by default: no validated channel metadata supplied"
    if any(not numerator.get(k) for k in required) or any(not denominator.get(k) for k in required):
        return False, "required units, energy bands, quantity, convention, and compatibility evidence are missing"
    try:
        n_low, n_high = map(float, numerator["energy_band_kev"])
        d_low, d_high = map(float, denominator["energy_band_kev"])
    except (TypeError, ValueError):
        return False, "energy bands must each contain two numeric keV bounds"
    if n_low < 0 or d_low < 0 or n_high <= n_low or d_high <= d_low:
        return False, "energy bands are invalid"
    if not numerator["ratio_compatible"] or not denominator["ratio_compatible"]:
        return False, "source metadata does not affirm a scientifically compatible ratio"
    if numerator["units"] != denominator["units"] or numerator["physical_quantity"] != denominator["physical_quantity"]:
        return False, "ratio channels have incompatible units or physical quantities"
    if numerator["measurement_convention"] != denominator["measurement_convention"]:
        return False, "ratio channels use incompatible measurement conventions"
    if d_high > n_low:
        return False, "bands overlap; a hard/soft ratio requires disjoint ordered bands"
    return True, None


def _add_validated_ratio(out, config, timestamps):
    enabled, reason = _ratio_metadata_valid(config)
    report = {"enabled": False, "invalid_ratio_count": 0, "reason": reason}
    if not enabled:
        return report
    if not all(c in out for c in ("helios_flux", "solexs_flux")):
        report["reason"] = "ratio metadata is compatible but one or both source channels are absent"
        return report
    numerator = pd.to_numeric(out["helios_flux"], errors="coerce").to_numpy(dtype=float)
    denominator = pd.to_numeric(out["solexs_flux"], errors="coerce").to_numpy(dtype=float)
    invalid = ~np.isfinite(numerator) | ~np.isfinite(denominator) | (numerator < 0) | (denominator <= 0)
    ratio = np.full(len(out), np.nan, dtype=float)
    valid = ~invalid
    ratio[valid] = numerator[valid] / denominator[valid]
    out["hard_soft_ratio"] = ratio
    out["hard_soft_ratio_change"] = pd.Series(ratio, index=out.index).diff()
    index = pd.DatetimeIndex(timestamps)
    for minutes in (5, 10, 30):
        minimum = max(config.min_rolling_samples, int(np.ceil(minutes * 60 / config.cadence_seconds * config.min_rolling_fraction)))
        rolling = pd.Series(ratio, index=index).rolling(f"{minutes}min", min_periods=minimum, closed="both")
        out[f"hard_soft_ratio_mean_{minutes}m"] = rolling.mean().to_numpy()
        out[f"hard_soft_ratio_var_{minutes}m"] = rolling.var().to_numpy()
        out[f"hard_soft_ratio_slope_{minutes}m"] = _elapsed_slope(pd.Series(ratio, index=index), index, pd.Timedelta(minutes=minutes), minimum, config.cadence_seconds)
    return {"enabled": True, "invalid_ratio_count": int(invalid.sum()), "reason": None,
            "numerator": "helios_flux", "denominator": "solexs_flux",
            "metadata": {"helios_flux": config.channel_metadata["helios_flux"],
                         "solexs_flux": config.channel_metadata["solexs_flux"]}}


def add_stl_features(df: pd.DataFrame, config: FeatureConfig = None) -> pd.DataFrame:
    """Compute a causal STL estimate from a fixed trailing time grid only."""
    cfg = _config(config)
    _validate_feature_input(df, cfg)
    from statsmodels.tsa.seasonal import STL

    out = df.copy()
    timestamps = pd.to_datetime(out["timestamp"], utc=True, errors="coerce", format="mixed")
    deltas = timestamps.diff().dropna().dt.total_seconds().to_numpy()
    if len(deltas) and not np.allclose(deltas, cfg.cadence_seconds, rtol=0, atol=max(1e-6, cfg.cadence_seconds * cfg.irregular_tolerance_fraction)):
        raise ValueError("STL features require a validated regular cadence; irregular timestamps are unsupported.")
    period = max(2, int(cfg.stl_period_minutes * 60 / cfg.cadence_seconds))
    width = period * 2
    for channel in CHANNELS:
        if channel not in out:
            continue
        values = pd.to_numeric(out[channel], errors="coerce").to_numpy(dtype=float)
        result = np.full((len(values), 3), np.nan, dtype=float)
        for end in range(width - 1, len(values)):
            window = values[end - width + 1:end + 1]
            if not np.isfinite(window).all() or np.any(window <= 0):
                continue
            stl = STL(np.log10(window), period=period, robust=True).fit()
            result[end] = stl.trend[-1], stl.seasonal[-1], stl.resid[-1]
        for i, name in enumerate(("trend", "seasonal", "resid")):
            out[f"{channel}_stl_{name}"] = result[:, i]
    return out


def _validate_feature_input(df: pd.DataFrame, config: FeatureConfig) -> None:
    if "timestamp" not in df:
        raise ValueError("Feature input requires a timestamp column.")
    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", format="mixed")
    if timestamps.isna().any():
        raise ValueError("Feature timestamps must be valid UTC-compatible values.")
    if timestamps.duplicated().any():
        raise ValueError("Feature timestamps must not contain duplicates.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Feature observations must be ordered by timestamp.")
    for channel in CHANNELS:
        if channel not in df:
            continue
        values = pd.to_numeric(df[channel], errors="coerce").to_numpy(dtype=float)
        malformed = df[channel].notna().to_numpy() & np.isnan(values)
        if malformed.any() or np.isinf(values).any():
            raise ValueError(f"Feature input {channel} contains non-numeric or infinite values.")


def get_feature_registry(df: pd.DataFrame, config: FeatureConfig = None) -> Dict[str, dict]:
    """Describe numeric model inputs from this schema; unknown provenance stays unknown."""
    cfg = _config(config)
    registry = {}
    excluded = {"timestamp", "state", "flare_class"}
    for name in df.columns:
        if name in excluded or name.startswith(("forecast_", "target_", "label_", "nowcast_")):
            continue
        if name in CHANNELS:
            # Preserve raw measurements as provenance/source columns. Model
            # inputs use their train-fitted scaled counterparts and derivatives.
            continue
        if not pd.api.types.is_numeric_dtype(df[name]):
            continue
        if name.startswith("hard_soft_ratio"):
            sources, instruments, definition = ["helios_flux", "solexs_flux"], ["HEL1OS", "SoLEXS"], "HEL1OS / SoLEXS; enabled only with compatible declared metadata"
        else:
            prefix = name.split("_", 1)[0]
            source = CHANNEL_PREFIX.get(prefix)
            sources = [source] if source else ([name] if name.endswith(("quality", "quality_flag")) else [])
            instruments = [CHANNELS[source]] if source else []
            if name in {"sampling_interval_seconds", "cadence_deviation_fraction", "irregular_sampling", "long_observation_gap"}:
                sources, instruments = ["timestamp"], list(CHANNELS.values())
            definition = _definition_for(name, source)
        duration_match = re.search(r"_(\d+)m(?:_|$)", name)
        source_meta = (cfg.channel_metadata or {}).get(sources[0], {}) if len(sources) == 1 else {}
        registry[name] = {
            "name": name, "description_formula": definition,
            "source_columns": sources, "source_instruments": instruments,
            "physical_quantity": source_meta.get("physical_quantity"), "units": source_meta.get("units"),
            "energy_band_kev": source_meta.get("energy_band_kev"),
            "rolling_window_minutes": int(duration_match.group(1)) if duration_match else None,
            "minimum_history_samples": (max(cfg.min_rolling_samples, int(np.ceil(int(duration_match.group(1)) * 60 / cfg.cadence_seconds * cfg.min_rolling_fraction))) if duration_match else None),
            "availability": "at_observation_timestamp; causal/trailing",
            "missing_value_policy": "retain NaN; model rows/windows require explicit valid inputs",
            "causal": True, "feature_version": FEATURE_SCHEMA_VERSION,
            "provenance_known": bool(sources),
            "allowed_experiments": instruments if instruments else [],
        }
    return registry


def get_feature_columns(df: pd.DataFrame, instruments: Optional[Sequence[str]] = None,
                        config: FeatureConfig = None) -> List[str]:
    """Return ordered numeric features, optionally enforcing provenance-based isolation."""
    registry = get_feature_registry(df, config)
    candidates = list(registry)
    if instruments is None:
        return candidates
    normalized = {_normalize_instrument(value) for value in instruments}
    if not normalized or not normalized.issubset(set(CHANNELS.values())):
        raise ValueError(f"Instrument selection must use known project instruments: {list(CHANNELS.values())}.")
    selected = []
    for feature in candidates:
        entry = registry[feature]
        sources = set(entry["source_instruments"])
        # Timestamp-only quality features are available regardless of which
        # flux instrument is selected.
        if entry["source_columns"] == ["timestamp"]:
            selected.append(feature)
            continue
        if not entry["provenance_known"]:
            raise ValueError(f"Cannot safely select feature {feature}: source instrument provenance is unknown.")
        if sources.issubset(normalized):
            selected.append(feature)
    return selected


def select_feature_columns(df: pd.DataFrame, input_configuration: str,
                           config: FeatureConfig = None) -> List[str]:
    selection = {
        "solexs_only": ["SoLEXS"],
        "helios_only": ["HEL1OS"],
        "fusion": ["SoLEXS", "HEL1OS"],
    }
    if input_configuration not in selection:
        raise ValueError(f"Unsupported instrument configuration: {input_configuration}")
    return get_feature_columns(df, selection[input_configuration], config)


def _normalize_instrument(value):
    normalized = str(value).strip().lower().replace("1", "1")
    aliases = {"solexs": "SoLEXS", "helios": "HEL1OS", "hel1os": "HEL1OS"}
    return aliases.get(normalized, str(value).strip())


def _definition_for(name, source):
    if not source:
        return "Numeric input with unknown instrument provenance; not eligible for isolated experiments."
    channel = source
    if name == channel or name == f"{channel}_scaled":
        return f"Input measurement or training-scaled {channel} value."
    if name.endswith("_missing"):
        return f"1 when {channel} is missing at this timestamp; otherwise 0."
    if "time_since_valid" in name:
        return "Elapsed minutes since the most recent valid observation; trailing only."
    if "missing_fraction" in name:
        return "Expected cadence samples missing in the trailing elapsed-time window."
    if "spike_score" in name:
        return "Signed deviation from prior-window median divided by 1.4826 times prior-window MAD."
    if "spike_flag" in name:
        return "1 when absolute causal robust score meets configured threshold; NaN with insufficient history."
    if "slope_" in name:
        return "Least-squares slope against elapsed minutes in the trailing time window."
    if "flux_diff_" in name:
        return "Current measurement minus latest valid measurement at or before the named elapsed lag."
    if "quality_flag" in name:
        return "Source-provided quality value; semantics are defined by source metadata."
    if name.endswith(("_diff1", "_pct_change")):
        return "Difference or relative change from the immediately preceding observation."
    if name in {"sampling_interval_seconds", "cadence_deviation_fraction", "irregular_sampling", "long_observation_gap"}:
        return "Timestamp-derived cadence or observation-gap quality measure."
    return f"Trailing time-based statistic derived from {channel}."
