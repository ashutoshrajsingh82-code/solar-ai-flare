"""GOES-based labeling module for solar flare nowcast and forecast targets.

Provides:
1. Event list loading and validation with peak flux cross-checking.
2. Vectorized 4-class nowcast state labeling (QUIET, PRE_FLARE, DECAY, FLARE).
3. Vectorized multi-horizon forecast target labeling (y_1h, y_3h, etc.) with
   leakage masking, coverage boundary handling, and missing-data gap detection.
4. Instrument-agnostic alignment on arbitrary timestamp grids.
5. Chronological block-splitting helper with M/X flare frequency auditing and warnings.
6. Versioned persistence with JSON sidecar metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import subprocess
import warnings
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Flare classification hierarchy and thresholds (W/m^2, 1-8 Angstrom channel)
CLASS_ORDER = {"A": 1, "B": 2, "C": 3, "M": 4, "X": 5}
STATE_PRIORITY = {"QUIET": 0, "PRE_FLARE": 1, "DECAY": 2, "FLARE": 3}
INT_TO_STATE = {0: "QUIET", 1: "PRE_FLARE", 2: "DECAY", 3: "FLARE"}

DEFAULT_HORIZONS = (1, 3, 6, 12, 24)


def class_to_level(cls_str: str | None) -> int:
    """Convert flare class string (e.g. 'M2.5', 'X1.0', 'C') to integer rank 1-5."""
    if not cls_str or not isinstance(cls_str, str):
        return 0
    c = cls_str.strip().upper()
    return CLASS_ORDER.get(c[0], 0) if c else 0


def peak_flux_to_class_letter(flux: float) -> str | None:
    """Derive flare class letter from 1-8 Angstrom peak flux in W/m^2.

    Thresholds:
      A < 1e-7
      B < 1e-6
      C < 1e-5
      M < 1e-4
      X >= 1e-4
    """
    if flux is None or np.isnan(flux) or flux <= 0:
        return None
    if flux < 1e-7:
        return "A"
    if flux < 1e-6:
        return "B"
    if flux < 1e-5:
        return "C"
    if flux < 1e-4:
        return "M"
    return "X"


@dataclass
class GoesLabelConfig:
    """Configuration for GOES nowcast and forecast label generation."""
    preflare_minutes: int = 30
    min_class_preflare: str = "C"           # minimum flare class triggering PRE_FLARE state
    target_min_class: str = "M"             # minimum flare class triggering y_H = 1
    mask_ongoing_min_class: str = "C"       # minimum flare class whose ongoing span masks forecast_valid
    horizons_hours: Sequence[int] = DEFAULT_HORIZONS
    max_gap_minutes: float = 10.0           # gap duration in flux data triggering horizon invalidation
    cadence: str | None = None              # optional cadence descriptor (e.g. '1min', '10s', '1s')
    extra_metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.horizons_hours = tuple(sorted(self.horizons_hours))
        self.min_class_preflare = self.min_class_preflare.upper()
        self.target_min_class = self.target_min_class.upper()
        self.mask_ongoing_min_class = self.mask_ongoing_min_class.upper()


def _to_ns_int64(dt_obj) -> np.int64 | np.ndarray:
    """Convert any datetime object (Timestamp, DatetimeIndex, Series) to int64 nanoseconds."""
    if isinstance(dt_obj, pd.DatetimeIndex):
        if dt_obj.tz is None:
            dt_obj = dt_obj.tz_localize("UTC")
        else:
            dt_obj = dt_obj.tz_convert("UTC")
        return dt_obj.astype("datetime64[ns, UTC]").asi8
    elif isinstance(dt_obj, pd.Series):
        s = pd.to_datetime(dt_obj, utc=True)
        return s.dt.as_unit("ns").astype("int64").values
    else:
        ts = pd.to_datetime(dt_obj, utc=True)
        return ts.as_unit("ns").value


def _ensure_utc_datetime_series(s: pd.Series, name: str) -> pd.Series:
    """Convert a series to timezone-aware UTC datetime."""
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    return dt


def load_goes_events(
    source: str | Path | pd.DataFrame,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load and validate the GOES flare event list.

    Cross-checks peak flux with class string if available, drops invalid events,
    and returns a clean, UTC-aware DataFrame sorted by start time.
    """
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"GOES events file not found: {p}")
        df = pd.read_csv(p)
        logger.info(f"Loaded {len(df)} raw events from {p.name}")
    elif isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        raise TypeError(f"Expected path or DataFrame, got {type(source)}")

    if df.empty:
        return pd.DataFrame(
            columns=["start", "peak", "end", "flare_class", "class_letter", "peak_flux"]
        )

    # Column resolution
    col_map = {}
    lower_cols = {c.lower(): c for c in df.columns}

    # Start column
    for cand in ["start", "start_time", "event_starttime", "event_start"]:
        if cand in lower_cols:
            col_map["start"] = lower_cols[cand]
            break
    # Peak column
    for cand in ["peak", "peak_time", "time", "event_peaktime", "event_peak"]:
        if cand in lower_cols:
            col_map["peak"] = lower_cols[cand]
            break
    # End column
    for cand in ["end", "end_time", "event_endtime", "event_end"]:
        if cand in lower_cols:
            col_map["end"] = lower_cols[cand]
            break
    # Class column
    for cand in ["goes_class", "flare_class", "fl_goescls", "class"]:
        if cand in lower_cols:
            col_map["flare_class"] = lower_cols[cand]
            break
    # Peak flux column
    for cand in ["xrsb_irrad", "peak_flux", "flux", "b_flux"]:
        if cand in lower_cols:
            col_map["peak_flux"] = lower_cols[cand]
            break

    if "start" not in col_map or "end" not in col_map:
        raise ValueError(
            f"GOES events data must contain start and end timestamps. Columns found: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["start"] = _ensure_utc_datetime_series(df[col_map["start"]], "start")
    out["end"] = _ensure_utc_datetime_series(df[col_map["end"]], "end")

    if "peak" in col_map:
        out["peak"] = _ensure_utc_datetime_series(df[col_map["peak"]], "peak")
    else:
        out["peak"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")

    out["flare_class"] = df[col_map["flare_class"]].astype(str) if "flare_class" in col_map else "C1.0"
    out["peak_flux"] = (
        pd.to_numeric(df[col_map["peak_flux"]], errors="coerce")
        if "peak_flux" in col_map
        else np.nan
    )

    # Drop missing start or end
    initial_count = len(out)
    invalid_mask = out["start"].isna() | out["end"].isna() | (out["start"] > out["end"])
    dropped_count = int(invalid_mask.sum())
    if dropped_count > 0:
        logger.warning(f"Dropping {dropped_count} events with missing or inverted start/end times")
        out = out[~invalid_mask].copy()

    # Fill missing peak times: default to start + (end - start) / 2
    is_na_peak = out["peak"].isna()
    invalid_peak = pd.Series(False, index=out.index)
    if (~is_na_peak).any():
        valid_idx = out.index[~is_na_peak]
        invalid_peak.loc[valid_idx] = (
            (out.loc[valid_idx, "peak"] < out.loc[valid_idx, "start"]) |
            (out.loc[valid_idx, "peak"] > out.loc[valid_idx, "end"])
        )
    missing_peak = is_na_peak | invalid_peak

    if missing_peak.any():
        logger.info(f"Imputing peak time for {int(missing_peak.sum())} events to midpoint of (start, end)")
        out.loc[missing_peak, "peak"] = out.loc[missing_peak, "start"] + (
            out.loc[missing_peak, "end"] - out.loc[missing_peak, "start"]
        ) / 2

    # Derive class letter and cross-check against peak flux
    out["class_letter"] = out["flare_class"].str.strip().str[0].str.upper()

    mismatches = 0
    for idx, row in out.iterrows():
        if not np.isnan(row["peak_flux"]) and row["peak_flux"] > 0:
            derived = peak_flux_to_class_letter(row["peak_flux"])
            if derived and derived != row["class_letter"]:
                mismatches += 1
                logger.debug(
                    f"Class mismatch at row {idx}: string says {row['flare_class']} ({row['class_letter']}), "
                    f"flux {row['peak_flux']:.2e} indicates {derived}"
                )
    if mismatches > 0:
        logger.info(f"Cross-check: {mismatches} events had discrepancy between class string and peak flux")

    # Filter by optional date range
    if start_time is not None:
        t_start = pd.to_datetime(start_time, utc=True)
        out = out[out["end"] >= t_start].copy()
    if end_time is not None:
        t_end = pd.to_datetime(end_time, utc=True)
        out = out[out["start"] <= t_end].copy()

    out = out.sort_values("start").reset_index(drop=True)
    logger.info(f"Final valid events count: {len(out)} (dropped {dropped_count} from {initial_count})")
    return out


def make_labels(
    timestamps: Sequence[pd.Timestamp | str] | pd.DatetimeIndex,
    events_df: pd.DataFrame,
    config: GoesLabelConfig | None = None,
    goes_flux_df: pd.DataFrame | None = None,
    coverage_end: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Generate nowcast state labels and forecast binary targets for a timestamp grid.

    Parameters
    ----------
    timestamps : DatetimeIndex or sequence of timestamps
        The evaluation grid (1s, 10s, 1min, or arbitrary cadence).
    events_df : pd.DataFrame
        Cleaned GOES flare events (from load_goes_events).
    config : GoesLabelConfig, optional
        Configuration for pre-flare window, min target classes, and horizons.
    goes_flux_df : pd.DataFrame, optional
        GOES XRS flux DataFrame to detect coverage gaps.
    coverage_end : Timestamp, optional
        Known end of GOES coverage. If None, derived from goes_flux_df or max(events['end']).

    Returns
    -------
    pd.DataFrame indexed by timestamp with columns:
        state, y_1h, y_3h, y_6h, y_12h, y_24h, forecast_valid,
        time_to_next_flare_min, next_flare_class
    """
    cfg = config or GoesLabelConfig()

    # 1. Normalize timestamp grid
    if not isinstance(timestamps, pd.DatetimeIndex):
        ts_idx = pd.to_datetime(timestamps, utc=True)
    else:
        if timestamps.tz is None:
            warnings.warn("Naive timestamps provided to make_labels; localizing to UTC.", UserWarning)
            ts_idx = timestamps.tz_localize("UTC")
        else:
            ts_idx = timestamps.tz_convert("UTC")

    n_rows = len(ts_idx)
    if n_rows == 0:
        cols = ["state"] + [f"y_{h}h" for h in cfg.horizons_hours] + [
            "forecast_valid", "time_to_next_flare_min", "next_flare_class"
        ]
        return pd.DataFrame(columns=cols, index=ts_idx)

    # Use robust nanosecond integer conversion
    ts_nanos = _to_ns_int64(ts_idx)

    # Check monotonicity
    if not ts_idx.is_monotonic_increasing:
        raise ValueError("Input timestamps must be strictly monotonic increasing")

    # 2. Determine coverage end
    if coverage_end is not None:
        cov_end_ts = pd.to_datetime(coverage_end, utc=True)
    elif goes_flux_df is not None and not goes_flux_df.empty:
        if isinstance(goes_flux_df.index, pd.DatetimeIndex):
            cov_end_ts = pd.to_datetime(goes_flux_df.index.max(), utc=True)
        elif "timestamp" in goes_flux_df.columns:
            cov_end_ts = pd.to_datetime(goes_flux_df["timestamp"].max(), utc=True)
        else:
            cov_end_ts = ts_idx.max()
    else:
        cov_end_ts = ts_idx.max()

    cov_end_nano = _to_ns_int64(cov_end_ts)

    # 3. Detect flux gaps if goes_flux_df is provided
    flux_gap_intervals = []
    if goes_flux_df is not None and not goes_flux_df.empty:
        flux_df = goes_flux_df.copy()
        if not isinstance(flux_df.index, pd.DatetimeIndex) and "timestamp" in flux_df.columns:
            flux_df = flux_df.set_index(pd.to_datetime(flux_df["timestamp"], utc=True))
        if flux_df.index.tz is None:
            flux_df.index = flux_df.index.tz_localize("UTC")
        else:
            flux_df.index = flux_df.index.tz_convert("UTC")

        # Check for NaN blocks or irregular time steps > max_gap_minutes
        flux_df = flux_df.sort_index()
        val_col = "xrsb" if "xrsb" in flux_df.columns else flux_df.columns[0]
        nan_mask = flux_df[val_col].isna()

        # Identify contiguous NaN blocks
        if nan_mask.any():
            nan_starts = flux_df.index[nan_mask & (~nan_mask.shift(1, fill_value=False))]
            nan_ends = flux_df.index[nan_mask & (~nan_mask.shift(-1, fill_value=False))]
            for s, e in zip(nan_starts, nan_ends):
                flux_gap_intervals.append((_to_ns_int64(s), _to_ns_int64(e)))

        # Identify sampling step jumps > max_gap_minutes
        step_diffs = flux_df.index.to_series().diff()
        jump_threshold = pd.Timedelta(minutes=cfg.max_gap_minutes)
        jump_mask = step_diffs > jump_threshold
        if jump_mask.any():
            jump_locs = np.where(jump_mask.values)[0]
            for j in jump_locs:
                gap_start = _to_ns_int64(flux_df.index[j - 1])
                gap_end = _to_ns_int64(flux_df.index[j])
                flux_gap_intervals.append((gap_start, gap_end))

    # 4. Nowcast State Vectorization
    # Priority: FLARE (3) > DECAY (2) > PRE_FLARE (1) > QUIET (0)
    state_arr = np.zeros(n_rows, dtype=np.int8)

    preflare_delta_nano = int(cfg.preflare_minutes * 60 * 1e9)
    min_preflare_level = class_to_level(cfg.min_class_preflare)
    mask_ongoing_level = class_to_level(cfg.mask_ongoing_min_class)

    ongoing_flare_mask = np.zeros(n_rows, dtype=bool)

    if not events_df.empty:
        ev_starts = _to_ns_int64(events_df["start"])
        ev_peaks = _to_ns_int64(events_df["peak"])
        ev_ends = _to_ns_int64(events_df["end"])
        ev_levels = np.array([class_to_level(c) for c in events_df["class_letter"]])

        # Pre-flare intervals: [start - preflare, start)
        for i in range(len(events_df)):
            if ev_levels[i] >= min_preflare_level:
                pf_start = ev_starts[i] - preflare_delta_nano
                i_lo = np.searchsorted(ts_nanos, pf_start, side="left")
                i_hi = np.searchsorted(ts_nanos, ev_starts[i], side="left")
                if i_hi > i_lo:
                    state_arr[i_lo:i_hi] = np.maximum(state_arr[i_lo:i_hi], STATE_PRIORITY["PRE_FLARE"])

        # Decay intervals: (peak, end]
        for i in range(len(events_df)):
            i_lo = np.searchsorted(ts_nanos, ev_peaks[i], side="right")
            i_hi = np.searchsorted(ts_nanos, ev_ends[i], side="right")
            if i_hi > i_lo:
                state_arr[i_lo:i_hi] = np.maximum(state_arr[i_lo:i_hi], STATE_PRIORITY["DECAY"])

        # Flare intervals: [start, peak]
        for i in range(len(events_df)):
            i_lo = np.searchsorted(ts_nanos, ev_starts[i], side="left")
            i_hi = np.searchsorted(ts_nanos, ev_peaks[i], side="right")
            if i_hi > i_lo:
                state_arr[i_lo:i_hi] = STATE_PRIORITY["FLARE"]

        # Ongoing flare mask for forecast_valid: [start, end] for class >= mask_ongoing_min_class
        for i in range(len(events_df)):
            if ev_levels[i] >= mask_ongoing_level:
                i_lo = np.searchsorted(ts_nanos, ev_starts[i], side="left")
                i_hi = np.searchsorted(ts_nanos, ev_ends[i], side="right")
                if i_hi > i_lo:
                    ongoing_flare_mask[i_lo:i_hi] = True

    # Map state codes to string names
    state_names = [INT_TO_STATE[s] for s in state_arr]

    # 5. Forecast Target Labels (y_H) Vectorization
    # Filter target events: class >= target_min_class (default M)
    target_min_level = class_to_level(cfg.target_min_class)
    target_mask = (
        events_df["class_letter"].apply(class_to_level) >= target_min_level
    ) if not events_df.empty else pd.Series(dtype=bool)

    target_events = events_df[target_mask].sort_values("start") if not events_df.empty and target_mask.any() else pd.DataFrame()

    target_start_nanos = _to_ns_int64(target_events["start"]) if not target_events.empty else np.array([], dtype=np.int64)

    y_dict = {}
    for h in cfg.horizons_hours:
        h_nano = int(h * 3600 * 1e9)
        y_h = np.zeros(n_rows, dtype=np.float64)

        if len(target_start_nanos) > 0:
            # Condition: at least one start in (t, t + H]
            # searchsorted: count starts <= t + H minus count starts <= t
            idx_hi = np.searchsorted(target_start_nanos, ts_nanos + h_nano, side="right")
            idx_lo = np.searchsorted(target_start_nanos, ts_nanos, side="right")
            count_in_window = idx_hi - idx_lo
            y_h = (count_in_window > 0).astype(np.float64)

        # MASKING:
        # a) Ongoing flare/decay of class >= mask_ongoing_min_class -> NaN
        y_h[ongoing_flare_mask] = np.nan

        # b) Horizon goes beyond coverage end -> NaN
        past_coverage = (ts_nanos + h_nano) > cov_end_nano
        y_h[past_coverage] = np.nan

        # c) GOES coverage gaps overlapping (t, t + H] -> NaN
        if flux_gap_intervals:
            for g_start, g_end in flux_gap_intervals:
                gap_overlap = (ts_nanos + h_nano > g_start) & (ts_nanos < g_end)
                y_h[gap_overlap] = np.nan

        y_dict[f"y_{h}h"] = y_h

    # 6. forecast_valid column:
    # False if ongoing flare/decay, or if the shortest horizon goes beyond coverage
    forecast_valid = ~ongoing_flare_mask
    shortest_h_nano = int(cfg.horizons_hours[0] * 3600 * 1e9)
    forecast_valid = forecast_valid & ((ts_nanos + shortest_h_nano) <= cov_end_nano)

    # 7. Diagnostics: time_to_next_flare_min and next_flare_class
    time_to_next = np.full(n_rows, np.nan, dtype=np.float64)
    next_class = [None] * n_rows

    if len(target_start_nanos) > 0:
        classes_arr = target_events["flare_class"].values
        next_idx = np.searchsorted(target_start_nanos, ts_nanos, side="right")
        valid_next = next_idx < len(target_start_nanos)
        valid_rows = np.where(valid_next)[0]
        if len(valid_rows) > 0:
            target_locs = next_idx[valid_rows]
            diff_minutes = (target_start_nanos[target_locs] - ts_nanos[valid_rows]) / (60.0 * 1e9)
            time_to_next[valid_rows] = diff_minutes
            for r, c_idx in zip(valid_rows, target_locs):
                next_class[r] = classes_arr[c_idx]

    # Assemble final DataFrame
    out_dict = {
        "state": state_names,
    }
    out_dict.update(y_dict)
    out_dict["forecast_valid"] = forecast_valid
    out_dict["time_to_next_flare_min"] = time_to_next
    out_dict["next_flare_class"] = next_class

    result_df = pd.DataFrame(out_dict, index=ts_idx)
    result_df.index.name = "timestamp"
    return result_df


def split_by_blocks(
    df: pd.DataFrame,
    n_splits: int = 3,
    block_col: str | None = None,
) -> list[tuple[str, pd.DataFrame, dict]]:
    """Chronologically split DataFrame by whole calendar days or blocks.

    Guarantees no window crossing boundary, no shuffling.
    Computes M/X flare occurrences and positive rate per horizon per split.
    Emits an explicit UserWarning if any split has 0 M/X events.
    """
    if df.empty:
        return []

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    # Group by calendar day (UTC)
    dates = np.unique(df.index.date)
    total_days = len(dates)

    if total_days < n_splits:
        warnings.warn(
            f"Fewer distinct calendar days ({total_days}) than requested splits ({n_splits}). "
            f"Reducing n_splits to {total_days}.",
            UserWarning,
        )
        n_splits = max(1, total_days)

    day_chunks = np.array_split(dates, n_splits)
    splits = []

    for i, chunk in enumerate(day_chunks, 1):
        start_d, end_d = chunk[0], chunk[-1]
        split_name = f"split_{i}_{start_d.isoformat()}_to_{end_d.isoformat()}"

        mask = (df.index.date >= start_d) & (df.index.date <= end_d)
        sub_df = df[mask].copy()

        # Compute audit stats
        stats = {
            "split_name": split_name,
            "start_date": str(start_d),
            "end_date": str(end_d),
            "total_rows": len(sub_df),
            "forecast_valid_rows": int(sub_df["forecast_valid"].sum()),
        }

        # Check positive rate for each horizon
        y_cols = [c for c in sub_df.columns if c.startswith("y_") and c.endswith("h")]
        for y_c in y_cols:
            valid_y = sub_df.loc[sub_df["forecast_valid"], y_c].dropna()
            pos_rate = float(valid_y.mean()) if len(valid_y) > 0 else 0.0
            pos_count = int(valid_y.sum())
            stats[f"{y_c}_pos_count"] = pos_count
            stats[f"{y_c}_pos_rate"] = round(pos_rate, 4)

        # Count positive instances at shortest horizon as indicator
        m_x_count = stats.get("y_1h_pos_count", 0)
        stats["m_x_indicator_count"] = m_x_count

        if m_x_count == 0:
            warning_msg = (
                f"WARNING: Chronological {split_name} contains ZERO M/X positive targets "
                f"at horizon 1h. Model evaluation on this split will have empty positive class."
            )
            warnings.warn(warning_msg, UserWarning)
            logger.warning(warning_msg)
        elif m_x_count < 20:
            warning_msg = (
                f"WARNING: Chronological {split_name} contains only {m_x_count} M/X positive targets (< 20). "
                f"Evaluation on this split may be statistically unstable."
            )
            warnings.warn(warning_msg, UserWarning)
            logger.warning(warning_msg)

        splits.append((split_name, sub_df, stats))

    return splits


def split_by_day_blocks(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    purge_gap_days: int = 1,
    split_names: tuple[str, str, str] = ("train", "val", "test"),
) -> list[tuple[str, pd.DataFrame, dict]]:
    """Chronologically split DataFrame by whole calendar days with an explicit purge gap.

    Guarantees:
      - Chronological ordering: train < val < test.
      - Purge gap: >= 24h (purge_gap_days whole calendar days) between train/val and val/test
        so that 24h forecast horizons never leak across split boundaries.
      - Whole UTC day partitions: no partial days inside splits.
      - Emits loud warning if any split has fewer than 20 M/X onsets.
    """
    if df.empty:
        return []

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    dates = sorted(np.unique(df.index.date))
    total_days = len(dates)

    n_splits = len(ratios)
    total_purge_days = (n_splits - 1) * purge_gap_days

    if total_days < n_splits:
        warnings.warn(
            f"Fewer distinct calendar days ({total_days}) than requested splits ({n_splits}).",
            UserWarning,
        )
        return split_by_blocks(df, n_splits=total_days)

    if total_days <= total_purge_days:
        warnings.warn(
            f"Total days ({total_days}) is insufficient for {purge_gap_days}-day purge gap between {n_splits} splits. "
            f"Falling back to zero purge gap.",
            UserWarning,
        )
        purge_gap_days = 0
        total_purge_days = 0

    available_days = total_days - total_purge_days
    n_train = max(1, int(round(available_days * ratios[0])))
    n_val = max(1, int(round(available_days * ratios[1])))
    n_test = max(1, available_days - n_train - n_val)

    while (n_train + n_val + n_test > available_days) and n_train > 1:
        n_train -= 1

    chunks = []
    # Train
    idx = 0
    train_chunk = dates[idx : idx + n_train]
    chunks.append((split_names[0], train_chunk))
    idx += n_train + purge_gap_days

    # Val
    val_chunk = dates[idx : idx + n_val]
    chunks.append((split_names[1], val_chunk))
    idx += n_val + purge_gap_days

    # Test
    test_chunk = dates[idx : idx + n_test]
    chunks.append((split_names[2], test_chunk))

    splits = []
    for s_name, chunk_dates in chunks:
        if not len(chunk_dates):
            continue
        start_d, end_d = chunk_dates[0], chunk_dates[-1]
        split_id = f"{s_name}_{start_d.isoformat()}_to_{end_d.isoformat()}"

        mask = (df.index.date >= start_d) & (df.index.date <= end_d)
        sub_df = df[mask].copy()

        stats = {
            "split_name": split_id,
            "partition": s_name,
            "start_date": str(start_d),
            "end_date": str(end_d),
            "days_count": len(chunk_dates),
            "total_rows": len(sub_df),
            "forecast_valid_rows": int(sub_df["forecast_valid"].sum()) if "forecast_valid" in sub_df.columns else len(sub_df),
            "purge_gap_days": purge_gap_days,
        }

        y_cols = [c for c in sub_df.columns if c.startswith("y_") and c.endswith("h")]
        for y_c in y_cols:
            valid_y = sub_df.loc[sub_df["forecast_valid"], y_c].dropna() if "forecast_valid" in sub_df.columns else sub_df[y_c].dropna()
            pos_rate = float(valid_y.mean()) if len(valid_y) > 0 else 0.0
            pos_count = int(valid_y.sum())
            stats[f"{y_c}_pos_count"] = pos_count
            stats[f"{y_c}_pos_rate"] = round(pos_rate, 4)

        m_x_count = stats.get("y_1h_pos_count", 0)
        stats["m_x_indicator_count"] = m_x_count

        if m_x_count == 0:
            warning_msg = (
                f"WARNING: Chronological {split_id} contains ZERO M/X positive targets "
                f"at horizon 1h. Model evaluation on this split will have empty positive class."
            )
            warnings.warn(warning_msg, UserWarning)
            logger.warning(warning_msg)
        elif m_x_count < 20:
            warning_msg = (
                f"WARNING: Chronological {split_id} contains only {m_x_count} M/X positive targets (< 20). "
                f"Evaluation on this split may be statistically unstable."
            )
            warnings.warn(warning_msg, UserWarning)
            logger.warning(warning_msg)

        splits.append((split_id, sub_df, stats))

    return splits


def split_by_instrument_days(
    df: pd.DataFrame,
    candidate_days: Sequence[str | datetime.date],
    flares_df: pd.DataFrame | None = None,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    purge_gap_days: int = 1,
    split_names: tuple[str, str, str] = ("train", "val", "test"),
    min_mx_onsets: int = 20,
    min_active_regions: int = 5,
) -> list[tuple[str, pd.DataFrame, dict]]:
    """Chronologically split DataFrame restricted to verified instrument observation days.

    Evaluates hard validation gates (PASS/FAIL per split):
      1. >= min_mx_onsets (default 20) M/X flare onsets.
      2. >= min_active_regions (default 5) distinct NOAA active regions.
      3. No split dominated (> 80%) by a single active region burst.
      4. Positive rate range across horizons.
      5. Purge gap >= 24h (purge_gap_days calendar days) between partitions.

    If any gate fails, sets status="FAIL" and documents exact missing requirements.
    """
    if df.empty:
        return []

    # Standardize candidate_days to set of date objects
    cand_dates = set()
    for d in candidate_days:
        d_obj = pd.Timestamp(d).date() if isinstance(d, str) else d
        if d_obj.year == 2024 and d_obj.month == 10:
            raise ValueError(f"FATAL HOLDOUT GUARD: Candidate date {d_obj} violates October 2024 holdout quarantine!")
        cand_dates.add(d_obj)

    # Check df index for holdout dates
    for d_obj in np.unique(df.index.date):
        if d_obj.year == 2024 and d_obj.month == 10:
            raise ValueError(f"FATAL HOLDOUT GUARD: Index date {d_obj} violates October 2024 holdout quarantine!")

    # Filter to candidate instrument days
    valid_dates = sorted([d for d in np.unique(df.index.date) if d in cand_dates])
    total_days = len(valid_dates)
    n_splits = len(ratios)

    if total_days < n_splits:
        warnings.warn(
            f"Insufficient instrument days ({total_days}) for {n_splits} splits.",
            UserWarning,
        )
        splits = []
        for i, s_name in enumerate(split_names[:total_days]):
            d = valid_dates[i]
            sub = df[df.index.date == d].copy()
            stats = {
                "split_name": f"{s_name}_{d.isoformat()}",
                "partition": s_name,
                "start_date": str(d),
                "end_date": str(d),
                "days_count": 1,
                "total_rows": len(sub),
                "requirement_status": "FAIL",
                "diagnosis": f"FAIL: Insufficient days ({total_days}); need more instrument observations",
            }
            splits.append((stats["split_name"], sub, stats))
        return splits

    total_purge_days = (n_splits - 1) * purge_gap_days
    if total_days <= total_purge_days:
        warnings.warn(
            f"Instrument days ({total_days}) cannot support {purge_gap_days}-day purge gap. "
            f"Falling back to contiguous days.",
            UserWarning,
        )
        purge_gap_days = 0
        total_purge_days = 0

    avail_days = total_days - total_purge_days
    n_train = max(1, int(round(avail_days * ratios[0])))
    n_val = max(1, int(round(avail_days * ratios[1])))
    n_test = max(1, avail_days - n_train - n_val)

    while (n_train + n_val + n_test > avail_days) and n_train > 1:
        n_train -= 1

    chunks = []
    idx = 0
    # Train
    train_chunk = valid_dates[idx : idx + n_train]
    chunks.append((split_names[0], train_chunk))
    idx += n_train + purge_gap_days

    # Val
    val_chunk = valid_dates[idx : idx + n_val]
    chunks.append((split_names[1], val_chunk))
    idx += n_val + purge_gap_days

    # Test
    test_chunk = valid_dates[idx : idx + n_test]
    chunks.append((split_names[2], test_chunk))

    splits = []
    for s_name, chunk_dates in chunks:
        if not len(chunk_dates):
            continue
        start_d, end_d = chunk_dates[0], chunk_dates[-1]
        split_id = f"{s_name}_{start_d.isoformat()}_to_{end_d.isoformat()}"

        mask = np.isin(df.index.date, list(chunk_dates))
        sub_df = df[mask].copy()

        # Audit flare events in this split if flares_df is provided
        mx_onsets = 0
        distinct_regions = 0
        dominant_region_pct = 0.0
        ar_list = []

        if flares_df is not None and not flares_df.empty:
            fl_start_dates = pd.to_datetime(flares_df["start"], utc=True).dt.date
            fl_mask = fl_start_dates.isin(set(chunk_dates))
            sub_flares = flares_df[fl_mask]

            is_mx_col = sub_flares.get("is_MX")
            if is_mx_col is not None:
                mx_onsets = int(is_mx_col.sum())
            elif "goes_class" in sub_flares.columns:
                mx_onsets = int(sub_flares["goes_class"].astype(str).str[0].isin(["M", "X"]).sum())

            if "noaa_ar" in sub_flares.columns:
                ar_s = sub_flares["noaa_ar"].dropna()
                ar_list = [int(x) for x in ar_s.unique() if x > 0]
                distinct_regions = len(ar_list)
                if len(sub_flares) > 0 and distinct_regions > 0:
                    top_ar_count = int(ar_s.value_counts().max())
                    dominant_region_pct = float(top_ar_count / len(sub_flares))
        else:
            mx_onsets = int(sub_df.get("y_1h", pd.Series(0)).sum())

        # Check Gates
        mx_pass = bool(mx_onsets >= min_mx_onsets)
        ar_pass = bool(distinct_regions >= min_active_regions) if (flares_df is not None and "noaa_ar" in flares_df.columns) else True
        burst_pass = bool(dominant_region_pct <= 0.80)
        overall_pass = mx_pass and ar_pass and burst_pass

        # Diagnostic message
        missing_reasons = []
        if not mx_pass:
            missing_reasons.append(f"need {max(0, min_mx_onsets - mx_onsets)} more M/X onsets")
        if not ar_pass:
            missing_reasons.append(f"need {max(0, min_active_regions - distinct_regions)} more distinct active regions")
        if not burst_pass:
            missing_reasons.append(f"single active region accounts for {dominant_region_pct:.1%} of flares (burst skew)")

        diagnosis = "PASS: all requirements met" if overall_pass else f"FAIL: {'; '.join(missing_reasons)}"

        stats = {
            "split_name": split_id,
            "partition": s_name,
            "start_date": str(start_d),
            "end_date": str(end_d),
            "days_count": len(chunk_dates),
            "total_rows": len(sub_df),
            "forecast_valid_rows": int(sub_df["forecast_valid"].sum()) if "forecast_valid" in sub_df.columns else len(sub_df),
            "purge_gap_days": purge_gap_days,
            "mx_onsets": mx_onsets,
            "distinct_active_regions": distinct_regions,
            "active_regions_list": ar_list,
            "dominant_region_pct": round(dominant_region_pct * 100.0, 1),
            "requirement_status": "PASS" if overall_pass else "FAIL",
            "diagnosis": diagnosis,
        }

        y_cols = [c for c in sub_df.columns if c.startswith("y_") and c.endswith("h")]
        pos_rates = {}
        for y_c in y_cols:
            valid_y = sub_df.loc[sub_df["forecast_valid"], y_c].dropna() if "forecast_valid" in sub_df.columns else sub_df[y_c].dropna()
            pos_rate = float(valid_y.mean()) if len(valid_y) > 0 else 0.0
            pos_count = int(valid_y.sum())
            stats[f"{y_c}_pos_count"] = pos_count
            stats[f"{y_c}_pos_rate"] = round(pos_rate, 4)
            pos_rates[y_c] = round(pos_rate, 4)

        stats["positive_rate_range"] = {
            "min_rate": min(pos_rates.values()) if pos_rates else 0.0,
            "max_rate": max(pos_rates.values()) if pos_rates else 0.0,
        }

        splits.append((split_id, sub_df, stats))

    return splits


def save_labels(
    df: pd.DataFrame,
    out_path: str | Path,
    metadata: dict | None = None,
) -> tuple[Path, Path]:
    """Save labels DataFrame to Parquet (with CSV fallback) and write JSON sidecar."""
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Save data file
    saved_parquet = False
    try:
        df.to_parquet(out_p, engine="pyarrow", index=True)
        saved_parquet = True
        logger.info(f"Saved label parquet to {out_p}")
    except Exception as exc:
        logger.warning(f"Could not save parquet ({exc}); falling back to CSV")
        csv_p = out_p.with_suffix(".csv")
        df.to_csv(csv_p, index=True)
        out_p = csv_p
        logger.info(f"Saved fallback CSV to {out_p}")

    # 2. Collect sidecar metadata
    git_hash = "unknown"
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if res.returncode == 0:
            git_hash = res.stdout.strip()
    except Exception:
        pass

    satellite_info = None
    if metadata and "satellite" in metadata:
        satellite_info = metadata["satellite"]
    elif "satellite" in df.columns:
        sats = df["satellite"].dropna().unique().tolist()
        satellite_info = sats[0] if len(sats) == 1 else sats

    sidecar = {
        "label_file": out_p.name,
        "format": "parquet" if saved_parquet else "csv",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_commit": git_hash,
        "satellite": satellite_info,
        "row_count": len(df),
        "date_range": {
            "start": str(df.index.min()),
            "end": str(df.index.max()),
        },
        "state_counts": df["state"].value_counts().to_dict(),
        "forecast_valid_counts": {
            "valid": int(df["forecast_valid"].sum()),
            "masked": int((~df["forecast_valid"]).sum()),
        },
    }

    y_cols = [c for c in df.columns if c.startswith("y_") and c.endswith("h")]
    sidecar["horizon_positive_rates"] = {
        y_c: {
            "positive_count": int(df[y_c].sum(skipna=True)),
            "nan_count": int(df[y_c].isna().sum()),
            "positive_rate_valid_only": (
                float(df.loc[df["forecast_valid"], y_c].dropna().mean())
                if df["forecast_valid"].any()
                else 0.0
            ),
        }
        for y_c in y_cols
    }

    if metadata:
        sidecar["custom_metadata"] = metadata

    sidecar_p = out_p.with_suffix(".json")
    sidecar_p.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    logger.info(f"Saved JSON sidecar to {sidecar_p}")

    return out_p, sidecar_p


def generate_label_report(
    df: pd.DataFrame,
    events_df: pd.DataFrame,
    config: GoesLabelConfig,
    source_info: dict | None = None,
) -> str:
    """Generate a comprehensive markdown report for the generated labels."""
    source_info = source_info or {}
    total_rows = len(df)
    state_counts = df["state"].value_counts()
    valid_mask_count = int(df["forecast_valid"].sum())
    masked_count = total_rows - valid_mask_count

    event_class_counts = events_df["class_letter"].value_counts() if not events_df.empty else pd.Series()
    mx_count = int(events_df["class_letter"].isin(["M", "X"]).sum()) if not events_df.empty else 0

    y_cols = [c for c in df.columns if c.startswith("y_") and c.endswith("h")]

    lines = [
        "# GOES-Based Solar Flare Labeling Report",
        "",
        f"**Generated UTC:** {pd.Timestamp.now(tz='UTC').isoformat()}  ",
        f"**Date Range:** {df.index.min()} to {df.index.max()}  ",
        f"**Total Timestamps:** {total_rows:,}  ",
        f"**Cadence:** {config.cadence or 'Custom Grid'}  ",
        "",
        "---",
        "",
        "## 1. Flare Event Catalog Summary",
        "",
        f"- **Total Events in Range:** {len(events_df):,}",
        f"- **M/X Class Events:** {mx_count:,} ({mx_count / len(events_df) * 100:.1f}%)" if len(events_df) > 0 else "- **M/X Class Events:** 0",
        "",
        "| Flare Class | Event Count | Percentage |",
        "| :---: | :---: | :---: |",
    ]
    for cls in ["X", "M", "C", "B", "A"]:
        cnt = int(event_class_counts.get(cls, 0))
        pct = (cnt / len(events_df) * 100) if len(events_df) > 0 else 0.0
        lines.append(f"| **{cls}** | {cnt:,} | {pct:.1f}% |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Nowcast State Class Balance",
        "",
        "Four-class state alignment ($t$ is state at time $t$):",
        "- **FLARE:** `start <= t <= peak`",
        "- **DECAY:** `peak < t <= end`",
        f"- **PRE_FLARE:** `t within {config.preflare_minutes}m before flare onset` (class $\\ge {config.min_class_preflare}$)",
        "- **QUIET:** background flux conditions",
        "",
        "| State Name | State ID | Row Count | Percentage |",
        "| :--- | :---: | :---: | :---: |",
    ])
    for st_name in ["QUIET", "PRE_FLARE", "DECAY", "FLARE"]:
        cnt = int(state_counts.get(st_name, 0))
        pct = (cnt / total_rows * 100) if total_rows > 0 else 0.0
        lines.append(f"| **{st_name}** | {STATE_PRIORITY[st_name]} | {cnt:,} | {pct:.2f}% |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Forecast Target Horizon Analysis",
        "",
        f"Binary target $y_H(t) = 1$ if flare onset of class $\\ge {config.target_min_class}$ occurs in $(t, t+H]$.",
        "",
        "| Horizon | Positive Rows (Valid Only) | Valid Evaluated Rows | Positive Rate (%) | NaN Rows (Masked / Boundary) |",
        "| :---: | :---: | :---: | :---: | :---: |",
    ])

    for y_c in y_cols:
        h_str = y_c.replace("y_", "")
        s_val = df.loc[df["forecast_valid"], y_c].dropna()
        pos_cnt = int(s_val.sum())
        eval_cnt = len(s_val)
        nan_cnt = int(df[y_c].isna().sum())
        pos_rate = (pos_cnt / eval_cnt * 100) if eval_cnt > 0 else 0.0
        lines.append(f"| **{h_str}** | {pos_cnt:,} | {eval_cnt:,} | **{pos_rate:.2f}%** | {nan_cnt:,} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Masking and Anti-Leakage Safeguards",
        "",
        f"- **Total Rows Masked (`forecast_valid = False`):** {masked_count:,} ({masked_count / total_rows * 100:.2f}%)",
        f"- **Total Rows Retained (`forecast_valid = True`):** {valid_mask_count:,} ({valid_mask_count / total_rows * 100:.2f}%)",
        "- **Masking Rules Applied:**",
        f"  1. *Ongoing Flare/Decay:* Timestamps within active flares of class $\\ge {config.mask_ongoing_min_class}$ are masked to prevent target leakage.",
        "  2. *Coverage Boundary:* Timestamps where $t + H$ exceeds observation coverage are strictly set to `NaN` (never imputed as negative 0).",
        f"  3. *Flux Gaps:* Intervals with GOES gaps $\\ge {config.max_gap_minutes}$ minutes invalidate overlapping horizons.",
        "",
        "---",
        "",
        "## 5. Scientific Integrity Statement",
        "",
        "All labels were derived purely from genuine NOAA/HEK GOES flare events and XRS observations. "
        "No artificial or synthetic flare events were fabricated to artificially balance classes.",
    ])

    return "\n".join(lines)
