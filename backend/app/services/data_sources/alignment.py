"""Gap-preserving alignment for already validated instrument time series."""
import pandas as pd


def align_observations(solexs: pd.DataFrame, hel1os: pd.DataFrame, cadence_seconds: int,
                       aggregation: str = "mean") -> tuple[pd.DataFrame, dict]:
    """Aggregate each source independently onto UTC bins and outer-join them.

    Bins are right-closed/right-labeled, so an output timestamp never includes
    samples later than that timestamp. Missing source bins remain NaN; this
    function never interpolates or treats different instrument bands as equal.
    """
    if cadence_seconds <= 0:
        raise ValueError("cadence_seconds must be a positive integer")
    if aggregation not in {"mean", "sum", "last"}:
        raise ValueError("aggregation must be one of: mean, sum, last")
    for name, frame in (("SoLEXS", solexs), ("HEL1OS", hel1os)):
        if "timestamp" not in frame:
            raise ValueError(f"{name} input is missing timestamp")
        if frame.empty:
            raise ValueError(f"{name} input is empty")
    solexs_times = pd.to_datetime(solexs["timestamp"], utc=True, errors="coerce")
    hel1os_times = pd.to_datetime(hel1os["timestamp"], utc=True, errors="coerce")
    if solexs_times.isna().any() or hel1os_times.isna().any():
        raise ValueError("Instrument inputs contain invalid timestamps")
    solexs_period = [solexs_times.min(), solexs_times.max()]
    hel1os_period = [hel1os_times.min(), hel1os_times.max()]

    def prepare(frame, prefix):
        data = frame.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
        if data["timestamp"].isna().any():
            raise ValueError(f"{prefix} input contains invalid timestamps")
        values = [column for column in data.select_dtypes(include="number").columns
                  if column not in {"original_time_unix_seconds"}]
        if not values:
            raise ValueError(f"{prefix} input has no numeric measurement columns")
        data = data.set_index("timestamp")[values].sort_index()
        if data.index.has_duplicates:
            raise ValueError(f"{prefix} input contains duplicate timestamps")
        frequency = f"{int(cadence_seconds)}s"
        sampled = getattr(data.resample(frequency, label="right", closed="right"), aggregation)()
        return sampled.rename(columns={column: f"{prefix}_{column}" for column in values})

    left = prepare(solexs, "solexs")
    right = prepare(hel1os, "hel1os")
    aligned = left.join(right, how="outer").sort_index()
    frequency = f"{int(cadence_seconds)}s"
    grid_start = aligned.index.min().ceil(frequency)
    grid_end = aligned.index.max().floor(frequency)
    if grid_start <= grid_end:
        grid = pd.date_range(grid_start, grid_end, freq=frequency, tz="UTC")
        aligned = aligned.reindex(aligned.index.union(grid)).sort_index()
    overlap_start = max(solexs_period[0], hel1os_period[0])
    overlap_end = min(solexs_period[1], hel1os_period[1])
    overlap = overlap_start <= overlap_end
    left_cols, right_cols = list(left.columns), list(right.columns)
    left_present = aligned[left_cols].notna().any(axis=1)
    right_present = aligned[right_cols].notna().any(axis=1)
    left_cadence = _median_cadence(solexs)
    right_cadence = _median_cadence(hel1os)
    aligned.index.name = "timestamp"
    aligned = aligned.reset_index()
    return aligned, {
        "status": "aligned" if overlap else "no_temporal_overlap",
        "time_standard": "UTC",
        "target_cadence_seconds": int(cadence_seconds),
        "aggregation": aggregation,
        "bin_convention": "right-closed and right-labeled; no future samples included",
        "solexs_observed_cadence_seconds_median": left_cadence,
        "hel1os_observed_cadence_seconds_median": right_cadence,
        "solexs_period": [solexs_period[0].isoformat(), solexs_period[1].isoformat()],
        "hel1os_period": [hel1os_period[0].isoformat(), hel1os_period[1].isoformat()],
        "overlap_period": [overlap_start.isoformat(), overlap_end.isoformat()] if overlap else None,
        "non_overlapping_bins": {
            "solexs_only": int((left_present & ~right_present).sum()),
            "hel1os_only": int((right_present & ~left_present).sum()),
            "union_bins": int(len(aligned)),
        },
        "missing_bins": {
            "solexs": int((~left_present).sum()),
            "hel1os": int((~right_present).sum()),
            "both": int((~left_present & ~right_present).sum()),
        },
        "interpolation": "none",
        "output_rows": int(len(aligned)),
    }


def _median_cadence(frame):
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").sort_values()
    diffs = timestamps.diff().dropna().dt.total_seconds()
    return float(diffs.median()) if not diffs.empty else None
