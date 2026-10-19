
"""Reader for NOAA/NCEI GOES XRS L2 Flare Report CSV reference events."""
from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd

from app.services.data_sources.base import IngestionResult


REQUIRED_COLUMNS = {"start_time", "time", "end_time", "flare_class"}
HORIZONS_HOURS = (1, 3, 6, 12, 24)


def read_goes_flare_report_csv(
    path, coverage_start, coverage_end
) -> IngestionResult:
    """Read a local NOAA/NCEI event report with explicit coverage.

    Event onset is `start_time`; `time` is the event peak.
    Invalid event rows are rejected and reported. Missing or incomplete
    reference coverage must never be interpreted as a negative label.
    """
    source_path = Path(path)

    if not source_path.is_file():
        raise FileNotFoundError(
            f"GOES flare report does not exist: {source_path}"
        )
    if source_path.suffix.lower() != ".csv":
        raise ValueError("GOES flare reference input must be a CSV file")

    start_coverage = _utc_timestamp(coverage_start, "coverage_start")
    end_coverage = _utc_timestamp(coverage_end, "coverage_end")

    if start_coverage >= end_coverage:
        raise ValueError(
            "GOES report coverage_start must be earlier than coverage_end"
        )

    try:
        raw = pd.read_csv(source_path)
    except Exception as exc:
        raise ValueError(
            "Unable to read GOES flare report CSV; "
            "it may be malformed or truncated"
        ) from exc

    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(
            f"GOES flare report is missing required fields: {sorted(missing)}"
        )
    if raw.empty:
        raise ValueError("GOES flare report contains no event rows")

    out = pd.DataFrame(index=raw.index)

    for source_name, target_name in (
        ("start_time", "start_time"),
        ("time", "peak_time"),
        ("end_time", "end_time"),
    ):
        out[target_name] = pd.to_datetime(
            raw[source_name], utc=True, errors="coerce"
        )

    raw_classes = (
        raw["flare_class"].fillna("").astype(str).str.strip().str.upper()
    )
    out["flare_class"] = raw_classes
    out["flare_class_group"] = raw_classes.str.extract(
        r"^([ABCMX])", expand=False
    )

    bad_times = out[
        ["start_time", "peak_time", "end_time"]
    ].isna().any(axis=1)

    bad_class = out["flare_class_group"].isna()

    bad_order = (~bad_times) & (
        (out["start_time"] > out["peak_time"])
        | (out["peak_time"] > out["end_time"])
    )

    bad_coverage = (~bad_times) & (
        (out["start_time"] < start_coverage)
        | (out["start_time"] > end_coverage)
    )

    invalid = bad_times | bad_class | bad_order | bad_coverage

    if "flare_id" in raw.columns:
        flare_ids = raw["flare_id"].fillna("").astype(str).str.strip()
        has_id = flare_ids.ne("")
        duplicate_id = has_id & flare_ids.duplicated(keep=False)

        if duplicate_id.any():
            rows = raw.index[duplicate_id].astype(int).tolist()
            raise ValueError(
                "GOES flare report contains duplicate flare_id "
                f"values at rows {rows}"
            )

    rejected = {
        "invalid_or_missing_times": int(bad_times.sum()),
        "missing_or_unknown_class": int(bad_class.sum()),
        "start_peak_end_order_invalid": int(bad_order.sum()),
        "event_outside_declared_coverage": int(bad_coverage.sum()),
    }

    accepted = out.loc[~invalid].reset_index(drop=True)

    if accepted.empty:
        raise ValueError(
            f"GOES report has no valid events after checks: {rejected}"
        )

    # Preserve optional source identifiers and saturation status for audit.
    for column in (
        "flare_id",
        "xrsb_irrad",
        "background_irrad",
        "peak_saturated",
        "sequential_flare_num",
    ):
        if column in raw.columns:
            accepted[column] = (
                raw.loc[~invalid, column].reset_index(drop=True)
            )

    digest = hashlib.sha256()
    with source_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    report = {
        "status": "validated_with_rejections" if invalid.any() else "validated",
        "source": "NOAA/NCEI GOES XRS L2 Flare Report",
        "filename": source_path.name,
        "file_sha256": digest.hexdigest(),
        "file_format": "CSV",
        "product_version": _version_from_filename(source_path.name),
        "reference_event_definition": (
            "GOES XRS L2 event report; event onset uses start_time; "
            "class from flare_class"
        ),
        "timestamp_standard": (
            "UTC (CSV datetime fields interpreted as UTC per product readme)"
        ),
        "timestamp_fields": {
            "start": "start_time",
            "peak": "time",
            "end": "end_time",
        },
        "irradiance_field": "xrsb_irrad (W/m^2) when present",
        "flare_class_field": "flare_class",
        "coverage_start": start_coverage.isoformat(),
        "coverage_end": end_coverage.isoformat(),
        "input_rows": int(len(raw)),
        "accepted_rows": int(len(accepted)),
        "rejected_rows": int(invalid.sum()),
        "rejected_row_indices": raw.index[invalid].astype(int).tolist(),
        "rejection_reasons": rejected,
        "known_limitations": [
            "GOES classes and magnitudes are from an independent "
            "geostationary soft-X-ray instrument, not SoLEXS or HEL1OS "
            "measurements.",
            "This report is the current documented product revision; "
            "validate satellite-era and version caveats for the chosen file.",
        ],
    }

    return IngestionResult(accepted, report)


def label_observations_from_goes(
    observations,
    events,
    coverage_start,
    coverage_end,
    horizons_hours=HORIZONS_HOURS,
    significant_classes=("M", "X"),
):
    """Create leakage-safe onset labels.

    A label is known only if the observation and its entire forecast window
    are inside the declared reference coverage. The forecast interval is
    (observation_time, observation_time + horizon].
    """
    obs = pd.to_datetime(
        observations["timestamp"], utc=True, errors="coerce"
    )

    if obs.isna().any():
        raise ValueError("Observation timestamps contain invalid values")

    start_coverage = _utc_timestamp(coverage_start, "coverage_start")
    end_coverage = _utc_timestamp(coverage_end, "coverage_end")

    if start_coverage >= end_coverage:
        raise ValueError(
            "GOES report coverage_start must be earlier than coverage_end"
        )

    # Explicitly normalize every timestamp to nanoseconds before converting
    # to int64. This avoids comparing different datetime resolutions.
    obs_ns = (
        obs.dt.as_unit("ns").astype("int64").to_numpy(dtype=np.int64)
    )
    coverage_start_ns = start_coverage.as_unit("ns").value
    coverage_end_ns = end_coverage.as_unit("ns").value

    if not events.empty:
        if "start_time" not in events.columns:
            raise ValueError("GOES events are missing start_time")
        if "flare_class_group" not in events.columns:
            raise ValueError("GOES events are missing flare_class_group")

        significant = events[
            events["flare_class_group"].isin(significant_classes)
        ].copy()

        event_times = pd.to_datetime(
            significant["start_time"], utc=True, errors="coerce"
        )

        if event_times.isna().any():
            raise ValueError(
                "GOES event onset timestamps contain invalid values"
            )

        event_ns = np.sort(
            event_times.dt.as_unit("ns")
            .astype("int64")
            .to_numpy(dtype=np.int64)
        )
    else:
        significant = events.copy()
        event_ns = np.array([], dtype=np.int64)

    labels = pd.DataFrame({"timestamp": obs})

    report = {
        "target_event": "M/X GOES flare onset",
        "target_time_field": "start_time",
        "forecast_window": "(observation_time, observation_time + horizon]",
        "coverage_start": start_coverage.isoformat(),
        "coverage_end": end_coverage.isoformat(),
        "significant_event_count": int(len(significant)),
        "horizon_counts": {},
    }

    for horizon in horizons_hours:
        if horizon <= 0:
            raise ValueError("Forecast horizons must be positive")

        horizon_ns = int(horizon * 3600 * 1_000_000_000)

        # A negative label is valid only if the full horizon is covered.
        known = (
            (obs_ns >= coverage_start_ns)
            & (obs_ns + horizon_ns <= coverage_end_ns)
        )

        target = np.full(len(obs_ns), np.nan, dtype=float)
        known_obs_ns = obs_ns[known]

        # side="right" excludes events starting exactly at observation time.
        next_event_indices = np.searchsorted(
            event_ns, known_obs_ns, side="right"
        )

        next_event_ns = np.full(
            int(known.sum()), np.iinfo(np.int64).max, dtype=np.int64
        )

        has_event = next_event_indices < len(event_ns)
        next_event_ns[has_event] = event_ns[
            next_event_indices[has_event]
        ]

        target[known] = (
            next_event_ns <= known_obs_ns + horizon_ns
        ).astype(float)

        labels[f"forecast_{horizon}h"] = target

        report["horizon_counts"][f"{horizon}h"] = {
            "known_rows": int(np.isfinite(target).sum()),
            "positive_rows": int(np.nansum(target)),
            "unknown_rows": int((~np.isfinite(target)).sum()),
        }

    return labels, report


def _utc_timestamp(value, name):
    result = pd.to_datetime(value, utc=True, errors="coerce")

    if pd.isna(result):
        raise ValueError(
            f"{name} must be a valid UTC-compatible timestamp"
        )

    return result


def _version_from_filename(filename):
    match = re.search(r"(v\d+(?:-\d+)+)", filename, flags=re.IGNORECASE)
    return match.group(1) if match else None
