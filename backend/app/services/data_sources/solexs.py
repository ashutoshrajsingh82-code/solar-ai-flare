"""Reader for SoLEXS Level-1 light-curve FITS products.

The supported columns (RATE.TIME Unix seconds and RATE.COUNTS counts/s) follow
a publicly discoverable copy of the SoLEXS user manual. Product compatibility
is provisional until checked against the authenticated PRADAN manual and a real
download of the matching product version.
"""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from astropy.io import fits

from app.services.data_sources.base import IngestionResult


def read_solexs_l1_light_curve(path: str | Path) -> IngestionResult:
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"SoLEXS file does not exist: {source_path}")
    if not (source_path.name.lower().endswith(".fits") or source_path.name.lower().endswith(".fits.gz")):
        raise ValueError("SoLEXS L1 LC input must be a FITS or FITS.gz file")

    try:
        with fits.open(source_path, mode="readonly", memmap=False) as hdus:
            rate_hdu = next((hdu for hdu in hdus if str(hdu.name).upper() == "RATE"), None)
            if rate_hdu is None or rate_hdu.data is None:
                raise ValueError("SoLEXS L1 LC FITS is missing a populated RATE table")
            names = {str(name).upper(): name for name in (rate_hdu.columns.names or [])}
            if not {"TIME", "COUNTS"}.issubset(names):
                raise ValueError("SoLEXS RATE table must contain TIME and COUNTS fields")
            time_values = np.asarray(rate_hdu.data[names["TIME"]], dtype="float64").reshape(-1)
            count_values = np.asarray(rate_hdu.data[names["COUNTS"]], dtype="float64").reshape(-1)
            if len(time_values) != len(count_values):
                raise ValueError("SoLEXS TIME and COUNTS fields have inconsistent lengths")
            header = dict(hdus[0].header)
            rate_header = dict(rate_hdu.header)
            time_unit = rate_hdu.columns[names["TIME"]].unit
            count_unit = rate_hdu.columns[names["COUNTS"]].unit
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Unable to read SoLEXS FITS file; it may be malformed or truncated") from exc

    if len(time_values) == 0:
        raise ValueError("SoLEXS RATE table is empty")
    if time_unit and time_unit.lower() not in {"s", "sec", "second", "seconds"}:
        raise ValueError(f"Unsupported SoLEXS TIME unit {time_unit!r}; expected Unix seconds")
    if count_unit and count_unit.lower().replace(" ", "") not in {"count/s", "counts/s", "countsec-1", "countssec-1", "count/s"}:
        raise ValueError(f"Unsupported SoLEXS COUNTS unit {count_unit!r}; expected counts/s")

    finite_time = np.isfinite(time_values)
    if not finite_time.all():
        raise ValueError("SoLEXS TIME contains non-finite values; refusing to discard rows")
    if len(np.unique(time_values)) != len(time_values):
        raise ValueError("SoLEXS TIME contains duplicate timestamps; refusing to deduplicate")
    if np.any(np.diff(time_values) < 0):
        raise ValueError("SoLEXS TIME is out of order; refusing to silently reorder samples")

    valid_counts = np.isfinite(count_values) & (count_values >= 0)
    rejected_rows = int((~valid_counts).sum())
    source_digest = hashlib.sha256()
    with source_path.open("rb") as source_stream:
        for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
            source_digest.update(chunk)
    accepted_times = time_values[valid_counts]
    accepted_counts = count_values[valid_counts]
    if not len(accepted_times):
        raise ValueError("SoLEXS COUNTS has no finite, nonnegative samples")
    cadence = float(np.median(np.diff(accepted_times))) if len(accepted_times) > 1 else None
    # The documented SoLEXS LC cadence is one second. Use that product
    # cadence for gap counts instead of letting a gap inflate the sample
    # median (especially for short files).
    nominal_cadence = 1.0
    gaps = []
    if cadence and cadence > 0:
        for left, right in zip(accepted_times[:-1], accepted_times[1:]):
            if right - left > nominal_cadence * 1.5:
                gaps.append({"start_unix_seconds": float(left), "end_unix_seconds": float(right),
                             "missing_intervals_estimate": max(1, int(round((right-left)/nominal_cadence))-1)})

    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(accepted_times, unit="s", utc=True),
        "original_time_unix_seconds": accepted_times,
        "solexs_counts_per_second": accepted_counts,
    })
    report = {
        "status": "validated_with_rejections" if rejected_rows else "validated",
        "source": "solexs",
        "product": "L1 light curve",
        "product_level": "L1",
        "product_version": header.get("VERSION") or rate_header.get("VERSION"),
        "filename": source_path.name,
        "file_sha256": source_digest.hexdigest(),
        "file_format": "FITS" + ("+gzip" if source_path.name.lower().endswith(".gz") else ""),
        "timestamp_field": "RATE.TIME",
        "timestamp_standard": "Unix seconds normalized to UTC",
        "measurement_field": "RATE.COUNTS",
        "measurement_unit": count_unit or "counts/s (manual-defined; FITS unit keyword absent)",
        "energy_band": "2–22 keV (SoLEXS L1 LC manual contract)",
        "cadence_seconds_median": cadence,
        "data_period_start": frame.timestamp.min().isoformat(),
        "data_period_end": frame.timestamp.max().isoformat(),
        "input_rows": int(len(time_values)),
        "accepted_rows": int(len(frame)),
        "rejected_rows": rejected_rows,
        "rejected_row_indices": np.flatnonzero(~valid_counts).astype(int).tolist(),
        "rejection_reasons": {"non_finite_or_negative_counts": rejected_rows},
        "missing_intervals": gaps,
        "quality_filter_applied": False,
        "limitations": ["GTI companion filtering is not implemented", "Real-product headers are not yet integration-tested"],
        "fits_primary_header": {key: str(header[key]) for key in ("TELESCOP", "INSTRUME", "DATE-OBS", "DATE-END", "ORIGIN") if key in header},
        "rate_header": {key: str(rate_header[key]) for key in ("EXTNAME", "TUNIT1", "TUNIT2", "TIMEUNIT", "TIMESYS", "TSTART", "TSTOP") if key in rate_header},
    }
    return IngestionResult(frame, report)
