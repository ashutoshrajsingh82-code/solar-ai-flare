"""Read-only Phase 3A audit for local SoLEXS, HEL1OS and GOES archives.

Nested archives are read in memory and are never extracted or modified.
SoLEXS/HEL1OS samples remain candidates until the matching registered product
manuals establish their GTI, quality and calibration semantics.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from astropy.io import fits
from netCDF4 import Dataset, num2date

GENERATOR = "phase3a_data_audit_v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def contiguous_intervals(timestamps, cadence_seconds=None, valid=None, max_gap_factor=1.5):
    """Summarize valid timestamps without bridging internal gaps."""
    times = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="coerce"))
    if times.isna().any():
        raise ValueError("Coverage timestamps must all be valid.")
    if len(times) == 0:
        return []
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Coverage timestamps must be unique and ordered.")
    mask = np.ones(len(times), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if len(mask) != len(times):
        raise ValueError("Coverage mask length does not match timestamps.")
    selected = np.flatnonzero(mask)
    if not len(selected):
        return []
    differences = np.diff(times.asi8) / 1e9
    cadence = cadence_seconds
    if cadence is None:
        cadence = float(np.median(differences)) if len(differences) else 1.0
    if cadence <= 0:
        raise ValueError("Cadence must be positive.")
    # A rejected row is itself a gap; do not bridge across it when summarizing
    # candidate-valid coverage (for example a one-minute bad GOES sample).
    split_points = np.flatnonzero(
        (np.diff(selected) > 1)
        | (np.diff(times.asi8[selected]) / 1e9 > cadence * max_gap_factor)
    ) + 1
    return [
        {
            "start_utc": times[chunk[0]].isoformat(),
            "end_utc": times[chunk[-1]].isoformat(),
            "sample_count": int(len(chunk)),
            "nominal_cadence_seconds": float(cadence),
        }
        for chunk in np.split(selected, split_points)
    ]


def gti_membership(timestamps, intervals, closed="both"):
    """Generic interval helper; never applied to mission products without manuals."""
    if closed not in {"both", "left", "right", "neither"}:
        raise ValueError("closed must be both, left, right, or neither.")
    times = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="coerce"))
    if times.isna().any():
        raise ValueError("GTI timestamps must be valid.")
    result = np.zeros(len(times), dtype=bool)
    for start, end in intervals:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        if start > end:
            raise ValueError("GTI start must not be after its end.")
        left = times >= start if closed in {"both", "left"} else times > start
        right = times <= end if closed in {"both", "right"} else times < end
        result |= left & right
    return result


def _table(hdus):
    return next((h for h in hdus if getattr(h, "columns", None) is not None and h.data is not None), None)


def _read_solexs_lc(compressed):
    payload = gzip.decompress(compressed)
    with fits.open(io.BytesIO(payload), mode="readonly", memmap=False) as hdus:
        rate = next((h for h in hdus if h.name.upper() == "RATE"), None)
        if rate is None or rate.data is None:
            raise ValueError("SoLEXS product has no populated RATE extension.")
        names = {n.upper(): n for n in (rate.columns.names or [])}
        if not {"TIME", "COUNTS"}.issubset(names):
            raise ValueError("SoLEXS RATE must expose TIME and COUNTS.")
        seconds = np.asarray(rate.data[names["TIME"]], dtype=float)
        values = np.asarray(rate.data[names["COUNTS"]], dtype=float)
        times = pd.DatetimeIndex(pd.to_datetime(seconds, unit="s", utc=True))
        valid = np.isfinite(seconds) & np.isfinite(values) & (values >= 0)
        return {
            "times": times, "values": values,
            "finite_time": int(np.isfinite(seconds).sum()),
            "candidate_valid": valid,
            "units": {c.name: c.unit for c in rate.columns},
            "header": {k: rate.header.get(k) for k in ("DATE-OBS", "DATE-END", "TIMESYS", "TIMEUNIT", "TIMEDEL", "TSTART", "TSTOP") if k in rate.header},
            "primary": {k: hdus[0].header.get(k) for k in ("TELESCOP", "INSTRUME", "DATE-OBS", "DATE-END") if k in hdus[0].header},
        }


def _read_gti(data):
    payload = gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
    with fits.open(io.BytesIO(payload), mode="readonly", memmap=False) as hdus:
        table = _table(hdus)
        if table is None:
            return {"table_present": False, "columns": []}
        return {
            "table_present": True, "extension": table.name, "rows": len(table.data),
            "columns": [{"name": c.name, "unit": c.unit, "format": c.format} for c in table.columns],
            "time_semantics": "UNVERIFIED; matching registered product manual unavailable",
        }


def _read_hel1os_lc(data):
    with fits.open(io.BytesIO(data), mode="readonly", memmap=False) as hdus:
        table = _table(hdus)
        if table is None:
            raise ValueError("HEL1OS light curve has no populated binary table.")
        names = {n.upper(): n for n in table.columns.names}
        if not {"MJD", "ISOT", "CTR", "STAT_ERR"}.issubset(names):
            raise ValueError(f"Unexpected HEL1OS columns: {list(table.columns.names)}")
        isot = pd.to_datetime(np.asarray(table.data[names["ISOT"]]).astype(str), utc=True, errors="coerce")
        values = np.asarray(table.data[names["CTR"]], dtype=float)
        error = np.asarray(table.data[names["STAT_ERR"]], dtype=float)
        valid = isot.notna() & np.isfinite(values) & (values >= 0) & np.isfinite(error) & (error >= 0)
        return {
            "times": pd.DatetimeIndex(isot), "values": values, "errors": error,
            "candidate_valid": np.asarray(valid, dtype=bool), "rows": len(values),
            "valid_count": int(np.asarray(valid, dtype=bool).sum()),
            "extension": table.name,
            "columns": [{"name": c.name, "unit": c.unit, "format": c.format} for c in table.columns],
            "header": {k: table.header.get(k) for k in ("TSTART", "TSTOP", "TIMESYS", "TIMEUNIT", "DATE-OBS", "DATE-END") if k in table.header},
            "primary": {k: hdus[0].header.get(k) for k in ("TELESCOP", "INSTRUME", "DATE-OBS", "DATE-END", "OBS_ID") if k in hdus[0].header},
        }


def _nc_times(nc):
    variable = nc.variables["time"]
    values = np.asarray(variable[:], dtype=float)
    decoded = num2date(values, variable.units, calendar=getattr(variable, "calendar", "standard"),
                       only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    return pd.DatetimeIndex(pd.to_datetime(decoded, utc=True))


def _flag_reasons(flag, variable):
    masks = np.asarray(getattr(variable, "flag_masks", []), dtype=np.uint64)
    values = np.asarray(getattr(variable, "flag_values", []), dtype=np.uint64)
    meanings = str(getattr(variable, "flag_meanings", "")).split()
    return [
        meaning for mask, value, meaning in zip(masks, values, meanings)
        if (int(flag) & int(mask)) == int(value) and meaning != "good_data"
    ]


def _audit_goes_nc(path):
    records = []
    summaries = {}
    with Dataset(path, "r") as nc:
        times = _nc_times(nc)
        for channel in ("xrsa", "xrsb"):
            value_var, flag_var = nc.variables[f"{channel}_flux"], nc.variables[f"{channel}_flag"]
            raw = np.asarray(value_var[:], dtype=float)
            flags = np.asarray(flag_var[:], dtype=int)
            fill = float(getattr(value_var, "_FillValue", -9999.0))
            finite, sentinel = np.isfinite(raw), raw == fill
            valid = finite & ~sentinel & (flags == 0)
            counts = Counter()
            for i, flag in enumerate(flags):
                reasons = _flag_reasons(int(flag), flag_var)
                if not finite[i]:
                    reasons.append("non_finite")
                if sentinel[i]:
                    reasons.append("fill_value")
                if not reasons:
                    reasons = ["valid_good_data"] if valid[i] else ["not_valid"]
                counts.update(set(reasons))
                records.append({
                    "timestamp_utc": times[i].isoformat(), "source_file": path.name,
                    "channel": channel,
                    "value_w_m2": None if (not finite[i] or sentinel[i]) else float(raw[i]),
                    "flag": int(flag), "quality_meanings": "|".join(reasons),
                    "candidate_valid": bool(valid[i]),
                    "flag_excluded_union": int(nc.variables[f"{channel}_flag_excluded"][i]),
                })
            summaries[channel] = {
                "raw_rows": len(raw), "finite_rows": int(finite.sum()),
                "sentinel_rows": int(sentinel.sum()), "good_flag_rows": int((flags == 0).sum()),
                "candidate_valid_rows": int(valid.sum()),
                "excluded_reason_record_counts": dict(sorted(counts.items())),
                "units": str(getattr(value_var, "units", "unknown")), "fill_value": fill,
                "flag_values": np.asarray(getattr(flag_var, "flag_values", [])).astype(int).tolist(),
                "flag_masks": np.asarray(getattr(flag_var, "flag_masks", [])).astype(int).tolist(),
                "flag_meanings": str(getattr(flag_var, "flag_meanings", "")),
            }
        corr = nc.variables["electron_correction_flag"]
        corr_values, corr_counts = np.unique(np.asarray(corr[:], dtype=int), return_counts=True)
        return {
            "path": str(path), "sha256": sha256_path(path), "filename": path.name,
            "satellite": "GOES-18" if "_g18_" in path.name.lower() else "UNRESOLVED",
            "product_version": path.name.rsplit("_v", 1)[-1].removesuffix(".nc"),
            "product": str(getattr(nc, "title", "unknown")),
            "institution": str(getattr(nc, "institution", "unknown")),
            "source": str(getattr(nc, "source", "unknown")),
            "time_units": str(nc.variables["time"].units),
            "start_utc": times.min().isoformat() if len(times) else None,
            "end_utc": times.max().isoformat() if len(times) else None,
            "rows": len(times),
            "cadence_seconds_median": float(np.median(np.diff(times.asi8)) / 1e9) if len(times) > 1 else None,
            "channels": summaries,
            "electron_correction_flag_counts": {str(int(k)): int(v) for k, v in zip(corr_values, corr_counts)},
            "timestamps": times, "records": records,
        }


def _compare_goes_csv(csv_path, daily):
    frame = pd.read_csv(csv_path)
    if not {"timestamp", "xrsa", "xrsb"}.issubset(frame.columns):
        raise ValueError("GOES CSV must contain timestamp, xrsa and xrsb.")
    csv_times = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True, errors="coerce"))
    nc_times = pd.DatetimeIndex(
        np.concatenate([item["timestamps"].asi8 for item in daily]).astype("datetime64[ns]")
    ).tz_localize("UTC")
    order = np.argsort(nc_times.asi8, kind="stable")
    nc_times = nc_times[order]
    by_channel = {name: [] for name in ("xrsa", "xrsb")}
    for item in daily:
        with Dataset(item["path"], "r") as nc:
            for channel in by_channel:
                var = nc.variables[f"{channel}_flux"]
                values = np.asarray(var[:], dtype=float)
                values[values == float(var._FillValue)] = np.nan
                by_channel[channel].extend(values.tolist())
    comparisons = {}
    for channel in by_channel:
        csv_values = pd.to_numeric(frame[channel], errors="coerce").to_numpy(float)
        nc_values = np.asarray(by_channel[channel], dtype=float)[order]
        same_len = len(csv_values) == len(nc_values)
        finite_csv, finite_nc = np.isfinite(csv_values), np.isfinite(nc_values)
        comparisons[channel] = {
            "rows": len(csv_values), "csv_finite": int(finite_csv.sum()),
            "netcdf_finite": int(finite_nc.sum()),
            "values_equal_nan_aware": bool(same_len and np.allclose(csv_values, nc_values, rtol=1e-7, atol=0.0, equal_nan=True)),
            "csv_finite_where_netcdf_fill": int((finite_csv & ~finite_nc).sum()) if same_len else None,
            "csv_missing_where_netcdf_finite": int((~finite_csv & finite_nc).sum()) if same_len else None,
        }
    return {
        "path": str(csv_path), "sha256": sha256_path(csv_path), "rows": len(frame),
        "columns": list(frame.columns), "start_utc": csv_times.min().isoformat() if len(csv_times) else None,
        "end_utc": csv_times.max().isoformat() if len(csv_times) else None,
        "duplicate_timestamps": int(csv_times.duplicated().sum()),
        "timestamps_equal_netcdf_exact": bool(len(csv_times) == len(nc_times) and np.array_equal(csv_times.asi8, nc_times.asi8)),
        "channel_comparison": comparisons, "quality_flags_present": False,
    }


def _audit_events(path, generator):
    frame = pd.read_csv(path)
    times = {name: pd.to_datetime(frame[name], utc=True, errors="coerce") for name in ("start", "peak", "end")}
    groups = frame.goes_class.fillna("").astype(str).str.upper().str.extract(r"^([ABCMX])", expand=False)
    mx = groups.isin({"M", "X"})
    is_mx = frame.is_MX.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    checks = {
        "invalid_start": int(times["start"].isna().sum()),
        "invalid_peak": int(times["peak"].isna().sum()),
        "invalid_end": int(times["end"].isna().sum()),
        "invalid_class": int(groups.isna().sum()),
        "time_order_violations": int(((times["start"] > times["peak"]) | (times["peak"] > times["end"])).sum()),
        "is_MX_disagreements": int((mx != is_mx).sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_start_class_rows": int(frame.assign(_s=times["start"], _c=groups).duplicated(["_s", "_c"], keep=False).sum()),
    }
    intervals = sorted(zip(times["start"].dropna().tolist(), times["end"].dropna().tolist()))
    overlaps, active = 0, []
    for start, end in intervals:
        active = [old for old in active if old >= start]
        overlaps += len(active)
        active.append(end)
    provenance = {
        "classification": "DERIVED_LOCAL_HEK_QUERY",
        "provider": None, "query_filters": None, "satellite_per_event": "not recorded",
        "product_version": None, "retrieval_time": None,
        "completeness": "UNVERIFIED_PENDING_COMPARISON_TO_OFFICIAL_NOAA_REPORT",
    }
    if generator and generator.is_file():
        code = generator.read_text(encoding="utf-8", errors="replace")
        required = ("a.hek.EventType", "a.hek.FL.GOESCls", "a.hek.OBS.Observatory", 'drop_duplicates(subset=["start", "goes_class"])')
        if all(part in code for part in required):
            provenance.update({
                "provider": "SunPy Fido querying Heliophysics Event Knowledgebase",
                "query_filters": ["event type FL", "GOES class > B1.0", "observatory GOES"],
                "field_mapping": {"event_starttime": "start", "event_peaktime": "peak", "event_endtime": "end", "fl_goescls": "goes_class", "ar_noaanum": "noaa_ar"},
                "dedupe_rule": "drop duplicate (start, goes_class), keeping first",
                "exact_generation_source_sha256": sha256_path(generator),
            })
    class_counts = groups.value_counts(dropna=False)
    return {
        "path": str(path), "sha256": sha256_path(path), "columns": list(frame.columns),
        "rows": len(frame), "start_min_utc": times["start"].min().isoformat(),
        "start_max_utc": times["start"].max().isoformat(), "end_max_utc": times["end"].max().isoformat(),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "mx_events": int(mx.sum()),
        "mx_events_by_start_date_utc": {str(k): int(v) for k, v in pd.Series(mx.to_numpy(), index=times["start"].dt.date).groupby(level=0).sum().items()},
        "quality_checks": checks, "overlapping_event_interval_pairs": int(overlaps),
        "provenance": provenance,
        "official_comparison": "NOT PERFORMED; NOAA 2026 Flare Report is not present locally.",
        "official_comparison_candidate": "sci_xrsf-l2-flrpt_geo_y2026_v1-0-1.csv plus metadata JSON",
    }


def _audit_solexs(path):
    members_out, coverage = [], []
    with ZipFile(path, "r") as outer:
        nested = sorted((i for i in outer.infolist() if i.filename.lower().endswith(".zip")), key=lambda i: i.filename)
        for nested_info in nested:
            nested_bytes = outer.read(nested_info)
            with ZipFile(io.BytesIO(nested_bytes), "r") as inner:
                members = [i for i in inner.infolist() if not i.is_dir()]
                lcs = [i for i in members if i.filename.lower().endswith(".lc.gz")]
                gtis = [i for i in members if i.filename.lower().endswith(".gti.gz")]
                for lc in lcs:
                    data = inner.read(lc)
                    parsed = _read_solexs_lc(data)
                    times = parsed["times"]
                    cadence = float(np.median(np.diff(times.asi8)) / 1e9) if len(times) > 1 else 1.0
                    intervals = contiguous_intervals(times, cadence, parsed["candidate_valid"])
                    product = {
                        "nested_archive": nested_info.filename,
                        "nested_archive_sha256": sha256_bytes(nested_bytes),
                        "member": lc.filename, "member_sha256": sha256_bytes(data),
                        "detector": "SDD2" if "/SDD2/" in lc.filename else "SDD1" if "/SDD1/" in lc.filename else None,
                        "format": "gzip-compressed FITS",
                        "raw_rows": len(times), "raw_start_utc": times.min().isoformat(),
                        "raw_end_utc": times.max().isoformat(), "cadence_seconds_median": cadence,
                        "finite_nonnegative_candidate_rows": int(parsed["candidate_valid"].sum()),
                        "excluded_nan_or_negative_rows": int((~parsed["candidate_valid"]).sum()),
                        "valid_rows_after_GTI": None, "GTI_applied": False,
                        "units": parsed["units"], "RATE_header": parsed["header"],
                        "primary_header": parsed["primary"], "candidate_intervals": intervals,
                        "GTI_companions": [
                            {"member": g.filename, "sha256": sha256_bytes(inner.read(g)), **_read_gti(inner.read(g))}
                            for g in gtis
                        ],
                        "quality_calibration_status": "UNVERIFIED: matching registered manual unavailable",
                    }
                    members_out.append(product)
                    coverage.extend({"instrument": "SoLEXS", "source_file": lc.filename,
                                     "status": "candidate_only_GTI_not_applied", **interval} for interval in intervals)
    return {
        "path": str(path), "sha256": sha256_path(path), "nested_archive_count": len(nested),
        "format": "ZIP of daily ZIPs; LC members are gzip-compressed FITS",
        "products": members_out, "raw_rows": sum(p["raw_rows"] for p in members_out),
        "candidate_rows": sum(p["finite_nonnegative_candidate_rows"] for p in members_out),
        "scientifically_valid_rows": None, "status": "BLOCKED_UNVERIFIED_INSTRUMENT_SEMANTICS",
    }, coverage


def _audit_hel1os(path):
    products, coverage = [], []
    with ZipFile(path, "r") as outer:
        nested = sorted((i for i in outer.infolist() if i.filename.lower().endswith(".zip")), key=lambda i: i.filename)
        for nested_info in nested:
            nested_bytes = outer.read(nested_info)
            with ZipFile(io.BytesIO(nested_bytes), "r") as inner:
                items = [i for i in inner.infolist() if not i.is_dir()]
                lcs = [i for i in items if re.search(r"lightcurve_.*\.fits$", i.filename, re.I)]
                gtis = [i.filename for i in items if re.search(r"/aux/gti.*\.fits$", i.filename, re.I)]
                spectra_count = sum("spectra" in i.filename.lower() for i in items)
                event_members = [i.filename for i in items if "/events/" in i.filename.lower()]
                for lc in sorted(lcs, key=lambda i: i.filename):
                    data = inner.read(lc)
                    parsed = _read_hel1os_lc(data)
                    times = parsed["times"]
                    diffs = np.diff(times.asi8) / 1e9 if len(times) > 1 else np.array([])
                    cadence = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else None
                    intervals = contiguous_intervals(times, cadence or 1.0, parsed["candidate_valid"])
                    version_match = re.search(r"_lev1_(V[^.]+)", nested_info.filename, re.I)
                    version = version_match.group(1) if version_match else None
                    record = {
                        "nested_archive": nested_info.filename,
                        "nested_archive_sha256": sha256_bytes(nested_bytes),
                        "member": lc.filename, "member_sha256": sha256_bytes(data),
                        "processing_version": version, "extension": parsed["extension"],
                        "raw_rows": parsed["rows"], "raw_start_utc": times.min().isoformat(),
                        "raw_end_utc": times.max().isoformat(), "cadence_seconds_median": cadence,
                        "candidate_finite_nonnegative_rate_error_rows": parsed["valid_count"],
                        "scientifically_valid_rows": None, "columns": parsed["columns"],
                        "extension_header": parsed["header"], "primary_header": parsed["primary"],
                        "candidate_intervals": intervals, "GTI_companions": gtis,
                        "spectra_member_count": spectra_count, "event_members": event_members,
                        "quality_calibration_status": "UNVERIFIED: matching registered manual unavailable",
                    }
                    products.append(record)
                    coverage.extend({"instrument": "HEL1OS", "source_file": lc.filename,
                                     "status": "candidate_only_quality_GTI_unverified",
                                     "processing_version": version, **interval} for interval in intervals)
    overlaps = []
    ordered = sorted(products, key=lambda p: (p["raw_start_utc"], p["member"]))
    for index, left in enumerate(ordered):
        ls, le = pd.Timestamp(left["raw_start_utc"]), pd.Timestamp(left["raw_end_utc"])
        for right in ordered[index + 1:]:
            rs, re_ = pd.Timestamp(right["raw_start_utc"]), pd.Timestamp(right["raw_end_utc"])
            if max(ls, rs) <= min(le, re_):
                overlaps.append({
                    "left_member": left["member"], "left_version": left["processing_version"],
                    "right_member": right["member"], "right_version": right["processing_version"],
                    "overlap_start_utc": max(ls, rs).isoformat(), "overlap_end_utc": min(le, re_).isoformat(),
                    "selection": "UNRESOLVED_NO_PRODUCT_DISCARDED",
                })
    return {
        "path": str(path), "sha256": sha256_path(path), "nested_archive_count": len(nested),
        "format": "outer ZIP with nested Level-1 ZIP/FITS products",
        "lightcurve_count": len(products), "products": products,
        "overlapping_products": overlaps, "version_policy": "UNRESOLVED; no processing version selected",
        "scientifically_valid_rows": None, "status": "BLOCKED_UNVERIFIED_INSTRUMENT_SEMANTICS",
    }, coverage


def _coverage_frame(rows):
    columns = ["instrument", "source_file", "status", "processing_version",
               "start_utc", "end_utc", "sample_count", "nominal_cadence_seconds"]
    frame = pd.DataFrame(rows)
    for col in columns:
        if col not in frame:
            frame[col] = None
    return frame[columns].sort_values(["instrument", "start_utc", "source_file"],
                                      kind="stable", na_position="last").reset_index(drop=True)


def _candidate_joint_intervals(coverage, events):
    """Intersect candidate instrument coverage; never call it verified coverage."""
    candidates = {}
    for instrument in ("SoLEXS", "HEL1OS"):
        candidates[instrument] = [
            (pd.Timestamp(row["start_utc"]), pd.Timestamp(row["end_utc"]))
            for row in coverage
            if row.get("instrument") == instrument
            and row.get("start_utc") and row.get("end_utc")
        ]
    goes_by_channel = {}
    for row in coverage:
        source = str(row.get("source_file", ""))
        if row.get("instrument") == "GOES-18" and row.get("start_utc") and row.get("end_utc"):
            channel = source.rsplit(":", 1)[-1]
            goes_by_channel.setdefault(channel, []).append(
                (pd.Timestamp(row["start_utc"]), pd.Timestamp(row["end_utc"]))
            )
    goes_common = []
    for a_start, a_end in goes_by_channel.get("xrsa", []):
        for b_start, b_end in goes_by_channel.get("xrsb", []):
            start, end = max(a_start, b_start), min(a_end, b_end)
            if start <= end:
                goes_common.append((start, end))
    events_frame = events
    overlaps = []
    for sx_start, sx_end in candidates["SoLEXS"]:
        for hl_start, hl_end in candidates["HEL1OS"]:
            start, end = max(sx_start, hl_start), min(sx_end, hl_end)
            if start > end:
                continue
            for goes_start, goes_end in goes_common:
                joint_start, joint_end = max(start, goes_start), min(end, goes_end)
                if joint_start > joint_end:
                    continue
                count = int(((events_frame["start"] >= joint_start) &
                             (events_frame["start"] <= joint_end)).sum())
                key = (joint_start.isoformat(), joint_end.isoformat())
                overlaps.append((key, count))
    unique = {}
    for key, count in overlaps:
        unique[key] = count
    return [
        {"start_utc": start, "end_utc": end, "candidate_goes_event_starts": count,
         "status": "CANDIDATE_ONLY_UNVERIFIED_INSTRUMENT_QUALITY_AND_LABELS"}
        for (start, end), count in sorted(unique.items())
    ]


def _markdown(report):
    events, goes, sx, hl = report["goes_events"], report["goes"], report["solexs"], report["hel1os"]
    lines = [
        f"<!-- {GENERATOR} -->",
        "# Phase 3A — Real solar data validation and training readiness",
        "",
        f"- Audit time UTC: {report['audit_time_utc']}",
        f"- Status: **{report['readiness_status']}**",
        "- Training started: **No**",
        "- Raw archives modified or extracted: **No**",
        "",
        "## Findings",
        "",
        f"- GOES-18 L2 has {goes['total_rows']} one-minute rows across {len(goes['netcdf_files'])} daily NetCDFs, from {goes['coverage_start_utc']} to {goes['coverage_end_utc']}. The CSV timestamps and values match NetCDF exactly (NaN-aware): {goes['csv_comparison']['timestamps_equal_netcdf_exact']}; A values {goes['csv_comparison']['channel_comparison']['xrsa']['values_equal_nan_aware']}; B values {goes['csv_comparison']['channel_comparison']['xrsb']['values_equal_nan_aware']}. The CSV drops product quality flags.",
        f"- The event catalog has {events['rows']} events and {events['mx_events']} M/X events. The local generation script identifies a SunPy HEK query (FL, GOES class > B1.0, observatory GOES) and deduplicates by start plus class. This is a derived local catalog, not the NOAA Flare Report file. Retrieval time, version and per-event satellite are absent.",
        "- The local catalog is UNVERIFIED for completeness and event revisions until compared row-by-row with NOAA/NCEI's current 2026 GOES Flare Report and its metadata JSON. The official report is available online, but is not in the local input set.",
        f"- SoLEXS contains {sx['nested_archive_count']} nested daily archives and {sx['raw_rows']} light-curve rows; {sx['candidate_rows']} have finite nonnegative values. GTI filtering was not applied. The FITS COUNTS field has no declared unit; no flux/rate relabeling was made.",
        f"- HEL1OS contains {hl['nested_archive_count']} nested archives and {hl['lightcurve_count']} light curves; {len(hl['overlapping_products'])} pairs have overlapping time coverage. MJD/ISOT, GTI, detector, quality and V111/V211 interpretation still need registered product documentation. No version was discarded.",
        f"- Cross-instrument interval intersection yields {len(report['joint_coverage']['candidate_intervals'])} candidate overlaps using rows with both GOES channels passing the conservative flag filter. This is a time-overlap count across detector/band products; it is not verified exposure or a count of independently labeled events.",
        "",
        "## GOES quality policy",
        "",
        "Candidate-valid samples require finite non-fill values and a channel flag of zero (good_data). Nonzero flags, including interpolation, and fill values are retained as exclusions in the record-quality CSV. The original NetCDF files remain unchanged.",
        "",
        "## Proposed input mapping",
        "",
        "- SoLEXS: retain detector-specific RATE.COUNTS for SDD2, pending manual confirmation of its physical meaning, GTIs and calibration.",
        "- HEL1OS: retain per-detector/per-band CTR and STAT_ERR in cts/sec; apply documented GTI and housekeeping rules after manual verification.",
        "- GOES: keep xrsa_flux/xrsb_flux as separate W/m² reference measurements. They are not a substitute for either Aditya instrument channel.",
        "- Do not map count-rate fields into the existing flux-named prediction inputs or form a cross-instrument ratio without documented compatible calibration.",
        "",
        "## Chronological training gate",
        "",
        "No three-way split is defensible from the verified data: event completeness and both Aditya measurement semantics are unresolved, HEL1OS coverage is partial, and the catalog spans only six days. No events were moved or rebalanced. Positive/negative/unknown split counts and valid-window counts are withheld because they would imply verified inputs that are not established.",
        "",
        "## Needed next evidence",
        "",
        "1. Matching registered SoLEXS and HEL1OS manuals with GTI endpoint, quality, detector, calibration and version definitions.",
        "2. NOAA/NCEI GOES Flare Report 2026 CSV or NetCDF plus metadata JSON; compare by event identifier, class and times, and document unmatched/revised/overlapping events.",
        "3. A longer quality-verified observation interval before date-based train/validation/test boundaries are proposed.",
        "",
        "## Checksums",
        "",
        f"- SoLEXS archive: {sx['sha256']}",
        f"- HEL1OS archive: {hl['sha256']}",
        f"- GOES event CSV: {events['sha256']}",
        f"- GOES measurement CSV: {goes['csv_comparison']['sha256']}",
        "- Daily NetCDF and nested FITS member checksums are recorded in the JSON.",
        "",
        "Additional output: phase3a_goes_record_quality.csv contains one row per GOES channel and timestamp.",
        "",
    ]
    return "\n".join(lines)


def _recognized(path, kind):
    try:
        if kind == "json":
            return json.loads(path.read_text(encoding="utf-8")).get("generator") == GENERATOR
        if kind == "markdown":
            return path.read_text(encoding="utf-8").startswith(f"<!-- {GENERATOR} -->")
        with path.open("r", encoding="utf-8", newline="") as stream:
            return "audit_generator" in next(csv.reader(stream))
    except Exception:
        return False


def _atomic_write(path, content, kind):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not _recognized(path, kind):
        raise FileExistsError(f"Refusing to overwrite an unrecognized existing file: {path}")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def run_audit(downloads_root, repo_root):
    downloads_root, repo_root = Path(downloads_root), Path(repo_root)
    goes_dir = downloads_root / "data" / "goes"
    solexs_path = downloads_root / "solexs_2026Oct23T184402685.zip"
    hel1os_path = downloads_root / "hel1os_2026Oct23T184518914.zip"
    event_path, csv_path = goes_dir / "goes_flares.csv", goes_dir / "goes_xrs_1min.csv"
    generator = downloads_root / "download_goes.py"
    required = [solexs_path, hel1os_path, event_path, csv_path]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required input files missing: " + ", ".join(missing))
    nc_paths = sorted((goes_dir / "raw").glob("*.nc"))
    if not nc_paths:
        raise FileNotFoundError(f"No NetCDF files found under {goes_dir / 'raw'}")
    sx, sx_coverage = _audit_solexs(solexs_path)
    hl, hl_coverage = _audit_hel1os(hel1os_path)
    daily = [_audit_goes_nc(path) for path in nc_paths]
    comparison = _compare_goes_csv(csv_path, daily)
    events = _audit_events(event_path, generator if generator.is_file() else None)
    coverage = sx_coverage + hl_coverage
    quality_records = []
    for item in daily:
        quality_records.extend(item["records"])
        for channel in ("xrsa", "xrsb"):
            rows = [r for r in item["records"] if r["channel"] == channel]
            times = pd.DatetimeIndex(pd.to_datetime([r["timestamp_utc"] for r in rows], utc=True))
            valid = np.asarray([r["candidate_valid"] for r in rows], dtype=bool)
            coverage.extend({"instrument": "GOES-18", "source_file": item["filename"] + ":" + channel,
                             "status": "finite_nonfill_flag0_candidate", **interval}
                            for interval in contiguous_intervals(times, 60, valid))
    coverage_df = _coverage_frame(coverage)
    quality_df = pd.DataFrame([{"audit_generator": GENERATOR, **r} for r in quality_records])
    report = {
        "generator": GENERATOR, "audit_time_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True, "training_started": False, "raw_files_modified": False,
        "readiness_status": "BLOCKED_UNVERIFIED_LABELS",
        "status_reason": "HEK-derived catalog not compared to NOAA's current 2026 report; SoLEXS/HEL1OS quality and measurement semantics remain unverified.",
        "input_paths": {
            "repo_root": str(repo_root), "goes_event_csv": str(event_path),
            "goes_measurement_csv": str(csv_path), "goes_netcdf_directory": str(goes_dir / "raw"),
            "solexs_archive": str(solexs_path), "hel1os_archive": str(hel1os_path),
            "goes_generation_script": str(generator) if generator.is_file() else None,
        },
        "goes": {
            "netcdf_file_count": len(daily), "total_rows": sum(x["rows"] for x in daily),
            "coverage_start_utc": min(x["start_utc"] for x in daily if x["start_utc"]),
            "coverage_end_utc": max(x["end_utc"] for x in daily if x["end_utc"]),
            "netcdf_files": daily, "csv_comparison": comparison,
            "validity_rule": "finite, not fill value, and flag zero; nonzero/interpolated flags excluded",
        },
        "goes_events": events, "solexs": sx, "hel1os": hl,
        "joint_coverage": {
            "scientifically_verified_intervals": [],
            "candidate_intervals": _candidate_joint_intervals(
                coverage,
                pd.read_csv(event_path).assign(
                    start=lambda f: pd.to_datetime(f["start"], utc=True, errors="coerce")
                ),
            ),
            "reason": "No interval currently meets all verified measurement, quality and independent-label requirements.",
        },
        "proposed_feature_mapping": [
            {"instrument": "SoLEXS", "source": "SDD2 RATE.COUNTS", "units": "not declared in FITS", "quality": "manual/GTI/calibration pending", "feature": "solexs_counts_sdd2_native", "inference_ready": False},
            {"instrument": "HEL1OS", "source": "per-detector/band CTR and STAT_ERR", "units": "cts/sec", "quality": "manual/GTI/housekeeping/version policy pending", "feature": "hel1os_detector_band_count_rate", "inference_ready": False},
            {"instrument": "GOES-18", "source": "xrsa_flux and xrsb_flux", "units": "W/m2", "quality": "filter by documented flags and fill values", "feature": "goes_xrs_channel_irradiance", "inference_ready": False},
        ],
        "training_readiness": {
            "training_allowed": False, "date_boundaries": None, "valid_windows": None,
            "positive_negative_unknown_counts": None,
            "reason": "Measurement semantics, event completeness and sufficient temporal span are not established.",
        },
    }
    return report, coverage_df, quality_df


def write_reports(report, coverage, quality, report_dir):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    coverage = coverage.assign(audit_generator=GENERATOR)
    coverage = coverage.loc[:, ["audit_generator", *[c for c in coverage.columns if c != "audit_generator"]]]
    quality = quality.loc[:, ["audit_generator", *[c for c in quality.columns if c != "audit_generator"]]]
    files = [
        (report_dir / "phase3a_data_validation.md", _markdown(report), "markdown"),
        (report_dir / "phase3a_data_audit.json", json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", "json"),
        (report_dir / "phase3a_coverage.csv", coverage.to_csv(index=False), "csv"),
        (report_dir / "phase3a_goes_record_quality.csv", quality.to_csv(index=False), "csv"),
    ]
    for path, content, kind in files:
        _atomic_write(path, content, kind)
    return [path for path, _, _ in files]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads-root", type=Path, default=Path(r"C:\Users\ASHUTOSH\Downloads"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report-dir", type=Path, default=Path(r"E:\SolarFlareData\reports"))
    args = parser.parse_args()
    report, coverage, quality = run_audit(args.downloads_root, args.repo_root)
    outputs = write_reports(report, coverage, quality, args.report_dir)
    print("Status:", report["readiness_status"])
    print("Training started:", report["training_started"])
    print("Event rows / M-X:", report["goes_events"]["rows"], report["goes_events"]["mx_events"])
    print("Coverage intervals / GOES channel records:", len(coverage), len(quality))
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
