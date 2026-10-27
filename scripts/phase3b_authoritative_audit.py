"""Phase 3B: authoritative event reconciliation and verified-data gate.

Reads Phase 3A audit products and original event inputs. It never extracts or
modifies the large instrument archives, and refuses to overwrite unrelated
reports. Native instrument semantics remain blocked pending registered manuals.
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
from zipfile import ZipFile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time

GENERATOR = "phase3b_authoritative_audit_v1"
NOAA_BASE = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/multi/l2/data/xrsf-l2-flrpt_science/csv"
NOAA_DOCS = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs"
PRADAN = "https://pradan.issdc.gov.in/al1/"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_series(values):
    """Parse documented UTC input values as timezone-aware timestamps."""
    return pd.to_datetime(values, utc=True, errors="coerce", format="mixed")


def normalize_class(value):
    """Normalize whitespace/case only; retain the magnitude without rounding."""
    original = "" if pd.isna(value) else str(value)
    raw = original.strip().upper()
    match = re.fullmatch(r"([ABCMX])\s*([0-9]+(?:\.[0-9]+)?)", raw)
    if not match:
        return {"original": original, "normalized": None, "family": None, "magnitude": None, "is_mx": None}
    family, magnitude = match.groups()
    normalized = family + magnitude
    return {"original": original, "normalized": normalized, "family": family,
            "magnitude": float(magnitude), "is_mx": family in {"M", "X"}}


def classify_horizon(origin, horizon_seconds, catalog_complete, event_starts, coverage_intervals):
    """A missing/unobserved target horizon is unknown, never a negative label."""
    origin = pd.Timestamp(origin)
    origin = origin.tz_localize("UTC") if origin.tzinfo is None else origin.tz_convert("UTC")
    end = origin + pd.Timedelta(seconds=horizon_seconds)
    found = [pd.Timestamp(t) for t in event_starts
             if origin < pd.Timestamp(t) <= end]
    if found:
        return {"label": 1, "reason": "verified_event_start_in_open_closed_horizon", "event_starts": found}
    for start, stop in coverage_intervals:
        start, stop = pd.Timestamp(start), pd.Timestamp(stop)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        stop = stop.tz_localize("UTC") if stop.tzinfo is None else stop.tz_convert("UTC")
        if start <= origin and stop >= end:
            if catalog_complete:
                return {"label": 0, "reason": "fully_covered_horizon_and_complete_event_catalog", "event_starts": []}
            return {"label": None, "reason": "event_catalog_completeness_unverified", "event_starts": []}
    return {"label": None, "reason": "missing_or_insufficient_horizon_coverage", "event_starts": []}


def reconcile_events(local: pd.DataFrame, official: pd.DataFrame, start, end, tolerance_seconds=60):
    """Match by start time within catalog resolution; preserve ambiguity and inputs."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    local = local.copy().reset_index(drop=True)
    official = official.copy().reset_index(drop=True)
    local["_row"] = np.arange(len(local))
    official["_row"] = np.arange(len(official))
    for frame, source in ((local, "local"), (official, "official")):
        time_name = "start" if source == "local" else "start_time"
        frame["_start_utc"] = utc_series(frame[time_name])
        peak_name = "peak" if source == "local" else "time"
        frame["_peak_utc"] = utc_series(frame[peak_name])
        end_name = "end" if source == "local" else "end_time"
        frame["_end_utc"] = utc_series(frame[end_name])
        class_name = "goes_class" if source == "local" else "flare_class"
        frame["_class"] = frame[class_name].map(normalize_class)
        frame["_class_family"] = frame["_class"].map(lambda x: x["family"])
        frame["_class_normalized"] = frame["_class"].map(lambda x: x["normalized"])

    local = local[(local._start_utc >= start) & (local._start_utc < end)].copy().reset_index(drop=True)
    official = official[(official._start_utc >= start) & (official._start_utc < end)].copy().reset_index(drop=True)
    local_candidates, official_candidates = {}, {}
    pairs = []
    official_times = official._start_utc
    for li, row in local.iterrows():
        if pd.isna(row._start_utc):
            local_candidates[li] = []
            continue
        delta = (official_times - row._start_utc).dt.total_seconds().abs()
        candidates = np.flatnonzero((delta <= tolerance_seconds).to_numpy()).tolist()
        local_candidates[li] = candidates
        for oi in candidates:
            official_candidates.setdefault(oi, []).append(li)
            pairs.append((li, oi, float((official.iloc[oi]._start_utc - row._start_utc).total_seconds())))

    pair_by_local = {}
    for li, oi, delta in pairs:
        pair_by_local.setdefault(li, []).append((oi, delta))
    def official_fields(oi):
        row = official.iloc[oi]
        return {
            "official_row_number": int(row._row) + 1,
            "official_flare_id": str(row.get("flare_id", "")),
            "official_start_raw": str(row.get("start_time", "")),
            "official_peak_raw": str(row.get("time", "")),
            "official_end_raw": "" if pd.isna(row.get("end_time")) else str(row.get("end_time")),
            "official_class_raw": str(row.get("flare_class", "")),
            "official_class_normalized": row._class_normalized or "",
            "official_class_family": row._class_family or "",
            "official_satellite": str(row.get("xrsb_irrad_source", "")),
            "official_active_region": "" if pd.isna(row.get("active_region")) else str(row.get("active_region")),
        }

    rows, matched_official = [], set()
    for li, row in local.iterrows():
        candidates = pair_by_local.get(li, [])
        candidate_ids = [str(official.iloc[oi].get("flare_id", "")) for oi, _ in candidates]
        entry = {
            "record_type": "local_event", "local_row_number": int(row._row) + 1,
            "local_start_raw": str(row.get("start", "")),
            "local_peak_raw": str(row.get("peak", "")),
            "local_end_raw": "" if pd.isna(row.get("end")) else str(row.get("end")),
            "local_class_raw": str(row.get("goes_class", "")),
            "local_class_normalized": row._class_normalized or "",
            "local_class_family": row._class_family or "",
            "local_noaa_ar": "" if pd.isna(row.get("noaa_ar")) else str(row.get("noaa_ar")),
            "local_is_MX_raw": str(row.get("is_MX", "")),
            "match_candidate_count": len(candidates), "candidate_official_flare_ids": "|".join(candidate_ids),
            "start_delta_seconds": "", "peak_delta_seconds": "", "end_delta_seconds": "",
            "class_family_match": "", "exact_normalized_class_match": "",
            "reconciliation_status": "", "local_event_identity_uniquely_verified": False,
        }
        if not candidates:
            entry["reconciliation_status"] = "unmatched_local"
        elif len(candidates) != 1 or len(official_candidates.get(candidates[0][0], [])) != 1:
            entry["reconciliation_status"] = "ambiguous_start_match_no_selection"
        else:
            oi, delta = candidates[0]
            matched_official.add(oi)
            o = official.iloc[oi]
            entry.update(official_fields(oi))
            entry["start_delta_seconds"] = delta
            entry["peak_delta_seconds"] = ((o._peak_utc - row._peak_utc).total_seconds()
                                           if pd.notna(o._peak_utc) and pd.notna(row._peak_utc) else "")
            entry["end_delta_seconds"] = ((o._end_utc - row._end_utc).total_seconds()
                                          if pd.notna(o._end_utc) and pd.notna(row._end_utc) else "")
            family_match = bool(row._class_family and row._class_family == o._class_family)
            exact_match = bool(row._class_normalized and row._class_normalized == o._class_normalized)
            entry["class_family_match"] = family_match
            entry["exact_normalized_class_match"] = exact_match
            entry["local_event_identity_uniquely_verified"] = family_match
            if not family_match:
                entry["reconciliation_status"] = "unique_start_match_class_family_mismatch"
            elif not exact_match:
                entry["reconciliation_status"] = "unique_start_match_same_family_magnitude_mismatch"
            else:
                entry["reconciliation_status"] = "unique_start_and_class_match"
        rows.append(entry)

    for oi, row in official.iterrows():
        if oi in matched_official:
            continue
        candidate_local = official_candidates.get(oi, [])
        entry = {"record_type": "official_event", "local_row_number": "",
                 "local_start_raw": "", "local_peak_raw": "", "local_end_raw": "",
                 "local_class_raw": "", "local_class_normalized": "", "local_class_family": "",
                 "local_noaa_ar": "", "local_is_MX_raw": "", "match_candidate_count": len(candidate_local),
                 "candidate_official_flare_ids": str(row.get("flare_id", "")), "start_delta_seconds": "",
                 "peak_delta_seconds": "", "end_delta_seconds": "", "class_family_match": "",
                 "exact_normalized_class_match": "", "local_event_identity_uniquely_verified": False,
                 "reconciliation_status": "ambiguous_start_match_no_selection" if candidate_local else "official_only"}
        entry.update(official_fields(oi))
        rows.append(entry)

    result = pd.DataFrame(rows)
    # Keep exact duplicates and same-source near-duplicate candidates separate.
    local_dups = int(local.duplicated(subset=["start", "goes_class"], keep=False).sum())
    official_dups = int(official.duplicated(subset=["flare_id"], keep=False).sum())
    invalid_local_order = int(((local._start_utc > local._peak_utc) | (local._peak_utc > local._end_utc)).fillna(False).sum())
    invalid_official_order = int(((official._start_utc > official._peak_utc) |
                                  (official._peak_utc > official._end_utc)).fillna(False).sum())
    return result, {
        "time_match_policy": f"candidate if absolute start-time difference <= {tolerance_seconds} seconds; tolerance reflects official 1-minute report resolution",
        "class_policy": "candidate matching is timestamp-only; exact normalized classes and family agreement are reported separately; no class is overwritten",
        "local_rows_in_coverage": len(local), "official_rows_in_coverage": len(official),
        "official_missing_end_time_rows": int(official._end_utc.isna().sum()),
        "candidate_pair_count": len(pairs), "local_with_candidate": len(pair_by_local),
        "official_with_candidate": len(official_candidates),
        "unique_mutual_start_matches": sum(len(pair_by_local.get(li, [])) == 1 and
                                            len(official_candidates.get(pair_by_local[li][0][0], [])) == 1
                                            for li in range(len(local))),
        "ambiguous_local_rows": sum(len(pair_by_local.get(li, [])) > 1 or
                                     (len(pair_by_local.get(li, [])) == 1 and
                                      len(official_candidates.get(pair_by_local[li][0][0], [])) > 1)
                                     for li in range(len(local))),
        "unmatched_local_rows": sum(not pair_by_local.get(li) for li in range(len(local))),
        "unmatched_official_rows": sum(not official_candidates.get(oi) for oi in range(len(official))),
        "local_duplicate_start_class_rows": local_dups, "official_duplicate_flare_id_rows": official_dups,
        "near_duplicate_start_candidates_within_tolerance_not_merged": _near_duplicate_candidates(local, tolerance_seconds) +
                                                                      _near_duplicate_candidates(official, tolerance_seconds),
        "local_invalid_time_order_rows": invalid_local_order,
        "official_invalid_time_order_rows": invalid_official_order,
        "family_mismatch_rows": int((result.get("reconciliation_status", pd.Series(dtype=str)) == "unique_start_match_class_family_mismatch").sum()),
        "same_family_magnitude_mismatch_rows": int((result.get("reconciliation_status", pd.Series(dtype=str)) == "unique_start_match_same_family_magnitude_mismatch").sum()),
        "ambiguous_pairs_were_selected": False,
    }, local, official


def _near_duplicate_candidates(frame, tolerance_seconds):
    out = []
    for i in range(len(frame)):
        for j in range(i + 1, len(frame)):
            left, right = frame.iloc[i], frame.iloc[j]
            if pd.isna(left._start_utc) or pd.isna(right._start_utc):
                continue
            delta = abs((right._start_utc - left._start_utc).total_seconds())
            if delta <= tolerance_seconds and left._class_family == right._class_family:
                id_col = "flare_id" if "flare_id" in frame else None
                out.append({"row_left": int(left._row) + 1, "row_right": int(right._row) + 1,
                            "stable_ids": [str(left[id_col]), str(right[id_col])] if id_col else None,
                            "start_delta_seconds": delta, "class_family": left._class_family,
                            "resolution": "CANDIDATE_ONLY_NOT_MERGED"})
    return out


def verified_event_table(official):
    rows, excluded = [], []
    for _, row in official.iterrows():
        label = row._class
        reasons = []
        if pd.isna(row._start_utc): reasons.append("invalid_start_time")
        if pd.isna(row._peak_utc): reasons.append("invalid_peak_time")
        if row._class_family is None: reasons.append("unparseable_class")
        if pd.isna(row.get("flare_id")): reasons.append("missing_stable_flare_id")
        if pd.notna(row._start_utc) and pd.notna(row._peak_utc) and row._start_utc > row._peak_utc:
            reasons.append("start_after_peak")
        if reasons:
            excluded.append({"flare_id": row.get("flare_id"), "reasons": "|".join(reasons)})
            continue
        rows.append({
            "source_dataset": "NOAA_NCEI_GOES_XRS_L2_FLARE_REPORT",
            "source_version": "1-0-1", "flare_id": str(row.flare_id),
            "event_start_utc": row._start_utc.isoformat(), "event_peak_utc": row._peak_utc.isoformat(),
            "event_end_utc": "" if pd.isna(row._end_utc) else row._end_utc.isoformat(),
            "original_class": str(row.get("flare_class", "")),
            "normalized_class": label["normalized"], "class_family": label["family"],
            "is_mx": bool(label["is_mx"]), "xrsb_irrad_w_m2": row.get("xrsb_irrad"),
            "xrsb_satellite": row.get("xrsb_irrad_source"), "peak_saturated": row.get("peak_saturated"),
            "active_region": row.get("active_region"),
            "event_key_basis": "NOAA flare_id (minute-resolution onset key)",
            "event_label_status": "verified_official_report_record",
            "negative_label_completeness": "NOT_ESTABLISHED_BY_REPORT_DOCUMENTATION",
        })
    return pd.DataFrame(rows), excluded


def _daily_goes_quality(quality_path, measurement_path, raw_files):
    quality = pd.read_csv(quality_path)
    quality["timestamp_utc"] = utc_series(quality.timestamp_utc)
    quality["date_utc"] = quality.timestamp_utc.dt.strftime("%Y-%m-%d")
    measurement = pd.read_csv(measurement_path)
    measurement["timestamp_utc"] = utc_series(measurement.timestamp)
    comparisons, days = [], []
    for (date, channel), group in quality.groupby(["date_utc", "channel"], sort=True):
        group = group.sort_values("timestamp_utc", kind="stable")
        times = pd.DatetimeIndex(group.timestamp_utc.dropna())
        dts = np.diff(times.asi8) / 1e9 if len(times) > 1 else np.array([])
        reasons = group.quality_meanings.fillna("").astype(str)
        flag_counts = {}
        for reason in reasons:
            for token in set(x for x in reason.split("|") if x):
                flag_counts[token] = flag_counts.get(token, 0) + 1
        finite_nonfill = pd.to_numeric(group.value_w_m2, errors="coerce").notna()
        candidate = group.candidate_valid.astype(bool)
        expected = 1440
        expected_times = pd.date_range(f"{date}T00:00:00Z", periods=expected, freq="min")
        missing_times = expected_times.difference(times)
        missing_runs = np.diff(np.r_[False, expected_times.isin(missing_times), False].astype(np.int8))
        run_starts, run_ends = np.flatnonzero(missing_runs == 1), np.flatnonzero(missing_runs == -1)
        longest_missing_run = int(np.max(run_ends - run_starts)) if len(run_starts) else 0
        excluded_union = group.flag_excluded_union.fillna(0).astype(int).value_counts().sort_index()
        raw_channel = raw_files.get((date, channel), {})
        row = {
            "utc_date": date, "instrument": "GOES-18", "channel": channel,
            "expected_rows": expected, "observed_rows": len(group),
            "missing_rows_vs_expected": max(0, expected - group.timestamp_utc.nunique()),
            "longest_missing_run_seconds": longest_missing_run * 60,
            "duplicate_timestamps": int(group.timestamp_utc.duplicated().sum()),
            "non_60s_intervals": int((np.abs(dts - 60) > 1e-6).sum()),
            "finite_nonfill_rows": int(finite_nonfill.sum()),
            "fill_or_nonfinite_rows": int((~finite_nonfill).sum()),
            "flag_zero_good_rows": int((group.flag == 0).sum()),
            "candidate_valid_rows": int(candidate.sum()),
            "candidate_coverage_fraction_of_expected": float(candidate.sum() / expected),
            "quality_reason_counts": flag_counts,
            "excluded_1s_flag_union_counts": {str(k): int(v) for k, v in excluded_union.items()},
            "source_file": raw_channel.get("filename"), "source_sha256": raw_channel.get("sha256"),
            "timestamp_status": "UTC_FROM_NETCDF_UNITS; VERIFIED_BY_PHASE3A_SAMPLE_COMPARISON",
        }
        days.append(row)
        left = group[["timestamp_utc", "value_w_m2"]].rename(columns={"value_w_m2": "netcdf_value"})
        right = measurement[["timestamp_utc", channel]].rename(columns={channel: "csv_value"})
        joined = left.merge(right, on="timestamp_utc", how="outer", validate="one_to_one", indicator=True)
        both = joined[joined._merge == "both"]
        nv = pd.to_numeric(both.netcdf_value, errors="coerce").to_numpy(float)
        cv = pd.to_numeric(both.csv_value, errors="coerce").to_numpy(float)
        paired = np.isfinite(nv) & np.isfinite(cv)
        close = np.isclose(nv[paired], cv[paired], rtol=1e-7, atol=0.0)
        comparisons.append({
            "utc_date": date, "channel": channel, "matched_timestamps": len(both),
            "csv_only_timestamps": int((joined._merge == "right_only").sum()),
            "netcdf_only_timestamps": int((joined._merge == "left_only").sum()),
            "both_finite_value_pairs": int(paired.sum()),
            "values_equal_within_rtol_1e-7": bool(close.all()),
            "value_mismatch_count": int((~close).sum()),
            "max_absolute_difference_w_m2": float(np.max(np.abs(nv[paired] - cv[paired]))) if paired.any() else None,
            "fill_rows_with_csv_finite": int((~np.isfinite(nv) & np.isfinite(cv)).sum()),
        })
    return days, comparisons


def _gti_table_info(member, payload):
    try:
        if payload[:2] == b"\x1f\x8b": payload = gzip.decompress(payload)
        with fits.open(io.BytesIO(payload), mode="readonly", memmap=False) as hdus:
            hdu = next((h for h in hdus if getattr(h, "columns", None) is not None and h.data is not None), None)
            if hdu is None: return {"member": member, "parse_status": "NO_POPULATED_TABLE"}
            names = {n.upper(): n for n in hdu.columns.names}
            vals = {}
            for key in ("START", "STOP"):
                if key in names:
                    arr = np.asarray(hdu.data[names[key]], dtype=float)
                    vals[key.lower()] = {"min_raw": float(np.nanmin(arr)) if np.isfinite(arr).any() else None,
                                         "max_raw": float(np.nanmax(arr)) if np.isfinite(arr).any() else None,
                                         "unit": next((c.unit for c in hdu.columns if c.name == names[key]), None)}
            return {"member": member, "parse_status": "STRUCTURE_READ_SEMANTICS_UNVERIFIED",
                    "extension": hdu.name, "rows": len(hdu.data),
                    "columns": [{"name": c.name, "format": c.format, "unit": c.unit} for c in hdu.columns],
                    "header_time_keys": {k: hdu.header.get(k) for k in ("TIMESYS", "TIMEUNIT", "TSTART", "TSTOP", "MJDREF", "TIMEZERO") if k in hdu.header},
                    "raw_interval_columns": vals}
    except Exception as exc:
        return {"member": member, "parse_status": "PARSE_ERROR", "error": type(exc).__name__}


def audit_aditya_archive_semantics(downloads_root):
    """Read MJD/ISOT and GTI structure in-place; apply no mission rules."""
    sx_path = downloads_root / "solexs_2026Oct23T184402685.zip"
    hl_path = downloads_root / "hel1os_2026Oct23T184518914.zip"
    hel_lc, hel_gtis, solexs_gtis = [], [], []
    with ZipFile(sx_path, "r") as outer:
        for nested in (x for x in outer.infolist() if x.filename.lower().endswith(".zip")):
            with ZipFile(io.BytesIO(outer.read(nested)), "r") as inner:
                for item in inner.infolist():
                    if item.is_dir() or not item.filename.lower().endswith(".gti.gz"): continue
                    info = _gti_table_info(item.filename, inner.read(item))
                    info.update({"instrument": "SoLEXS", "nested_archive": nested.filename})
                    solexs_gtis.append(info)
    with ZipFile(hl_path, "r") as outer:
        for nested in (x for x in outer.infolist() if x.filename.lower().endswith(".zip")):
            with ZipFile(io.BytesIO(outer.read(nested)), "r") as inner:
                for item in inner.infolist():
                    if item.is_dir(): continue
                    name = item.filename.lower()
                    if "gti" in name and name.endswith(".fits"):
                        info = _gti_table_info(item.filename, inner.read(item))
                        info.update({"instrument": "HEL1OS", "nested_archive": nested.filename})
                        hel_gtis.append(info)
                    if not re.search(r"lightcurve_.*\.fits$", name): continue
                    payload = inner.read(item)
                    with fits.open(io.BytesIO(payload), mode="readonly", memmap=False) as hdus:
                        table = next((h for h in hdus if getattr(h, "columns", None) is not None and h.data is not None and
                                      {n.upper() for n in (h.columns.names or [])}.issuperset({"MJD", "ISOT"})), None)
                        if table is None: continue
                        names = {n.upper(): n for n in table.columns.names}
                        mjd_all = np.asarray(table.data[names["MJD"]], dtype=float)
                        # FITS light curves can contain many millions of rows. A deterministic,
                        # evenly spaced sample checks representation consistency without a
                        # prohibitively slow whole-file datetime conversion.
                        sample_indices = np.unique(np.linspace(
                            0, max(0, len(mjd_all) - 1), min(len(mjd_all), 1000), dtype=np.int64))
                        mjd = mjd_all[sample_indices]
                        isot = utc_series(np.asarray(table.data[names["ISOT"]][sample_indices]).astype(str))
                        finite = np.isfinite(mjd) & ~pd.isna(isot)
                        if finite.any():
                            converted = pd.DatetimeIndex(pd.to_datetime(
                                Time(mjd[finite], format="mjd", scale="utc").to_datetime(timezone=timezone.utc), utc=True))
                            deltas = np.abs(converted.asi8 - pd.DatetimeIndex(isot[finite]).asi8) / 1e9
                            delta_summary = {"compared_rows": int(finite.sum()), "mjd_isot_exact_rows": int((deltas == 0).sum()),
                                             "mjd_isot_within_1ms_rows": int((deltas <= 0.001).sum()),
                                             "mjd_isot_over_1s_rows": int((deltas > 1.0).sum()),
                                             "max_abs_difference_seconds": float(np.max(deltas)),
                                             "median_abs_difference_seconds": float(np.median(deltas))}
                        else:
                            delta_summary = {"compared_rows": 0, "mjd_isot_exact_rows": 0,
                                             "mjd_isot_within_1ms_rows": 0, "mjd_isot_over_1s_rows": None,
                                             "max_abs_difference_seconds": None, "median_abs_difference_seconds": None}
                        hel_lc.append({"nested_archive": nested.filename, "member": item.filename,
                                       "processing_version": (re.search(r"_lev1_(V[^.]+)", nested.filename, re.I).group(1)
                                                              if re.search(r"_lev1_(V[^.]+)", nested.filename, re.I) else None),
                                       "extension": table.name, "rows": len(mjd_all),
                                       "comparison_method": f"deterministic evenly spaced sample of up to 1000 rows from {len(mjd_all)}; first/middle/last included",
                                       "time_scale_assumption_for_comparison": "UTC (provisional; not established by registered manual)",
                                       "mjd_isot_comparison": delta_summary})
    deltas = [x["mjd_isot_comparison"] for x in hel_lc if x["mjd_isot_comparison"]["compared_rows"]]
    return {
        "solexs_gti_tables": solexs_gtis, "solexs_gti_table_count": len(solexs_gtis),
        "solexs_gti_total_rows": sum(x.get("rows", 0) for x in solexs_gtis),
        "hel1os_gti_tables": hel_gtis, "hel1os_gti_table_count": len(hel_gtis),
        "hel1os_gti_total_rows": sum(x.get("rows", 0) for x in hel_gtis),
        "hel1os_lightcurve_time_checks": hel_lc, "hel1os_lightcurve_count_checked": len(hel_lc),
        "hel1os_mjd_isot_total_rows_compared": sum(x["compared_rows"] for x in deltas),
        "hel1os_mjd_isot_rows_within_1ms": sum(x["mjd_isot_within_1ms_rows"] for x in deltas),
        "hel1os_mjd_isot_rows_over_1s": sum(x["mjd_isot_over_1s_rows"] for x in deltas),
        "hel1os_mjd_isot_max_abs_difference_seconds": max((x["max_abs_difference_seconds"] for x in deltas), default=None),
        "interpretation": "Structural sample consistency only. MJD was compared using an assumed UTC scale; authority, FITS time reference, GTI closure and application remain UNVERIFIED without the registered instrument manuals.",
        "raw_archive_modified": False, "archives_extracted": False,
    }


def _event_in_candidate_segments(event_times, segments):
    return sum(1 for event in event_times if any(pd.Timestamp(s) <= event <= pd.Timestamp(e) for s, e in segments))


def build_coverage(phase3a, verified_labels, quality_days, archive_semantics):
    rows, events = [], list(utc_series(verified_labels.event_start_utc)) if len(verified_labels) else []
    for product in phase3a["solexs"]["products"]:
        intervals = product.get("candidate_intervals", [])
        bounds = [(i["start_utc"], i["end_utc"]) for i in intervals]
        member = product.get("member", "")
        gti = next((g for g in archive_semantics["solexs_gti_tables"]
                    if g.get("nested_archive") == product.get("nested_archive")
                    and ("/SDD2/" in g.get("member", "") or "SDD2" in member)), None)
        rows.append({
            "date_utc": product["nested_archive"].split("_")[-2] if "_" in product["nested_archive"] else None,
            "instrument": "SoLEXS", "detector": product.get("detector"), "energy_band": "UNVERIFIED",
            "processing_version": None, "source_file": product["member"],
            "raw_start_utc": product["raw_start_utc"], "raw_end_utc": product["raw_end_utc"],
            "expected_rows": product["raw_rows"], "observed_rows": product["raw_rows"],
            "candidate_quality_rows": product["finite_nonnegative_candidate_rows"],
            "valid_after_GTI": None, "GTI_rows_observed": gti.get("rows") if gti else None,
            "GTI_applied": False, "candidate_intervals_json": json.dumps(bounds),
            "candidate_event_starts": _event_in_candidate_segments(events, bounds),
            "verified_coverage": False,
            "coverage_status": "UNVERIFIED_TIME_REFERENCE_UNITS_GTI_QUALITY_CALIBRATION",
        })
    for product in phase3a["hel1os"]["products"]:
        ext = product.get("extension", "")
        detector = ext.split("_LC_", 1)[0] if "_LC_" in ext else "UNRESOLVED"
        band = ext.split("_BAND_", 1)[1] if "_BAND_" in ext else "UNRESOLVED"
        intervals = product.get("candidate_intervals", [])
        bounds = [(i["start_utc"], i["end_utc"]) for i in intervals]
        rows.append({
            "date_utc": product.get("raw_start_utc", "")[:10], "instrument": "HEL1OS",
            "detector": detector, "energy_band": band,
            "processing_version": product.get("processing_version"), "source_file": product["member"],
            "raw_start_utc": product["raw_start_utc"], "raw_end_utc": product["raw_end_utc"],
            "expected_rows": None, "observed_rows": product["raw_rows"],
            "candidate_quality_rows": product["candidate_finite_nonnegative_rate_error_rows"],
            "valid_after_GTI": None, "GTI_rows_observed": None, "GTI_applied": False,
            "candidate_intervals_json": json.dumps(bounds),
            "candidate_event_starts": _event_in_candidate_segments(events, bounds),
            "verified_coverage": False,
            "coverage_status": "UNVERIFIED_TIME_REFERENCE_GTI_QUALITY_VERSION_CALIBRATION",
        })
    for day in quality_days:
        rows.append({
            "date_utc": day["utc_date"], "instrument": "GOES-18", "detector": None,
            "energy_band": "XRS-A" if day["channel"] == "xrsa" else "XRS-B",
            "processing_version": "2-2-2", "source_file": day["source_file"],
            "raw_start_utc": None, "raw_end_utc": None,
            "expected_rows": day["expected_rows"], "observed_rows": day["observed_rows"],
            "candidate_quality_rows": day["candidate_valid_rows"],
            "valid_after_GTI": None, "GTI_rows_observed": None,
            "GTI_applied": None, "candidate_intervals_json": None,
            "candidate_event_starts": None, "verified_coverage": True,
            "coverage_status": "VERIFIED_GOES_FLAG0_FINITE_NONFILL;SEE_DAILY_QUALITY_JSON",
        })
    return pd.DataFrame(rows)


def make_window_counts():
    rows = []
    configurations = [
        ("project_default_native_aditya_pair", "SoLEXS+HEL1OS", 60, 60),
        ("project_default_native_aditya_pair_plus_goes_reference", "SoLEXS+HEL1OS+GOES-18", 60, 60),
    ]
    for config_id, streams, sequence_length, cadence in configurations:
        for hours in (1, 3, 6, 12, 24):
            rows.append({
                "configuration_id": config_id, "required_streams": streams,
                "lookback_samples_project_default": sequence_length,
                "cadence_seconds_project_default": cadence,
                "lookback_duration_first_to_last": f"{(sequence_length - 1) * cadence} seconds",
                "forecast_horizon_hours": hours,
                "target_definition": f"verified M/X event start in (t, t+{hours}h]",
                "min_input_valid_fraction": None, "max_allowed_gap": None,
                "candidate_windows": None, "verified_valid_windows": 0,
                "positive_windows": None, "negative_windows": None,
                "unknown_or_excluded_windows": None,
                "status": "NOT_COMPUTABLE_UNVERIFIED_REQUIRED_ADITYA_STREAMS",
                "reason": "No GTI-applied, documented-quality SoLEXS/HEL1OS streams; no scientifically justified valid-fraction/gap policy. Negative labels also require complete event-report and target-horizon coverage.",
            })
    return pd.DataFrame(rows)


def _source_entry(source_id, name, path, product, coverage_start, coverage_end, status,
                  root, official_url=None, relevant_metadata=None, uncertainties=None):
    path = Path(path) if path else None
    if path and path.is_file():
        try: relative = str(path.relative_to(root))
        except ValueError: relative = str(path)
        checksum = sha256_path(path)
        size = path.stat().st_size
    else:
        relative, checksum, size = None, None, None
    return {
        "source_id": source_id, "dataset_or_document": name, "product_version": product,
        "official_source_url": official_url, "retrieval_date_utc": datetime.now(timezone.utc).date().isoformat() if path and path.is_file() else None,
        "path_relative_to_data_root_when_possible": relative, "sha256": checksum, "size_bytes": size,
        "coverage_start_utc": coverage_start, "coverage_end_utc": coverage_end,
        "relevant_metadata": relevant_metadata or [], "verification_status": status,
        "outstanding_uncertainties": uncertainties or [],
    }


def _register(downloads_root, data_root, report_root, phase3a, official_metadata):
    docs = data_root / "source_docs"
    report_json = report_root / "phase3a_data_audit.json"
    events_path = downloads_root / "data/goes/goes_flares.csv"
    xrs_csv = downloads_root / "data/goes/goes_xrs_1min.csv"
    nc_dir = downloads_root / "data/goes/raw"
    sx_path = downloads_root / "solexs_2026Oct23T184402685.zip"
    hl_path = downloads_root / "hel1os_2026Oct23T184518914.zip"
    gen_path = downloads_root / "download_goes.py"
    official_csv = data_root / "goes_official/sci_xrsf-l2-flrpt_geo_y2026_v1-0-1.csv"
    official_meta = data_root / "goes_official/sci_xrsf-l2-flrpt_geo_metadata.json"
    entries = [
        _source_entry("official_noaa_goes_2026", "NOAA/NCEI GOES XRS L2 science-quality flare report", official_csv,
                      official_metadata["global_attributes"]["flrpt_algorithm_version"],
                      "2026-01-01T00:00:00Z", "2026-10-22T23:59:59Z", "AUTHORITATIVE_OFFICIAL_REPORT_RECORDS",
                      data_root, f"{NOAA_BASE}/sci_xrsf-l2-flrpt_geo_y2026_v1-0-1.csv",
                      [str(official_meta), "companion metadata JSON"], ["NOAA docs do not claim complete flare population beyond report scope; negative-label completeness is not established"]),
        _source_entry("official_noaa_goes_metadata_2026", "NOAA/NCEI GOES XRS L2 flare report metadata", official_meta,
                      "1-0-1", official_metadata["global_attributes"].get("time_coverage_start"),
                      official_metadata["global_attributes"].get("time_coverage_end"), "AUTHORITATIVE_METADATA",
                      data_root, f"{NOAA_BASE}/sci_xrsf-l2-flrpt_geo_metadata.json"),
        _source_entry("noaa_flare_readme", "GOES Flare Report ReadMe", docs / "GOES_Flare_Report_ReadMe.pdf",
                      "dated 2026-06-10; report release 1-0-1 dated 2026-06-24", None, None, "OFFICIAL_DOCUMENT_RETRIEVED",
                      data_root, f"{NOAA_DOCS}/GOES_Flare_Report_ReadMe.pdf"),
        _source_entry("noaa_xrs_readme", "GOES-R XRS L2 Data ReadMe", docs / "GOES-R_XRS_L2_Data_ReadMe.pdf",
                      "dated 2026-10-06; product 2-2-2", None, None, "OFFICIAL_DOCUMENT_RETRIEVED",
                      data_root, f"{NOAA_DOCS}/GOES-R_XRS_L2_Data_ReadMe.pdf"),
        _source_entry("noaa_xrs_users_guide", "GOES-R XRS L2 Data Users Guide", docs / "GOES-R_XRS_L2_Data_Users_Guide.pdf",
                      "dated 2026-10-06", None, None, "OFFICIAL_DOCUMENT_RETRIEVED",
                      data_root, f"{NOAA_DOCS}/GOES-R_XRS_L2_Data_Users_Guide.pdf"),
        _source_entry("pradan_alpps_guide", "Aditya-L1 ALPPS User Guide", docs / "ALPPS_UserGuide_V1.0.pdf",
                      "V1.0; effective 2026-01-01", None, None, "GENERAL_PORTAL_GUIDE_ONLY_INSTRUMENT_PRODUCT_MANUALS_UNAVAILABLE",
                      data_root, "https://alpps.issdc.gov.in/web/extensions/Aditya-l1/resources/docs/ALPPS_UserGuide_V1.0.pdf",
                      uncertainties=["Guide covers portal/general process, not detailed SoLEXS/HEL1OS L1 FITS semantics"]),
        _source_entry("local_hek_flare_csv", "Local SunPy/HEK-derived event CSV", events_path, None,
                      phase3a["goes"]["coverage_start_utc"], phase3a["goes"]["coverage_end_utc"], "DERIVED_LOCAL_UNVERIFIED_SOURCE_LABELS_RECONCILED",
                      data_root, relevant_metadata=[str(gen_path), "download_goes.py query filters and dedupe rule"]),
        _source_entry("local_goes_measurement_csv", "Local GOES XRS 1-minute CSV", xrs_csv, None,
                      phase3a["goes"]["coverage_start_utc"], phase3a["goes"]["coverage_end_utc"], "VALUES_AND_TIMESTAMPS_MATCH_RAW_NETCDF",
                      data_root, relevant_metadata=[str(p) for p in sorted(nc_dir.glob("*.nc"))]),
        _source_entry("local_solexs_l1_archive", "Local SoLEXS Level-1 nested archive", sx_path, "archive-version-as-named",
                      phase3a["solexs"]["products"][0]["raw_start_utc"], phase3a["solexs"]["products"][-1]["raw_end_utc"],
                      "STRUCTURE_OBSERVED_PRODUCT_SEMANTICS_UNVERIFIED", data_root,
                      uncertainties=["TIME reference/epoch, COUNTS unit, GTI endpoint semantics, detector/calibration and flags"]),
        _source_entry("local_hel1os_l1_archive", "Local HEL1OS Level-1 nested archive", hl_path, "multiple V111/V211 products",
                      min(x["raw_start_utc"] for x in phase3a["hel1os"]["products"]), max(x["raw_end_utc"] for x in phase3a["hel1os"]["products"]),
                      "STRUCTURE_OBSERVED_PRODUCT_SEMANTICS_AND_VERSION_POLICY_UNVERIFIED", data_root,
                      uncertainties=["MJD scale/authoritative time field, GTI, flags/deadtime/calibration, duplicate version precedence"]),
    ]
    for path in sorted(nc_dir.glob("*.nc")):
        entries.append(_source_entry("local_goes_netcdf_" + path.stem, "GOES-18 XRS L2 1-minute NetCDF", path,
                                     path.stem.rsplit("_v", 1)[-1], None, None,
                                     "SCIENCE_QUALITY_NETCDF_METADATA_AND_FLAGS_AUDITED", data_root,
                                     "https://www.ncei.noaa.gov/products/goes-r-extreme-ultraviolet-xray-irradiance"))
    for source_id, name, url in (
        ("solexs_l1_manual_unavailable", "SoLEXS L1 product manual / SIS", PRADAN,
         ),
        ("hel1os_l1_manual_unavailable", "HEL1OS L1 product manual / SIS", PRADAN,
         ),
    ):
        entries.append({"source_id": source_id, "dataset_or_document": name, "product_version": None,
                        "official_source_url": url, "retrieval_date_utc": None, "path_relative_to_data_root_when_possible": None,
                        "sha256": None, "size_bytes": None, "coverage_start_utc": None, "coverage_end_utc": None,
                        "relevant_metadata": [], "verification_status": "BLOCKED_REQUIRES_REGISTERED_PRADAN_OTHER_DOWNLOADS",
                        "outstanding_uncertainties": ["No public instrument-specific L1 manual was available unauthenticated; do not infer semantics"]})
    return {"generator": GENERATOR, "configured_data_root": str(data_root),
            "retrieval_date_utc": datetime.now(timezone.utc).isoformat(), "sources": entries,
            "documented_facts": {
                "noaa_report": ["one event per reported flare", "report updated daily", "flare_id is an onset-derived event key", "CSV times are UTC minute-resolution fields", "v1.0.1 revision includes added SWPC locations/active regions and sequential flares >2"],
                "goes_xrs": ["XRS-A 0.05-0.4 nm", "XRS-B 0.1-0.8 nm", "science-quality data are in W/m2", "L2 avg1m flag meanings come from official XRS L2 guide"],
                "aditya_public_portal": ["public ALPPS guide says SoLEXS/HEL1OS science products are distributed via PRADAN; it is a general access guide, not the detailed product manual"],
            },
            "observed_facts": ["archive/member structures and table columns recorded in Phase3A JSON", "event and sample values reported from identified local files"],
            "assumptions": [],
            "manual_retrieval_steps": ["Sign in to the Aditya-L1 PRADAN portal using your own existing account.", "Open Other Downloads and locate SoLEXS and HEL1OS product/user manuals and Level-1 data product SIS/archive documentation.", "Download manual, FITS schema, calibration, quality and GTI documents without placing credentials/cookies in the repository.", "Save unmodified documents under E:\\SolarFlareData\\source_docs\\isro\\ and record title, revision, portal URL/item ID and SHA-256."],
            "report_root": str(report_root)}


def build_report(downloads_root, data_root, repo_root, report_root):
    phase3a = json.loads((report_root / "phase3a_data_audit.json").read_text(encoding="utf-8"))
    official_meta = json.loads((data_root / "goes_official/sci_xrsf-l2-flrpt_geo_metadata.json").read_text(encoding="utf-8"))
    local = pd.read_csv(downloads_root / "data/goes/goes_flares.csv")
    official = pd.read_csv(data_root / "goes_official/sci_xrsf-l2-flrpt_geo_y2026_v1-0-1.csv")
    start = pd.Timestamp(phase3a["goes"]["coverage_start_utc"])
    end = pd.Timestamp(phase3a["goes"]["coverage_end_utc"]) + pd.Timedelta(minutes=1)
    reconciliation, event_audit, local_target, official_target = reconcile_events(local, official, start, end)
    verified, excluded = verified_event_table(official_target)
    event_audit.update({
        "target_coverage_start_utc": start.isoformat(), "target_coverage_end_exclusive_utc": end.isoformat(),
        "official_product_id": "sci_xrsf-l2-flrpt_geo_y2026_v1-0-1",
        "official_product_algorithm_version": official_meta["global_attributes"]["flrpt_algorithm_version"],
        "verified_official_event_rows": len(verified),
        "verified_official_MX_events": int(verified.is_mx.sum()) if len(verified) else 0,
        "verified_official_class_counts": verified.class_family.value_counts().sort_index().to_dict() if len(verified) else {},
        "verified_event_exclusion_rows": len(excluded),
        "negative_label_completeness_established": False,
        "negative_label_reason": "NOAA documentation defines a science-quality composite event report and one entry per reported flare, but does not establish a completeness guarantee for all possible events/measurement gaps in this interval.",
    })

    raw_files = {}
    for item in phase3a["goes"]["netcdf_files"]:
        date = item["start_utc"][:10]
        for channel in ("xrsa", "xrsb"):
            raw_files[(date, channel)] = {"filename": item["filename"], "sha256": item["sha256"]}
    quality_days, comparisons = _daily_goes_quality(
        report_root / "phase3a_goes_record_quality.csv",
        downloads_root / "data/goes/goes_xrs_1min.csv", raw_files)
    archive_semantics = audit_aditya_archive_semantics(downloads_root)
    coverage = build_coverage(phase3a, verified, quality_days, archive_semantics)
    windows = make_window_counts()

    root_reg = _register(downloads_root, data_root, report_root, phase3a, official_meta)
    source_hashes = {x["source_id"]: x["sha256"] for x in root_reg["sources"] if x["sha256"]}
    readiness = {
        "status": "BLOCKED_INSTRUMENT_SEMANTICS",
        "training_started": False, "model_tuning_or_evaluation_started": False,
        "final_split_boundaries_defined": False,
        "reason": "Registered SoLEXS and HEL1OS L1 product manuals are unavailable in public unauthenticated sources; count units, time references, GTI/quality/calibration and HEL1OS version precedence remain unverified. No verified joined input windows can be counted.",
        "verified_label_rows": len(verified), "verified_MX_event_rows": int(verified.is_mx.sum()) if len(verified) else 0,
        "verified_valid_windows": 0, "negative_window_count": None,
        "distinct_event_threshold_proposal": "For a future exploratory split only, first test the repository's existing >=5 distinct M/X events per partition as a floor; it is not a universal adequacy claim. A defensible evaluation needs a predeclared larger event-powered rationale and longer coverage.",
        "project_default_window_context": {"sequence_length_samples": 60, "cadence_seconds": 60,
                                           "forecast_horizons_hours": [1, 3, 6, 12, 24],
                                           "source": "backend/ml/features/windowing.py WindowConfig; feature cadence in backend/ml/features/engineering.py FeatureConfig"},
        "quality_policy": {"GOES": "finite, non-fill and flag==0; exclude interpolated flag records; no fill/forward-fill/interpolation", "SoLEXS": "not applied pending manual", "HEL1OS": "not applied pending manual"},
        "negative_labels_allowed": False,
        "chronological_split_safeguards": "Existing causal trailing rolling features, chronological partitions, horizon-tail purging, train-only scaler fitting and no shuffling were inspected and preserved; no boundaries selected.",
    }
    summary = {
        "generator": GENERATOR, "audit_time_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(data_root), "repo_root": str(repo_root),
        "phase3a_input_json_sha256": sha256_path(report_root / "phase3a_data_audit.json"),
        "read_only_for_raw_sources": True, "raw_sources_modified": False,
        "event_reconciliation": event_audit,
        "verified_official_events_by_day_utc": verified.assign(day=verified.event_start_utc.str[:10]).groupby("day").agg(
            rows=("flare_id", "size"), mx_events=("is_mx", "sum")).reset_index().to_dict("records") if len(verified) else [],
        "goes_measurement_quality": {
            "measurement_csv_matches_raw_values": bool(all(x["values_equal_within_rtol_1e-7"] for x in comparisons)),
            "max_value_mismatch_rows": int(sum(x["value_mismatch_count"] for x in comparisons)),
            "candidate_valid_rows_per_channel": {c: int(sum(x["candidate_valid_rows"] for x in quality_days if x["channel"] == c)) for c in ("xrsa", "xrsb")},
            "csv_comparison_rule": "exact timestamp join; pairwise relative tolerance 1e-7 for NetCDF float32/CSV decimal representation; fill values never replaced",
        },
        "instrument_semantics": {"solexs": "UNVERIFIED; GTI numeric endpoints observed but units/closure/reference undocumented", "hel1os": "UNVERIFIED; MJD/ISOT structurally inspectable but authoritative time field/scale and quality/GTI rules undocumented", "processing_versions": "V111/V211 overlaps retained; precedence unresolved"},
        "aditya_archive_structural_checks": archive_semantics,
        "solexs_candidate_rows": phase3a["solexs"]["candidate_rows"],
        "official_events_missing_end_time": int(event_audit.get("official_missing_end_time_rows", 0)),
        "window_count_status": "WITHHELD; no input stream meets the verification gate", "readiness": readiness,
        "source_checksums": source_hashes,
        "reproducibility_command": "python scripts/phase3b_authoritative_audit.py --downloads-root C:\\Users\\ASHUTOSH\\Downloads --data-root E:\\SolarFlareData --repo-root C:\\Users\\ASHUTOSH\\Downloads\\solar-ai-flare-main\\solar-ai-flare-main",
    }
    return root_reg, reconciliation, verified, {"generator": GENERATOR, "official_metadata_sha256": source_hashes.get("official_noaa_goes_metadata_2026"),
        "quality_policy": "candidate valid iff finite non-fill AND status flag==0; interpolated samples excluded", "quality_reasoning": "NOAA XRS L2 Users Guide Table 3 defines good_data, eclipse, bad_data and interpolated over 1-2 minute gaps; conservative predictive-use audit retains only flag-zero rows.",
        "daily_channel_counts": quality_days, "csv_netcdf_matched_sample_comparison": comparisons,
        "all_compared_values_match_within_rtol_1e-7": summary["goes_measurement_quality"]["measurement_csv_matches_raw_values"],
        "invalid_values_not_replaced": True}, coverage, windows, readiness, summary


def _markdown(readiness, summary, reg, reconciliation, quality, coverage, windows):
    ev = summary["event_reconciliation"]
    mx_by_day = summary["verified_official_events_by_day_utc"]
    days_text = ", ".join(f"{d['day']}: {d['mx_events']}" for d in mx_by_day)
    return "\n".join([
        f"<!-- {GENERATOR} -->", "# Phase 3B — Authoritative data reconciliation and verified-window audit", "",
        f"- Audit time UTC: {summary['audit_time_utc']}", f"- Readiness: **{readiness['status']}**",
        "- Training, tuning, and evaluation started: **No**", "- Raw source files changed: **No**", "",
        "## Event reconciliation", "",
        f"The current NOAA/NCEI 2026 report v{ev['official_product_algorithm_version']} contributes {ev['official_rows_in_coverage']} events in the GOES measurement interval; the local HEK-derived file has {ev['local_rows_in_coverage']}. Under the declared ±60-second onset candidate rule, {ev['unique_mutual_start_matches']} are unique mutual matches, {ev['same_family_magnitude_mismatch_rows']} have same-family magnitude differences, {ev['unmatched_official_rows']} are NOAA-only, {ev['unmatched_local_rows']} are local-only, and {ev['ambiguous_local_rows']} local rows are ambiguous. No ambiguous event was selected.",
        f"The verified NOAA report rows include {summary['event_reconciliation']['verified_official_MX_events']} M/X events. UTC-day M/X counts: {days_text}.",
        "The reconciliation CSV preserves original local and NOAA times/classes and stable NOAA flare IDs. Local class strings are not overwritten; NOAA class is used as the reported label. Eight matched class strings differ in magnitude but retain the same C/M/X family. The official-only records are retained as official events.",
        "NOAA documents the report as a daily-updated, science-quality composite with one record per reported flare and a minute-resolution time axis. The documentation does not promise an exhaustive population guarantee for every possible flare or measurement gap; the report therefore does not authorize negative labels by itself.", "",
        "## GOES XRS quality", "",
        f"The local measurement CSV matches the six raw NetCDFs on timestamps and values within relative tolerance 1e-7: {summary['goes_measurement_quality']['measurement_csv_matches_raw_values']}; value mismatches: {summary['goes_measurement_quality']['max_value_mismatch_rows']}. Daily quality counts and per-day/channel comparisons are in the GOES quality JSON. Fill values remain missing and are not replaced; nonzero status flags, including interpolated flags, are excluded from candidate-valid samples.",
        "The NOAA L2 guide says to use electron-corrected irradiance for most purposes; the audited `xrsa_flux`/`xrsb_flux` variables are the documented corrected W/m² series. Keep the GOES event report separate from predictors.", "",
        "## Aditya instrument gate", "",
        "The public ALPPS guide is a general portal guide, not a SoLEXS/HEL1OS L1 product manual. No unauthenticated registered manuals were available in the public source pages. The audit retains structural measurements only; no SoLEXS GTI or HEL1OS GTI/version policy was applied. The numeric SoLEXS GTI START/STOP columns have no declared units in the FITS tables. HEL1OS `MJD` and `ISOT` are both present, but their authoritative time scale/field and validity rules still need the registered SIS/manual. Count/rate units, detector calibration, dead-time, saturation, and quality interpretation remain unresolved.",
        "Manual retrieval: sign into PRADAN with your existing account, open **Other Downloads**, and retrieve the SoLEXS and HEL1OS L1 product/user manuals, SIS, GTI/quality and calibration references. Save unmodified documents under `E:\\SolarFlareData\\source_docs\\isro\\` and record portal item, revision, retrieval time, and SHA-256. No credentials or cookies should be copied into project files.",
        "HEL1OS V111/V211 overlapping products remain separate; no filename-based version was preferred. No native product parser changes were made because the documentation gate is not met.", "",
        "## Coverage and windows", "",
        f"The instrument coverage CSV contains {len(coverage)} rows including raw/candidate product coverage and daily GOES channel quality. Structurally, SoLEXS has six 86,400-row daily SDD2 products with 1-second timestamp spacing; only {summary.get('solexs_candidate_rows', 'see Phase 3A JSON')} are finite/nonnegative before GTI application. HEL1OS products cover only partial days and include 44 light curves and duplicate-version overlaps. These are not verified scientific exposure intervals.",
        f"Structural archive checks recorded {summary['aditya_archive_structural_checks']['solexs_gti_table_count']} SoLEXS GTI tables and {summary['aditya_archive_structural_checks']['hel1os_gti_table_count']} HEL1OS GTI tables. HEL1OS MJD/ISOT compared {summary['aditya_archive_structural_checks']['hel1os_mjd_isot_total_rows_compared']} rows under provisional UTC; these checks establish representation consistency only, not mission-authorized time semantics.",
        "The project window contract was read from the existing code: sequence length 60 at the project’s 60-second feature cadence, and 1/3/6/12/24-hour forecast horizons. Valid window and positive/negative counts are withheld because neither Aditya stream has documented timestamp/GTI/quality semantics, the required stream alignment is unresolved, and minimum-valid-fraction/gap criteria have not been scientifically justified. Negative windows are never inferred from absent events or missing samples.",
        "No split boundaries were chosen. Existing trailing causal rolling features, chronological splits, per-horizon boundary purging, and train-only scaling were inspected and left unchanged. Six days are insufficient evidence of generalization.", "",
        "## Verdict and next action", "", f"**{readiness['status']}** — retrieve the registered SoLEXS/HEL1OS manuals and validate MJD/ISOT agreement, GTI endpoints/time reference, field units, quality rules, calibration and version precedence. Then recalculate verified coverage and window counts. The NOAA report reconciles the event source but does not make the instrument streams usable.",
        "", "## Reproducibility", "", f"`{summary['reproducibility_command']}`", "",
        "Outputs include the document register, event reconciliation, verified official event table, GOES quality audit, instrument coverage, and horizon/window gate counts. Checksums are in the register/summary JSON.", "",
    ])


def _recognized(path, kind):
    try:
        if kind == "json": return json.loads(path.read_text(encoding="utf-8")).get("generator") == GENERATOR
        if kind == "markdown": return path.read_text(encoding="utf-8").startswith(f"<!-- {GENERATOR} -->")
        with path.open("r", encoding="utf-8", newline="") as stream: return "audit_generator" in next(csv.reader(stream))
    except Exception:
        return False


def _write(path, content, kind):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not _recognized(path, kind):
        raise FileExistsError(f"Refusing to overwrite an unrecognized existing file: {path}")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def write_reports(outputs, report_root):
    reg, reconciliation, verified, quality, coverage, windows, readiness, summary = outputs
    reconciliation = reconciliation.copy(); reconciliation.insert(0, "audit_generator", GENERATOR)
    verified = verified.copy(); verified.insert(0, "audit_generator", GENERATOR)
    coverage = coverage.copy(); coverage.insert(0, "audit_generator", GENERATOR)
    windows = windows.copy(); windows.insert(0, "audit_generator", GENERATOR)
    files = [
        ("phase3b_document_register.json", json.dumps(reg, indent=2, sort_keys=True, default=str) + "\n", "json"),
        ("phase3b_event_reconciliation.csv", reconciliation.to_csv(index=False), "csv"),
        ("phase3b_verified_labels.csv", verified.to_csv(index=False), "csv"),
        ("phase3b_goes_quality_audit.json", json.dumps(quality, indent=2, sort_keys=True, default=str) + "\n", "json"),
        ("phase3b_instrument_coverage.csv", coverage.to_csv(index=False), "csv"),
        ("phase3b_valid_window_counts.csv", windows.to_csv(index=False), "csv"),
        ("phase3b_scientific_readiness.md", _markdown(readiness, summary, reg, reconciliation, quality, coverage, windows), "markdown"),
        ("phase3b_audit_summary.json", json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", "json"),
    ]
    for name, text, kind in files: _write(report_root / name, text, kind)
    return [str(report_root / name) for name, _, _ in files]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads-root", type=Path, default=Path(r"C:\Users\ASHUTOSH\Downloads"))
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\SolarFlareData"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report-root", type=Path)
    args = parser.parse_args()
    report_root = args.report_root or args.data_root / "reports"
    outputs = build_report(args.downloads_root, args.data_root, args.repo_root, report_root)
    paths = write_reports(outputs, report_root)
    print("Readiness:", outputs[6]["status"])
    print("Event rows local/official/verified:", outputs[7]["event_reconciliation"]["local_rows_in_coverage"],
          outputs[7]["event_reconciliation"]["official_rows_in_coverage"], outputs[6]["verified_label_rows"])
    print("Unique matches / official-only / ambiguous:", outputs[7]["event_reconciliation"]["unique_mutual_start_matches"],
          outputs[7]["event_reconciliation"]["unmatched_official_rows"], outputs[7]["event_reconciliation"]["ambiguous_local_rows"])
    for path in paths: print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
