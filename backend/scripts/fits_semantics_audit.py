"""Read-only structural audit for Aditya-L1 FITS products.

Audit is the default and never filters observations. The optional filtering API
fails closed unless explicit verified rules are supplied; it never infers them
from FITS headers or candidate column names.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import re
import zipfile
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from astropy.io import fits


TIME_KEYS = ("TIMEZERO", "TIMEUNIT", "MJDREF", "MJDREFI", "MJDREFF", "TIMESYS", "TIMEPIXR")
IDENTITY_KEYS = ("TELESCOP", "INSTRUME", "DETECTOR", "DETNAM", "OBS_ID", "DATE-OBS", "DATE-END",
                 "TSTART", "TSTOP", "CREATOR", "DATE", "CALDBVER", "PROCVER", "VERSION",
                 "HDUVERS", "HDUVERS1", "TIMEDEL", "STARETIM", "TIMEBIN", "ELOW", "EHIGH")
SCIENCE_NAMES = {"EVENTS_CDTE", "EVENTS_CZT", "LIGHTCURVE", "PHA_SPECTRA", "RATE", "SPECTRUM"}


def _safe(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    return str(value)


def _source_bytes(path: Path, archive_member: str | None = None) -> bytes:
    if archive_member is not None:
        parts = archive_member.split("!/")
        payload = None
        for index, part in enumerate(parts):
            source = path if index == 0 else io.BytesIO(payload)
            with zipfile.ZipFile(source) as zf:
                payload = zf.read(part)
        return payload
    raw = path.read_bytes()
    if path.name.lower().endswith(".fits.gz") or path.name.lower().endswith(".fit.gz"):
        return gzip.decompress(raw)
    return raw


def _sha256(path: Path, member: str | None = None) -> str:
    h = hashlib.sha256()
    if member:
        h.update(_source_bytes(path, member))
    else:
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


def _header_snapshot(header: fits.Header) -> dict[str, Any]:
    keys = set(TIME_KEYS + IDENTITY_KEYS + ("BUNIT", "TELESCOP", "ORIGIN", "OBJECT"))
    return {key: _safe(header[key]) for key in sorted(keys) if key in header}


def _col_info(hdu: fits.BinTableHDU | fits.TableHDU) -> list[dict[str, Any]]:
    return [{"name": col.name, "format": col.format, "unit": col.unit,
             "dim": col.dim, "null": _safe(col.null)} for col in hdu.columns]


def _gti_validation(data: Any, columns: list[str]) -> dict[str, Any]:
    upper = {c.upper(): c for c in columns}
    start_name=upper.get("START",upper.get("TSTART")); stop_name=upper.get("STOP",upper.get("TSTOP"))
    if start_name is None or stop_name is None:
        missing=[label for label,found in (("START/TSTART",start_name),("STOP/TSTOP",stop_name)) if found is None]
        return {"interval_count": int(len(data)), "missing_columns": missing,
                "malformed_rows": None, "reversed_rows": None, "nonfinite_rows": None,
                "duplicate_intervals": None, "overlapping_pairs": None, "sorted_by_start": None,
                "semantics": "STILL_UNVERIFIED; no inclusion mask applied"}
    starts = np.asarray(data[start_name], dtype=float)
    stops = np.asarray(data[stop_name], dtype=float)
    finite = np.isfinite(starts) & np.isfinite(stops)
    reversed_rows = finite & (stops < starts)
    pairs = sorted((float(a), float(b)) for a, b, ok in zip(starts, stops, finite) if ok and b >= a)
    duplicates = sum(n - 1 for n in Counter(pairs).values() if n > 1)
    overlaps = sum(1 for i, (a, b) in enumerate(pairs) for c, d in pairs[i + 1:] if c < b)
    original_valid = [(float(a), float(b)) for a, b, ok in zip(starts, stops, finite) if ok and b >= a]
    return {"interval_count": int(len(data)), "missing_columns": [],
            "malformed_rows": int((~finite | reversed_rows).sum()),
            "reversed_rows": int(reversed_rows.sum()), "nonfinite_rows": int((~finite).sum()),
            "duplicate_intervals": int(duplicates), "overlapping_pairs": int(overlaps),
            "sorted_by_start": original_valid == sorted(original_valid),
            "semantics": "STILL_UNVERIFIED; validation only, no endpoint convention or sample filter applied"}


def _candidate_columns(data: Any, columns: list[str]) -> list[dict[str, Any]]:
    result = []
    for name in columns:
        upper = name.upper()
        is_time = upper in {"TIME", "TIME_MJD", "MJD", "ISOT"} or upper.endswith("_TIME")
        is_quality = (upper in {"QUALITY", "STATUS"} or upper.endswith(("_QUALITY", "_STATUS", "_FLAG", "STAT"))
                      or "FLAG" in upper or ("STAT" in upper and upper != "STAT_ERR") or "SATCTR" in upper or "PILECTR" in upper)
        if not is_time and not is_quality:
            continue
        arr = np.asarray(data[name])
        values, counts = np.unique(arr, return_counts=True) if is_quality and arr.size and arr.dtype.kind in "biufUS" else ([], [])
        nulls = int(np.asarray(data[name]).dtype.kind == "f" and np.isnan(np.asarray(data[name], dtype=float)).sum()) if arr.size else 0
        result.append({"name": name, "candidate_kind": "time" if is_time else "quality_status", "dtype": str(arr.dtype), "count": int(arr.size), "null_count": nulls,
                       "value_counts": [{"value": _safe(v), "count": int(n)} for v, n in zip(values[:256], counts[:256])],
                       "unique_truncated": len(values) > 256,
                       "interpretation": "CANDIDATE_ONLY; no scientific semantics inferred"})
    return result


def _is_science_extension(name: str) -> bool:
    upper=name.upper()
    # Generic names are supported alongside the actual HEL1OS Level-1 names
    # present in this dataset (manual §2.1 and Appendix B, pp.5, 46-52).
    return (upper in SCIENCE_NAMES or "_LC_BAND_" in upper
            or upper in {"CDTE1-EVENTS","CDTE2-EVENTS","CZT1-EVENTS","CZT2-EVENTS"})


def _instrument_product(path: str, hdus: list[dict[str, Any]]) -> tuple[str | None, str | None, str]:
    upper=path.upper()
    header_values=[str(h.get("header_metadata",{}).get("INSTRUME","" )).upper() for h in hdus]
    if "HEL1OS" in upper or any("HEL1OS" in v for v in header_values): instrument="HEL1OS"
    elif "SOLEXS" in upper or "SLX" in upper or any("SOLEXS" in v for v in header_values): instrument="SoLEXS"
    else: instrument=None
    extnames=[str(h.get("extension_name","" )).upper() for h in hdus]
    if any("GTI" in n for n in extnames) or "/GTI" in upper or ".GTI.GZ" in upper: product="GTI"
    elif any(n=="HLSHK" for n in extnames) or "/HK.FITS" in upper: product="HOUSEKEEPING"
    elif any("_LC_BAND_" in n for n in extnames) or "LIGHTCURVE" in upper or ".LC.GZ" in upper or any(n=="RATE" for n in extnames): product="LIGHTCURVE"
    elif any("EVENTS" in n for n in extnames) or "EVT.FITS" in upper: product="EVENTS"
    elif any(n=="SPECTRUM" for n in extnames) or "/SPECTRA/" in upper: product="SPECTRA"
    else: product="OTHER_OR_UNRESOLVED"
    basis="archive filename/member path and/or FITS INSTRUME/EXTNAME/DETNAM metadata; association only"
    return instrument,product,basis


def _product_version(path: str) -> str | None:
    for pattern in (r"_LEV1_(V\d+)",r"_V(\d+\.\d+)(?:\.ZIP)?",r"_V(\d+\.\d+)"):
        match=re.search(pattern,path,re.I)
        if match:return match.group(1)
    return None


def process_aditya_l1_level1(fits_filepath: str | Path, *, mode: str = "audit",
                             verified_rules: dict[str, Any] | None = None,
                             diagnostic_override: str | None = None) -> dict[str, Any]:
    """Inspect one FITS product, retaining the original script's requested stages.

    Returns primary metadata, science extensions, GTIs, candidate masks,
    rejection accounting, and unresolved conditions. Audit mode never filters.
    Filtering requires explicit matrix-backed rules, except a loudly named
    diagnostic override which is never science-ready.
    """
    path = Path(fits_filepath)
    if mode not in {"audit", "filtering"}:
        raise ValueError("mode must be 'audit' or 'filtering'")
    if mode == "filtering":
        if diagnostic_override == "UNVERIFIED_DIAGNOSTIC_ONLY":
            raise RuntimeError("Diagnostic filtering is intentionally not implemented: provisional masks could be mistaken for scientific filtering.")
        if not verified_rules:
            raise RuntimeError("Filtering refused: verified instrument/product rules are absent; audit mode remains available.")
        raise RuntimeError("Filtering refused: this utility is audit-only. Apply filters only through a separately reviewed implementation after the matrix rules are authoritative and verified.")
    unresolved = []
    source_name = str(path)
    try:
        payload = path.read_bytes()
        if path.name.lower().endswith(".gz"):
            payload = gzip.decompress(payload)
        with fits.open(io.BytesIO(payload), mode="readonly", memmap=False, checksum=True) as hdul:
            primary = _header_snapshot(hdul[0].header)
            hdus, gtis, science = [], [], []
            for index, hdu in enumerate(hdul):
                name = str(hdu.header.get("EXTNAME", "PRIMARY" if index == 0 else "")).strip()
                data = hdu.data
                cols = _col_info(hdu) if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)) else []
                names = [c["name"] for c in cols]
                row_count = int(len(data)) if data is not None and cols else (0 if data is None else None)
                headers = _header_snapshot(hdu.header)
                candidates = _candidate_columns(data, names) if data is not None and cols else []
                record = {"hdu_index": index, "hdu_type": type(hdu).__name__, "extension_name": name,
                          "row_count": row_count, "columns": cols, "header_metadata": headers,
                          "candidate_columns": candidates}
                hdus.append(record)
                if _is_science_extension(name):
                    science.append(record)
                is_gti = "GTI" in name.upper() or ("START" in {x.upper() for x in names} and "STOP" in {x.upper() for x in names})
                if is_gti:
                    validation = _gti_validation(data, names) if data is not None and cols else {
                        "interval_count": 0, "missing_columns": ["START", "STOP"], "malformed_rows": None,
                        "reversed_rows": None, "nonfinite_rows": None, "duplicate_intervals": None,
                        "overlapping_pairs": None, "sorted_by_start": None,
                        "semantics": "STILL_UNVERIFIED; no inclusion mask applied"}
                    gtis.append({**record, **validation})
            if not gtis:
                unresolved.append("No GTI-like extension/START-STOP table found; absence does not imply all samples are valid.")
            if not science:
                unresolved.append("No recognized science extension name found; inspect HDU inventory without schema assumptions.")
            if not any(any(k in h.get("header_metadata", {}) for k in TIME_KEYS) for h in hdus):
                unresolved.append("No recognized time-reference keyword found in primary or extension headers.")
            if not any(c["name"].upper() in {"QUALITY", "STATUS"} for h in hdus for c in h["columns"]):
                unresolved.append("No QUALITY/STATUS column found; this is missing metadata, not confirmed clean data.")
            # Candidate masks are accounting placeholders only; no unverified
            # scientific predicate is applied in default audit mode.
            n_records = sum(h["row_count"] for h in science)
            return {"source_file": source_name, "file_size_bytes": path.stat().st_size, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "status": "AUDITED_STRUCTURALLY_NOT_SCIENCE_READY", "mode": mode,
                    "primary_metadata": primary, "hdus": hdus, "science_extensions": science,
                    "gti_extensions": gtis, "raw_record_count": n_records,
                    "candidate_masks": {"time": "NOT_APPLIED_UNVERIFIED_SEMANTICS", "gti": "NOT_APPLIED_UNVERIFIED_ENDPOINTS_OR_POLICY",
                                        "quality": "NOT_APPLIED_UNVERIFIED_FLAG_MEANINGS"},
                    "rejections": {"gti_only": None, "quality_only": None, "both_masks": None, "combined_unique": None,
                                   "final_retained": None, "accounting_status": "NOT_COMPUTED_AUDIT_MODE_MASKS_NOT_APPLIED"},
                    "unresolved_conditions": unresolved,
                    "diagnostic_override": diagnostic_override if mode == "filtering" else None}
    except Exception as exc:
        return {"source_file": source_name, "file_sha256": _sha256(path) if path.exists() else None,
                "status": "INSPECTION_FAILED", "mode": mode, "error_type": type(exc).__name__,
                "error": str(exc), "raw_record_count": None, "hdus": [], "gti_extensions": [],
                "rejections": None, "unresolved_conditions": ["File could not be inspected; no records were inferred."]}


def _iter_archive(path: Path, archive_bytes: bytes | None = None, prefix: str = ""):
    """Yield FITS members through nested ZIPs, never writing extracted files."""
    source = path if archive_bytes is None else io.BytesIO(archive_bytes)
    with zipfile.ZipFile(source) as zf:
        for info in sorted(zf.infolist(), key=lambda item: item.filename):
            if info.is_dir():
                continue
            chain = f"{prefix}!/{info.filename}" if prefix else info.filename
            lower = info.filename.lower()
            if lower.endswith((".fits", ".fit", ".fits.gz", ".fit.gz", ".lc.gz", ".gti.gz", ".pi.gz")):
                yield path, chain
            elif lower.endswith((".zip", ".tar", ".tgz", ".tar.gz")):
                # ZIP nesting is supported without extracting. TAR members are
                # inventoried elsewhere but not parsed by this FITS utility yet.
                if lower.endswith(".zip"):
                    nested = zf.read(info)
                    yield from _iter_archive(path, nested, chain)


def _iter_candidates(root: Path | str):
    root = Path(root)
    paths = sorted(root.rglob("*")) if root.is_dir() else [root]
    for path in paths:
        if not path.is_file():
            continue
        lower = path.name.lower()
        if lower.endswith((".fits", ".fit", ".fits.gz", ".fit.gz", ".lc.gz", ".gti.gz", ".pi.gz")):
            yield path, None
        elif lower.endswith(".zip"):
            try:
                yield from _iter_archive(path)
            except Exception:
                # Invalid archives will be recorded as failed inspection.
                yield path, "__ARCHIVE_OPEN_FAILED__"


def audit_root(data_root: str | Path, input_paths: list[str | Path] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Audit FITS files in directories and zip members without extraction."""
    inventory, hdus, gtis, quality = [], [], [], []
    candidates = (candidate for source in input_paths for candidate in _iter_candidates(Path(source))) if input_paths else _iter_candidates(Path(data_root))
    for path, member in candidates:
        try:
            if member == "__ARCHIVE_OPEN_FAILED__":
                raise zipfile.BadZipFile("Archive could not be opened or its central directory is corrupt.")
            raw = _source_bytes(path, member)
            if member:
                # Temporary in-memory source is deliberately not materialized on disk.
                result = _audit_payload(path, raw, f"{path}!{member}")
                digest = hashlib.sha256(raw).hexdigest()
                result["file_sha256"] = digest
            else:
                result = process_aditya_l1_level1(path)
            result["archive_member"] = member
        except Exception as exc:
            result = {"source_file": str(path), "archive_member": member, "status": "INSPECTION_FAILED",
                      "error_type": type(exc).__name__, "error": str(exc), "hdus": [], "gti_extensions": [],
                      "rejections": None, "unresolved_conditions": ["File could not be inspected; no records were inferred."]}
        instrument,product,basis=_instrument_product(result.get("source_file",str(path)),result.get("hdus",[]))
        result.update({"instrument":instrument,"product_association":product,"processing_version":_product_version(result.get("source_file",str(path))),"association_basis":basis,
                       "archive_status":"nested_archive_member_not_extracted" if member else ("original_archive" if path.suffix.lower()==".zip" else "direct_file"),
                       "extracted":False})
        inventory.append({k: result.get(k) for k in ("source_file","archive_member","file_size_bytes","file_sha256","status","instrument","product_association","processing_version","association_basis","archive_status","extracted","raw_record_count","error_type","error","primary_metadata","unresolved_conditions")})
        for hdu in result.get("hdus", []):
            hdus.append({"source_file": result["source_file"], "archive_member": member, **hdu})
            for candidate in hdu.get("candidate_columns", []):
                if candidate["candidate_kind"] == "quality_status":
                    quality.append({"source_file": result["source_file"], "archive_member": member,
                                    "extension_name": hdu["extension_name"], **candidate})
        for gti in result.get("gti_extensions", []):
            gtis.append({"source_file": result["source_file"], "archive_member": member,
                         "extension_name": gti["extension_name"], "interval_count": gti["interval_count"],
                         "missing_columns": gti["missing_columns"], "malformed_rows": gti["malformed_rows"],
                         "nonfinite_rows": gti["nonfinite_rows"], "reversed_rows": gti["reversed_rows"],
                         "duplicate_intervals": gti["duplicate_intervals"], "overlapping_pairs": gti["overlapping_pairs"],
                         "sorted_by_start": gti["sorted_by_start"], "semantics": gti["semantics"]})
    return {"file_inventory": inventory, "hdu_inventory": hdus, "gti_validation": gtis, "quality_distribution": quality}


def _audit_payload(path: Path, payload: bytes, display_name: str) -> dict[str, Any]:
    """Inspect an archive member while keeping its bytes in memory."""
    try:
        data = gzip.decompress(payload) if display_name.lower().endswith(".gz") else payload
        with fits.open(io.BytesIO(data), mode="readonly", memmap=False, checksum=True) as hdul:
            primary, hdus, gtis, science, unresolved = _parse_hdus(hdul)
        return {"source_file": display_name, "file_size_bytes":len(payload), "file_sha256": hashlib.sha256(payload).hexdigest(),
                "status": "AUDITED_STRUCTURALLY_NOT_SCIENCE_READY", "mode": "audit", "primary_metadata": primary,
                "hdus": hdus, "science_extensions": science, "gti_extensions": gtis,
                "raw_record_count": sum(h["row_count"] for h in science),
                "candidate_masks": {"time": "NOT_APPLIED_UNVERIFIED_SEMANTICS", "gti": "NOT_APPLIED_UNVERIFIED_ENDPOINTS_OR_POLICY", "quality": "NOT_APPLIED_UNVERIFIED_FLAG_MEANINGS"},
                "rejections": {"gti_only": None, "quality_only": None, "both_masks": None, "combined_unique": None, "final_retained": None,
                               "accounting_status": "NOT_COMPUTED_AUDIT_MODE_MASKS_NOT_APPLIED"},
                "unresolved_conditions": unresolved}
    except Exception as exc:
        return {"source_file": display_name, "file_sha256": hashlib.sha256(payload).hexdigest(), "status": "INSPECTION_FAILED",
                "error_type": type(exc).__name__, "error": str(exc), "hdus": [], "gti_extensions": [],
                "unresolved_conditions": ["File could not be inspected; no records were inferred."]}


def _parse_hdus(hdul):
    primary = _header_snapshot(hdul[0].header); hdus=[]; gtis=[]; science=[]; unresolved=[]
    for i,hdu in enumerate(hdul):
        name=str(hdu.header.get("EXTNAME", "PRIMARY" if i==0 else "")).strip(); d=hdu.data
        cols=_col_info(hdu) if isinstance(hdu,(fits.BinTableHDU,fits.TableHDU)) else []; names=[c["name"] for c in cols]
        n=int(len(d)) if d is not None and cols else (0 if d is None else None)
        rec={"hdu_index":i,"hdu_type":type(hdu).__name__,"extension_name":name,"row_count":n,"columns":cols,
             "header_metadata":_header_snapshot(hdu.header),"candidate_columns":_candidate_columns(d,names) if d is not None and cols else []}
        hdus.append(rec)
        if _is_science_extension(name): science.append(rec)
        if "GTI" in name.upper() or {"START","STOP"}.issubset({x.upper() for x in names}):
            gtis.append({**rec,**(_gti_validation(d,names) if d is not None and cols else {"interval_count":0,"missing_columns":["START","STOP"],"malformed_rows":None,"nonfinite_rows":None,"reversed_rows":None,"duplicate_intervals":None,"overlapping_pairs":None,"sorted_by_start":None,"semantics":"STILL_UNVERIFIED; no inclusion mask applied"})})
    if not gtis: unresolved.append("No GTI-like extension/START-STOP table found; absence does not imply all samples are valid.")
    if not science: unresolved.append("No recognized science extension name found; inspect HDU inventory without schema assumptions.")
    if not any(any(k in h["header_metadata"] for k in TIME_KEYS) for h in hdus): unresolved.append("No recognized time-reference keyword found in primary or extension headers.")
    if not any(c["name"].upper() in {"QUALITY","STATUS"} for h in hdus for c in h["columns"]): unresolved.append("No QUALITY/STATUS column found; this is missing metadata, not confirmed clean data.")
    return primary,hdus,gtis,science,unresolved


def write_reports(data_root: str | Path, report_root: str | Path, input_paths: list[str | Path] | None = None) -> dict[str, Path]:
    report_root=Path(report_root); report_root.mkdir(parents=True,exist_ok=True); result=audit_root(data_root,input_paths)
    result["audit_metadata"]={"audit_time_utc":datetime.now(timezone.utc).isoformat(),"configured_data_root":str(data_root),
                              "explicit_input_paths":[str(x) for x in input_paths] if input_paths else [str(data_root)],
                              "extraction_performed":False,"filtering_performed":False}
    paths={}
    layouts={"file_inventory":("fits_file_inventory.csv",["source_file","archive_member","file_size_bytes","file_sha256","status","instrument","product_association","processing_version","association_basis","archive_status","extracted","raw_record_count","error_type","error","primary_metadata","unresolved_conditions"]),
             "hdu_inventory":("fits_hdu_inventory.csv",["source_file","archive_member","hdu_index","hdu_type","extension_name","row_count","columns","header_metadata","candidate_columns"]),
             "gti_validation":("fits_gti_validation.csv",["source_file","archive_member","extension_name","interval_count","missing_columns","malformed_rows","nonfinite_rows","reversed_rows","duplicate_intervals","overlapping_pairs","sorted_by_start","semantics"]),
             "quality_distribution":("fits_quality_distribution.csv",["source_file","archive_member","extension_name","name","dtype","count","null_count","value_counts","unique_truncated","interpretation"])}
    for key,(name,default_fields) in layouts.items():
        path=report_root/name; rows=result[key]
        fields=sorted({k for row in rows for k in row}) if rows else default_fields
        with path.open("w",newline="",encoding="utf-8-sig") as fh:
            writer=csv.DictWriter(fh,fieldnames=fields or ["status"],extrasaction="ignore"); writer.writeheader()
            for row in rows: writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
        paths[key]=path
    report_root.joinpath("fits_audit_summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,default=_safe)+"\n",encoding="utf-8")
    paths["summary"]=report_root/"fits_audit_summary.json"
    phase4e_matrix=Path(__file__).resolve().parents[1]/"reports"/"phase4e_semantics_evidence_matrix.csv"
    matrix_path=phase4e_matrix if phase4e_matrix.exists() else Path(data_root)/"reports"/"phase3c_instrument_semantics_matrix.csv"
    if not matrix_path.exists():matrix_path=Path(__file__).resolve().parents[1]/"reports"/"phase3c_instrument_semantics_matrix.csv"
    evidence=[]
    if matrix_path.exists():
        with matrix_path.open(encoding="utf-8-sig",newline="") as fh:
            for rule in csv.DictReader(fh):
                status=rule.get("verification_status",rule.get("status","STILL_UNVERIFIED"))
                evidence.append({"instrument":rule.get("instrument"),"product_variant":rule.get("product_variant"),
                                 "field_or_keyword":rule.get("field_or_keyword"),"document":rule.get("document_filename_version_section_page", "Not available"),
                                 "page_or_section":rule.get("document_filename_version_section_page","Not available"),
                                 "document_path":rule.get("evidence_document_path"),"document_sha256":rule.get("evidence_sha256"),
                                 "audit_observation":rule.get("phase4e_supported_interpretation",rule.get("supporting_fits_header_or_observed_evidence")),
                                 "confidence":"Document-backed within cited scope; partial/unverified limitations retained",
                                 "status":status,"needed_evidence":rule.get("phase4e_unresolved_limits",rule.get("notes_unresolved_conflicts"))})
    evidence_out=report_root/"fits_document_evidence.csv"
    fields=sorted({k for row in evidence for k in row})
    with evidence_out.open("w",newline="",encoding="utf-8-sig") as fh:
        writer=csv.DictWriter(fh,fieldnames=fields or ["status"]);writer.writeheader();writer.writerows(evidence)
    paths["document_evidence"]=evidence_out
    counts=Counter(row.get("status") for row in result["file_inventory"])
    instrument_counts=Counter(row.get("instrument") or "UNRESOLVED" for row in result["file_inventory"])
    product_counts=Counter(row.get("product_association") or "UNRESOLVED" for row in result["file_inventory"])
    evidence_counts=Counter(row.get("status") for row in evidence)
    quality_names=Counter(row.get("name") for row in result["quality_distribution"])
    gti_bad=sum((row.get("malformed_rows") or 0) for row in result["gti_validation"] if isinstance(row.get("malformed_rows"),int))
    failures=[r for r in result["file_inventory"] if r.get("status")=="INSPECTION_FAILED"]
    md=["# Read-only Aditya-L1 FITS structural audit","","## Summary",
        f"- Candidate FITS products: {len(result['file_inventory'])}",
        f"- Structurally inspected: {counts.get('AUDITED_STRUCTURALLY_NOT_SCIENCE_READY',0)}",
        f"- Failed inspection: {len(failures)}",f"- GTI-like tables: {len(result['gti_validation'])}",
        f"- Instrument counts: {dict(instrument_counts)}",f"- Product/file counts: {dict(product_counts)}",
        f"- GTI-like tables: {len(result['gti_validation'])}; malformed/reversed/nonfinite GTI rows: {gti_bad}",
        f"- Candidate quality/status fields: {dict(quality_names)}",
        f"- Phase 3C evidence rows: {len(evidence)}; document-backed statuses: {dict(evidence_counts)}.","",
        "## Interpretation limits","This is a structural audit. No timestamp conversion, GTI mask, or quality mask is applied. Missing GTI/quality metadata is not treated as valid exposure or clean data. Candidate values and headers do not establish scientific semantics. A documented rule may be verified only within the specific product scope cited in the evidence CSV.","",
        "## Verified source paths",
        *[f"- `{path}`" for path in result["audit_metadata"]["explicit_input_paths"]],
        "## Available inputs and evidence status",
        f"The audit inspected the explicit paths listed above, including nested archive members in memory. Extraction performed: `{result['audit_metadata']['extraction_performed']}`. See the per-file inventory for exact member names, hashes, and structural status.",
        f"Instrument documentation evidence is drawn from `{matrix_path}`. The per-rule dispositions are listed in `fits_document_evidence.csv`; statuses remain scoped to cited rules and do not establish overall training readiness.",
        "No timestamp was converted or compared, no GTI endpoint rule was inferred, and no quality/status value was classified as good or bad. The reported rejection counts and final retained counts are null in audit mode because no masks were applied.","",
        "## Outputs"]+[f"- {name}" for name in ("fits_file_inventory.csv","fits_hdu_inventory.csv","fits_gti_validation.csv","fits_quality_distribution.csv","fits_document_evidence.csv","fits_audit_summary.json")]
    if failures: md.extend(["","## Inspection failures"]+[f"- `{r['source_file']}`: {r.get('error_type')}: {r.get('error')}" for r in failures])
    report_path=report_root/"fits_audit_report.md";report_path.write_text("\n".join(md)+"\n",encoding="utf-8");paths["report"]=report_path
    return paths


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",required=True, help="Configured root, also used to locate the Phase 3C evidence matrix")
    parser.add_argument("--report-root",required=True)
    parser.add_argument("--input",action="append",help="Explicit file or directory to audit; repeat for multiple verified source paths")
    args=parser.parse_args()
    outputs=write_reports(args.data_root,args.report_root,args.input)
    with outputs["file_inventory"].open(encoding="utf-8-sig", newline="") as fh:
        files_audited = sum(1 for _ in csv.DictReader(fh))
    print(json.dumps({"outputs":{k:str(v) for k,v in outputs.items()},"files_audited":files_audited}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
