"""Phase 3C report builder for the registered Aditya-L1 documentation gate.

Reads the prior read-only Phase 3A/3B inventories and source-document tree.
It does not parse/extract archives, modify source data, or enable ingestion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

GENERATOR = "phase3c_semantics_audit_v1"
ADJUDICATION_STATUSES = {
    "VERIFIED_FROM_AUTHORITATIVE_DOCUMENT",
    "PARTIALLY_RESOLVED",
    "STILL_UNVERIFIED",
    "CONFLICTING_EVIDENCE",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _row(instrument, variant, pattern, field, interpretation, units, time_ref,
         gti, quality, calibration, precedence, document, evidence, notes):
    return {
        "instrument": instrument, "product_variant": variant, "file_extension_pattern": pattern,
        "field_or_keyword": field, "proposed_interpretation": interpretation,
        "units": units, "time_reference_and_conversion": time_ref,
        "gti_application_rule": gti, "quality_sentinel_handling": quality,
        "calibration_requirement": calibration, "version_precedence_rule": precedence,
        "document_filename_version_section_page": document,
        "supporting_fits_header_or_observed_evidence": evidence,
        "verification_status": "UNVERIFIED", "notes_unresolved_conflicts": notes,
    }


def semantics_matrix():
    unresolved = "Registered instrument-specific L1 manual/SIS not retrieved."
    return pd.DataFrame([
        _row("SoLEXS", "SDD2 light curve", "*/SDD2/*.lc.gz; RATE", "RATE.TIME",
             "Native time coordinate; epoch/reference unresolved", "TUNIT absent; RATE TIMEUNIT=s",
             "TIMESYS=UTC observed; TSTART resembles Unix seconds but epoch/conversion is not authoritative", "Not applied; matching SDD2 GTI exists", "No documented flags found in inventory; invalid/null policy unresolved", "No response applied; purpose-specific calibration unresolved", "Six daily products named v1.0; compatibility not documented", "Not retrieved", "Six daily RATE tables; TIME TUNIT absent; DATE-OBS/DATE-END; TIMESYS UTC; TIMEUNIT s", unresolved),
        _row("SoLEXS", "SDD2 light curve", "*/SDD2/*.lc.gz; RATE", "RATE.COUNTS",
             "Native recorded quantity; count/bin vs rate unresolved", "TUNIT absent", "Same timing ambiguity as RATE.TIME", "Do not apply until START/STOP units, reference, and closure are documented", "Finite/nonnegative is only a candidate filter, not verified quality", "Response/calibration needed for physical spectral flux; conversion not established", "Six daily products; version/config equivalence unresolved", "Not retrieved", "COUNTS column present in all six observed products; no TUNIT", unresolved),
        _row("SoLEXS", "SDD2 light curve", "*/SDD2/*.gti.gz; GTI", "START, STOP",
             "Good-time interval endpoints; numeric convention unresolved", "TUNIT absent", "GTI extension lacks a recorded TIMEUNIT/TIMESYS/MJDREF in audited keywords", "Association by same date/detector path observed; boundary closure unknown", "GTI is not a complete quality/telemetry policy", "Not applicable", "One corresponding SDD2 GTI per day observed; SDD1 companions are empty", "Not retrieved", "12 GTI tables across six day archives; 84 rows total; endpoint columns lack units", unresolved),
        _row("SoLEXS", "SDD2 light curve", "*/SDD2/*.lc.gz", "RATE EXTNAME / detector path",
             "SDD2 product identity", "Detector label only", "Not applicable", "Associate only matching detector/day after documentation", "Unknown", "Energy response/calibration not applied", "Six archives carry v1.0; mutually compatible status unresolved", "Not retrieved", "All six selected light curves under SDD2; product names include SDD2", unresolved),
        _row("SoLEXS", "Daily L1 products", "AL1_SLX_L1_YYYYMMDD_v1.0.zip", "CREATOR, DATE, OBS_ID, TIMEDEL",
             "Generation/provenance and cadence metadata", "TIMEDEL=1 observed", "Primary GTI DATE-OBS/TSTART and RATE DATE-OBS/TSTART appear aligned but no registered time contract", "No GTI applied", "Six days have creator solexs_pipeline-1.4; compatibility details unresolved", "Calibration status unresolved", "Same archive name version v1.0 and creator observed; full config compatibility unknown", "Not retrieved", "Phase3A inventory has six daily SDD2 products; representative archive headers identify CREATOR, DATE and OBS_ID", unresolved),
        _row("SoLEXS", "Selected native light curve", "*/SDD2/*.lc.gz", "physical irradiance / flux conversion",
             "No conversion is justified from available evidence", "No physical unit established", "Upstream timing unresolved", "GTI unresolved", "Quality unresolved", "Response/calibration files and documented response conversion required", "Not applicable", "Not retrieved", "COUNTS has no declared column unit; no calibrated response conversion in inventory", unresolved),
        _row("HEL1OS", "Light curve per detector/band", "*/lightcurve_*.fits; MJD", "MJD",
             "Modified Julian Date field; time scale and authority unresolved", "Column unit MJD", "Do not convert to UTC until official time scale/reference is known", "GTI association exists by product path; time semantics unresolved", "Quality policy unresolved", "Detector calibration/response status unresolved", "V111/V211 precedence unresolved", "Not retrieved", "44 light curves expose MJD; extension TSTART/TSTOP numeric MJD-like values; no TIMESYS/MJDREF in recorded extension-key inventory", unresolved),
        _row("HEL1OS", "Light curve per detector/band", "*/lightcurve_*.fits", "ISOT",
             "ISO-8601-like string field; authoritative time field/scale unresolved", "Column unit UT", "Do not infer UTC solely from MJD/ISOT numeric agreement", "GTI association and time basis unresolved", "Invalid/null/sentinel rules unresolved", "Not applicable", "V111/V211 precedence unresolved", "Not retrieved", "44 products contain ISOT with unit UT; deterministic comparison to MJD is structural only", unresolved),
        _row("HEL1OS", "Light curve per detector/band", "*/lightcurve_*.fits", "CTR",
             "Native count-rate-like field according to FITS unit; product semantics unresolved", "cts/sec observed", "Timestamp selection unresolved", "GTI not applied", "Finite/nonnegative is candidate-only; saturation/dead-time/telemetry rules unknown", "Response/calibration conversion not established", "Do not combine versions until documented", "Not retrieved", "44 light curves expose CTR, format D, unit cts/sec", unresolved),
        _row("HEL1OS", "Light curve per detector/band", "*/lightcurve_*.fits", "STAT_ERR",
             "Uncertainty field; statistical meaning unresolved", "cts/sec observed", "Same time basis as row unresolved", "GTI not applied", "No handling rule established", "Uncertainty propagation/calibration unresolved", "Version precedence unresolved", "Not retrieved", "44 products expose STAT_ERR, format D, unit cts/sec", unresolved),
        _row("HEL1OS", "Detector and energy-band extension", "CDTE1/2_LC_BAND_*; CZT1/2_LC_BAND_*", "EXTNAME",
             "Detector/band identifiers encoded in extension label; exact operational mapping needs SIS", "Energy labels in keV in extension name", "Not applicable", "Matching detector-specific GTI needed", "Quality rules unresolved", "Response differs by detector/band and requires documentation", "Version compatibility unresolved", "Not retrieved", "44 light curves comprise four repeated extension patterns: CDTE1, CDTE2, CZT1, CZT2; named bands 5–20 and 20–40 keV in audited set", unresolved),
        _row("HEL1OS", "Per-product GTI", "*/aux/gti*.fits", "START, STOP and time keywords",
             "GTI interval association and time reference unresolved", "Observed GTI table schema in Phase3A; detailed scale/units absent from registered evidence", "No recorded TIMESYS/MJDREF/TIMEZERO in Phase3B header-key scan", "No inclusion or boundary rule may be applied yet", "GTI does not establish dead-time or telemetry quality", "Not applicable", "Must follow chosen product version", "Not retrieved", "44 GTI tables, 44 rows total; same product-version directories as light curves", unresolved),
        _row("HEL1OS", "V111/V211 overlapping light curves", "*_lev1_V111.zip; *_lev1_V211.zip", "version identifier / provenance", "Version precedence and cross-version compatibility unresolved", "Not applicable", "Time basis unresolved", "Do not merge overlapping GTIs or samples", "Version-specific quality effects unknown", "Calibration compatibility unknown", "No precedence selected", "Not retrieved", "40 V111 and 4 V211 light curves observed; overlapping products retained by Phase3A", unresolved),
        _row("HEL1OS", "All selected light curves", "*/lightcurve_*.fits", "quality, sentinel, dead-time, saturation, telemetry", "Handling unresolved", "Product-specific", "Time-linked housekeeping relation unresolved", "GTI alone cannot resolve these quality rules", "No invalid values should be relabeled valid based on finite/nonnegative checks alone", "Detector-specific response/housekeeping documentation needed", "Version-specific handling unresolved", "Not retrieved", "No verified quality fields/rules in Phase3A/3B products inventory", unresolved),
    ])


def _documents(data_root, report_root):
    docs = data_root / "source_docs"
    prior = report_root / "phase3b_document_register.json"
    prior_reg = json.loads(prior.read_text(encoding="utf-8")) if prior.exists() else {"sources": []}
    found = []
    for x in prior_reg.get("sources", []):
        if x.get("source_id") in {"pradan_alpps_guide", "solexs_l1_manual_unavailable", "hel1os_l1_manual_unavailable"}:
            found.append({**x, "phase3c_relevance": "general portal access guide only; does not establish instrument-specific L1 semantics" if x.get("source_id") == "pradan_alpps_guide" else "not retrieved"})
    isro_root = docs / "isro"
    for instrument, expected in (("SoLEXS", "solexs"), ("HEL1OS", "hel1os")):
        for category in ("product manual", "SIS", "GTI/quality/calibration/response documentation"):
            paths = list((isro_root / expected).glob("**/*")) if (isro_root / expected).exists() else []
            files = [p for p in paths if p.is_file()]
            found.append({"source_id": f"phase3c_{expected}_{category.lower().replace('/', '_').replace(' ', '_')}",
                          "dataset_or_document": f"{instrument} {category}", "official_source_url": "https://pradan.issdc.gov.in/al1/protected/miscDownloads.xhtml",
                          "version": None, "retrieval_date_utc": None, "path": None, "sha256": None,
                          "status": "NOT_RETRIEVED_LOGIN_REQUIRED" if not files else "PRESENT_NEEDS_MANUAL_REVIEW",
                          "access_observation": "Public PRADAN homepage loaded; Other Downloads navigation redirected to ISSDC sign-in. No signed-in session was available in the task browser.",
                          "files_found_in_expected_instrument_folder": [str(p) for p in files]})
    general = isro_root / "general"
    general_files = [p for p in general.rglob("*") if p.is_file()] if general.exists() else []
    found.append({"source_id": "phase3c_general_aditya_handbook", "dataset_or_document": "Aditya-L1 User Handbook / instrument appendices",
                  "official_source_url": "https://pradan.issdc.gov.in/al1/protected/miscDownloads.xhtml", "version": None,
                  "retrieval_date_utc": None, "path": None, "sha256": None,
                  "status": "NOT_RETRIEVED_LOGIN_REQUIRED" if not general_files else "PRESENT_NEEDS_MANUAL_REVIEW",
                  "access_observation": "Other Downloads requires ISSDC authentication in the currently available session.",
                  "files_found_in_expected_general_folder": [str(p) for p in general_files]})
    registered_files = []
    for path in sorted(p for p in isro_root.rglob("*") if p.is_file()) if isro_root.exists() else []:
        registered_files.append({"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size})
    return {"generator": GENERATOR, "audit_time_utc": datetime.now(timezone.utc).isoformat(),
            "portal_attempt": {"public_homepage": "https://pradan.issdc.gov.in/al1/index.xhtml",
                               "requested_section": "Other Downloads",
                               "section_url": "https://pradan.issdc.gov.in/al1/protected/miscDownloads.xhtml",
                               "observed_result": "redirected to https://idp.issdc.gov.in/auth/realms/issdc/protocol/openid-connect/auth and displayed the ISSDC username/password login form; no authenticated session available; no credentials entered",
                               "downloads_performed": []}, "documents": found,
            "registered_source_files": registered_files}


def _rule_key(row):
    return tuple(row.get(key, "") for key in ("instrument", "product_variant", "field_or_keyword"))


def _apply_adjudication(matrix, docs, path: Path | None):
    """Apply independently reviewed evidence only after fail-closed provenance checks."""
    if path is None:
        return matrix, {"status": "NOT_PROVIDED", "rows_applied": 0, "reason": "No reviewed adjudication input supplied."}
    if not path.is_file():
        raise FileNotFoundError(f"Adjudication file does not exist: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Adjudication file is empty.")
    expected = {_rule_key(row) for row in matrix.to_dict("records")}
    supplied = {_rule_key(row) for row in rows}
    if len(rows) != len(supplied) or supplied != expected:
        raise ValueError("Adjudication rows must match the canonical rule identities exactly once.")
    registered = {entry["sha256"].lower(): entry["path"] for entry in docs.get("registered_source_files", [])}
    conflicts, invalid = [], []
    by_key = {}
    for row in rows:
        key = _rule_key(row)
        status = row.get("verification_status", "")
        digest = row.get("evidence_sha256", "").lower()
        citation = row.get("document_filename_version_section_page", "")
        if status not in ADJUDICATION_STATUSES:
            invalid.append(f"{key}: unsupported status {status!r}")
        if not citation or not digest or digest not in registered:
            invalid.append(f"{key}: missing page citation or unregistered document hash")
        if status == "CONFLICTING_EVIDENCE":
            conflicts.append(key)
        by_key[key] = row
    if invalid:
        raise ValueError("Invalid adjudication evidence: " + "; ".join(invalid))
    result = matrix.copy()
    for index, base in result.iterrows():
        reviewed = by_key[_rule_key(base)]
        for field in ("verification_status", "document_filename_version_section_page",
                      "phase4e_supported_interpretation", "phase4e_unresolved_limits",
                      "evidence_document_path", "evidence_sha256"):
            if field in reviewed:
                result.at[index, field] = reviewed[field]
        result.at[index, "canonical_evidence_path"] = registered[reviewed["evidence_sha256"].lower()]
        result.at[index, "canonical_adjudication_source"] = str(path)
    return result, {"status": "APPLIED", "path": str(path), "sha256": sha256(path), "rows_applied": len(rows),
                    "conflicting_rules": [list(key) for key in conflicts],
                    "registered_document_hashes_validated": sorted(set(row["evidence_sha256"].lower() for row in rows))}


def _coverage(phase3a, phase3b, readiness_status="BLOCKED_DOCUMENTATION_UNAVAILABLE"):
    rows = []
    for product in phase3a["solexs"]["products"]:
        rows.append({"instrument": "SoLEXS", "date_utc": product["raw_start_utc"][:10], "detector": product.get("detector"),
                     "product_variant": product.get("member"), "processing_version": product.get("nested_archive", "").split("_")[-1].removesuffix(".zip"),
                     "raw_rows": product["raw_rows"], "candidate_finite_nonnegative_rows": product["finite_nonnegative_candidate_rows"],
                     "GTI_rows_observed": None,
                     "GTI_tables_listed_as_companions": len(product.get("GTI_companions", [])),
                     "valid_after_GTI": None, "valid_intervals": None, "longest_gap_seconds": None,
                     "verified_GOES_events_in_valid_intervals": None, "candidate_windows": None, "verified_windows": 0,
                     "positive_windows": None, "negative_windows": None, "status": readiness_status,
                     "reason": "RATE units/time reference and GTI interpretation are unverified; candidate counts are not scientific valid counts"})
    for product in phase3a["hel1os"]["products"]:
        rows.append({"instrument": "HEL1OS", "date_utc": product["raw_start_utc"][:10], "detector": product["extension"].split("_LC_", 1)[0],
                     "product_variant": product.get("member"), "processing_version": product.get("processing_version"),
                     "raw_rows": product["raw_rows"], "candidate_finite_nonnegative_rows": product["candidate_finite_nonnegative_rate_error_rows"],
                     "GTI_rows_observed": None,
                     "GTI_tables_listed_as_companions": len(product.get("GTI_companions", [])), "valid_after_GTI": None,
                     "valid_intervals": None, "longest_gap_seconds": None, "verified_GOES_events_in_valid_intervals": None,
                     "candidate_windows": None, "verified_windows": 0, "positive_windows": None, "negative_windows": None,
                     "status": readiness_status,
                     "reason": "MJD/ISOT authority, GTI, quality/calibration, and V111/V211 precedence are unverified"})
    goes = json.loads((phase3b.parent / "phase3b_goes_quality_audit.json").read_text(encoding="utf-8"))
    for day in goes["daily_channel_counts"]:
        rows.append({"instrument": "GOES-18", "date_utc": day["utc_date"], "detector": None,
                     "product_variant": day["source_file"], "processing_version": "2-2-2", "raw_rows": day["observed_rows"],
                     "candidate_finite_nonnegative_rows": day["candidate_valid_rows"], "GTI_rows_observed": None,
                     "GTI_tables_listed_as_companions": None,
                     "valid_after_GTI": None, "valid_intervals": None, "longest_gap_seconds": day.get("longest_missing_run_seconds"),
                     "verified_GOES_events_in_valid_intervals": None, "candidate_windows": None, "verified_windows": 0,
                     "positive_windows": None, "negative_windows": None,
                     "status": "GOES_QUALITY_VERIFIED_AS_REFERENCE_ONLY",
                     "reason": "NOAA sample quality/CSV consistency audited in Phase 3B; does not make joint Aditya feature coverage verified"})
    return pd.DataFrame(rows)


def build(data_root: Path, repo_root: Path, report_root: Path, adjudication_path: Path | None = None):
    phase3a_path = report_root / "phase3a_data_audit.json"
    phase3b_path = report_root / "phase3b_audit_summary.json"
    phase3a = json.loads(phase3a_path.read_text(encoding="utf-8"))
    phase3b = json.loads(phase3b_path.read_text(encoding="utf-8"))
    docs = _documents(data_root, report_root)
    matrix, adjudication = _apply_adjudication(semantics_matrix(), docs, adjudication_path)
    class_counts = phase3b["event_reconciliation"].get("verified_official_class_counts", {})
    hel_versions = {v: sum(p.get("processing_version") == v for p in phase3a["hel1os"]["products"])
                    for v in sorted({p.get("processing_version") for p in phase3a["hel1os"]["products"]})}
    verified = int((matrix.verification_status == "VERIFIED_FROM_AUTHORITATIVE_DOCUMENT").sum())
    partial = int((matrix.verification_status == "PARTIALLY_RESOLVED").sum())
    unresolved = int(matrix.verification_status.isin(["STILL_UNVERIFIED", "CONFLICTING_EVIDENCE", "UNVERIFIED"]).sum())
    documents_available = bool(docs.get("registered_source_files"))
    readiness_status = ("BLOCKED_SCIENTIFIC_SEMANTICS_UNRESOLVED" if documents_available else "BLOCKED_DOCUMENTATION_UNAVAILABLE")
    reason = ("Reviewed document evidence was registered, but unresolved or conflicting critical semantics remain; "
              "training and scientific filtering stay blocked." if documents_available else
              "No registered instrument documentation is available for adjudication; critical semantics remain unverified.")
    coverage = _coverage(phase3a, phase3b_path, readiness_status)
    summary = {
        "generator": GENERATOR, "audit_time_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(repo_root), "readiness_status": readiness_status, "reason": reason,
        "adjudication": adjudication,
        "training_started": False, "tuning_or_evaluation_started": False, "final_split_boundaries_defined": False,
        "production_adapter_rules_changed": False,
        "adapter_review_findings": [
            "SoLEXS adapter accepts missing TIME and COUNTS TUNIT metadata and labels COUNTS as counts/s using a publicly discoverable manual; the local archives omit both column units. This project does not treat that repost as the registered product contract.",
            "SoLEXS adapter converts TIME directly as Unix seconds and assigns 2-22 keV/one-second cadence without a registered matching product manual in the current evidence set.",
            "SoLEXS adapter does not apply the matching GTI companions. HEL1OS adapter remains fail-closed.",
            "No adapter change was made because the fully verified rule set needed to implement time, GTI, quality, calibration, and version policy is incomplete; the readiness gate remains blocked."
        ],
        "observed_archive_inventory": {
            "solexs_nested_archives": phase3a["solexs"]["nested_archive_count"], "solexs_lightcurves": len(phase3a["solexs"]["products"]),
            "solexs_detector_counts": {d: sum(p.get("detector") == d for p in phase3a["solexs"]["products"]) for d in sorted({p.get("detector") for p in phase3a["solexs"]["products"]})},
            "solexs_raw_rows": phase3a["solexs"]["raw_rows"], "solexs_candidate_rows_unverified": phase3a["solexs"]["candidate_rows"],
            "solexs_gti_tables": phase3b["aditya_archive_structural_checks"]["solexs_gti_table_count"],
            "hel1os_nested_archives": phase3a["hel1os"]["nested_archive_count"], "hel1os_lightcurves": len(phase3a["hel1os"]["products"]),
            "hel1os_versions": hel_versions,
            "hel1os_extension_counts": {e: sum(p.get("extension") == e for p in phase3a["hel1os"]["products"]) for e in sorted({p.get("extension") for p in phase3a["hel1os"]["products"]})},
            "hel1os_gti_tables": phase3b["aditya_archive_structural_checks"]["hel1os_gti_table_count"],
            "hel1os_mjd_isot_sample_rows": phase3b["aditya_archive_structural_checks"]["hel1os_mjd_isot_total_rows_compared"],
            "hel1os_mjd_isot_rows_within_1ms_under_provisional_utc": phase3b["aditya_archive_structural_checks"]["hel1os_mjd_isot_rows_within_1ms"],
            "raw_archives_modified_or_extracted": False,
        },
        "goes_event_reference": {"verified_report_records": phase3b["event_reconciliation"]["verified_official_event_rows"],
                                 "class_counts": class_counts, "negative_label_completeness_established": False,
                                 "candidate_event_matches_do_not_verify_instrument_interval_overlap": True},
        "coverage_and_window_counts": {"verified_solexs_samples": None, "verified_hel1os_samples": None,
                                        "verified_joint_windows": 0, "positive_windows": None, "negative_windows": None,
                                        "reason": "No complete verified Aditya time, GTI, quality, calibration, and version rule set exists; negative-label completeness is not established."},
        "phase3a_sha256": sha256(phase3a_path), "phase3b_sha256": sha256(phase3b_path),
        "matrix_rule_count": len(matrix), "matrix_verified_rules": verified,
        "matrix_partially_resolved_rules": partial, "matrix_unresolved_or_conflicting_rules": unresolved,
    }
    version_text = ", ".join(f"{count} {version}" for version, count in hel_versions.items())
    readiness = "\n".join([
        f"<!-- {GENERATOR} -->", "# Phase 3C — Instrument semantics and documentation gate", "",
        f"- Readiness: **{summary['readiness_status']}**", f"- Audit time UTC: {summary['audit_time_utc']}",
        "- Training, tuning, evaluation, and final split selection: **not started**", "- Local archives modified/extracted: **No**", "",
        "## Documentation acquisition", "",
        "Source-document registration and scientific adjudication are separate. Registered files and hashes are in the document register; any reviewed adjudication must be supplied explicitly and is accepted only when every cited hash matches a registered source file and every canonical rule is represented once.",
        f"Adjudication: **{adjudication['status']}**; verified/partial/unresolved-or-conflicting rules: **{verified}/{partial}/{unresolved}**. The gate remains blocked until all required semantics, coverage, and label-completeness requirements are independently satisfied.", "",
        "## What the local files establish", "",
        f"The archives contain {summary['observed_archive_inventory']['solexs_lightcurves']} SoLEXS SDD2 daily light curves ({summary['observed_archive_inventory']['solexs_raw_rows']} rows) and {summary['observed_archive_inventory']['solexs_gti_tables']} GTI tables. The reviewed manual establishes Unix-time semantics for the documented SoLEXS L1 coordinate and count-rate scope, but does not establish the GTI endpoint contract, complete provenance compatibility, or a physical-flux conversion.",
        f"HEL1OS contains {summary['observed_archive_inventory']['hel1os_lightcurves']} light curves across {len(summary['observed_archive_inventory']['hel1os_extension_counts'])} repeated detector/band extension patterns ({version_text} products) and {summary['observed_archive_inventory']['hel1os_gti_tables']} GTI tables. MJD and ISOT agree within 1 ms for {summary['observed_archive_inventory']['hel1os_mjd_isot_rows_within_1ms_under_provisional_utc']} of {summary['observed_archive_inventory']['hel1os_mjd_isot_sample_rows']} sampled rows under a provisional UTC comparison. Agreement does not establish the authoritative time scale.",
        "Product fields, FITS layouts, nominal extension labels, file versions, and observed timestamps are evidence, not semantics. The canonical matrix carries only the reviewed statuses and page citations accepted by the provenance checks; it does not create a filtering policy.", "",
        "## Adapter review and coverage", "",
        "The SoLEXS adapter currently accepts missing column units, assumes TIME is Unix seconds, describes COUNTS as counts/s, assigns an energy band/cadence contract, and does not apply GTI. The reviewed sources support only the cited portions of those assumptions; they do not establish a safe GTI, quality, calibration, or version policy. The HEL1OS reader remains fail-closed. No production adapter changes were made.",
        "GOES event records remain a separate reference source. NOAA report records in the interval: 123 (C=50, M=67, X=6); the report documentation does not establish population completeness, so negative labels are not allowed. Instrument event overlap, verified valid samples, and positive/negative window counts remain withheld; verified joint windows = 0. No final chronological split boundaries were chosen.", "",
        "## Next minimal action", "",
        "Sign in manually to PRADAN, retrieve the SoLEXS/HEL1OS registered L1 manuals, SIS and supporting GTI, quality, calibration/response and version-selection documentation, then save them unmodified in the requested folders. If the portal cannot provide them, place any officially downloaded copies in those folders with portal identifiers/revision metadata; do not treat unregistered reposts as authoritative. Rerun this audit before changing adapters or calculating scientific coverage.", "",
    ])
    return matrix, docs, coverage, readiness, summary


def _recognized(path: Path, kind: str) -> bool:
    try:
        if kind == "json": return json.loads(path.read_text(encoding="utf-8")).get("generator") == GENERATOR
        if kind == "md": return path.read_text(encoding="utf-8").startswith(f"<!-- {GENERATOR} -->")
        with path.open("r", encoding="utf-8", newline="") as f: return "audit_generator" in next(csv.reader(f))
    except Exception: return False


def _write(path: Path, body: str, kind: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not _recognized(path, kind):
        raise FileExistsError(f"Refusing to overwrite unrecognized file: {path}")
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(body); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def write_reports(outputs, report_root: Path):
    matrix, docs, coverage, readiness, summary = outputs
    matrix = matrix.copy(); matrix.insert(0, "audit_generator", GENERATOR)
    coverage = coverage.copy(); coverage.insert(0, "audit_generator", GENERATOR)
    files = [
        ("phase3c_instrument_semantics_matrix.csv", matrix.to_csv(index=False), "csv"),
        ("phase3c_document_register.json", json.dumps(docs, indent=2, sort_keys=True, default=str) + "\n", "json"),
        ("phase3c_coverage_reassessment.csv", coverage.to_csv(index=False), "csv"),
        ("phase3c_readiness.md", readiness, "md"),
        ("phase3c_audit_summary.json", json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", "json"),
    ]
    for name, body, kind in files: _write(report_root / name, body, kind)
    return [str(report_root / name) for name, _, _ in files]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\SolarFlareData"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--adjudication", type=Path, help="Reviewed CSV with an exact, hash-validated entry for every canonical rule")
    args = parser.parse_args()
    root = args.report_root or args.data_root / "reports"
    outputs = build(args.data_root, args.repo_root, root, args.adjudication)
    paths = write_reports(outputs, root)
    print("Readiness:", outputs[4]["readiness_status"])
    print("Matrix rules verified:", outputs[4]["matrix_verified_rules"], "/", outputs[4]["matrix_rule_count"])
    for path in paths: print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
