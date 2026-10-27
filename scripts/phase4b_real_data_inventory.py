"""Consolidate read-only Phase 3 audits into a Phase 4B readiness inventory.

This script does not extract archives, alter observations, infer instrument
semantics, create labels, or train a model. Candidate event overlaps remain
explicitly provisional and are copied from the prior structural audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def file_record(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        return {"label": label, "path": str(path), "exists": False}
    return {
        "label": label,
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def rule_next_action(rule: dict[str, str]) -> str:
    instrument = rule["instrument"]
    field = rule["field_or_keyword"]
    if instrument == "SoLEXS":
        return (
            "Retrieve the registered SoLEXS L1 manual/SIS and applicable time, GTI, "
            "quality, calibration/response, and processing-version documentation; "
            "record the controlling section/page, then rerun Phase 3C."
        )
    return (
        "Retrieve the registered HEL1OS L1 manual/SIS and applicable time-scale, "
        "GTI, detector/band, quality/dead-time, calibration/response, and V111/V211 "
        "precedence documentation; record the controlling section/page, then rerun Phase 3C."
    )


def collect(data_root: Path, downloads_root: Path, repo_root: Path) -> dict[str, Any]:
    reports = data_root / "reports"
    phase3c = json_file(reports / "phase3c_audit_summary.json")
    phase3b = json_file(reports / "phase3b_audit_summary.json")
    phase3a = json_file(reports / "phase3a_data_audit.json")
    document_register = json_file(reports / "phase3c_document_register.json")
    matrix = csv_rows(reports / "phase3c_instrument_semantics_matrix.csv")
    coverage_rows = csv_rows(reports / "phase3b_instrument_coverage.csv")
    reconciliation = csv_rows(reports / "phase3b_event_reconciliation.csv")

    archive_paths = {
        "solexs_outer_archive": downloads_root / "solexs_2026Oct23T184402685.zip",
        "hel1os_outer_archive": downloads_root / "hel1os_2026Oct23T184518914.zip",
        "local_goes_events": downloads_root / "data" / "goes" / "goes_flares.csv",
        "local_goes_measurements": downloads_root / "data" / "goes" / "goes_xrs_1min.csv",
        "official_goes_events": data_root / "goes_official" / "sci_xrsf-l2-flrpt_geo_y2026_v1-0-1.csv",
        "official_goes_metadata": data_root / "goes_official" / "sci_xrsf-l2-flrpt_geo_metadata.json",
    }
    provenance = [file_record(path, label) for label, path in archive_paths.items()]

    instrument_counts: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in coverage_rows:
        grouped[row["instrument"]].append(row)
    for instrument, rows in grouped.items():
        instrument_counts[instrument] = {
            "product_rows": len(rows),
            "observed_rows_sum_across_products": sum(int(float(r["observed_rows"] or 0)) for r in rows),
            "candidate_quality_rows_sum_across_products": sum(int(float(r["candidate_quality_rows"] or 0)) for r in rows),
            "candidate_event_start_occurrences_sum_across_products": sum(int(float(r["candidate_event_starts"] or 0)) for r in rows),
            "verified_coverage_product_rows": sum(r["verified_coverage"].lower() == "true" for r in rows),
            "candidate_interval_time_basis": "provisional; do not interpret as authoritative UTC coverage",
            "products": [
                {
                    "date": r["date_utc"],
                    "detector": r["detector"],
                    "energy_band": r["energy_band"],
                    "processing_version": r["processing_version"],
                    "source_product": r["source_file"],
                    "provisional_start": r["raw_start_utc"],
                    "provisional_end": r["raw_end_utc"],
                    "observed_rows": int(float(r["observed_rows"] or 0)),
                    "candidate_quality_rows": int(float(r["candidate_quality_rows"] or 0)),
                    "candidate_event_start_occurrences": int(float(r["candidate_event_starts"] or 0)),
                    "gti_applied": r["GTI_applied"].lower() == "true",
                    "verified_coverage": r["verified_coverage"].lower() == "true",
                    "status": r["coverage_status"],
                    "candidate_intervals": json.loads(r["candidate_intervals_json"] or "[]"),
                }
                for r in rows
            ],
        }

    reconciliation_counts = Counter(r["reconciliation_status"] for r in reconciliation)
    class_counts = dict(phase3b["event_reconciliation"]["verified_official_class_counts"])
    source_docs = sorted(p for p in (data_root / "source_docs").rglob("*") if p.is_file())
    instrument_doc_records = [
        doc for doc in document_register["documents"]
        if "solexs" in doc.get("dataset_or_document", "").casefold()
        or "hel1os" in doc.get("dataset_or_document", "").casefold()
    ]
    registered_instrument_docs_present = any(
        doc.get("verification_status") == "VERIFIED" or doc.get("status") == "VERIFIED"
        for doc in instrument_doc_records
    )

    blockers = []
    for rule in matrix:
        if rule["instrument"] == "SoLEXS":
            relevant_files = [
                str(archive_paths["solexs_outer_archive"]),
                "nested AL1_SLX_L1_YYYYMMDD_v1.0.zip: SDD2/*.lc.gz RATE.TIME/RATE.COUNTS",
                "nested AL1_SLX_L1_YYYYMMDD_v1.0.zip: SDD2/*.gti.gz GTI.START/GTI.STOP",
            ]
        else:
            relevant_files = [
                str(archive_paths["hel1os_outer_archive"]),
                "nested *_lev1_V111.zip and *_lev1_V211.zip: lightcurve_*.fits MJD/ISOT/CTR/STAT_ERR/EXTNAME",
                "nested *_lev1_V111.zip and *_lev1_V211.zip: aux/gti*.fits START/STOP",
            ]
        relevant_files.extend(
            [
                str(reports / "phase3a_data_audit.json"),
                str(reports / "phase3c_instrument_semantics_matrix.csv"),
            ]
        )
        blockers.append(
            {
                "instrument": rule["instrument"],
                "product_variant": rule["product_variant"],
                "rule": rule["field_or_keyword"],
                "unresolved_scientific_question": (
                    f"What authoritative interpretation applies to {rule['field_or_keyword']}? "
                    f"The current evidence proposes: {rule['proposed_interpretation']}"
                ),
                "relevant_instrument_files_and_metadata": relevant_files,
                "observed_local_evidence": rule["supporting_fits_header_or_observed_evidence"],
                "authoritative_documentation_required": (
                    "Registered instrument-specific Level-1 manual/SIS with the field definition and applicable timing, "
                    "GTI, quality, calibration/response, and product-version sections; include supporting calibration "
                    "or housekeeping documentation where relevant."
                ),
                "can_resolve_from_existing_local_evidence": False,
                "why_not": rule["notes_unresolved_conflicts"],
                "next_concrete_action": rule_next_action(rule),
                "status": rule["verification_status"],
            }
        )

    product_inventory = phase3c["observed_archive_inventory"]
    product_file_inventory = {
        "SoLEXS": [
            {
                "nested_archive": p["nested_archive"],
                "nested_archive_sha256": p["nested_archive_sha256"],
                "member": p["member"],
                "member_sha256": p["member_sha256"],
                "format": p["format"],
                "detector": p["detector"],
                "raw_rows": p["raw_rows"],
                "finite_nonnegative_candidate_rows": p["finite_nonnegative_candidate_rows"],
                "excluded_nan_or_negative_rows": p["excluded_nan_or_negative_rows"],
                "time_units": p["units"]["TIME"],
                "count_units": p["units"]["COUNTS"],
                "time_header": p["RATE_header"],
                "candidate_segments": p["candidate_intervals"],
                "candidate_segment_break_count": max(0, len(p["candidate_intervals"]) - 1),
                "duplicate_timestamp_rows": 0,
                "duplicate_check_basis": "Phase 3A structural validator rejects duplicate or non-monotonic parsed timestamps; time epoch remains unverified.",
                "gti_applied": p["GTI_applied"],
                "gti_companions": p["GTI_companions"],
                "scientifically_valid_rows": p["valid_rows_after_GTI"],
            }
            for p in phase3a["solexs"]["products"]
        ],
        "HEL1OS": [
            {
                "nested_archive": p["nested_archive"],
                "nested_archive_sha256": p["nested_archive_sha256"],
                "member": p["member"],
                "member_sha256": p["member_sha256"],
                "format": phase3a["hel1os"]["format"],
                "processing_version": p["processing_version"],
                "extension": p["extension"],
                "raw_rows": p["raw_rows"],
                "candidate_finite_nonnegative_rows": p["candidate_finite_nonnegative_rate_error_rows"],
                "columns": p["columns"],
                "candidate_segments": p["candidate_intervals"],
                "candidate_segment_break_count": max(0, len(p["candidate_intervals"]) - 1),
                "duplicate_timestamp_rows": 0,
                "duplicate_check_basis": "Phase 3A structural validator rejects duplicate or non-monotonic parsed timestamps; MJD/ISOT time scale remains unverified.",
                "gti_companions": p["GTI_companions"],
                "scientifically_valid_rows": p["scientifically_valid_rows"],
            }
            for p in phase3a["hel1os"]["products"]
        ],
    }
    inventory = {
        "schema_version": "phase4b-real-data-inventory-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_status": "BLOCKED_NOT_STARTED",
        "source_data_modified": False,
        "archive_extraction_performed": False,
        "data_root": str(data_root),
        "downloads_root": str(downloads_root),
        "provenance_files": provenance,
        "available_source_documents": [
            {"path": str(p), "bytes": p.stat().st_size, "filename": p.name} for p in source_docs
        ],
        "registered_solexs_hel1os_l1_document_found": registered_instrument_docs_present,
        "registered_instrument_document_register": instrument_doc_records,
        "audit_inputs": {
            "phase3a_report": file_record(reports / "phase3a_data_audit.json", "phase3a audit"),
            "phase3b_report": file_record(reports / "phase3b_audit_summary.json", "phase3b audit"),
            "phase3c_report": file_record(reports / "phase3c_audit_summary.json", "phase3c audit"),
            "phase3c_rule_matrix": file_record(reports / "phase3c_instrument_semantics_matrix.csv", "phase3c rule matrix"),
            "phase3b_coverage": file_record(reports / "phase3b_instrument_coverage.csv", "phase3b candidate coverage"),
        },
        "archive_inventory": {
            **product_inventory,
            "product_files": product_file_inventory,
            "source_archive_bytes_unchanged_from_phase3b": phase3b["raw_sources_modified"] is False,
            "note": "Nested archive members, FITS table structures and candidate counts are inherited from Phase 3A-3C read-only audits; no files were extracted by this consolidation script.",
        },
        "structural_validation": {
            "solexs_duplicate_or_nonmonotonic_timestamp_rows": 0,
            "hel1os_duplicate_or_nonmonotonic_timestamp_rows": 0,
            "time_check_caveat": "These are checks on structurally parsed candidate time arrays; they do not verify the authoritative epoch or time scale.",
            "solexs_candidate_segments": sum(len(p["candidate_intervals"]) for p in phase3a["solexs"]["products"]),
            "solexs_candidate_segment_breaks": sum(max(0, len(p["candidate_intervals"]) - 1) for p in phase3a["solexs"]["products"]),
            "solexs_rows_excluded_as_nan_or_negative_candidate_values": sum(p["excluded_nan_or_negative_rows"] for p in phase3a["solexs"]["products"]),
            "hel1os_candidate_segments": sum(len(p["candidate_intervals"]) for p in phase3a["hel1os"]["products"]),
            "hel1os_candidate_segment_breaks": sum(max(0, len(p["candidate_intervals"]) - 1) for p in phase3a["hel1os"]["products"]),
            "goes_measurement_rows": phase3a["goes"]["total_rows"],
            "goes_csv_duplicate_timestamp_rows": phase3a["goes"]["csv_comparison"]["duplicate_timestamps"],
            "goes_channel_missing_counts": {
                channel: stats["rows"] - stats["csv_finite"]
                for channel, stats in phase3a["goes"]["csv_comparison"]["channel_comparison"].items()
            },
            "goes_event_record_quality": phase3a["goes_events"]["quality_checks"],
            "candidate_segment_interpretation": "Segments split at rejected candidate values and/or parsed time gaps; cadence/time conversion is provisional and segment breaks are not verified GTI gaps.",
        },
        "observation_coverage": {
            "per_instrument": instrument_counts,
            "verified_solexs_observations": None,
            "verified_hel1os_observations": None,
            "verified_joint_windows": 0,
            "eligible_positive_windows": None,
            "eligible_negative_windows": None,
            "unknown_label_windows": None,
            "split_feasibility": "NOT_ESTABLISHED; zero verified joint windows and no defensible event/negative window counts",
            "candidate_counts_are_not_eligible_counts": True,
        },
        "goes_event_reconciliation": {
            "official_catalog": "NOAA_NCEI_GOES_XRS_L2_FLARE_REPORT",
            "official_algorithm_version": phase3b["event_reconciliation"]["official_product_algorithm_version"],
            "official_records_in_candidate_interval": phase3b["event_reconciliation"]["verified_official_event_rows"],
            "official_class_counts": class_counts,
            "official_mx_event_records": phase3b["event_reconciliation"]["verified_official_MX_events"],
            "local_event_rows": phase3b["event_reconciliation"]["local_rows_in_coverage"],
            "local_official_reconciliation_counts": dict(reconciliation_counts),
            "local_to_official_candidate_matches": phase3b["event_reconciliation"]["candidate_pair_count"],
            "matching_policy": phase3b["event_reconciliation"]["time_match_policy"],
            "official_events_missing_end_time": phase3b["event_reconciliation"]["official_missing_end_time_rows"],
            "catalog_negative_completeness_established": False,
            "candidate_instrument_event_start_occurrences": {
                name: counts["candidate_event_start_occurrences_sum_across_products"]
                for name, counts in instrument_counts.items() if name in {"SoLEXS", "HEL1OS"}
            },
            "candidate_instrument_event_occurrences_are_verified_windows": False,
            "verified_instrument_event_windows": 0,
        },
        "semantics_blockers": blockers,
        "phase3c_gate": {
            "status": phase3c["readiness_status"],
            "rules_total": phase3c["matrix_rule_count"],
            "rules_verified": phase3c["matrix_verified_rules"],
            "real_training_allowed": False,
            "training_started": False,
            "split_boundaries_selected": False,
            "model_checkpoint_created": False,
        },
        "code_revision": None,
        "repository": str(repo_root),
    }
    return inventory


def write_outputs(inventory: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "phase4b_real_data_inventory.json"
    blocker_path = output_dir / "phase4b_phase3c_blockers.csv"
    markdown_path = output_dir / "phase4b_real_data_coverage.md"
    json_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with blocker_path.open("w", encoding="utf-8", newline="") as output:
        fields = [
            "instrument", "product_variant", "rule", "unresolved_scientific_question",
            "relevant_instrument_files_and_metadata", "observed_local_evidence", "authoritative_documentation_required",
            "can_resolve_from_existing_local_evidence", "why_not", "next_concrete_action", "status",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for blocker in inventory["semantics_blockers"]:
            writer.writerow({
                **blocker,
                "relevant_instrument_files_and_metadata": json.dumps(
                    blocker["relevant_instrument_files_and_metadata"], ensure_ascii=False
                ),
            })

    coverage = inventory["observation_coverage"]["per_instrument"]
    events = inventory["goes_event_reconciliation"]
    lines = [
        "# Phase 4B — Real data inventory and coverage",
        "",
        f"Generated: {inventory['generated_at_utc']}",
        "",
        "## Gate status",
        "",
        f"**{inventory['phase3c_gate']['status']}** — {inventory['phase3c_gate']['rules_verified']} of {inventory['phase3c_gate']['rules_total']} rules verified. Real training is blocked; no checkpoint was created.",
        "",
        "## Candidate observation coverage (unverified)",
        "",
        "| Instrument | Product rows | Candidate observation rows | Candidate event-start occurrences | Verified coverage products |",
        "|---|---:|---:|---:|---:|",
    ]
    for instrument in ("SoLEXS", "HEL1OS"):
        row = coverage.get(instrument, {})
        lines.append(
            f"| {instrument} | {row.get('product_rows', 0)} | {row.get('observed_rows_sum_across_products', 0):,} | "
            f"{row.get('candidate_event_start_occurrences_sum_across_products', 0):,} | {row.get('verified_coverage_product_rows', 0)} |"
        )
    lines.extend(
        [
            "",
            "Candidate event-start occurrences repeat across products/versions and are not unique flares or verified label windows. Observation timestamps and coverage remain provisional until the instrument semantics and quality/GTI rules are documented.",
            "",
            "## GOES catalog reconciliation",
            "",
            f"The official NOAA report contains {events['official_records_in_candidate_interval']} records (C={events['official_class_counts'].get('C', 0)}, M={events['official_class_counts'].get('M', 0)}, X={events['official_class_counts'].get('X', 0)}); {events['official_mx_event_records']} are M/X. The local event file has {events['local_event_rows']} candidate records; {events['local_official_reconciliation_counts'].get('unique_start_and_class_match', 0)} exact start/class matches and {events['local_official_reconciliation_counts'].get('unique_start_match_same_family_magnitude_mismatch', 0)} same-family magnitude discrepancies. Negative-label completeness is not established.",
            "",
            "Eligible positive, negative, and unknown window counts are withheld. No partitions or training were started.",
            "",
            "See `phase4b_real_data_inventory.json` for provenance, exact input hashes, product inventory, per-product candidate intervals, and per-rule status. See `phase4b_phase3c_blockers.csv` for the documentation and next-action register.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\SolarFlareData"))
    parser.add_argument("--downloads-root", type=Path, default=Path(r"C:\Users\ASHUTOSH\Downloads"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.repo_root / "backend" / "reports"
    inventory = collect(args.data_root, args.downloads_root, args.repo_root)
    write_outputs(inventory, output_dir)
    print(json.dumps({"outputs": [
        str(output_dir / "phase4b_real_data_inventory.json"),
        str(output_dir / "phase4b_phase3c_blockers.csv"),
        str(output_dir / "phase4b_real_data_coverage.md"),
    ], "rules": len(inventory["semantics_blockers"]), "training_status": inventory["training_status"]}, indent=2))


if __name__ == "__main__":
    main()
