"""Generate a read-only Phase 4H data-readiness report from canonical evidence.

This script reports evidence status; it neither selects data nor creates filters,
labels, splits, models, or checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


GENERATOR = "phase4h_data_readiness_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _item(name: str, status: str, evidence: str, rationale: str) -> dict:
    return {"item": name, "status": status, "evidence": evidence, "rationale": rationale}


def build(canonical_root: Path, data_root: Path) -> dict:
    summary_path = canonical_root / "phase3c_audit_summary.json"
    register_path = canonical_root / "phase3c_document_register.json"
    if not summary_path.is_file() or not register_path.is_file():
        raise FileNotFoundError("Canonical Phase 3C summary and document register are required.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    register = json.loads(register_path.read_text(encoding="utf-8"))
    validated, mismatches = [], []
    for source in register.get("registered_source_files", []):
        path = Path(source["path"])
        actual = sha256(path) if path.is_file() else None
        row = {"path": str(path), "expected_sha256": source["sha256"], "actual_sha256": actual,
               "matches": actual == source["sha256"], "size_bytes": source.get("size_bytes")}
        validated.append(row)
        if not row["matches"]:
            mismatches.append(str(path))
    inventory = summary["observed_archive_inventory"]
    rules = {"verified": summary["matrix_verified_rules"], "partially_resolved": summary["matrix_partially_resolved_rules"],
             "unresolved_or_conflicting": summary["matrix_unresolved_or_conflicting_rules"]}
    checks = [
        _item("Canonical source-document provenance", "PASS" if validated and not mismatches else "BLOCKED",
              "Registered manual paths and SHA-256 values in canonical document register.",
              "Only hash-matching local canonical documents may support adjudication."),
        _item("Archive structural inventory", "PASS",
              "Phase 4E: 178 structurally readable FITS products, 0 inspection failures; extraction/filtering disabled.",
              "Structural readability is not scientific eligibility."),
        _item("Valid observation timestamps and time reference", "BLOCKED",
              "SoLEXS Unix-time claim is scoped; HEL1OS MJD/ISOT authority and disagreement handling remain partial.",
              "No combined observation-time policy is established."),
        _item("Applicable GTI intervals and endpoint semantics", "BLOCKED",
              "SoLEXS endpoint contract is missing; HEL1OS L1 GTI can be ignored only in the documented L1 scope.",
              "No GTI mask or endpoint convention may be applied."),
        _item("Quality and saturation handling", "BLOCKED",
              "Candidate HEL1OS HK/status fields were structurally inventoried, but no L1 bit meaning, threshold, or rejection rule is documented.",
              "Missing metadata is not evidence of clean data."),
        _item("Instrument/product/version precedence", "BLOCKED",
              "Fourteen HEL1OS V111/V211 counterpart groups have no authoritative supersession rule.",
              "Products remain separate; row counts and apparent recency are not precedence evidence."),
        _item("Calibration eligibility and physical-flux conversion", "BLOCKED",
              "SoLEXS manual identifies ARF/RMF for spectral analysis but no applicable RATE-to-irradiance conversion; HEL1OS response policy remains incomplete.",
              "Count rate is not promoted to physical flux."),
        _item("Observation coverage and gaps", "BLOCKED",
              f"Inventory includes {inventory['solexs_lightcurves']} SoLEXS and {inventory['hel1os_lightcurves']} HEL1OS light curves, but valid exposure rules are incomplete.",
              "File presence and raw rows do not establish usable coverage."),
        _item("Verified flare-event labels and provenance", "BLOCKED",
              f"Phase 3B records {summary['goes_event_reference']['verified_report_records']} reference events, but instrument interval overlap is not verified.",
              "Reference catalog provenance does not establish instrument-positive windows."),
        _item("Completeness of negative labels", "BLOCKED",
              "Canonical summary explicitly records negative-label completeness as false.",
              "Unobserved or uncatalogued intervals are not negative examples."),
        _item("Chronological split feasibility without leakage", "BLOCKED",
              "Canonical summary: no final split boundaries; verified joint windows = 0.",
              "No eligible observation/label population exists for a leakage assessment."),
    ]
    registered_names = {Path(source["path"]).name for source in register.get("registered_source_files", [])}
    authenticated_pradan_evidence = {"SoLEXS_Release_Updates.txt", "HEL1OS_ReleaseNotes.pdf", "hel1os-level2proc.zip"}
    authenticated_outcome = "AUTHENTICATED_ACCESS_SUCCEEDED" if authenticated_pradan_evidence <= registered_names else "ACCESS_RESTRICTED"
    authenticated_limitation = (
        "Authenticated normal-use acquisition succeeded for the registered Phase 4I artifacts; no access control was bypassed. "
        "The acquired artifacts still do not include the missing Level-1 SIS/endpoint/flag/precedence contracts."
        if authenticated_outcome == "AUTHENTICATED_ACCESS_SUCCEEDED" else
        "ISSDC authenticated session required; no access control was bypassed."
    )
    access_routes = [
        {"title": "PRADAN Aditya-L1 FAQ", "issuer": "ISSDC/ISTRAC", "url": "https://pradan1.issdc.gov.in/al1/faq.xhtml",
         "access_outcome": "PUBLICLY_ACCESSIBLE_REVIEWED", "supports": "Access route only", "limitations": "HEL1OS FAQ directs technical specifications to protected OtherDownloads; no field-level contract."},
        {"title": "Aditya-L1 Proposal Processing System User Guide V1.0", "issuer": "ISSDC", "url": "https://alpps.issdc.gov.in/web/extensions/Aditya-l1/resources/docs/ALPPS_UserGuide_V1.0.pdf",
         "access_outcome": "PUBLICLY_ACCESSIBLE_REVIEWED", "supports": "POC/public-product context", "limitations": "No GTI, time-field, quality-bit, response-conversion, or version-precedence contract."},
        {"title": "PRADAN Other Downloads", "issuer": "ISSDC", "url": "https://pradan.issdc.gov.in/al1/protected/miscDownloads.xhtml",
         "access_outcome": authenticated_outcome, "supports": "Official route for instrument handbook/SIS/release material", "limitations": authenticated_limitation},
    ]
    return {"generator": GENERATOR, "audit_time_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_input": {"summary": str(summary_path), "summary_sha256": sha256(summary_path),
                                "document_register": str(register_path), "document_register_sha256": sha256(register_path)},
            "canonical_readiness_status": summary["readiness_status"], "rule_counts": rules,
            "document_hash_validation": validated, "document_hash_mismatches": mismatches,
            "official_access_routes": access_routes, "readiness_checks": checks,
            "safety": {"training_or_tuning_performed": False, "evaluation_performed": False,
                       "production_checkpoint_activity": False, "production_activation": False,
                       "gti_or_quality_filters_applied": False, "version_counterparts_merged": False,
                       "negative_labels_promoted": False}}


def render(report: dict) -> str:
    lines = [f"<!-- {GENERATOR} -->", "# Phase 4H - data-readiness gate", "",
             f"- Canonical readiness: **{report['canonical_readiness_status']}**",
             f"- Rules (verified / partial / unresolved): **{report['rule_counts']['verified']} / {report['rule_counts']['partially_resolved']} / {report['rule_counts']['unresolved_or_conflicting']}**", "",
             "## Gate assessment", "", "| Item | Status | Rationale |", "|---|---|---|"]
    lines += [f"| {row['item']} | **{row['status']}** | {row['rationale']} |" for row in report["readiness_checks"]]
    lines += ["", "## Official acquisition routes", "", "| Source | Outcome | Scope |", "|---|---|---|"]
    lines += [f"| [{row['title']}]({row['url']}) | {row['access_outcome']} | {row['limitations']} |" for row in report["official_access_routes"]]
    lines += ["", "## Safety", "", "No training, tuning, evaluation, checkpoint activity, production activation, GTI/quality filtering, counterpart merge, or negative-label promotion was performed.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\SolarFlareData"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.canonical_root, args.data_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "phase4h_data_readiness.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "phase4h_data_readiness.md").write_text(render(report), encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root), "canonical_readiness": report["canonical_readiness_status"],
                      "blocked_checks": sum(row["status"] == "BLOCKED" for row in report["readiness_checks"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
