import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "phase4h_data_readiness.py"
SPEC = importlib.util.spec_from_file_location("phase4h_data_readiness", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_readiness_report_blocks_missing_semantic_prerequisites_and_validates_hashes(tmp_path):
    canonical = tmp_path / "canonical"; canonical.mkdir()
    source = tmp_path / "manual.pdf"; source.write_bytes(b"manual")
    (canonical / "phase3c_audit_summary.json").write_text(json.dumps({
        "readiness_status": "BLOCKED_SCIENTIFIC_SEMANTICS_UNRESOLVED",
        "matrix_verified_rules": 7, "matrix_partially_resolved_rules": 6, "matrix_unresolved_or_conflicting_rules": 1,
        "observed_archive_inventory": {"solexs_lightcurves": 6, "hel1os_lightcurves": 44},
        "goes_event_reference": {"verified_report_records": 123},
    }), encoding="utf-8")
    (canonical / "phase3c_document_register.json").write_text(json.dumps({"registered_source_files": [{
        "path": str(source), "sha256": module.sha256(source), "size_bytes": source.stat().st_size}]}), encoding="utf-8")
    report = module.build(canonical, tmp_path)
    assert report["document_hash_mismatches"] == []
    assert report["rule_counts"] == {"verified": 7, "partially_resolved": 6, "unresolved_or_conflicting": 1}
    assert all(row["status"] in {"PASS", "BLOCKED", "NOT_ASSESSED"} for row in report["readiness_checks"])
    assert any(row["item"] == "Completeness of negative labels" and row["status"] == "BLOCKED" for row in report["readiness_checks"])


def test_readiness_report_fails_document_provenance_when_hash_changes(tmp_path):
    canonical = tmp_path / "canonical"; canonical.mkdir()
    source = tmp_path / "manual.pdf"; source.write_bytes(b"changed")
    (canonical / "phase3c_audit_summary.json").write_text(json.dumps({
        "readiness_status": "BLOCKED_SCIENTIFIC_SEMANTICS_UNRESOLVED", "matrix_verified_rules": 7,
        "matrix_partially_resolved_rules": 6, "matrix_unresolved_or_conflicting_rules": 1,
        "observed_archive_inventory": {"solexs_lightcurves": 6, "hel1os_lightcurves": 44},
        "goes_event_reference": {"verified_report_records": 123}}), encoding="utf-8")
    (canonical / "phase3c_document_register.json").write_text(json.dumps({"registered_source_files": [{
        "path": str(source), "sha256": "0" * 64, "size_bytes": source.stat().st_size}]}), encoding="utf-8")
    report = module.build(canonical, tmp_path)
    provenance = next(row for row in report["readiness_checks"] if row["item"] == "Canonical source-document provenance")
    assert provenance["status"] == "BLOCKED"


def test_readiness_reports_authenticated_pradan_acquisition_only_when_registered(tmp_path):
    canonical = tmp_path / "canonical"; canonical.mkdir()
    paths = []
    for name in ("SoLEXS_Release_Updates.txt", "HEL1OS_ReleaseNotes.pdf", "hel1os-level2proc.zip"):
        source = tmp_path / name
        source.write_bytes(name.encode("utf-8"))
        paths.append({"path": str(source), "sha256": module.sha256(source), "size_bytes": source.stat().st_size})
    (canonical / "phase3c_audit_summary.json").write_text(json.dumps({
        "readiness_status": "BLOCKED_SCIENTIFIC_SEMANTICS_UNRESOLVED", "matrix_verified_rules": 7,
        "matrix_partially_resolved_rules": 6, "matrix_unresolved_or_conflicting_rules": 1,
        "observed_archive_inventory": {"solexs_lightcurves": 6, "hel1os_lightcurves": 44},
        "goes_event_reference": {"verified_report_records": 123}}), encoding="utf-8")
    (canonical / "phase3c_document_register.json").write_text(
        json.dumps({"registered_source_files": paths}), encoding="utf-8")

    report = module.build(canonical, tmp_path)

    pradan = next(row for row in report["official_access_routes"] if row["title"] == "PRADAN Other Downloads")
    assert pradan["access_outcome"] == "AUTHENTICATED_ACCESS_SUCCEEDED"
