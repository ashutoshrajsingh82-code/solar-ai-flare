import csv
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "phase3c_semantics_audit.py"
SPEC = importlib.util.spec_from_file_location("phase3c_semantics_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_instrument_rules_without_registered_documents_remain_unverified():
    matrix = audit.semantics_matrix()
    assert len(matrix) >= 10
    assert set(matrix.verification_status) == {"UNVERIFIED"}
    solexs_counts = matrix[matrix.field_or_keyword == "RATE.COUNTS"].iloc[0]
    assert solexs_counts.units == "TUNIT absent"
    assert "count/bin vs rate unresolved" in solexs_counts.proposed_interpretation
    assert "Registered instrument-specific" in solexs_counts.notes_unresolved_conflicts


def test_missing_portal_documents_are_registered_as_access_blocked(tmp_path):
    report_root = tmp_path / "reports"
    report_root.mkdir()
    (report_root / "phase3b_document_register.json").write_text(
        json.dumps({"sources": []}), encoding="utf-8")

    register = audit._documents(tmp_path, report_root)
    assert register["portal_attempt"]["downloads_performed"] == []
    assert "login form" in register["portal_attempt"]["observed_result"]
    missing = [d for d in register["documents"] if d["status"] == "NOT_RETRIEVED_LOGIN_REQUIRED"]
    assert len(missing) >= 7
    assert all(d["sha256"] is None and d["version"] is None for d in missing)


def test_document_registration_hashes_each_canonical_file(tmp_path):
    report_root = tmp_path / "reports"
    report_root.mkdir()
    (report_root / "phase3b_document_register.json").write_text(
        json.dumps({"sources": []}), encoding="utf-8")
    source = tmp_path / "source_docs" / "isro" / "hel1os" / "HEL1OS_ReleaseNotes.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"official-document-bytes")

    register = audit._documents(tmp_path, report_root)

    assert register["registered_source_files"] == [{
        "path": str(source),
        "sha256": audit.sha256(source),
        "size_bytes": len(b"official-document-bytes"),
    }]


def test_unknown_instrument_semantics_never_become_valid_or_negative_windows(tmp_path):
    report_root = tmp_path / "reports"
    report_root.mkdir()
    (report_root / "phase3b_goes_quality_audit.json").write_text(
        json.dumps({"daily_channel_counts": []}), encoding="utf-8")
    phase3a = {
        "solexs": {"products": [{"raw_start_utc": "2026-01-01T00:00:00Z", "detector": "SDD2",
                                    "member": "sample.lc.gz", "nested_archive": "sample_v1.0.zip",
                                    "raw_rows": 4, "finite_nonnegative_candidate_rows": 3,
                                    "GTI_companions": [{"rows": 2}]}]},
        "hel1os": {"products": [{"raw_start_utc": "2026-01-01T00:00:00Z", "extension": "CZT1_LC_BAND_20.00KEV_TO_40.00KEV",
                                    "member": "lightcurve.fits", "processing_version": "V111",
                                    "raw_rows": 4, "candidate_finite_nonnegative_rate_error_rows": 3,
                                    "GTI_companions": ["gti.fits"]}]},
    }
    rows = audit._coverage(phase3a, report_root / "phase3b_audit_summary.json")
    aditya = rows[rows.instrument.isin(["SoLEXS", "HEL1OS"])]
    assert aditya.valid_after_GTI.isna().all()
    assert aditya.verified_windows.eq(0).all()
    assert aditya.positive_windows.isna().all()
    assert aditya.negative_windows.isna().all()
    assert aditya.status.eq("BLOCKED_DOCUMENTATION_UNAVAILABLE").all()


def test_report_writer_refuses_to_replace_unrelated_existing_file(tmp_path):
    target = tmp_path / "phase3c_readiness.md"
    target.write_text("user notes", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        audit._write(target, "replacement", "md")
    assert target.read_text(encoding="utf-8") == "user notes"


def _reviewed_adjudication(tmp_path, status="PARTIALLY_RESOLVED", digest="a" * 64):
    rows = audit.semantics_matrix().to_dict("records")
    for row in rows:
        row.update({
            "verification_status": status,
            "document_filename_version_section_page": "Manual.pdf p.1 §1",
            "phase4e_supported_interpretation": "Reviewed, scoped evidence.",
            "phase4e_unresolved_limits": "No filtering decision follows.",
            "evidence_document_path": "C:/original/Manual.pdf",
            "evidence_sha256": digest,
        })
    path = tmp_path / "reviewed.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    return path


def test_reviewed_adjudication_requires_registered_hash_and_preserves_provenance(tmp_path):
    digest = "a" * 64
    source = tmp_path / "source_docs" / "isro" / "solexs" / "Manual.pdf"
    source.parent.mkdir(parents=True); source.write_bytes(b"manual")
    reviewed, trail = audit._apply_adjudication(
        audit.semantics_matrix(), {"registered_source_files": [{"path": str(source), "sha256": digest}]},
        _reviewed_adjudication(tmp_path, digest=digest))
    assert trail["status"] == "APPLIED"
    assert trail["rows_applied"] == 14
    assert set(reviewed.verification_status) == {"PARTIALLY_RESOLVED"}
    assert set(reviewed.canonical_evidence_path) == {str(source)}


def test_adjudication_rejects_unregistered_hash_and_records_conflicts(tmp_path):
    with pytest.raises(ValueError, match="unregistered document hash"):
        audit._apply_adjudication(audit.semantics_matrix(), {"registered_source_files": []},
                                  _reviewed_adjudication(tmp_path))
    digest = "b" * 64
    source = tmp_path / "source_docs" / "isro" / "hel1os" / "Manual.pdf"
    source.parent.mkdir(parents=True); source.write_bytes(b"manual")
    _, trail = audit._apply_adjudication(
        audit.semantics_matrix(), {"registered_source_files": [{"path": str(source), "sha256": digest}]},
        _reviewed_adjudication(tmp_path, status="CONFLICTING_EVIDENCE", digest=digest))
    assert len(trail["conflicting_rules"]) == 14
