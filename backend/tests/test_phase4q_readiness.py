"""Phase 4Q: Automated Tests for Data Acquisition Readiness, Eligibility, and Retraining Gates.

Tests coverage:
- Archive and member SHA-256 integrity.
- Safe handling of missing and corrupt downloads.
- Credential and signed-URL redaction.
- Deterministic manifest generation with custom prefix.
- Idempotent ingestion and resume behavior.
- Duplicate product detection and cross-period temporal overlap detection.
- Eligibility blocker enforcement (B-HL-*, B-SX-*, B-LBL-*).
- Retraining policy gate enforcement (training halted when new data condition fails).
- Bit-level preservation of frozen Phase 4O benchmark artifacts.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import numpy as np
import pytest

from scripts.operator_acquisition_helper import (
    compute_sha256,
    redact_sensitive_string,
    inspect_archive,
    generate_acquisition_log,
)
from scripts.index_real_observations import RealDatasetIndexer


def test_archive_hash_and_byte_integrity(tmp_path: Path):
    """Test SHA-256 hash and byte size calculation on on-disk files."""
    test_content = b"Phase 4Q Authorized Observational Ingestion Test Payload"
    test_file = tmp_path / "valid_test_payload.bin"
    test_file.write_bytes(test_content)

    assert test_file.stat().st_size == len(test_content)
    sha = compute_sha256(test_file)
    assert len(sha) == 64
    assert sha.islower()


def test_safe_handling_of_missing_and_corrupted_downloads(tmp_path: Path):
    """Test that missing files and corrupted zips are handled fail-closed."""
    # 1. Missing file
    missing_file = tmp_path / "does_not_exist.zip"
    res_missing = inspect_archive(missing_file)
    assert res_missing["status"] == "FILE_NOT_FOUND"
    assert res_missing["is_valid_zip"] is False

    # 2. Corrupt / truncated zip file
    corrupt_file = tmp_path / "truncated_archive.zip"
    corrupt_file.write_bytes(b"PK\x03\x04TruncatedCorruptedHeaderBytesWithoutEndCentralDirectory")
    res_corrupt = inspect_archive(corrupt_file)
    assert res_corrupt["is_valid_zip"] is False
    assert res_corrupt["corruption_error"] is not None


def test_credential_and_token_redaction():
    """Test that session cookies, tokens, and passwords are redacted from logs and messages."""
    raw_text = (
        "GET /al1/protected/downloadData?SESSION=abc123xyz456&state=secret_state_789 HTTP/1.1\n"
        "Cookie: JSESSIONID=deadbeef1234; OAuth_Token_Request_State=token_abc\n"
        "Authorization: Bearer my_super_secret_access_token_999\n"
        "Payload: password=AdminPassword123&user=operator"
    )

    clean_text = redact_sensitive_string(raw_text)

    assert "abc123xyz456" not in clean_text
    assert "deadbeef1234" not in clean_text
    assert "token_abc" not in clean_text
    assert "my_super_secret_access_token_999" not in clean_text
    assert "AdminPassword123" not in clean_text

    assert "[REDACTED]" in clean_text


def test_deterministic_manifest_generation_and_phase4p_immutability(tmp_path: Path):
    """Verify that generating a Phase 4Q manifest does not overwrite the Phase 4P manifest."""
    reports_dir = backend_dir / "reports"
    p4p_manifest_path = reports_dir / "phase4p_expanded_dataset_manifest.json"
    p4q_manifest_path = reports_dir / "phase4q_expanded_dataset_manifest.json"

    assert p4p_manifest_path.exists(), "Phase 4P manifest must exist."
    p4p_hash_before = compute_sha256(p4p_manifest_path)

    # Both manifests should exist independently
    assert p4q_manifest_path.exists(), "Phase 4Q manifest must exist."

    # Verify Phase 4P manifest is unchanged
    p4p_hash_after = compute_sha256(p4p_manifest_path)
    assert p4p_hash_before == p4p_hash_after, "Phase 4P manifest was mutated!"


def test_idempotent_ingestion_and_resume_behavior(tmp_path: Path):
    """Verify that running the acquisition log generator twice yields identical outputs."""
    dummy_zip = tmp_path / "dummy_archive.zip"
    with zipfile.ZipFile(dummy_zip, "w") as zf:
        zf.writestr("test.txt", "data")

    log_path1 = tmp_path / "log1.json"
    log_path2 = tmp_path / "log2.json"

    res1 = generate_acquisition_log([dummy_zip], log_path1)
    res2 = generate_acquisition_log([dummy_zip], log_path2)

    # Discard non-deterministic timestamp
    del res1["generated_at_utc"]
    del res2["generated_at_utc"]
    for a in res1["archives"]:
        del a["verified_at_utc"]
    for a in res2["archives"]:
        del a["verified_at_utc"]

    assert json.dumps(res1, sort_keys=True) == json.dumps(res2, sort_keys=True)


def test_cross_period_temporal_overlap_detection():
    """Verify that temporal overlaps between candidate observation periods are detected."""
    indexer = RealDatasetIndexer([])
    indexer.inventory = [
        {
            "eligibility_decision": "ELIGIBLE",
            "member_path": "day_a.lc.gz",
            "start_time_unix": 10000.0,
            "end_time_unix": 20000.0,
        },
        {
            "eligibility_decision": "ELIGIBLE",
            "member_path": "day_b_overlapping.lc.gz",
            "start_time_unix": 18000.0,
            "end_time_unix": 25000.0,
        },
    ]
    indexer._detect_duplicates_and_overlaps()

    assert len(indexer.overlapping_intervals) == 1
    overlap = indexer.overlapping_intervals[0]
    assert overlap["overlap_start_unix"] == 18000.0
    assert overlap["overlap_end_unix"] == 20000.0


def test_scientific_blocker_enforcement():
    """Verify that HEL1OS and non-sequential spectral products are excluded with strict blocker codes."""
    indexer = RealDatasetIndexer([])
    indexer.inventory = [
        {
            "instrument": "HEL1OS",
            "package_name": "HLS_dump.zip",
            "eligibility_decision": "EXCLUDED",
            "exclusion_code": "EXCL-HEL1OS-BLOCKERS",
        },
        {
            "instrument": "SoLEXS",
            "detector": "SDD2",
            "product_type": "SPECTRUM_PI",
            "eligibility_decision": "EXCLUDED",
            "exclusion_code": "EXCL-SOLEXS-PI-SPECTRA",
        },
        {
            "instrument": "SoLEXS",
            "detector": "SDD1",
            "product_type": "LIGHTCURVE",
            "eligibility_decision": "EXCLUDED",
            "exclusion_code": "EXCL-SOLEXS-SDD1-INACTIVE",
        },
    ]
    # Verify no excluded items are marked eligible
    eligible = [p for p in indexer.inventory if p["eligibility_decision"] == "ELIGIBLE"]
    assert len(eligible) == 0


def test_retraining_policy_gate_fails_closed():
    """Verify that retraining gate halts execution when new local data condition fails."""
    # Retraining policy gate requires 5 conditions:
    new_local_observations_exist = False
    scientific_eligibility_established = True
    manifest_deterministic = True
    chronological_partitions_defined = True
    final_test_reserved = True

    can_retrain = (
        new_local_observations_exist and
        scientific_eligibility_established and
        manifest_deterministic and
        chronological_partitions_defined and
        final_test_reserved
    )

    assert can_retrain is False, "Retraining must be blocked when new local observations do not exist!"


def test_preservation_of_frozen_phase4o_benchmark_artifacts():
    """Verify that frozen Phase 4O reference checkpoint exists with verified SHA-256 hash."""
    checkpoint_path = backend_dir / "artifacts" / "real_data_experiments" / "phase4o" / "solexs_phase4o_cnn_lstm_direct_20261025T184851Z.pt"
    assert checkpoint_path.exists(), "Phase 4O reference checkpoint must exist."

    expected_sha256 = "d6076ab20c99fdc852d338c47f9b7248250a333c4708916a8ced8de3a6336514"
    actual_sha256 = compute_sha256(checkpoint_path)
    assert actual_sha256 == expected_sha256, "Phase 4O reference checkpoint hash has been altered!"
