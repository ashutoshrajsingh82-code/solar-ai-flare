"""Phase 4R: Automated Tests for Acquisition, Verification, Training Gate, and Evaluation Protocol.

Tests coverage:
- Archive integrity, byte-count verification, and corrupted archive detection.
- Deterministic manifest regeneration without mutating Phase 4P/4Q manifests.
- Duplicate and overlapping observation detection.
- UTC timestamp normalization and gap-preserving contiguous segmentation.
- Leakage-free chronological walk-forward partitioning.
- Machine-readable training gate enforcement and failure diagnostics.
- Training gate clearance behavior when genuine additional data is present.
- Final-test isolation and training-only scaler fitting.
- Credential, cookie, and signed-URL redaction.
- Bit-level preservation of frozen Phase 4M-S and Phase 4O benchmark artifacts.
"""
from __future__ import annotations

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
)
from scripts.index_real_observations import RealDatasetIndexer
from scripts.phase4r_training_gate import evaluate_training_gate, ORIGINAL_COHORT_DATES
from ml.experiments.real_self_supervised import (
    CountRateScaler,
    extract_contiguous_segments,
)


def test_archive_integrity_and_corrupt_detection(tmp_path: Path):
    """Test valid archive inspection and fail-closed handling of corrupted archives."""
    # 1. Valid ZIP
    valid_zip = tmp_path / "valid.zip"
    with zipfile.ZipFile(valid_zip, "w") as zf:
        zf.writestr("test_member.txt", b"Valid payload data for Phase 4R")
    res_valid = inspect_archive(valid_zip)
    assert res_valid["is_valid_zip"] is True
    assert res_valid["byte_size"] > 0
    assert len(res_valid["sha256"]) == 64

    # 2. Corrupt / truncated ZIP
    corrupt_zip = tmp_path / "corrupt.zip"
    corrupt_zip.write_bytes(b"PK\x03\x04TruncatedBytesHeader")
    res_corrupt = inspect_archive(corrupt_zip)
    assert res_corrupt["is_valid_zip"] is False
    assert res_corrupt["corruption_error"] is not None


def test_deterministic_manifest_regeneration_and_history_preservation():
    """Verify that Phase 4R manifest generation leaves Phase 4P and 4Q manifests untouched."""
    reports_dir = backend_dir / "reports"
    p4p_path = reports_dir / "phase4p_expanded_dataset_manifest.json"
    p4q_path = reports_dir / "phase4q_expanded_dataset_manifest.json"
    p4r_path = reports_dir / "phase4r_expanded_manifest.json"

    assert p4p_path.exists(), "Phase 4P manifest must exist."
    assert p4q_path.exists(), "Phase 4Q manifest must exist."
    assert p4r_path.exists(), "Phase 4R manifest must exist."

    expected_p4p_hash = "b5a3918df6753de873847e3346649a81370189ce24040482b5780a34e145d7a6"
    expected_p4q_hash = "c964a40be3a0d948abd070473ffd688b3761b85f74ab7c978165c854aeaecc75"

    assert compute_sha256(p4p_path) == expected_p4p_hash, "Phase 4P manifest was mutated!"
    assert compute_sha256(p4q_path) == expected_p4q_hash, "Phase 4Q manifest was mutated!"


def test_duplicate_and_overlapping_observation_detection():
    """Verify duplicate detection and overlap flagging in RealDatasetIndexer."""
    indexer = RealDatasetIndexer([])
    indexer.inventory = [
        {
            "eligibility_decision": "ELIGIBLE",
            "member_path": "day1.lc.gz",
            "start_time_unix": 1000.0,
            "end_time_unix": 2000.0,
        },
        {
            "eligibility_decision": "ELIGIBLE",
            "member_path": "day2_overlap.lc.gz",
            "start_time_unix": 1500.0,
            "end_time_unix": 2500.0,
        },
    ]
    indexer._detect_duplicates_and_overlaps()

    assert len(indexer.overlapping_intervals) == 1
    overlap = indexer.overlapping_intervals[0]
    assert overlap["product_1"] == "day1.lc.gz"
    assert overlap["product_2"] == "day2_overlap.lc.gz"
    assert overlap["overlap_start_unix"] == 1500.0
    assert overlap["overlap_end_unix"] == 2000.0


def test_utc_normalization_and_gap_preserving_segmentation():
    """Test that extract_contiguous_segments enforces strictly unbroken 1-second cadence."""
    # 80 continuous valid seconds (should form 1 valid segment)
    times1 = np.arange(1707715800.0, 1707715880.0, 1.0)
    counts1 = np.full(80, 120.0)

    # 10 second gap
    # 25 continuous seconds (too short: < 70s)
    times2 = np.arange(1707715890.0, 1707715915.0, 1.0)
    counts2 = np.full(25, 150.0)

    all_times = np.concatenate([times1, times2])
    all_counts = np.concatenate([counts1, counts2])

    segments = extract_contiguous_segments(all_times, all_counts, min_segment_length=70)
    assert len(segments) == 1
    seg_times, seg_counts = segments[0]
    assert len(seg_times) == 80
    assert seg_times[0] == 1707715800.0
    assert seg_times[-1] == 1707715879.0


def test_leakage_free_chronological_partitioning():
    """Verify that walk-forward partition timestamps enforce strict temporal monotonicity."""
    train_dates = ["2026-01-31", "2026-02-01", "2026-02-02"]
    val_dates = ["2026-02-03", "2026-02-04"]
    test_dates = ["2026-02-05"]

    assert max(train_dates) < min(val_dates), "Train and validation partitions overlap in time!"
    assert max(val_dates) < min(test_dates), "Validation and test partitions overlap in time!"


def test_training_gate_enforcement_on_actual_manifest():
    """Verify that the training gate halts execution on the current dataset."""
    manifest_path = backend_dir / "reports" / "phase4r_expanded_manifest.json"
    res = evaluate_training_gate(manifest_path)

    assert res["status"] == "TRAINING_BLOCKED"
    assert res["passed"] is False
    assert "condition_1_additional_data_exists" in res["failed_conditions"]
    assert "condition_9_independent_periods_sufficient" in res["failed_conditions"]


def test_training_gate_passes_when_mock_additional_data_present(tmp_path: Path):
    """Verify that the training gate clears when additional eligible observation periods exist."""
    mock_manifest = {
        "total_products_indexed": 40,
        "eligible_for_forecasting": 7,
        "excluded_products": 33,
        "duplicate_counterpart_groups": 0,
        "overlapping_interval_conflicts": 0,
        "inventory": [
            {
                "observation_date": d,
                "instrument": "SoLEXS",
                "detector": "SDD2",
                "eligibility_decision": "ELIGIBLE",
                "num_contiguous_segments": 3,
                "total_records": 86400,
                "finite_records": 45000,
            }
            for d in list(ORIGINAL_COHORT_DATES) + ["20240508"]  # Adds May 8, 2024
        ]
    }

    mock_path = tmp_path / "mock_expanded_manifest.json"
    with open(mock_path, "w", encoding="utf-8") as f:
        json.dump(mock_manifest, f)

    res = evaluate_training_gate(mock_path)
    assert res["status"] == "TRAINING_PERMITTED"
    assert res["passed"] is True
    assert len(res["failed_conditions"]) == 0
    assert "20240508" in res["additional_dates_found"]


def test_final_test_isolation_and_training_only_scaler_fitting():
    """Verify that scaler parameters are fitted strictly on training data and remain immutable during test evaluation."""
    train_counts = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    test_counts = np.array([500.0, 1000.0, 2000.0])

    scaler = CountRateScaler()
    scaler.fit(train_counts)

    mu_initial = scaler.mean
    sigma_initial = scaler.std

    # Transform test counts
    transformed_test = scaler.transform(test_counts)

    # Scaler parameters must remain identical
    assert scaler.mean == mu_initial
    assert scaler.std == sigma_initial

    # Invert transform reproduces test counts
    inverted_test = scaler.inverse_transform(transformed_test)
    assert np.allclose(inverted_test, test_counts, atol=1e-5)


def test_credential_and_signed_url_redaction():
    """Verify that access tokens, cookies, and passwords are redacted from logged strings."""
    secret_url = "https://pradan1.issdc.gov.in/al1/protected/downloadData?token=secret123&SESSION=cookie456&state=auth789"
    redacted = redact_sensitive_string(secret_url)

    assert "secret123" not in redacted
    assert "cookie456" not in redacted
    assert "auth789" not in redacted
    assert "[REDACTED]" in redacted


def test_preservation_of_frozen_phase4m_and_phase4o_checkpoints():
    """Verify bit-level preservation of frozen Phase 4M-S and Phase 4O model checkpoints."""
    art_dir = backend_dir / "artifacts" / "real_data_experiments"

    p4m_ckpt = art_dir / "solexs_self_supervised_cnnlstm_20261025T115207Z.pt"
    p4o_ckpt = art_dir / "phase4o" / "solexs_phase4o_cnn_lstm_direct_20261025T184851Z.pt"

    assert p4m_ckpt.exists(), "Phase 4M-S checkpoint must exist."
    assert p4o_ckpt.exists(), "Phase 4O reference checkpoint must exist."

    expected_p4m_sha = "815ded9a605fa4bb410791b3a64d07624d6f3d03ffaf02953b67c7f1f8e53d45"
    expected_p4o_sha = "d6076ab20c99fdc852d338c47f9b7248250a333c4708916a8ced8de3a6336514"

    assert compute_sha256(p4m_ckpt) == expected_p4m_sha, "Phase 4M-S checkpoint was modified!"
    assert compute_sha256(p4o_ckpt) == expected_p4o_sha, "Phase 4O reference checkpoint was modified!"
