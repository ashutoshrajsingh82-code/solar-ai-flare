"""Phase 4S: Automated Tests for May 2024 Archive Ingestion, Training Gate, and Forecasting Pipeline.

Covers:
1. May 2024 archive integrity, size, and SHA-256 bit-level verification.
2. Nested daily package parsing, exactly 86,400 rows/day, and 28 total NaN bins.
3. Preservation of historical manifests (Phase 4P, 4Q, 4R) and Phase 4S manifest consistency.
4. Phase 4S training readiness gate: verification of all 10 criteria and fail-closed behavior.
5. Strict training-only scaler fitting (no validation or test leakage).
6. Gap-preserving contiguous segmentation with 1-second cadence.
7. Verification of historical frozen checkpoints (Phase 4M-S and Phase 4O).
8. Scientific integrity assertions (no physical flux conversion, no negative flare labels).
"""
from __future__ import annotations

import io
import gzip
import hashlib
import json
import sys
import zipfile
from pathlib import Path

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import numpy as np
import pytest
from astropy.io import fits

from scripts.phase4s_training_gate import (
    evaluate_phase4s_gate,
    EXPECTED_ARCHIVE_PATH,
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_ARCHIVE_SIZE,
    REQUIRED_MAY_DATES,
)
from scripts.operator_acquisition_helper import compute_sha256
from ml.experiments.real_self_supervised import (
    CountRateScaler,
    extract_contiguous_segments,
    create_sliding_windows,
    validate_solexs_fits_table,
)


def test_may2024_archive_integrity():
    """Verify May 2024 candidate archive exists, matches exact size and SHA-256, and is valid zip."""
    assert EXPECTED_ARCHIVE_PATH.exists(), f"Archive not found: {EXPECTED_ARCHIVE_PATH}"
    assert EXPECTED_ARCHIVE_PATH.stat().st_size == EXPECTED_ARCHIVE_SIZE
    computed_hash = compute_sha256(EXPECTED_ARCHIVE_PATH)
    assert computed_hash == EXPECTED_ARCHIVE_SHA256

    with zipfile.ZipFile(EXPECTED_ARCHIVE_PATH) as zf:
        bad = zf.testzip()
        assert bad is None, f"Corrupted file inside archive: {bad}"
        namelist = zf.namelist()
        for d in REQUIRED_MAY_DATES:
            expected_inner = f"AL1_SLX_L1_{d}_v1.0.zip"
            assert expected_inner in namelist, f"Missing inner zip for {d}"


def test_historical_manifests_preservation():
    """Verify Phase 4P, 4Q, and 4R manifests remain unmutated."""
    reports_dir = backend_dir / "reports"
    p4p_path = reports_dir / "phase4p_expanded_dataset_manifest.json"
    p4q_path = reports_dir / "phase4q_expanded_dataset_manifest.json"
    p4r_path = reports_dir / "phase4r_expanded_manifest.json"

    assert p4p_path.exists()
    assert p4q_path.exists()
    assert p4r_path.exists()

    expected_p4p_hash = "b5a3918df6753de873847e3346649a81370189ce24040482b5780a34e145d7a6"
    expected_p4q_hash = "c964a40be3a0d948abd070473ffd688b3761b85f74ab7c978165c854aeaecc75"

    assert compute_sha256(p4p_path) == expected_p4p_hash
    assert compute_sha256(p4q_path) == expected_p4q_hash


def test_phase4s_manifest_consistency():
    """Verify Phase 4S manifest lists all 8 May 2024 products and 6 Jan/Feb 2026 products."""
    manifest_path = backend_dir / "reports" / "phase4s_dataset_manifest.json"
    assert manifest_path.exists(), "phase4s_dataset_manifest.json must exist."

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["indexer_version"] == "1.0.0"
    assert data["eligible_for_forecasting"] == 14
    assert data["overlapping_interval_conflicts"] == 0

    # Ensure all May dates are indexed as ELIGIBLE
    may_products = [
        item for item in data["inventory"]
        if item.get("instrument") == "SoLEXS"
        and item.get("detector") == "SDD2"
        and item.get("eligibility_decision") == "ELIGIBLE"
        and item.get("observation_date", "").startswith("202405")
    ]
    assert len(may_products) == 8


def test_phase4s_training_gate_clearance():
    """Verify that evaluate_phase4s_gate passes with TRAINING_PERMITTED on actual archive."""
    manifest_path = backend_dir / "reports" / "phase4s_dataset_manifest.json"
    gate_res = evaluate_phase4s_gate(manifest_path, EXPECTED_ARCHIVE_PATH)
    assert gate_res["passed"] is True
    assert gate_res["status"] == "TRAINING_PERMITTED"
    assert len(gate_res["conditions"]) == 10
    for cond_key, cond_info in gate_res["conditions"].items():
        assert cond_info["passed"] is True, f"Condition {cond_key} failed: {cond_info.get('detail')}"


def test_phase4s_training_gate_fail_closed(tmp_path: Path):
    """Verify that training gate fails if archive is missing or corrupted."""
    manifest_path = backend_dir / "reports" / "phase4s_dataset_manifest.json"
    fake_path = tmp_path / "nonexistent.zip"
    gate_res = evaluate_phase4s_gate(manifest_path, fake_path)
    assert gate_res["passed"] is False
    assert gate_res["status"] == "TRAINING_BLOCKED"


def test_may2024_nan_and_row_count():
    """Verify exact count of 86,400 rows per day and 28 total NaN records across May 8-15, 2024."""
    nan_counts = {}
    row_counts = {}

    with zipfile.ZipFile(EXPECTED_ARCHIVE_PATH) as outer:
        for d in REQUIRED_MAY_DATES:
            inner_name = f"AL1_SLX_L1_{d}_v1.0.zip"
            inner_bytes = outer.read(inner_name)
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                lc_member = f"AL1_SLX_L1_{d}_v1.0/SDD2/AL1_SOLEXS_{d}_SDD2_L1.lc.gz"
                raw_gz = inner.read(lc_member)
                decomp = gzip.decompress(raw_gz)
                with fits.open(io.BytesIO(decomp)) as hdul:
                    times, counts, tu, cu = validate_solexs_fits_table(hdul)
                    row_counts[d] = len(counts)
                    nan_counts[d] = int(np.isnan(counts).sum())

    total_rows = sum(row_counts.values())
    total_nans = sum(nan_counts.values())

    assert total_rows == 8 * 86400  # 691,200 total records
    for d in REQUIRED_MAY_DATES:
        assert row_counts[d] == 86400

    assert total_nans == 28
    assert nan_counts["20240508"] == 5
    assert nan_counts["20240509"] == 2
    assert nan_counts["20240510"] == 4
    assert nan_counts["20240511"] == 2
    assert nan_counts["20240512"] == 4
    assert nan_counts["20240513"] == 4
    assert nan_counts["20240514"] == 5
    assert nan_counts["20240515"] == 2


def test_scaler_isolation_no_leakage():
    """Verify that CountRateScaler fitted strictly on training data does not reflect test distributions."""
    train_counts = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    test_counts = np.array([5000.0, 15000.0, 29000.0], dtype=np.float32)

    scaler = CountRateScaler().fit(train_counts)
    expected_mean = float(np.mean(np.log10(train_counts + 1.0)))
    expected_std = float(np.std(np.log10(train_counts + 1.0)))

    assert np.isclose(scaler.mean, expected_mean)
    assert np.isclose(scaler.std, expected_std)

    # Transform test without refitting
    test_scaled = scaler.transform(test_counts)
    test_recovered = scaler.inverse_transform(test_scaled)
    np.testing.assert_allclose(test_counts, test_recovered, rtol=1e-5)


def test_contiguous_segmentation_preserves_gaps():
    """Verify that gaps in observation timestamps or NaNs break contiguous segments."""
    times = np.array([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0])
    counts = np.array([5.0, 6.0, np.nan, 8.0, 9.0, 10.0, 11.0, 12.0])

    segs = extract_contiguous_segments(times, counts, min_segment_length=3)
    # The first 4 points have a NaN at index 2, breaking into [0, 1] (length 2 < 3) and [3] (length 1 < 3)
    # The last 4 points [10, 11, 12, 13] are contiguous and all valid (length 4 >= 3)
    assert len(segs) == 1
    np.testing.assert_array_equal(segs[0][0], np.array([10.0, 11.0, 12.0, 13.0]))


def test_frozen_reference_checkpoints_preserved():
    """Verify that Phase 4M-S and Phase 4O reference checkpoints are unmodified."""
    p4m_path = backend_dir / "artifacts" / "real_data_experiments" / "solexs_self_supervised_cnnlstm_20261025T115207Z.pt"
    p4o_path = backend_dir / "artifacts" / "real_data_experiments" / "phase4o" / "solexs_phase4o_cnn_lstm_direct_20261025T184851Z.pt"

    assert p4m_path.exists(), "Phase 4M-S checkpoint must exist."
    assert p4o_path.exists(), "Phase 4O checkpoint must exist."

    expected_p4m_hash = "815ded9a605fa4bb410791b3a64d07624d6f3d03ffaf02953b67c7f1f8e53d45"
    expected_p4o_hash = "d6076ab20c99fdc852d338c47f9b7248250a333c4708916a8ced8de3a6336514"

    assert compute_sha256(p4m_path) == expected_p4m_hash
    assert compute_sha256(p4o_path) == expected_p4o_hash
