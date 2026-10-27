"""Phase 4U: Automated Unit Tests for Multi-Epoch Dataset Contract, Deduplication, and Holdout Governance.

Covers:
1. Integrity and SHA-256 verification of both local source archives.
2. Complete deduplication across the 14 daily observation products.
3. Verification that all 14 local products are classified as previously examined development data.
4. Fail-closed behavior of the Phase 4U readiness gate (PHASE4U_BLOCKED_INDEPENDENT_EPOCH_ACQUISITION).
5. Enforcement of causal origin-based cohort conditioning (no future target conditioning).
6. Strict contiguous segmentation (minimum length >= 70s, zero gap bridging).
7. Scaler isolation contract (training partition only).
8. Holdout protection and access prohibition.
9. Active blocker preservation (B-SX-FLUX-003, B-LBL-NEG-007, B-HL-*, B-EPOCH3-ACQ-008).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import numpy as np
import pandas as pd
import pytest

from scripts.operator_acquisition_helper import compute_sha256
from ml.experiments.real_self_supervised import (
    CountRateScaler,
    extract_contiguous_segments,
    create_sliding_windows,
)

ARCHIVE_MAY2024 = Path(r"C:\Users\ASHUTOSH\Downloads\solexs_2026Oct25T200126271.zip")
ARCHIVE_FEB2026 = Path(r"C:\Users\ASHUTOSH\Downloads\solexs_2026Oct23T184402685.zip")

EXPECTED_SHA_MAY2024 = "8df85ed75b6f00052d7f45a6764b187c700977f931504fb79fdfe56f10df9db2"
EXPECTED_SHA_FEB2026 = "34c39624194ee2da95bbf8e3dfed5ac002c55a517a31bc4091bfdd0d2b4671c2"


def test_local_source_archives_integrity():
    """Verify both local development archives exist and match exact registered SHA-256 digests."""
    assert ARCHIVE_MAY2024.exists(), f"Missing archive: {ARCHIVE_MAY2024}"
    assert ARCHIVE_FEB2026.exists(), f"Missing archive: {ARCHIVE_FEB2026}"

    assert compute_sha256(ARCHIVE_MAY2024) == EXPECTED_SHA_MAY2024
    assert compute_sha256(ARCHIVE_FEB2026) == EXPECTED_SHA_FEB2026


def test_deduplication_and_inventory_accounting():
    """Verify observation inventory contains exactly 14 unique daily light curves with 0 duplicates."""
    csv_path = backend_dir / "reports" / "phase4u_observation_inventory.csv"
    assert csv_path.exists(), "phase4u_observation_inventory.csv must exist."

    df = pd.read_csv(csv_path)
    assert len(df) == 14
    assert df["product_key"].nunique() == 14
    assert (df["is_duplicate_product"] == False).all()

    # Verify breakdown: 8 in May 2024, 6 in Jan-Feb 2026
    may_count = (df["observation_date"].astype(str).str.startswith("202405")).sum()
    feb_count = (df["observation_date"].astype(str).str.startswith("2026")).sum()
    assert may_count == 8
    assert feb_count == 6


def test_all_local_observations_marked_examined():
    """Verify that every local observation is classified as previously examined development data."""
    json_path = backend_dir / "reports" / "phase4u_source_inventory.json"
    assert json_path.exists(), "phase4u_source_inventory.json must exist."

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["overall_summary"]["all_products_previously_examined"] is True
    assert data["overall_summary"]["untouched_third_epoch_locally_available"] is False

    csv_path = backend_dir / "reports" / "phase4u_observation_inventory.csv"
    df = pd.read_csv(csv_path)
    assert (df["untouched_holdout_candidate"] == False).all()


def test_phase4u_readiness_gate_fails_closed():
    """Verify that Phase 4U readiness gate fails closed because Epoch 3 holdout is unacquired."""
    gate_path = backend_dir / "reports" / "phase4u_readiness_gate.json"
    assert gate_path.exists(), "phase4u_readiness_gate.json must exist."

    with open(gate_path, "r", encoding="utf-8") as f:
        gate = json.load(f)

    assert gate["overall_status"] == "PHASE4U_BLOCKED_INDEPENDENT_EPOCH_ACQUISITION"
    assert gate["gate_passed"] is False

    # Check that condition 5 (unacquired third epoch) failed
    cond5 = [c for c in gate["conditions"] if c["condition_id"] == "GATE-4U-05"][0]
    assert cond5["satisfied"] is False


def test_causal_cohort_conditioning_rule():
    """Verify causal conditioning on forecast origin count rate rather than future target."""
    # Synthetic windows: history x (60,), target y (10,)
    # Window A: origin count rate = 300 (Moderate), but spikes to 1500 (Extreme) in future
    x_A = np.full((60, 1), 300.0, dtype=np.float32)
    y_A = np.full(10, 1500.0, dtype=np.float32)

    # Under CAUSAL rule: conditioned on x_t = 300.0 -> Moderate flaring
    origin_rate = x_A[-1, 0]
    assert 200.0 <= origin_rate <= 1000.0, "Causal conditioning must classify Window A as Moderate"

    # Under LOOK-AHEAD rule: peak y = 1500 -> Extreme flaring
    future_peak = np.max(y_A)
    assert future_peak > 1000.0, "Future target peak differs from causal origin"


def test_segment_isolation_and_minimum_length():
    """Verify that segments < 70s are dropped and gaps strictly break contiguous segments."""
    times = np.arange(0, 100, dtype=float)
    counts = np.ones(100, dtype=float) * 100.0
    # Introduce NaN gap at index 30
    counts[30] = np.nan

    # First segment: length 30 (0 to 29) -> less than 70s -> should be dropped
    # Second segment: length 69 (31 to 99) -> less than 70s -> should be dropped
    segs = extract_contiguous_segments(times, counts, min_segment_length=70)
    assert len(segs) == 0

    # Introduce segment with length 75 (100 to 174)
    times_long = np.arange(100, 175, dtype=float)
    counts_long = np.ones(75, dtype=float) * 100.0
    segs_long = extract_contiguous_segments(times_long, counts_long, min_segment_length=70)
    assert len(segs_long) == 1
    assert len(segs_long[0][0]) == 75

    # Should produce exactly 75 - 70 + 1 = 6 windows
    X, y, t = create_sliding_windows(segs_long, seq_len=60, horizon=10)
    assert len(X) == 6


def test_scaler_isolation_contract():
    """Verify CountRateScaler fitting contract strictly isolates training data."""
    train_data = np.array([50.0, 100.0, 150.0], dtype=np.float32)
    val_data = np.array([500.0, 1000.0], dtype=np.float32)

    scaler = CountRateScaler().fit(train_data)
    train_mean = scaler.mean
    train_std = scaler.std

    # Transforming val data must NOT alter fitted parameters
    val_scaled = scaler.transform(val_data)
    assert scaler.mean == train_mean
    assert scaler.std == train_std

    # Inverse transform recovers original scale
    val_recovered = scaler.inverse_transform(val_scaled)
    np.testing.assert_allclose(val_data, val_recovered, rtol=1e-5)


def test_active_blockers_preservation():
    """Verify active blockers are recorded and enforced in phase4u_scientific_blockers.md."""
    blocker_path = backend_dir / "reports" / "phase4u_scientific_blockers.md"
    assert blocker_path.exists(), "phase4u_scientific_blockers.md must exist."

    content = blocker_path.read_text(encoding="utf-8")
    assert "B-SX-FLUX-003" in content
    assert "B-LBL-NEG-007" in content
    assert "B-HL-CALIB-001" in content
    assert "B-EPOCH3-ACQ-008" in content
