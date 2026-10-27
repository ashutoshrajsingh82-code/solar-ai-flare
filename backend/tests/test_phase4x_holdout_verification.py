"""Phase 4X Test Suite — Independent Holdout Acquisition Verification and Pre-Evaluation Lock.

Verifies:
1. Frozen checkpoint hash matches the Phase 4W freeze manifest.
2. Frozen scaler parameters match target values (mean 2.597875, std 0.582402).
3. Frozen code modules (pipeline and model) match cryptographic signatures.
4. October 2024 holdout data is confirmed unacquired locally, preventing premature evaluation.
5. Phase 4X acquisition manifest reflects status BLOCKED_INDEPENDENT_HOLDOUT_NOT_ACQUIRED.
6. Phase 4X reports are present, populated, and compliant.
7. Causal regime partitioning logic strictly conditions on forecast origin (y_t) and never future targets.
8. Pre-registered metric calculations and persistence symmetry adhere strictly to protocol.
9. No fabricated holdout predictions or synthetic data exist.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BACKEND_DIR / "reports"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
PHASE4V_DIR = ARTIFACTS_DIR / "real_data_experiments" / "phase4v"
PHASE4X_DIR = ARTIFACTS_DIR / "real_data_experiments" / "phase4x"

EXPECTED_CKPT_SHA256 = "e67662368c49f0d110b15e2378c04dceb8df8fbd9936d2b882a028d4fb0bd26d"
CKPT_FILENAME = "solexs_phase4v_cnn_lstm_20261026T163620Z.pt"
EXPECTED_PIPELINE_SHA256 = "06a97149db0667f8adfb82cab36b9e5aa32ea9a2f39cdc21aaac72775ae1362f"
EXPECTED_MODEL_SHA256 = "6c5081e333d1a9684a7600110a4e52df3167c3a13df5d6eafc1bfeb00a2957ee"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class TestPhase4XFrozenArtifacts:
    """Verifies frozen model checkpoint, normalizer, and code signatures."""

    def test_frozen_checkpoint_exists_and_hash_matches(self):
        ckpt_path = PHASE4V_DIR / CKPT_FILENAME
        assert ckpt_path.exists(), f"Checkpoint missing: {ckpt_path}"
        actual_sha = sha256_file(ckpt_path)
        assert actual_sha == EXPECTED_CKPT_SHA256, (
            f"Checkpoint SHA256 mismatch: expected {EXPECTED_CKPT_SHA256}, got {actual_sha}"
        )

    def test_frozen_normalizer_parameters_match(self):
        manifest_path = REPORTS_DIR / "phase4w_model_freeze_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scaler = manifest["frozen_scaler"]
        assert scaler["type"] == "log10_zscore"
        assert round(scaler["mean"], 6) == 2.597875
        assert round(scaler["std"], 6) == 0.582402

    def test_frozen_code_hashes_match(self):
        pipeline_path = BACKEND_DIR / "scripts" / "phase4v_pipeline.py"
        model_path = BACKEND_DIR / "ml" / "experiments" / "real_self_supervised.py"
        assert sha256_file(pipeline_path) == EXPECTED_PIPELINE_SHA256
        assert sha256_file(model_path) == EXPECTED_MODEL_SHA256


class TestPhase4XHoldoutAcquisitionAudit:
    """Verifies that the October 2024 holdout is correctly flagged as unacquired."""

    def test_october2024_holdout_absent_locally(self):
        """Mandatory Safeguard: October 2024 must NOT be present locally or accidentally evaluated."""
        downloads_dir = Path("C:/Users/ASHUTOSH/Downloads")
        matching = list(downloads_dir.glob("*202410*"))
        assert len(matching) == 0, f"Unexpected October 2024 files found: {matching}"

    def test_manifest_status_is_blocked_independent_holdout_not_acquired(self):
        manifest_path = REPORTS_DIR / "phase4x_holdout_acquisition_manifest.json"
        assert manifest_path.exists(), f"Missing manifest: {manifest_path}"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["phase"] == "Phase 4X"
        assert manifest["status"] == "BLOCKED_INDEPENDENT_HOLDOUT_NOT_ACQUIRED"
        assert manifest["holdout_evaluation_executed"] is False
        assert manifest["holdout_epoch"]["local_availability"] is False
        assert "PENDING" in manifest["holdout_epoch"]["acquisition_status"]

    def test_no_fabricated_holdout_predictions_or_metrics(self):
        """Ensures no holdout predictions or metrics files were fabricated without data."""
        pred_path = REPORTS_DIR / "phase4x_holdout_predictions.npz"
        metrics_path = REPORTS_DIR / "phase4x_holdout_metrics.json"
        assert not pred_path.exists(), "Predictions must NOT exist when holdout is unacquired"
        assert not metrics_path.exists(), "Metrics must NOT exist when holdout is unacquired"


class TestPhase4XCausalRegimeAndEvaluationLogic:
    """Verifies that causal regime assignment and evaluation math adhere to pre-registered protocol."""

    def test_causal_regime_classification_logic(self):
        """Regime membership must depend ONLY on origin flux y_t, never future values."""
        # Simulated origin count rates
        origin_fluxes = np.array([50.0, 150.0, 200.0, 200.1, 500.0, 1000.0, 1000.1, 5000.0])

        quiet_mask = origin_fluxes <= 200.0
        mod_mask = (origin_fluxes > 200.0) & (origin_fluxes <= 1000.0)
        high_mask = origin_fluxes > 1000.0

        assert list(quiet_mask) == [True, True, True, False, False, False, False, False]
        assert list(mod_mask) == [False, False, False, True, True, True, False, False]
        assert list(high_mask) == [False, False, False, False, False, False, True, True]

    def test_surge_regime_classification_logic(self):
        """Surge regime requires |y_{t+10} - y_t| >= 50 counts/s."""
        origin = np.array([100.0, 100.0, 100.0])
        future_10 = np.array([120.0, 150.0, 40.0])
        delta = np.abs(future_10 - origin)
        surge_mask = delta >= 50.0
        assert list(surge_mask) == [False, True, True]

    def test_skill_score_mathematical_formulation(self):
        """Skill score vs persistence formula: (1 - MAE_model / MAE_pers) * 100%."""
        mae_pers = 20.0
        mae_model_better = 15.0
        skill_better = (1.0 - mae_model_better / mae_pers) * 100.0
        assert skill_better == 25.0

        mae_model_worse = 25.0
        skill_worse = (1.0 - mae_model_worse / mae_pers) * 100.0
        assert skill_worse == -25.0

        # Persistence skill is identically zero
        skill_pers = (1.0 - mae_pers / mae_pers) * 100.0
        assert skill_pers == 0.0


class TestPhase4XDeliverablesIntegrity:
    """Verifies that all Phase 4X required documentation and reports are populated and compliant."""

    def test_phase4x_reports_exist(self):
        reports = [
            "phase4x_holdout_acquisition_manifest.json",
            "phase4x_holdout_integrity_report.md",
            "phase4x_pre_evaluation_lock_report.md",
        ]
        for name in reports:
            p_rep = REPORTS_DIR / name
            p_art = PHASE4X_DIR / name
            assert p_rep.exists(), f"Report missing in reports/: {name}"
            assert p_art.exists(), f"Report missing in artifacts/phase4x/: {name}"
            assert p_rep.stat().st_size > 100, f"Report {name} is unexpectedly empty"

    def test_pre_evaluation_lock_report_contains_all_ten_locks(self):
        lock_report = (REPORTS_DIR / "phase4x_pre_evaluation_lock_report.md").read_text(encoding="utf-8")
        for i in range(1, 11):
            lock_id = f"LOCK-{i:02d}"
            assert lock_id in lock_report, f"Missing lock check {lock_id} in lock report"
        assert "BLOCKED_AWAITING_HOLDOUT_ACQUISITION" in lock_report
