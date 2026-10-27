"""Phase 4W Test Suite — Final Development Benchmark, Error Analysis, and Model Freeze.

Verifies:
1. Phase 4V checkpoint file and SHA256 integrity.
2. Metadata file, training mode, and production registry exclusion.
3. Holdout quarantine (October 2024 not accessed/unsealed).
4. Phase 4W model freeze manifest structure, checksums, and predeclared metrics.
5. Benchmark status determination (PHASE4W_NEURAL_SKILL_SUPPORTED_IN_DEFINED_REGIMES).
6. Metrics mathematical integrity (MAE <= RMSE, non-negative, persistence skill == 0).
7. Defined-regime skill validation (Quiet & Moderate regimes show > 20% skill in both epochs).
8. All required Phase 4W reports are present, populated, and compliant.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BACKEND_DIR / "reports"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
PHASE4V_DIR = ARTIFACTS_DIR / "real_data_experiments" / "phase4v"

EXPECTED_CKPT_SHA256 = "e67662368c49f0d110b15e2378c04dceb8df8fbd9936d2b882a028d4fb0bd26d"
CKPT_FILENAME = "solexs_phase4v_cnn_lstm_20261026T163620Z.pt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class TestPhase4WCheckpointAndMetadata:
    """Task 1: Checkpoint hash and metadata verification."""

    def test_checkpoint_exists_and_hash_matches(self):
        ckpt_path = PHASE4V_DIR / CKPT_FILENAME
        assert ckpt_path.exists(), f"Checkpoint missing: {ckpt_path}"
        actual_sha = sha256_file(ckpt_path)
        assert actual_sha == EXPECTED_CKPT_SHA256, (
            f"Checkpoint SHA256 mismatch: expected {EXPECTED_CKPT_SHA256}, got {actual_sha}"
        )

    def test_run_metadata_matches_checkpoint(self):
        meta_path = PHASE4V_DIR / "run_metadata.json"
        assert meta_path.exists(), f"Metadata missing: {meta_path}"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta.get("checkpoint_sha256") == EXPECTED_CKPT_SHA256
        assert meta.get("training_mode") == "DEVELOPMENT_ONLY_NOT_FINAL_MODEL"
        assert meta.get("production_registry_eligible") is False
        assert "LOCKED_NOT_ACQUIRED" in meta.get("holdout_epoch3_status", "")

    def test_holdout_october2024_not_accessed(self):
        """Mandatory Safeguard: October 2024 must remain locked and unaccessed."""
        meta_path = PHASE4V_DIR / "run_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert "October 2024" in meta.get("holdout_epoch3_status", "")
        # Confirm no October 2024 data was loaded into evaluation metrics
        metrics_csv = REPORTS_DIR / "phase4v_metrics.csv"
        assert metrics_csv.exists()
        df = pd.read_csv(metrics_csv)
        epochs = df["epoch_id"].unique()
        assert "EPOCH3_OCT2024" not in epochs
        assert all("OCT" not in ep for ep in epochs)


class TestPhase4WMetricsIntegrity:
    """Task 2: Independent verification of development metrics."""

    @pytest.fixture(scope="class")
    def metrics_df(self):
        csv_path = REPORTS_DIR / "phase4v_metrics.csv"
        assert csv_path.exists()
        return pd.read_csv(csv_path)

    def test_required_columns(self, metrics_df):
        expected_cols = {
            "epoch_id", "partition", "cohort", "model", "horizon_step",
            "n_samples", "mae_counts_per_s", "rmse_counts_per_s",
            "rel_mae_pct", "target_mean_counts_per_s", "skill_vs_persistence_pct"
        }
        assert expected_cols.issubset(set(metrics_df.columns))

    def test_mathematical_bounds(self, metrics_df):
        # Non-negative MAE and RMSE
        assert (metrics_df["mae_counts_per_s"] >= 0).all()
        assert (metrics_df["rmse_counts_per_s"] >= 0).all()
        # MAE <= RMSE
        assert (metrics_df["mae_counts_per_s"] <= metrics_df["rmse_counts_per_s"] + 1e-5).all()

    def test_persistence_skill_is_zero(self, metrics_df):
        pers_df = metrics_df[metrics_df["model"] == "Persistence"]
        assert (pers_df["skill_vs_persistence_pct"].abs() < 1e-4).all()

    def test_all_ten_horizons_present(self, metrics_df):
        horizons = sorted(metrics_df["horizon_step"].unique().tolist())
        assert horizons == list(range(1, 11))


class TestPhase4WDefinedRegimeSkill:
    """Task 3 & 4: Defined-regime skill confirmation."""

    @pytest.fixture(scope="class")
    def metrics_df(self):
        return pd.read_csv(REPORTS_DIR / "phase4v_metrics.csv")

    def test_quiet_regime_positive_skill_both_epochs(self, metrics_df):
        """Quiet regime (<= 200 c/s) must show > 20% skill in both epochs at H=1 and H=10."""
        cnn_quiet = metrics_df[
            (metrics_df["model"] == "CNN_LSTM_MultiEpoch") &
            (metrics_df["cohort"] == "Causal_REG_QUIET")
        ]
        for epoch_id in ["EPOCH1_MAY2024", "EPOCH2_FEB2026"]:
            for h in [1, 10]:
                row = cnn_quiet[(cnn_quiet["epoch_id"] == epoch_id) & (cnn_quiet["horizon_step"] == h)]
                assert len(row) == 1
                skill = row["skill_vs_persistence_pct"].values[0]
                assert skill > 20.0, f"{epoch_id} quiet H={h} skill was {skill} <= 20%"

    def test_moderate_regime_positive_skill_both_epochs(self, metrics_df):
        """Moderate regime (200-1000 c/s) must show > 20% skill in both epochs at H=1 and H=10."""
        cnn_mod = metrics_df[
            (metrics_df["model"] == "CNN_LSTM_MultiEpoch") &
            (metrics_df["cohort"] == "Causal_REG_MOD")
        ]
        for epoch_id in ["EPOCH1_MAY2024", "EPOCH2_FEB2026"]:
            for h in [1, 10]:
                row = cnn_mod[(cnn_mod["epoch_id"] == epoch_id) & (cnn_mod["horizon_step"] == h)]
                assert len(row) == 1
                skill = row["skill_vs_persistence_pct"].values[0]
                assert skill > 20.0, f"{epoch_id} moderate H={h} skill was {skill} <= 20%"

    def test_epoch2_overall_positive_skill(self, metrics_df):
        """February 2026 test partition must show positive overall skill (> 20%)."""
        cnn_e2_all = metrics_df[
            (metrics_df["model"] == "CNN_LSTM_MultiEpoch") &
            (metrics_df["cohort"] == "ALL") &
            (metrics_df["epoch_id"] == "EPOCH2_FEB2026")
        ]
        for h in [1, 10]:
            row = cnn_e2_all[cnn_e2_all["horizon_step"] == h]
            assert len(row) == 1
            skill = row["skill_vs_persistence_pct"].values[0]
            assert skill > 20.0, f"Epoch 2 ALL H={h} skill was {skill} <= 20%"

    def test_rmse_mae_ratio_near_gaussian_in_quiet(self, metrics_df):
        """Quiet regime RMSE/MAE ratio should be near-Gaussian (1.20 - 1.40)."""
        cnn_quiet = metrics_df[
            (metrics_df["model"] == "CNN_LSTM_MultiEpoch") &
            (metrics_df["cohort"] == "Causal_REG_QUIET") &
            (metrics_df["horizon_step"] == 1)
        ]
        for _, row in cnn_quiet.iterrows():
            ratio = row["rmse_counts_per_s"] / row["mae_counts_per_s"]
            assert 1.20 <= ratio <= 1.40, f"Ratio {ratio} outside expected range [1.20, 1.40]"


class TestPhase4WFreezeManifestAndReports:
    """Task 5 & 6: Freeze manifest and report generation audit."""

    def test_manifest_structure_and_hashes(self):
        manifest_path = REPORTS_DIR / "phase4w_model_freeze_manifest.json"
        assert manifest_path.exists(), f"Manifest missing: {manifest_path}"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert manifest["phase"] == "Phase 4W"
        assert manifest["status"] == "PHASE4W_NEURAL_SKILL_SUPPORTED_IN_DEFINED_REGIMES"
        assert manifest["frozen_checkpoint"]["sha256"] == EXPECTED_CKPT_SHA256
        assert manifest["frozen_checkpoint"]["verified"] is True
        assert manifest["frozen_scaler"]["type"] == "log10_zscore"
        assert manifest["frozen_architecture"]["class"] == "SelfSupervisedCNNLSTM"
        assert manifest["frozen_preprocessing"]["sequence_length_seconds"] == 60
        assert manifest["frozen_preprocessing"]["forecast_horizon_seconds"] == 10
        assert manifest["holdout_governance"]["holdout_status"] == "LOCKED_NOT_ACQUIRED"
        assert manifest["holdout_governance"]["holdout_accessed_during_phase4w"] is False
        assert len(manifest["predeclared_holdout_metrics"]) >= 4

    def test_all_phase4w_reports_exist_and_non_empty(self):
        required_reports = [
            "phase4w_final_development_benchmark.md",
            "phase4w_error_analysis.md",
            "phase4w_model_freeze_manifest.json",
            "phase4w_holdout_evaluation_protocol.md",
        ]
        for fname in required_reports:
            p = REPORTS_DIR / fname
            assert p.exists(), f"Required report missing: {fname}"
            assert p.stat().st_size > 100, f"Report {fname} is unexpectedly small or empty"

    def test_analysis_results_json_valid(self):
        analysis_path = REPORTS_DIR / "phase4w_analysis_results.json"
        assert analysis_path.exists()
        data = json.loads(analysis_path.read_text(encoding="utf-8"))
        assert data["status"] == "PHASE4W_NEURAL_SKILL_SUPPORTED_IN_DEFINED_REGIMES"
        assert data["checkpoint_verification"]["overall_passed"] is True
        assert data["metric_verification"]["overall_passed"] is True
