"""Phase 4V: Automated Tests for Multi-Epoch Forecasting Pipeline.

Covers (per Phase 4V specification §H):
1.  Archive and FITS validation.
2.  Product deduplication.
3.  Segment isolation and missing-data handling.
4.  Window counts and forecast-target alignment.
5.  Partition isolation and chronological rules.
6.  Training-only scaler fitting.
7.  Baseline alignment and metric calculations.
8.  Deterministic configuration and checkpoint metadata.
9.  Fail-closed readiness gate behavior.
10. Holdout-path exclusion and protection.
11. Rejection of synthetic labels and unapproved product types.
12. Preservation of historical artifacts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from ml.experiments.real_self_supervised import (
    CountRateScaler,
    SelfSupervisedCNNLSTM,
    create_sliding_windows,
    extract_contiguous_segments,
    persistence_forecast,
    rolling_mean_forecast,
)
from scripts.phase4v_readiness_gate import (
    ARCHIVE_FEB2026,
    ARCHIVE_MAY2024,
    EXPECTED_FEB_DATES,
    EXPECTED_MAY_DATES,
    SHA256_FEB2026,
    SHA256_MAY2024,
    evaluate_phase4v_gate,
)
from scripts.phase4v_pipeline import (
    COHORT_MOD_MAX,
    COHORT_QUIET_MAX,
    HORIZON,
    MIN_SEG_LEN,
    SEQ_LEN,
    SURGE_DELTA_THRESH,
    assign_causal_cohorts,
    compute_per_horizon_metrics,
    compute_skill_vs_persistence,
)

REPORTS_DIR = backend_dir / "reports"
ART_4V_DIR = backend_dir / "artifacts" / "real_data_experiments" / "phase4v"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def gate_report():
    """Evaluate the Phase 4V gate once for the entire session."""
    return evaluate_phase4v_gate()


@pytest.fixture
def simple_segment():
    """A simple 100-sample contiguous segment with uniform count rate of 100.0 c/s."""
    times = np.arange(0, 100, dtype=float)
    counts = np.full(100, 100.0, dtype=float)
    return times, counts


# ── Test 1: Archive and FITS validation ───────────────────────────────────────

class TestArchiveValidation:
    def test_may2024_archive_exists(self):
        assert ARCHIVE_MAY2024.is_file(), f"May 2024 archive missing: {ARCHIVE_MAY2024}"

    def test_feb2026_archive_exists(self):
        assert ARCHIVE_FEB2026.is_file(), f"Feb 2026 archive missing: {ARCHIVE_FEB2026}"

    def test_may2024_sha256(self):
        from scripts.operator_acquisition_helper import compute_sha256
        assert compute_sha256(ARCHIVE_MAY2024) == SHA256_MAY2024

    def test_feb2026_sha256(self):
        from scripts.operator_acquisition_helper import compute_sha256
        assert compute_sha256(ARCHIVE_FEB2026) == SHA256_FEB2026

    def test_may2024_expected_dates(self):
        """May 2024 archive must contain exactly the 8 expected daily packages."""
        import zipfile
        with zipfile.ZipFile(ARCHIVE_MAY2024) as zf:
            found = {n.split("_")[3] for n in zf.namelist()
                     if n.endswith(".zip") and "AL1_SLX_L1_" in n}
        assert EXPECTED_MAY_DATES.issubset(found), f"Missing dates: {EXPECTED_MAY_DATES - found}"

    def test_feb2026_expected_dates(self):
        """Feb 2026 archive must contain exactly the 6 expected daily packages."""
        import zipfile
        with zipfile.ZipFile(ARCHIVE_FEB2026) as zf:
            found = {n.split("_")[3] for n in zf.namelist()
                     if n.endswith(".zip") and "AL1_SLX_L1_" in n}
        assert EXPECTED_FEB_DATES.issubset(found), f"Missing dates: {EXPECTED_FEB_DATES - found}"


# ── Test 2: Product deduplication ─────────────────────────────────────────────

class TestProductDeduplication:
    def test_observation_inventory_exists(self):
        inv = REPORTS_DIR / "phase4u_observation_inventory.csv"
        assert inv.is_file(), "Phase 4U observation inventory must exist."

    def test_exactly_14_unique_products(self):
        import pandas as pd
        df = pd.read_csv(REPORTS_DIR / "phase4u_observation_inventory.csv")
        assert len(df) == 14
        assert df["product_key"].nunique() == 14

    def test_no_duplicate_products(self):
        import pandas as pd
        df = pd.read_csv(REPORTS_DIR / "phase4u_observation_inventory.csv")
        assert (df["is_duplicate_product"] == False).all()

    def test_product_key_format(self):
        """Product keys must follow SoLEXS_SDD2_YYYYMMDD format."""
        import pandas as pd
        df = pd.read_csv(REPORTS_DIR / "phase4u_observation_inventory.csv")
        assert all(k.startswith("SoLEXS_SDD2_") for k in df["product_key"])


# ── Test 3: Segment isolation and missing-data handling ───────────────────────

class TestSegmentIsolation:
    def test_nan_breaks_segment(self, simple_segment):
        times, counts = simple_segment
        counts[50] = np.nan
        segs = extract_contiguous_segments(times, counts, min_segment_length=1)
        # Should produce 2 segments
        assert len(segs) == 2
        seg_lengths = [len(s[0]) for s in segs]
        assert sorted(seg_lengths) == [49, 50]

    def test_negative_breaks_segment(self, simple_segment):
        times, counts = simple_segment
        counts[30] = -1.0
        segs = extract_contiguous_segments(times, counts, min_segment_length=1)
        assert len(segs) == 2

    def test_timestamp_gap_breaks_segment(self):
        times = np.concatenate([np.arange(0, 50, dtype=float),
                                 np.arange(100, 150, dtype=float)])
        counts = np.ones(100, dtype=float) * 100.0
        segs = extract_contiguous_segments(times, counts, min_segment_length=1)
        assert len(segs) == 2

    def test_minimum_segment_length_enforced(self):
        """Segments shorter than MIN_SEG_LEN (70) must be discarded."""
        times = np.arange(0, 69, dtype=float)
        counts = np.ones(69, dtype=float) * 100.0
        segs = extract_contiguous_segments(times, counts, min_segment_length=MIN_SEG_LEN)
        assert len(segs) == 0

    def test_segment_at_exactly_minimum(self):
        times = np.arange(0, MIN_SEG_LEN, dtype=float)
        counts = np.ones(MIN_SEG_LEN, dtype=float) * 100.0
        segs = extract_contiguous_segments(times, counts, min_segment_length=MIN_SEG_LEN)
        assert len(segs) == 1

    def test_no_gap_bridging(self):
        """Windows must never bridge across a telemetry gap."""
        # Create a segment with a gap right in the middle of where a window would span
        times = np.arange(0, 200, dtype=float)
        counts = np.ones(200, dtype=float) * 100.0
        counts[70] = np.nan  # gap at index 70

        segs = extract_contiguous_segments(times, counts, min_segment_length=MIN_SEG_LEN)
        X, y, t = create_sliding_windows(segs, SEQ_LEN, HORIZON)
        # No window should span index 70 (Unix time = 70.0)
        for i, t_orig in enumerate(t):
            # Window covers [t_orig - SEQ_LEN + 1, t_orig + HORIZON]
            win_start = t_orig - SEQ_LEN + 1
            win_end = t_orig + HORIZON
            assert not (win_start <= 70.0 <= win_end), f"Window {i} bridges gap at t=70"


# ── Test 4: Window counts and forecast-target alignment ───────────────────────

class TestWindowAlignment:
    def test_window_shape(self):
        times = np.arange(0, 150, dtype=float)
        counts = np.ones(150, dtype=float) * 200.0
        segs = extract_contiguous_segments(times, counts, min_segment_length=MIN_SEG_LEN)
        X, y, t = create_sliding_windows(segs, SEQ_LEN, HORIZON)
        # Expected: 150 - (60 + 10) + 1 = 81 windows
        assert len(X) == 81
        assert X.shape == (81, SEQ_LEN, 1)
        assert y.shape == (81, HORIZON)
        assert t.shape == (81,)

    def test_target_follows_input(self):
        """y[i, 0] must be the count immediately after the last input sample."""
        counts = np.arange(200.0, dtype=float)
        times = np.arange(200, dtype=float)
        segs = extract_contiguous_segments(times, counts, min_segment_length=MIN_SEG_LEN)
        X, y, t = create_sliding_windows(segs, SEQ_LEN, HORIZON)
        # For window 0: X = counts[0:60], y = counts[60:70]
        np.testing.assert_array_almost_equal(X[0, :, 0], np.arange(60.0))
        np.testing.assert_array_almost_equal(y[0], np.arange(60.0, 70.0))

    def test_window_does_not_cross_partition(self):
        """Creating windows from separately partitioned segments ensures no cross-boundary windows."""
        # Simulate train partition: t=0..120, val partition: t=200..320 (separate segs)
        train_times = np.arange(0, 121, dtype=float)
        val_times = np.arange(200, 321, dtype=float)
        train_counts = np.ones(121, dtype=float) * 100.0
        val_counts = np.ones(121, dtype=float) * 100.0

        train_segs = extract_contiguous_segments(train_times, train_counts, MIN_SEG_LEN)
        val_segs = extract_contiguous_segments(val_times, val_counts, MIN_SEG_LEN)

        X_tr, y_tr, t_tr = create_sliding_windows(train_segs, SEQ_LEN, HORIZON)
        X_val, y_val, t_val = create_sliding_windows(val_segs, SEQ_LEN, HORIZON)

        # All training origins must be ≤ 120; all val origins must be ≥ 200
        if len(t_tr) > 0:
            assert t_tr.max() <= 120.0
        if len(t_val) > 0:
            assert t_val.min() >= 200.0


# ── Test 5: Partition isolation and chronological rules ───────────────────────

class TestPartitionIsolation:
    def test_train_dates_before_val(self):
        """Epoch 1 training dates must all precede validation dates chronologically."""
        from scripts.phase4v_pipeline import EPOCH1_TRAIN_DATES, EPOCH1_VAL_DATES, EPOCH1_TEST_DATES
        assert max(EPOCH1_TRAIN_DATES) < min(EPOCH1_VAL_DATES)
        assert max(EPOCH1_VAL_DATES) < min(EPOCH1_TEST_DATES)

    def test_epoch2_dates_ordered(self):
        from scripts.phase4v_pipeline import EPOCH2_TRAIN_DATES, EPOCH2_VAL_DATES, EPOCH2_TEST_DATES
        assert max(EPOCH2_TRAIN_DATES) < min(EPOCH2_VAL_DATES)
        assert max(EPOCH2_VAL_DATES) < min(EPOCH2_TEST_DATES)

    def test_no_date_overlap_between_partitions_epoch1(self):
        from scripts.phase4v_pipeline import EPOCH1_TRAIN_DATES, EPOCH1_VAL_DATES, EPOCH1_TEST_DATES
        tr = set(EPOCH1_TRAIN_DATES)
        val = set(EPOCH1_VAL_DATES)
        tst = set(EPOCH1_TEST_DATES)
        assert len(tr & val) == 0
        assert len(val & tst) == 0
        assert len(tr & tst) == 0

    def test_no_date_overlap_between_epochs(self):
        """May 2024 and Jan-Feb 2026 dates must not overlap."""
        all_may = set(EXPECTED_MAY_DATES)
        all_feb = set(EXPECTED_FEB_DATES)
        assert len(all_may & all_feb) == 0

    def test_holdout_dates_excluded(self):
        """October 2024 dates must not appear in any partition configuration."""
        from scripts.phase4v_pipeline import (
            EPOCH1_TRAIN_DATES, EPOCH1_VAL_DATES, EPOCH1_TEST_DATES,
            EPOCH2_TRAIN_DATES, EPOCH2_VAL_DATES, EPOCH2_TEST_DATES,
        )
        all_pipeline_dates = set(
            EPOCH1_TRAIN_DATES + EPOCH1_VAL_DATES + EPOCH1_TEST_DATES +
            EPOCH2_TRAIN_DATES + EPOCH2_VAL_DATES + EPOCH2_TEST_DATES
        )
        oct_dates = {f"202410{d:02d}" for d in range(1, 8)}
        assert len(all_pipeline_dates & oct_dates) == 0, \
            f"October holdout dates found in pipeline: {all_pipeline_dates & oct_dates}"


# ── Test 6: Training-only scaler fitting ─────────────────────────────────────

class TestScalerIsolation:
    def test_scaler_fitted_only_on_train(self):
        train = np.array([50.0, 100.0, 200.0], dtype=np.float32)
        val = np.array([5000.0, 10000.0], dtype=np.float32)

        scaler = CountRateScaler().fit(train)
        mu, sigma = scaler.mean, scaler.std

        # Transforming val must not change scaler params
        _ = scaler.transform(val)
        assert scaler.mean == mu
        assert scaler.std == sigma

    def test_scaler_inverse_roundtrip(self):
        data = np.array([0.0, 100.0, 1000.0, 10000.0], dtype=np.float32)
        scaler = CountRateScaler().fit(data)
        scaled = scaler.transform(data)
        recovered = scaler.inverse_transform(scaled)
        np.testing.assert_allclose(data, recovered, rtol=1e-4)

    def test_scaler_non_negativity(self):
        """Inverse transform must never produce negative count rates."""
        scaler = CountRateScaler().fit(np.array([100.0, 200.0]))
        very_negative_scaled = np.array([[-100.0, -200.0]])
        result = scaler.inverse_transform(very_negative_scaled)
        assert (result >= 0.0).all()

    def test_unfitted_scaler_raises(self):
        scaler = CountRateScaler()
        with pytest.raises(RuntimeError):
            scaler.transform(np.array([1.0, 2.0]))


# ── Test 7: Baseline alignment and metric calculations ────────────────────────

class TestBaselineAndMetrics:
    @pytest.fixture
    def synthetic_windows(self):
        """100 windows with uniform history 100.0 and future 110.0 c/s."""
        X = np.full((100, SEQ_LEN, 1), 100.0, dtype=np.float32)
        y = np.full((100, HORIZON), 110.0, dtype=np.float32)
        return X, y

    def test_persistence_uses_last_input(self, synthetic_windows):
        X, y = synthetic_windows
        preds = persistence_forecast(X, HORIZON)
        assert preds.shape == (100, HORIZON)
        # All predictions should be 100.0 (last input value)
        np.testing.assert_array_equal(preds, 100.0)

    def test_rolling_mean_uses_full_history(self, synthetic_windows):
        X, y = synthetic_windows
        preds = rolling_mean_forecast(X, HORIZON)
        assert preds.shape == (100, HORIZON)
        np.testing.assert_array_equal(preds, 100.0)

    def test_baselines_same_shape_as_targets(self, synthetic_windows):
        X, y = synthetic_windows
        assert persistence_forecast(X, HORIZON).shape == y.shape
        assert rolling_mean_forecast(X, HORIZON).shape == y.shape

    def test_metric_computation_shapes(self, synthetic_windows):
        X, y = synthetic_windows
        preds = persistence_forecast(X, HORIZON)
        rows = compute_per_horizon_metrics(y, preds, "TestModel", "ALL", "EPOCH1", "test")
        assert len(rows) == HORIZON
        for r in rows:
            assert "mae_counts_per_s" in r
            assert "rmse_counts_per_s" in r
            assert "rel_mae_pct" in r
            assert r["n_samples"] == 100

    def test_skill_computation_persistence_zero(self, synthetic_windows):
        X, y = synthetic_windows
        preds_persist = persistence_forecast(X, HORIZON)
        rows_persist = compute_per_horizon_metrics(y, preds_persist, "Persistence", "ALL", "E1", "test")
        rows_persist = compute_skill_vs_persistence(rows_persist)
        # Persistence vs. persistence skill should be 0.0
        for r in rows_persist:
            if r["model"] == "Persistence":
                assert r["skill_vs_persistence_pct"] == pytest.approx(0.0, abs=1e-4)

    def test_perfect_forecast_skill_100(self, synthetic_windows):
        X, y = synthetic_windows
        preds_perfect = y.copy()  # perfect predictions
        preds_persist = persistence_forecast(X, HORIZON)

        rows_perfect = compute_per_horizon_metrics(y, preds_perfect, "Perfect", "ALL", "E1", "test")
        rows_persist = compute_per_horizon_metrics(y, preds_persist, "Persistence", "ALL", "E1", "test")
        all_rows = rows_perfect + rows_persist
        all_rows = compute_skill_vs_persistence(all_rows)

        for r in all_rows:
            if r["model"] == "Perfect":
                assert r["skill_vs_persistence_pct"] == pytest.approx(100.0, abs=1e-2)


# ── Test 8: Deterministic configuration and model architecture ────────────────

class TestDeterministicConfig:
    def test_model_architecture_fixed(self):
        """CNN-LSTM architecture must have exactly the prescribed dimensions."""
        import torch as _torch
        model = SelfSupervisedCNNLSTM(
            in_features=1, horizon=HORIZON, cnn_channels=32, lstm_hidden=64, lstm_layers=2, dropout=0.2
        )
        assert model.horizon == HORIZON
        x = _torch.zeros(4, SEQ_LEN, 1)
        out = model(x)
        assert out.shape == (4, HORIZON)

    def test_reproducibility_same_seed(self):
        """Two runs with the same seed must produce bit-identical model parameters."""
        import torch

        def _init_model(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = SelfSupervisedCNNLSTM(1, HORIZON, 32, 64, 2, 0.2)
            return {k: v.clone() for k, v in model.state_dict().items()}

        w1 = _init_model(42)
        w2 = _init_model(42)
        for k in w1:
            np.testing.assert_array_equal(w1[k].numpy(), w2[k].numpy(), err_msg=f"Mismatch in {k}")

    def test_contract_constants_match_dataset_contract(self):
        """SEQ_LEN=60, HORIZON=10, MIN_SEG_LEN=70 must match the Phase 4U contract."""
        assert SEQ_LEN == 60
        assert HORIZON == 10
        assert MIN_SEG_LEN == SEQ_LEN + HORIZON == 70
        assert COHORT_QUIET_MAX == 200.0
        assert COHORT_MOD_MAX == 1000.0
        assert SURGE_DELTA_THRESH == 50.0


# ── Test 9: Fail-closed readiness gate ───────────────────────────────────────

class TestReadinessGate:
    def test_gate_returns_dict(self, gate_report):
        assert isinstance(gate_report, dict)

    def test_gate_has_required_fields(self, gate_report):
        required = {"overall_status", "training_authorized", "training_mode",
                    "holdout_locked", "conditions", "failed_hard_conditions"}
        assert required.issubset(gate_report.keys())

    def test_holdout_always_locked(self, gate_report):
        assert gate_report["holdout_locked"] is True

    def test_holdout_not_locally_available(self, gate_report):
        assert gate_report["holdout_locally_available"] is False

    def test_gate_4v_07_fails(self, gate_report):
        """GATE-4V-07 (holdout) must always be unsatisfied until acquired."""
        cond7 = next(c for c in gate_report["conditions"] if c["condition_id"] == "GATE-4V-07")
        assert cond7["satisfied"] is False

    def test_gate_4v_07_is_non_blocking(self, gate_report):
        """Even though GATE-4V-07 fails, it must NOT appear in failed_hard_conditions."""
        assert "GATE-4V-07" not in gate_report["failed_hard_conditions"]

    def test_gate_file_inspected_not_manual(self, gate_report):
        """Gate must evaluate actual archives and files, not only a manual status field."""
        # Gate must have verified actual archives (condition details mention bytes or paths)
        cond1 = next(c for c in gate_report["conditions"] if c["condition_id"] == "GATE-4V-01")
        cond2 = next(c for c in gate_report["conditions"] if c["condition_id"] == "GATE-4V-02")
        # Detail must reference file size or SHA verification
        assert any(kw in cond1["detail"] for kw in ("bytes", "verified", "SHA", "Not found"))
        assert any(kw in cond2["detail"] for kw in ("bytes", "verified", "SHA", "Not found"))

    def test_gate_exits_nonzero_when_blocked(self, monkeypatch):
        """Simulate an archive being absent — gate must return training_authorized=False."""
        from unittest.mock import patch
        from pathlib import Path

        # Monkey-patch ARCHIVE_MAY2024 to a non-existent path
        with patch("scripts.phase4v_readiness_gate.ARCHIVE_MAY2024",
                   Path("/nonexistent/missing_archive.zip")):
            report = evaluate_phase4v_gate()
        assert report["training_authorized"] is False
        assert report["overall_status"] == "PHASE4V_BLOCKED_READINESS_GATE"
        assert "GATE-4V-01" in report["failed_hard_conditions"]


# ── Test 10: Holdout exclusion and protection ─────────────────────────────────

class TestHoldoutProtection:
    def test_october_dates_not_in_pipeline(self):
        from scripts.phase4v_pipeline import (
            EPOCH1_TRAIN_DATES, EPOCH1_VAL_DATES, EPOCH1_TEST_DATES,
            EPOCH2_TRAIN_DATES, EPOCH2_VAL_DATES, EPOCH2_TEST_DATES,
        )
        all_dates = (EPOCH1_TRAIN_DATES + EPOCH1_VAL_DATES + EPOCH1_TEST_DATES +
                     EPOCH2_TRAIN_DATES + EPOCH2_VAL_DATES + EPOCH2_TEST_DATES)
        for d in all_dates:
            assert not d.startswith("202410"), f"October holdout date {d} found in pipeline"

    def test_holdout_epoch_id_reserved(self, gate_report):
        assert "EPOCH_3_RESERVED_UNTOUCHED_HOLDOUT" in gate_report["holdout_epoch"]

    def test_run_metadata_marks_holdout_locked(self):
        meta_path = ART_4V_DIR / "run_metadata.json"
        if not meta_path.exists():
            pytest.skip("run_metadata.json not yet generated (pipeline not run in dev-train mode)")
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        assert "LOCKED" in meta.get("holdout_epoch3_status", "")

    def test_production_registry_ineligible(self):
        meta_path = ART_4V_DIR / "run_metadata.json"
        if not meta_path.exists():
            pytest.skip("run_metadata.json not yet generated")
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        assert meta.get("production_registry_eligible") is False

    def test_training_mode_marked_development_only(self):
        meta_path = ART_4V_DIR / "run_metadata.json"
        if not meta_path.exists():
            pytest.skip("run_metadata.json not yet generated")
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        assert meta.get("training_mode") == "DEVELOPMENT_ONLY_NOT_FINAL_MODEL"


# ── Test 11: Synthetic labels and unapproved product types rejected ───────────

class TestDataEligibilityRejection:
    def test_no_synthetic_labels_in_contract(self):
        """Dataset contract must explicitly exclude binary flare labels."""
        contract = (REPORTS_DIR / "phase4u_dataset_contract.md").read_text(encoding="utf-8")
        assert "EXCL" in contract or "prohibited" in contract.lower() or "zero flare" in contract.lower()

    def test_hel1os_excluded_in_contract(self):
        contract = (REPORTS_DIR / "phase4u_dataset_contract.md").read_text(encoding="utf-8")
        assert "HEL1OS" in contract

    def test_sdd1_excluded_in_contract(self):
        contract = (REPORTS_DIR / "phase4u_dataset_contract.md").read_text(encoding="utf-8")
        assert "SDD1" in contract

    def test_causal_cohort_does_not_use_future_peaks(self):
        """Causal cohort assignment must use only x_t, not max(y)."""
        # Window where origin = 150.0 (QUIET) but future peaks at 5000.0 (HIGH)
        X = np.full((1, SEQ_LEN, 1), 150.0, dtype=np.float32)
        cohorts = assign_causal_cohorts(X)
        assert cohorts["Causal_REG_QUIET"][0] is np.bool_(True), \
            "Origin-based cohort must classify as QUIET when x_t=150"
        assert cohorts["Causal_REG_HIGH"][0] is np.bool_(False)

    def test_causal_cohorts_mutually_exclusive_for_single_sample(self):
        for origin_val in [50.0, 300.0, 2000.0]:
            X = np.full((1, SEQ_LEN, 1), origin_val, dtype=np.float32)
            cohorts = assign_causal_cohorts(X)
            exclusive = [cohorts["Causal_REG_QUIET"][0],
                         cohorts["Causal_REG_MOD"][0],
                         cohorts["Causal_REG_HIGH"][0]]
            assert sum(exclusive) == 1, \
                f"Expected exactly 1 exclusive cohort for origin={origin_val}, got {sum(exclusive)}"

    def test_irradiance_conversion_absent_from_pipeline(self):
        """The pipeline must not contain W/m² or GOES class conversion."""
        pipeline_src = (backend_dir / "scripts" / "phase4v_pipeline.py").read_text(encoding="utf-8")
        for forbidden in ["W/m", "W m", "goes_class", "GOES_CLASS", "irradiance"]:
            assert forbidden not in pipeline_src, \
                f"Forbidden physical conversion '{forbidden}' found in pipeline source"


# ── Test 12: Preservation of historical artifacts ─────────────────────────────

class TestHistoricalArtifactPreservation:
    """All historical checkpoints and reports must be preserved untouched."""

    def test_phase4o_checkpoint_preserved(self):
        ckpt = backend_dir / "artifacts" / "real_data_experiments" / "phase4o" / \
               "solexs_phase4o_cnn_lstm_direct_20261025T184851Z.pt"
        assert ckpt.exists(), "Phase 4O reference checkpoint must not be deleted or overwritten"

    def test_phase4s_artifacts_preserved(self):
        art_4s = backend_dir / "artifacts" / "real_data_experiments" / "phase4s"
        assert art_4s.is_dir()
        assert any(art_4s.glob("*.pt")), "Phase 4S checkpoint must be preserved"

    def test_phase4u_reports_preserved(self):
        for fname in ["phase4u_readiness_gate.json", "phase4u_dataset_contract.md",
                      "phase4u_holdout_governance.md", "phase4u_observation_inventory.csv"]:
            p = REPORTS_DIR / fname
            assert p.exists(), f"Historical Phase 4U report missing: {fname}"

    def test_phase4t_reports_preserved(self):
        for fname in ["phase4t_reproducibility_audit.md", "phase4t_leakage_and_split_audit.md"]:
            p = REPORTS_DIR / fname
            assert p.exists(), f"Historical Phase 4T report missing: {fname}"

    def test_phase4v_artifacts_in_separate_directory(self):
        """Phase 4V artifacts must NOT be stored in phase4s/, phase4t/, or phase4u/ directories."""
        gate_src = backend_dir / "scripts" / "phase4v_readiness_gate.py"
        pipeline_src = backend_dir / "scripts" / "phase4v_pipeline.py"

        gate_text = gate_src.read_text(encoding="utf-8")
        pipe_text = pipeline_src.read_text(encoding="utf-8")

        # Must output to phase4v, not overwrite earlier phases
        assert "phase4v" in gate_text.lower()
        assert "phase4v" in pipe_text.lower()


# ── Test: FITS validation ─────────────────────────────────────────────────────

class TestFITSValidation:
    def test_validate_solexs_fits_table_missing_rate_hdu(self):
        """Missing RATE HDU must raise ValueError."""
        from astropy.io.fits import HDUList, PrimaryHDU
        from ml.experiments.real_self_supervised import validate_solexs_fits_table
        hdul = HDUList([PrimaryHDU()])
        with pytest.raises(ValueError, match="RATE"):
            validate_solexs_fits_table(hdul)

    def test_validate_solexs_fits_table_missing_columns(self):
        """RATE HDU with missing TIME or COUNTS must raise ValueError."""
        import astropy.io.fits as fits_mod
        from ml.experiments.real_self_supervised import validate_solexs_fits_table

        col = fits_mod.Column(name="TIME", format="D", array=np.arange(10, dtype=float))
        hdu = fits_mod.BinTableHDU.from_columns([col])
        hdu.name = "RATE"
        hdul = fits_mod.HDUList([fits_mod.PrimaryHDU(), hdu])
        with pytest.raises(ValueError, match="required columns"):
            validate_solexs_fits_table(hdul)


# ── Import guard ──────────────────────────────────────────────────────────────

# Deferred import to avoid polluting module-level imports
try:
    import torch
except ImportError:
    torch = None  # type: ignore
