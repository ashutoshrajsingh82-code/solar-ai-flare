"""Unit tests for GOES multi-month labeling v2 enhancements (v3 test suite).

Tests cover:
 1. Holdout quarantine enforcement across all new code paths.
 2. Purge-gap validation (>= 24h between train/val and val/test).
 3. Baseline metric verification against analytical ground truth.
 4. Calendar-day block-bootstrap resampling logic.
 5. PRADAN overlap scanning with mocked file manifests.
"""
from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pytest

from ml.labels.goes_labels import (
    split_by_day_blocks,
    split_by_blocks,
)
from scripts.download_goes_chunks import (
    assert_no_holdout,
    audit_class_vs_peak_flux,
)
from scripts.build_goes_labels import assert_no_holdout_timestamps
from scripts.generate_goes_v2_report import (
    compute_binary_metrics,
    block_bootstrap_metrics,
)
from scripts.scan_instrument_overlap import (
    scan_pradan_downloads,
    analyze_split_and_epoch_overlap,
)


class TestHoldoutQuarantineNewPaths:
    """Verify that all new code paths strictly reject the October 2024 holdout."""

    def test_assert_no_holdout_raises_on_october_overlap(self):
        with pytest.raises(ValueError, match="FATAL HOLDOUT GUARD"):
            assert_no_holdout(pd.Timestamp("2024-10-01", tz="UTC"), pd.Timestamp("2024-10-31", tz="UTC"))

    def test_assert_no_holdout_raises_on_boundary_touch(self):
        with pytest.raises(ValueError, match="FATAL HOLDOUT GUARD"):
            assert_no_holdout(pd.Timestamp("2024-09-25", tz="UTC"), pd.Timestamp("2024-10-05", tz="UTC"))

    def test_assert_no_holdout_allows_valid_ranges(self):
        # May 2024 is pre-holdout
        assert_no_holdout(pd.Timestamp("2024-05-01", tz="UTC"), pd.Timestamp("2024-05-31", tz="UTC"))
        # November 2024 is post-holdout
        assert_no_holdout(pd.Timestamp("2024-11-01", tz="UTC"), pd.Timestamp("2024-11-30", tz="UTC"))

    def test_assert_no_holdout_timestamps_detects_contamination(self):
        dates = pd.date_range("2024-09-30", "2024-10-02", freq="1D", tz="UTC")
        with pytest.raises(ValueError, match="HOLDOUT CONTAMINATION"):
            assert_no_holdout_timestamps(dates)

    def test_assert_no_holdout_timestamps_passes_clean_data(self):
        dates = pd.date_range("2024-05-01", "2024-05-10", freq="1D", tz="UTC")
        assert_no_holdout_timestamps(dates)


class TestPurgeGapEnforcement:
    """Verify that split_by_day_blocks guarantees >= 24h purge gap."""

    @pytest.fixture
    def multi_day_df(self) -> pd.DataFrame:
        dates = pd.date_range("2025-01-01", "2025-01-20 23:59:00", freq="1h", tz="UTC")
        df = pd.DataFrame(index=dates)
        df["forecast_valid"] = True
        df["y_1h"] = 0
        df.iloc[10:15, df.columns.get_loc("y_1h")] = 1
        return df

    def test_purge_gap_between_consecutive_splits(self, multi_day_df):
        splits = split_by_day_blocks(
            multi_day_df,
            ratios=(0.70, 0.15, 0.15),
            purge_gap_days=1,
            split_names=("train", "val", "test"),
        )
        assert len(splits) == 3

        train_name, train_df, train_stats = splits[0]
        val_name, val_df, val_stats = splits[1]
        test_name, test_df, test_stats = splits[2]

        train_end = train_df.index.max()
        val_start = val_df.index.min()
        val_end = val_df.index.max()
        test_start = test_df.index.min()

        # Check gap between train and val is >= 24h
        train_val_gap = (val_start - train_end).total_seconds() / 3600.0
        assert train_val_gap >= 24.0, f"Train-val gap is {train_val_gap}h (< 24h)"

        # Check gap between val and test is >= 24h
        val_test_gap = (test_start - val_end).total_seconds() / 3600.0
        assert val_test_gap >= 24.0, f"Val-test gap is {val_test_gap}h (< 24h)"

    def test_splits_are_strictly_chronological_and_disjoint(self, multi_day_df):
        splits = split_by_day_blocks(multi_day_df, purge_gap_days=1)
        train_df, val_df, test_df = splits[0][1], splits[1][1], splits[2][1]

        assert train_df.index.max() < val_df.index.min()
        assert val_df.index.max() < test_df.index.min()

        train_set = set(train_df.index)
        val_set = set(val_df.index)
        test_set = set(test_df.index)

        assert train_set.isdisjoint(val_set)
        assert val_set.isdisjoint(test_set)
        assert train_set.isdisjoint(test_set)


class TestAnalyticalBaselineMetrics:
    """Verify metric computation against exact analytical formulas."""

    def test_known_confusion_matrix_metrics(self):
        # 20 TP, 10 FP, 10 FN, 60 TN
        # P = 30, N = 70, Total = 100
        y_true = np.array([1] * 20 + [0] * 10 + [1] * 10 + [0] * 60)
        y_pred = np.array([1] * 20 + [1] * 10 + [0] * 10 + [0] * 60)

        res = compute_binary_metrics(y_true, y_pred)

        assert res["tp"] == 20
        assert res["fp"] == 10
        assert res["fn"] == 10
        assert res["tn"] == 60

        expected_pod = 20 / 30  # 2/3 ~ 0.6667
        expected_pofd = 10 / 70  # 1/7 ~ 0.1429
        expected_far = 10 / 30  # 1/3 ~ 0.3333
        expected_tss = expected_pod - expected_pofd  # 11/21 ~ 0.5238

        assert np.isclose(res["pod"], expected_pod)
        assert np.isclose(res["pofd"], expected_pofd)
        assert np.isclose(res["far"], expected_far)
        assert np.isclose(res["tss"], expected_tss)

        # Expected HSS: 2*(TP*TN - FP*FN) / ((TP+FN)*(FN+TN) + (TP+FP)*(FP+TN))
        num = 2 * (20 * 60 - 10 * 10)  # 2 * (1200 - 100) = 2200
        den = (30 * 70) + (30 * 70)  # 2100 + 2100 = 4200
        expected_hss = 2200 / 4200  # 11/21 ~ 0.5238
        assert np.isclose(res["hss"], expected_hss)

        # Brier: mean squared error = (10 + 10) / 100 = 0.20
        assert np.isclose(res["brier"], 0.20)

    def test_all_negative_climatology_metrics(self):
        # If model always predicts 0
        y_true = np.array([1] * 10 + [0] * 90)
        y_pred = np.zeros(100, dtype=int)
        y_prob = np.full(100, fill_value=0.10)

        res = compute_binary_metrics(y_true, y_pred, y_prob)
        assert res["pod"] == 0.0
        assert res["pofd"] == 0.0
        assert res["tss"] == 0.0
        assert res["hss"] == 0.0
        # Brier score: 0.10 * (1 - 0.10)^2 + 0.90 * (0.10 - 0)^2 = 0.1*0.81 + 0.9*0.01 = 0.081 + 0.009 = 0.09
        assert np.isclose(res["brier"], 0.09)


class TestDayBlockBootstrap:
    """Verify block-bootstrap resampling logic."""

    def test_bootstrap_produces_monotonic_ci(self):
        dates = np.array(["2025-01-01"] * 50 + ["2025-01-02"] * 50 + ["2025-01-03"] * 50)
        y_true = np.random.choice([0, 1], size=150, p=[0.7, 0.3])
        y_pred = np.random.choice([0, 1], size=150, p=[0.7, 0.3])

        ci = block_bootstrap_metrics(dates, y_true, y_pred, n_bootstrap=100, seed=123)

        assert "tss" in ci
        assert "hss" in ci
        assert "brier" in ci
        for m in ("tss", "hss", "brier"):
            low, high = ci[m]
            assert low <= high, f"CI for {m} inverted: {low} > {high}"


class TestPradanOverlapScanner:
    """Verify PRADAN overlap scanning with mocked file names."""

    def test_mock_pradan_scan(self, tmp_path):
        # Create mock SoLEXS zip
        solexs_zip = tmp_path / "solexs_2026Oct23T184402685.zip"
        with zipfile.ZipFile(solexs_zip, "w") as zf:
            zf.writestr("AL1_SLX_L1_20260201_v1.0.zip", b"dummy")
            zf.writestr("AL1_SLX_L1_20260202_v1.0.zip", b"dummy")

        # Create mock HEL1OS zip with one full day and one partial day
        helios_zip = tmp_path / "hel1os_2026Oct23T184518914.zip"
        with zipfile.ZipFile(helios_zip, "w") as zf:
            # 20260201: two 12h files = 86,380s (> 80%)
            zf.writestr("HLS_20260201_000000_43190sec_lev1_V111.zip", b"dummy")
            zf.writestr("HLS_20260201_120000_43190sec_lev1_V111.zip", b"dummy")
            # 20260202: only one 5h file = 18,000s (< 80%)
            zf.writestr("HLS_20260202_000000_18000sec_lev1_V111.zip", b"dummy")

        df = scan_pradan_downloads(tmp_path)
        assert len(df) == 2

        row_01 = df[df["date"] == "2026-02-01"].iloc[0]
        assert bool(row_01["solexs_ok"]) is True
        assert bool(row_01["helios_ok"]) is True
        assert bool(row_01["both"]) is True

        row_02 = df[df["date"] == "2026-02-02"].iloc[0]
        assert bool(row_02["solexs_ok"]) is True
        assert bool(row_02["helios_ok"]) is False
        assert bool(row_02["both"]) is False
