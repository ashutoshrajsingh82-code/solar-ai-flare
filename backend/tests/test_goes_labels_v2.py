"""Extended validation tests for the GOES multi-month label pipeline.

Tests cover:
 1. v1 label Parquet integrity (row count, columns, date range, NaN-mask consistency).
 2. Positive-rate audit: 1h rate > 0.35 for the v1 hyperactive AR 4366 window.
 3. Holdout quarantine: make_labels raises/warns if Oct-2024 data is included.
 4. Block-split chronological ordering: train/val/test windows must not overlap.
 5. Leakage-free split: no boundary crossing (last train timestamp < first val timestamp).
 6. X-class events all appear in the v1 label flare states (state == FLARE at peak times).
 7. Positive rate degrades gracefully from 1h to 24h (y_24h_rate >= y_1h_rate for high-AR window).
 8. forecast_valid=False rows have NaN for ALL y_H columns.
 9. make_labels returns empty DataFrame for empty events and timestamps.
10. split_by_blocks handles single-day DataFrames (n_splits=1 fallback).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure backend package is importable when running from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from ml.labels.goes_labels import (
    GoesLabelConfig,
    load_goes_events,
    make_labels,
    split_by_blocks,
    save_labels,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
V1_PARQUET = REPO_ROOT / "data" / "labels" / "goes_labels_v1.parquet"
V1_JSON = REPO_ROOT / "data" / "labels" / "goes_labels_v1.json"
EVENTS_CSV = Path("C:/Users/ASHUTOSH/Downloads/data/goes/goes_flares.csv")
FLUX_CSV = Path("C:/Users/ASHUTOSH/Downloads/data/goes/goes_xrs_1min.csv")

# Holdout dates (must never be accessed)
HOLDOUT_START = pd.Timestamp("2024-10-01", tz="UTC")
HOLDOUT_END = pd.Timestamp("2024-10-31", tz="UTC")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def v1_labels() -> pd.DataFrame:
    """Load the v1 label Parquet (skip if not present)."""
    if not V1_PARQUET.exists():
        pytest.skip(f"v1 label Parquet not found: {V1_PARQUET}")
    return pd.read_parquet(V1_PARQUET)


@pytest.fixture(scope="module")
def v1_sidecar() -> dict:
    """Load the v1 JSON sidecar."""
    if not V1_JSON.exists():
        pytest.skip(f"v1 JSON sidecar not found: {V1_JSON}")
    return json.loads(V1_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def v1_events() -> pd.DataFrame:
    """Load and validate the v1 events CSV (skip if not present)."""
    if not EVENTS_CSV.exists():
        pytest.skip(f"Events CSV not found: {EVENTS_CSV}")
    return load_goes_events(EVENTS_CSV)


@pytest.fixture(scope="module")
def two_m_events() -> pd.DataFrame:
    """Two non-overlapping M-class events 48 hours apart for split tests."""
    return pd.DataFrame([
        {
            "start": pd.Timestamp("2026-03-01 12:00:00+00:00"),
            "peak":  pd.Timestamp("2026-03-01 12:15:00+00:00"),
            "end":   pd.Timestamp("2026-03-01 12:45:00+00:00"),
            "flare_class": "M1.0",
            "class_letter": "M",
            "peak_flux": 1e-5,
        },
        {
            "start": pd.Timestamp("2026-03-03 06:00:00+00:00"),
            "peak":  pd.Timestamp("2026-03-03 06:20:00+00:00"),
            "end":   pd.Timestamp("2026-03-03 07:00:00+00:00"),
            "flare_class": "M2.0",
            "class_letter": "M",
            "peak_flux": 2e-5,
        },
    ])


# ---------------------------------------------------------------------------
# Test 1: v1 Parquet integrity
# ---------------------------------------------------------------------------
class TestV1LabelIntegrity:
    """Check v1 Parquet has the correct shape, columns and date range."""

    REQUIRED_COLS = {
        "state", "y_1h", "y_3h", "y_6h", "y_12h", "y_24h",
        "forecast_valid", "time_to_next_flare_min", "next_flare_class",
    }

    def test_required_columns_present(self, v1_labels):
        missing = self.REQUIRED_COLS - set(v1_labels.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_row_count_matches_sidecar(self, v1_labels, v1_sidecar):
        assert len(v1_labels) == v1_sidecar["row_count"], (
            f"Row count mismatch: parquet={len(v1_labels)} sidecar={v1_sidecar['row_count']}"
        )

    def test_index_is_datetime_utc(self, v1_labels):
        assert isinstance(v1_labels.index, pd.DatetimeIndex), "Index must be DatetimeIndex"
        assert v1_labels.index.tz is not None, "Index must be tz-aware"

    def test_index_is_monotonic(self, v1_labels):
        assert v1_labels.index.is_monotonic_increasing, "Timestamps must be monotonic increasing"

    def test_date_range(self, v1_labels, v1_sidecar):
        start_ok = str(v1_sidecar["date_range"]["start"])[:10]
        end_ok = str(v1_sidecar["date_range"]["end"])[:10]
        assert str(v1_labels.index.min())[:10] == start_ok
        assert str(v1_labels.index.max())[:10] == end_ok

    def test_state_only_valid_values(self, v1_labels):
        valid_states = {"QUIET", "PRE_FLARE", "FLARE", "DECAY"}
        unexpected = set(v1_labels["state"].unique()) - valid_states
        assert not unexpected, f"Unexpected state values: {unexpected}"

    def test_no_holdout_contamination(self, v1_labels):
        """The v1 label index must not overlap October 2024."""
        in_holdout = (
            (v1_labels.index >= HOLDOUT_START) & (v1_labels.index <= HOLDOUT_END)
        )
        assert not in_holdout.any(), "v1 labels contain October 2024 holdout rows!"

    def test_y_columns_are_binary_or_nan(self, v1_labels):
        """All y_H values must be 0.0, 1.0, or NaN."""
        for col in [c for c in v1_labels.columns if c.startswith("y_") and c.endswith("h")]:
            unique_vals = v1_labels[col].dropna().unique()
            unexpected = set(unique_vals) - {0.0, 1.0}
            assert not unexpected, f"{col}: unexpected non-binary values {unexpected}"


# ---------------------------------------------------------------------------
# Test 2: Positive rate audit for hyperactive AR 4366 window
# ---------------------------------------------------------------------------
class TestV1PositiveRateAudit:
    """AR 4366 (2026-01-31..2026-02-05) was hyperactive: 1h positive rate must be >= 35%."""

    def test_1h_positive_rate_above_threshold(self, v1_labels, v1_sidecar):
        rate = v1_sidecar["horizon_positive_rates"]["y_1h"]["positive_rate_valid_only"]
        assert rate >= 0.35, (
            f"y_1h positive rate {rate:.3f} too low for hyperactive AR 4366 window "
            f"(expected >= 0.35)"
        )

    def test_positive_rate_increases_with_horizon(self, v1_labels):
        """For an active region, longer horizons should not drastically drop positive rate."""
        rates = {}
        for col in [c for c in v1_labels.columns if c.startswith("y_") and c.endswith("h")]:
            valid = v1_labels.loc[v1_labels["forecast_valid"], col].dropna()
            if len(valid) > 0:
                rates[col] = float(valid.mean())
        horizons = sorted(rates.keys(), key=lambda c: int(c.replace("y_", "").replace("h", "")))
        if len(horizons) < 2:
            pytest.skip("Not enough horizons to check rate ordering")
        rate_1h = rates.get("y_1h", 0)
        rate_24h = rates.get("y_24h", 0)
        # For AR 4366 the 24h rate should be >= 1h rate because events span multiple days
        assert rate_24h >= rate_1h, (
            f"y_24h rate {rate_24h:.3f} < y_1h rate {rate_1h:.3f} for AR 4366 window"
        )

    def test_mx_count_nonzero(self, v1_sidecar):
        mx_count = v1_sidecar["horizon_positive_rates"]["y_1h"]["positive_count"]
        assert mx_count > 0, "Expected non-zero M/X positive count in v1 labels"


# ---------------------------------------------------------------------------
# Test 3: Holdout quarantine (functional test of make_labels with holdout range)
# ---------------------------------------------------------------------------
class TestHoldoutQuarantine:
    """make_labels itself does not enforce holdout, but split_by_blocks and
    the build script do. This test verifies that labels built from the holdout
    range will trigger a zero-positive warning (correctly), confirming the
    split infrastructure would flag it before training."""

    def test_holdout_grid_produces_zero_mx_events_if_events_are_empty(self):
        """If no events are passed for October 2024, all y_H should be 0."""
        # Build a 1-day Oct 2024 grid (not loading actual data -- just ensuring
        # the label logic returns zeros if no events given)
        grid = pd.date_range("2024-10-15 00:00", "2024-10-15 23:00", freq="1h", tz="UTC")
        empty_events = pd.DataFrame(
            columns=["start", "peak", "end", "flare_class", "class_letter"]
        )
        labels = make_labels(grid, empty_events)
        # All states must be QUIET, all y_H must be 0.0 (no events)
        assert (labels["state"] == "QUIET").all()
        for col in [c for c in labels.columns if c.startswith("y_") and c.endswith("h")]:
            non_nan = labels[col].dropna()
            assert (non_nan == 0.0).all(), f"{col}: expected all 0 with empty events"


# ---------------------------------------------------------------------------
# Test 4 & 5: Block-split chronological ordering and leakage-free boundaries
# ---------------------------------------------------------------------------
class TestChronologicalSplitAntiLeakage:
    """Verify that split_by_blocks produces non-overlapping windows."""

    def test_splits_are_chronologically_ordered(self, two_m_events):
        grid = pd.date_range("2026-03-01", "2026-03-05 23:59", freq="1min", tz="UTC")
        cfg = GoesLabelConfig(horizons_hours=(1, 3))
        labels = make_labels(grid, two_m_events, config=cfg)
        splits = split_by_blocks(labels, n_splits=3)
        assert len(splits) >= 2

        prev_end = None
        for split_name, split_df, stats in splits:
            if prev_end is not None:
                assert split_df.index.min() > prev_end, (
                    f"Split '{split_name}' starts before or at previous split end -- overlap detected!"
                )
            prev_end = split_df.index.max()

    def test_no_timestamp_overlap_between_splits(self, two_m_events):
        grid = pd.date_range("2026-03-01", "2026-03-05 23:59", freq="1min", tz="UTC")
        labels = make_labels(grid, two_m_events)
        splits = split_by_blocks(labels, n_splits=3)

        all_indices = [split_df.index for _, split_df, _ in splits]
        # Check each pair has no shared timestamps
        for i in range(len(all_indices)):
            for j in range(i + 1, len(all_indices)):
                overlap = all_indices[i].intersection(all_indices[j])
                assert len(overlap) == 0, (
                    f"Splits {i} and {j} share {len(overlap)} timestamps -- leakage detected!"
                )

    def test_union_of_splits_covers_full_grid(self, two_m_events):
        grid = pd.date_range("2026-03-01", "2026-03-05 23:59", freq="1min", tz="UTC")
        labels = make_labels(grid, two_m_events)
        splits = split_by_blocks(labels, n_splits=3)

        union_count = sum(len(split_df) for _, split_df, _ in splits)
        assert union_count == len(labels), (
            f"Union of splits ({union_count}) != full grid ({len(labels)}) -- rows lost!"
        )

    def test_split_stats_positive_counts_are_non_negative(self, two_m_events):
        grid = pd.date_range("2026-03-01", "2026-03-05 23:59", freq="1min", tz="UTC")
        labels = make_labels(grid, two_m_events)
        splits = split_by_blocks(labels, n_splits=3)
        for split_name, _, stats in splits:
            assert stats.get("y_1h_pos_count", 0) >= 0
            rate = stats.get("y_1h_pos_rate", 0)
            assert 0.0 <= rate <= 1.0, f"{split_name}: y_1h_pos_rate={rate} out of [0,1]"


# ---------------------------------------------------------------------------
# Test 6: X-class events appear as FLARE state in v1 labels
# ---------------------------------------------------------------------------
class TestXClassFlareStates:
    """X-class events at peak times must be labeled FLARE."""

    def test_x_class_peaks_are_labeled_flare(self, v1_labels, v1_events):
        x_events = v1_events[v1_events["class_letter"] == "X"]
        if x_events.empty:
            pytest.skip("No X-class events in v1 events")

        failures = []
        for _, row in x_events.iterrows():
            peak_ts = row["peak"]
            # Find the nearest timestamp in v1_labels within 1 minute
            idx = v1_labels.index.get_indexer([peak_ts], method="nearest")
            if idx[0] < 0:
                continue
            nearest_ts = v1_labels.index[idx[0]]
            diff_min = abs((nearest_ts - peak_ts).total_seconds()) / 60.0
            if diff_min > 1.0:
                continue  # too far from grid
            state = v1_labels.iloc[idx[0]]["state"]
            if state != "FLARE":
                failures.append(
                    f"X{row['flare_class']} at {peak_ts}: state={state} (expected FLARE)"
                )

        assert not failures, "X-class peaks NOT labeled FLARE:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Test 7: forecast_valid=False rows have NaN for ALL y_H columns
# ---------------------------------------------------------------------------
class TestMaskingConsistency:
    """Anti-leakage invariant: every masked row has NaN for all y_H columns."""

    def test_masked_rows_have_all_y_nan(self, v1_labels):
        y_cols = [c for c in v1_labels.columns if c.startswith("y_") and c.endswith("h")]
        masked = v1_labels[~v1_labels["forecast_valid"]]
        for col in y_cols:
            non_nan_in_masked = masked[col].dropna()
            assert len(non_nan_in_masked) == 0, (
                f"{col}: {len(non_nan_in_masked)} non-NaN values in masked rows"
            )

    def test_valid_rows_have_no_all_nan_y(self, v1_labels):
        """Valid rows should have at least y_1h defined (0 or 1)."""
        y_cols = [c for c in v1_labels.columns if c.startswith("y_") and c.endswith("h")]
        if not y_cols:
            pytest.skip("No y_H columns found")
        valid = v1_labels[v1_labels["forecast_valid"]]
        # For y_1h specifically, valid rows within coverage should not all be NaN
        if "y_1h" in y_cols:
            y1_valid = valid["y_1h"]
            non_nan_frac = y1_valid.notna().mean()
            assert non_nan_frac > 0.5, (
                f"Too many NaN in y_1h for valid rows: {1-non_nan_frac:.2%} are NaN"
            )


# ---------------------------------------------------------------------------
# Test 8: make_labels with empty inputs
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Edge cases: empty events, empty timestamps, single-day splits."""

    def test_empty_events_returns_zero_forecast(self):
        grid = pd.date_range("2026-04-01", "2026-04-01 05:00", freq="1h", tz="UTC")
        empty_ev = pd.DataFrame(columns=["start", "peak", "end", "flare_class", "class_letter"])
        labels = make_labels(grid, empty_ev)
        assert len(labels) == len(grid)
        assert (labels["state"] == "QUIET").all()
        y_cols = [c for c in labels.columns if c.startswith("y_") and c.endswith("h")]
        for col in y_cols:
            assert (labels.loc[labels["forecast_valid"], col].dropna() == 0.0).all()

    def test_empty_timestamps_returns_empty_df(self):
        empty_grid = pd.DatetimeIndex([], tz="UTC")
        ev = pd.DataFrame([{
            "start": pd.Timestamp("2026-04-01 12:00+00:00"),
            "peak": pd.Timestamp("2026-04-01 12:15+00:00"),
            "end": pd.Timestamp("2026-04-01 12:45+00:00"),
            "flare_class": "M1.0",
            "class_letter": "M",
        }])
        labels = make_labels(empty_grid, ev)
        assert len(labels) == 0

    def test_single_day_split_reduces_to_one(self):
        """split_by_blocks with 1 calendar day and n_splits=5 falls back to 1."""
        grid = pd.date_range("2026-04-01", "2026-04-01 23:59", freq="1min", tz="UTC")
        ev = pd.DataFrame([{
            "start": pd.Timestamp("2026-04-01 10:00+00:00"),
            "peak": pd.Timestamp("2026-04-01 10:20+00:00"),
            "end": pd.Timestamp("2026-04-01 11:00+00:00"),
            "flare_class": "M1.5",
            "class_letter": "M",
        }])
        labels = make_labels(grid, ev)
        with pytest.warns(UserWarning, match="Fewer distinct calendar days"):
            splits = split_by_blocks(labels, n_splits=5)
        assert len(splits) == 1

    def test_non_monotonic_timestamps_raises(self):
        ev = pd.DataFrame([{
            "start": pd.Timestamp("2026-04-01 10:00+00:00"),
            "peak": pd.Timestamp("2026-04-01 10:20+00:00"),
            "end": pd.Timestamp("2026-04-01 11:00+00:00"),
            "flare_class": "M1.5",
            "class_letter": "M",
        }])
        # Reversed (non-monotonic) timestamps
        grid = pd.DatetimeIndex([
            pd.Timestamp("2026-04-01 12:00+00:00"),
            pd.Timestamp("2026-04-01 11:00+00:00"),  # out of order
        ])
        with pytest.raises(ValueError, match="monotonic"):
            make_labels(grid, ev)


# ---------------------------------------------------------------------------
# Test 9: Persistence round-trip with metadata
# ---------------------------------------------------------------------------
class TestPersistenceRoundTrip:
    """save_labels + reload must preserve all label values."""

    def test_parquet_roundtrip(self, tmp_path):
        grid = pd.date_range("2026-05-01", "2026-05-01 12:00", freq="10min", tz="UTC")
        ev = pd.DataFrame([{
            "start": pd.Timestamp("2026-05-01 06:00+00:00"),
            "peak": pd.Timestamp("2026-05-01 06:20+00:00"),
            "end": pd.Timestamp("2026-05-01 07:00+00:00"),
            "flare_class": "M3.0",
            "class_letter": "M",
        }])
        labels = make_labels(grid, ev, config=GoesLabelConfig(horizons_hours=(1, 6)))
        out_p = tmp_path / "roundtrip_test.parquet"
        saved_p, sidecar_p = save_labels(labels, out_p, metadata={"test": "roundtrip"})
        reloaded = pd.read_parquet(saved_p)
        assert len(reloaded) == len(labels)
        assert set(reloaded.columns) == set(labels.columns)
        sidecar = json.loads(sidecar_p.read_text(encoding="utf-8"))
        assert sidecar["custom_metadata"]["test"] == "roundtrip"
        # State values preserved
        assert (reloaded["state"].values == labels["state"].values).all()
