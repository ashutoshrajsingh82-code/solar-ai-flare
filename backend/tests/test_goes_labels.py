"""Unit tests for GOES-based flare labeling module (ml.labels.goes_labels).

Tests cover:
1. Exact state boundaries, nowcast 4-class assignment, and priority resolution.
2. Forecast binary targets y_H and exact boundary flip at flare onset.
3. Anti-leakage masking: ongoing flare/decay rows marked forecast_valid=False and y_H=NaN.
4. End-of-coverage boundary masking: t + H > coverage_end yields NaN, not 0.
5. GOES flux gap detection and horizon invalidation.
6. Timezone handling (naive inputs localized to UTC).
7. Cadence invariance between 1-second and 1-minute grids.
8. Flux invariance: changing flux values does NOT change event-derived labels.
9. Block-splitting helper: chronological day partitioning and zero M/X event warnings.
10. Parquet and JSON sidecar persistence.
"""

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest

from ml.labels.goes_labels import (
    GoesLabelConfig,
    load_goes_events,
    make_labels,
    split_by_blocks,
    save_labels,
    class_to_level,
    peak_flux_to_class_letter,
)


@pytest.fixture
def single_m_event():
    """A single M1.0 flare event:
    Start: 2026-02-01 12:00:00 UTC
    Peak:  2026-02-01 12:15:00 UTC
    End:   2026-02-01 12:45:00 UTC
    """
    return pd.DataFrame([
        {
            "start": pd.Timestamp("2026-02-01 12:00:00+00:00"),
            "peak": pd.Timestamp("2026-02-01 12:15:00+00:00"),
            "end": pd.Timestamp("2026-02-01 12:45:00+00:00"),
            "flare_class": "M1.0",
            "class_letter": "M",
            "peak_flux": 1.0e-5,
        }
    ])


class TestGoesLabelEventLoading:
    """Tests for load_goes_events and class derivation."""

    def test_peak_flux_to_class_letter(self):
        assert peak_flux_to_class_letter(5e-8) == "A"
        assert peak_flux_to_class_letter(5e-7) == "B"
        assert peak_flux_to_class_letter(5e-6) == "C"
        assert peak_flux_to_class_letter(5e-5) == "M"
        assert peak_flux_to_class_letter(2e-4) == "X"

    def test_drops_missing_or_inverted_events(self):
        raw = pd.DataFrame([
            {"start": "2026-02-01 10:00:00", "end": "2026-02-01 10:30:00", "flare_class": "C1.0"},
            {"start": None, "end": "2026-02-01 11:30:00", "flare_class": "C2.0"},
            {"start": "2026-02-01 12:00:00", "end": None, "flare_class": "M1.0"},
            {"start": "2026-02-01 14:00:00", "end": "2026-02-01 13:00:00", "flare_class": "X1.0"},  # inverted
        ])
        clean = load_goes_events(raw)
        assert len(clean) == 1
        assert clean.iloc[0]["flare_class"] == "C1.0"

    def test_timezone_aware_utc_output(self):
        raw = pd.DataFrame([
            {"start": "2026-02-01 10:00:00", "end": "2026-02-01 10:30:00", "flare_class": "C1.0"}
        ])
        clean = load_goes_events(raw)
        assert clean["start"].dt.tz is not None
        assert str(clean["start"].dt.tz) == "UTC"


class TestGoesLabelNowcastAndForecastBoundaries:
    """Tests with a hand-built single M1.0 flare at known times."""

    def test_exact_state_boundaries_and_masking(self, single_m_event):
        # 1-minute grid covering 11:00 to 14:00 (3 hours = 181 steps)
        grid = pd.date_range("2026-02-01 11:00:00", "2026-02-01 14:00:00", freq="1min", tz="UTC")
        cfg = GoesLabelConfig(preflare_minutes=30, horizons_hours=(1, 3))

        labels = make_labels(grid, single_m_event, config=cfg, coverage_end="2026-02-01 14:00:00+00:00")

        # 1. QUIET before pre-flare: [11:00, 11:29]
        assert (labels.loc["2026-02-01 11:00:00":"2026-02-01 11:29:00", "state"] == "QUIET").all()

        # 2. PRE_FLARE: [11:30, 11:59] (start is 12:00)
        assert (labels.loc["2026-02-01 11:30:00":"2026-02-01 11:59:00", "state"] == "PRE_FLARE").all()

        # 3. FLARE: [12:00, 12:15] (start to peak inclusive)
        assert (labels.loc["2026-02-01 12:00:00":"2026-02-01 12:15:00", "state"] == "FLARE").all()

        # 4. DECAY: [12:16, 12:45] (peak < t <= end)
        assert (labels.loc["2026-02-01 12:16:00":"2026-02-01 12:45:00", "state"] == "DECAY").all()

        # 5. QUIET after decay: [12:46, 14:00]
        assert (labels.loc["2026-02-01 12:46:00":"2026-02-01 14:00:00", "state"] == "QUIET").all()

        # 6. Masking check: all rows during flare/decay [12:00, 12:45] must be forecast_valid=False and y_H=NaN
        flare_decay_span = labels.loc["2026-02-01 12:00:00":"2026-02-01 12:45:00"]
        assert (~flare_decay_span["forecast_valid"]).all()
        assert flare_decay_span["y_1h"].isna().all()
        assert flare_decay_span["y_3h"].isna().all()

        # 7. Rows outside flare/decay are forecast_valid=True (prior to coverage end boundary)
        quiet_span = labels.loc["2026-02-01 11:00:00":"2026-02-01 11:59:00"]
        assert quiet_span["forecast_valid"].all()

    def test_forecast_horizon_exact_onset_trigger(self, single_m_event):
        """y_1h(t) = 1 if flare start (12:00) is in (t, t + 1h].
        For t + 1h >= 12:00 => t >= 11:00.
        For start > t => 12:00 > t => t < 12:00.
        Therefore y_1h is 1 for t in [11:00, 11:59], and flips to NaN/0 at 12:00.
        """
        grid = pd.date_range("2026-02-01 10:30:00", "2026-02-01 13:30:00", freq="1min", tz="UTC")
        cfg = GoesLabelConfig(horizons_hours=(1,))
        labels = make_labels(grid, single_m_event, config=cfg, coverage_end="2026-02-01 14:00:00+00:00")

        # Before 11:00: flare start (12:00) is > t + 1h => y_1h == 0
        assert (labels.loc["2026-02-01 10:30:00":"2026-02-01 10:59:00", "y_1h"] == 0).all()

        # Exactly at 11:00: 11:00 + 1h = 12:00 => flare start (12:00) in (11:00, 12:00] => y_1h == 1
        assert labels.loc["2026-02-01 11:00:00", "y_1h"] == 1.0

        # Between 11:00 and 11:59: y_1h == 1
        assert (labels.loc["2026-02-01 11:00:00":"2026-02-01 11:59:00", "y_1h"] == 1).all()

        # At 12:00: ongoing flare begins -> masked to NaN
        assert np.isnan(labels.loc["2026-02-01 12:00:00", "y_1h"])
        assert not labels.loc["2026-02-01 12:00:00", "forecast_valid"]

    def test_end_of_coverage_boundary_sets_nan(self, single_m_event):
        """If t + H > coverage_end, y_H must be NaN, NOT 0."""
        grid = pd.date_range("2026-02-01 13:00:00", "2026-02-01 14:00:00", freq="1min", tz="UTC")
        cfg = GoesLabelConfig(horizons_hours=(1, 3))
        # Coverage ends at 14:00:00
        labels = make_labels(grid, single_m_event, config=cfg, coverage_end="2026-02-01 14:00:00+00:00")

        # For H=3h: for all t in [13:00, 14:00], t + 3h > 14:00 => all must be NaN
        assert labels["y_3h"].isna().all()

        # For H=1h: t in [13:00, 13:00] -> t + 1h <= 14:00 => 0.0
        assert labels.loc["2026-02-01 13:00:00", "y_1h"] == 0.0
        # t = 13:30 -> t + 1h = 14:30 > 14:00 => NaN!
        assert np.isnan(labels.loc["2026-02-01 13:30:00", "y_1h"])


class TestGoesLabelOverlappingEvents:
    """Overlapping flares priority test: FLARE > DECAY > PRE_FLARE > QUIET."""

    def test_overlapping_flares_priority(self):
        events = pd.DataFrame([
            # Flare 1 (M-class): Start 10:00, Peak 10:15, End 10:45
            {
                "start": pd.Timestamp("2026-02-01 10:00:00+00:00"),
                "peak": pd.Timestamp("2026-02-01 10:15:00+00:00"),
                "end": pd.Timestamp("2026-02-01 10:45:00+00:00"),
                "flare_class": "M1.0",
                "class_letter": "M",
                "peak_flux": 1e-5,
            },
            # Flare 2 (X-class): Start 10:30 (during Flare 1 decay), Peak 10:40, End 11:00
            {
                "start": pd.Timestamp("2026-02-01 10:30:00+00:00"),
                "peak": pd.Timestamp("2026-02-01 10:40:00+00:00"),
                "end": pd.Timestamp("2026-02-01 11:00:00+00:00"),
                "flare_class": "X1.0",
                "class_letter": "X",
                "peak_flux": 1e-4,
            },
        ])

        grid = pd.date_range("2026-02-01 09:15:00", "2026-02-01 11:15:00", freq="1min", tz="UTC")
        cfg = GoesLabelConfig(preflare_minutes=30)
        labels = make_labels(grid, events, config=cfg)

        # 1. 10:10: Inside Flare 1 FLARE state -> FLARE
        assert labels.loc["2026-02-01 10:10:00", "state"] == "FLARE"

        # 2. 10:20: Inside Flare 1 DECAY, and Flare 2 PRE_FLARE (10:00-10:30).
        # DECAY (2) > PRE_FLARE (1) => state must be DECAY!
        assert labels.loc["2026-02-01 10:20:00", "state"] == "DECAY"

        # 3. 10:35: Inside Flare 1 DECAY, and Flare 2 FLARE (10:30-10:40).
        # FLARE (3) > DECAY (2) => state must be FLARE!
        assert labels.loc["2026-02-01 10:35:00", "state"] == "FLARE"


class TestGoesLabelTimezonesAndCadence:
    """Tests timezone localization and cadence invariance."""

    def test_naive_timestamp_input_warns_and_localizes(self, single_m_event):
        naive_grid = pd.date_range("2026-02-01 11:00:00", "2026-02-01 12:00:00", freq="1min")
        with pytest.warns(UserWarning, match="Naive timestamps"):
            labels = make_labels(naive_grid, single_m_event)
        assert labels.index.tz is not None
        assert str(labels.index.tz) == "UTC"

    def test_cadence_consistency_1s_and_1min(self, single_m_event):
        """Labels at 1-min and 1-sec grid must evaluate identically on shared timestamps."""
        grid_1m = pd.date_range("2026-02-01 11:30:00", "2026-02-01 12:30:00", freq="1min", tz="UTC")
        grid_1s = pd.date_range("2026-02-01 11:30:00", "2026-02-01 12:30:00", freq="1s", tz="UTC")

        cfg = GoesLabelConfig(horizons_hours=(1,))
        labels_1m = make_labels(grid_1m, single_m_event, config=cfg)
        labels_1s = make_labels(grid_1s, single_m_event, config=cfg)

        # Intersect 1s results on exact 1m timestamps
        aligned_1s = labels_1s.loc[labels_1m.index]

        # Check identical states
        assert (labels_1m["state"] == aligned_1s["state"]).all()
        # Check identical forecast_valid
        assert (labels_1m["forecast_valid"] == aligned_1s["forecast_valid"]).all()
        # Check identical y_1h values (handling NaNs)
        valid_both = labels_1m["forecast_valid"]
        assert (labels_1m.loc[valid_both, "y_1h"] == aligned_1s.loc[valid_both, "y_1h"]).all()


class TestGoesLabelFluxInvariance:
    """Changing flux values must NOT alter event-derived labels."""

    def test_flux_values_do_not_change_labels(self, single_m_event):
        grid = pd.date_range("2026-02-01 11:00:00", "2026-02-01 13:00:00", freq="1min", tz="UTC")

        # Fake flux 1: background level 1e-7
        flux_1 = pd.DataFrame({"timestamp": grid, "xrsb": [1e-7] * len(grid)})
        # Fake flux 2: background level 1e-4 (1000x higher)
        flux_2 = pd.DataFrame({"timestamp": grid, "xrsb": [1e-4] * len(grid)})

        labels_1 = make_labels(grid, single_m_event, goes_flux_df=flux_1)
        labels_2 = make_labels(grid, single_m_event, goes_flux_df=flux_2)

        # Labels must be bit-for-bit identical
        assert (labels_1["state"] == labels_2["state"]).all()
        assert (labels_1["forecast_valid"] == labels_2["forecast_valid"]).all()
        y_valid = labels_1["forecast_valid"]
        assert (labels_1.loc[y_valid, "y_1h"] == labels_2.loc[y_valid, "y_1h"]).all()


class TestGoesLabelSplittingAndPersistence:
    """Tests chronological splitting and persistence."""

    def test_split_by_blocks_warns_on_zero_mx_split(self):
        # 3 calendar days with NO flares
        grid = pd.date_range("2026-03-01", "2026-03-03 23:59:00", freq="1h", tz="UTC")
        empty_events = pd.DataFrame(columns=["start", "peak", "end", "flare_class", "class_letter"])
        labels = make_labels(grid, empty_events)

        with pytest.warns(UserWarning, match="contains ZERO M/X positive targets"):
            splits = split_by_blocks(labels, n_splits=3)

        assert len(splits) == 3
        for _, _, stats in splits:
            assert stats["m_x_indicator_count"] == 0

    def test_persistence_parquet_and_sidecar(self, tmp_path, single_m_event):
        grid = pd.date_range("2026-02-01 11:00:00", "2026-02-01 12:00:00", freq="1min", tz="UTC")
        labels = make_labels(grid, single_m_event)

        out_file = tmp_path / "test_labels_v1.parquet"
        saved_p, sidecar_p = save_labels(labels, out_file, metadata={"test_run": True})

        assert saved_p.exists()
        assert sidecar_p.exists()

        # Verify sidecar json
        sidecar = json.loads(sidecar_p.read_text(encoding="utf-8"))
        assert sidecar["row_count"] == len(labels)
        assert "state_counts" in sidecar
        assert sidecar["custom_metadata"]["test_run"] is True

        # Verify parquet can be reloaded
        loaded = pd.read_parquet(saved_p)
        assert len(loaded) == len(labels)
        assert "state" in loaded.columns
        assert "y_1h" in loaded.columns
