import numpy as np
import pandas as pd
import pytest

from ml.features.windowing import (
    WindowConfig,
    build_split_audit,
    build_windows,
    chronological_split,
)


def _times(n=12):
    return pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")


def test_chronological_split_sorts_and_reproducibly_obeys_explicit_boundaries():
    frame = pd.DataFrame({"timestamp": _times(), "value": np.arange(12)})
    shuffled = frame.sample(frac=1, random_state=12)
    train, validation, test = chronological_split(
        shuffled,
        train_end="2026-01-01T04:00:00Z",
        val_end="2026-01-01T08:00:00+00:00",
    )
    assert train.timestamp.max() < validation.timestamp.min()
    assert validation.timestamp.max() < test.timestamp.min()
    assert len(train) == len(validation) == len(test) == 4
    second = chronological_split(frame, train_end="2026-01-01T04:00:00Z", val_end="2026-01-01T08:00:00Z")
    assert [part.timestamp.tolist() for part in (train, validation, test)] == [part.timestamp.tolist() for part in second]


@pytest.mark.parametrize("timestamps, message", [
    (["not-a-date", "2026-01-01", "2026-01-02"], "valid UTC"),
    (["2026-01-01", "2026-01-01", "2026-01-02"], "duplicates"),
])
def test_chronological_split_rejects_invalid_or_duplicate_timestamps(timestamps, message):
    with pytest.raises(ValueError, match=message):
        chronological_split(pd.DataFrame({"timestamp": timestamps, "value": [1, 2, 3]}))


def test_explicit_split_requires_three_nonempty_ordered_periods():
    frame = pd.DataFrame({"timestamp": _times(5)})
    with pytest.raises(ValueError, match="non-empty"):
        chronological_split(frame, train_end="2026-01-01T00:00Z", val_end="2026-01-01T02:00Z")
    with pytest.raises(ValueError, match="strictly ordered"):
        chronological_split(frame, train_end="2026-01-01T03:00Z", val_end="2026-01-01T02:00Z")


def test_window_partition_uses_only_past_context_and_returns_origins_in_partition():
    times = _times(9)
    frame = pd.DataFrame({
        "timestamp": times,
        "x": np.arange(9, dtype=float),
        "state": [0] * 9,
        "forecast_1h": [0.0] * 9,
    })
    train, validation, test = chronological_split(frame, train_frac=1/3, val_frac=1/3)
    cfg = WindowConfig(sequence_length=3, channel_cols=["x"], horizon_cols=["forecast_1h"])
    context = frame[frame.timestamp < validation.timestamp.min()]
    X, _, _, origins = build_windows(validation, cfg, context_df=context)
    assert len(X) == len(validation)
    assert list(origins) == list(validation.timestamp)
    assert np.all(X[:, -1, 0] >= validation.index.min())

    changed = frame.copy()
    changed.loc[changed.timestamp > validation.timestamp.iloc[0], "x"] = 999
    changed_context = changed[changed.timestamp < validation.timestamp.min()]
    changed_validation = changed[changed.timestamp.isin(validation.timestamp)]
    changed_X, _, _, _ = build_windows(changed_validation, cfg, context_df=changed_context)
    np.testing.assert_array_equal(X[0], changed_X[0])
    assert train.timestamp.max() < validation.timestamp.min() < test.timestamp.min()


def test_split_scoped_labels_cannot_see_later_partition_events():
    frame = pd.DataFrame({
        "timestamp": _times(12), "x": np.arange(12, dtype=float),
        "state": [0] * 11 + [2], "flare_class": ["-"] * 11 + ["X"],
        "forecast_1h": [0.0] * 12,
    })
    train, validation, test = chronological_split(frame, train_frac=0.5, val_frac=0.25)
    # The pipeline generates labels within each origin partition, then marks
    # its horizon tail unsupported/unknown.
    from app.services.preprocess_service import generate_split_scoped_labels
    from ml.features.labels import LabelConfig
    labeled, purged = generate_split_scoped_labels(
        frame, {"train": train, "validation": validation, "test": test},
        LabelConfig(cadence_seconds=3600, horizons_hours=[1]),
    )
    train_labels = labeled[labeled.timestamp.isin(train.timestamp)].forecast_1h
    assert train_labels.iloc[:-1].eq(0).all()
    assert pd.isna(train_labels.iloc[-1])
    assert purged["train"]["forecast_1h"] == 1
    assert test.iloc[-1].flare_class == "X"
    assert validation.timestamp.max() < test.timestamp.min()


def test_split_audit_counts_physical_events_separately_and_marks_fixture_synthetic():
    frame = pd.DataFrame({
        "timestamp": _times(7), "state": [0, 2, 2, 3, 0, 2, 2],
        "flare_class": ["-", "M1", "M1", "M1", "-", "X1", "X1"],
        "forecast_1h": [0, 1, 1, 1, 0, 1, 1],
    })
    audit = build_split_audit({"test": frame}, ["forecast_1h"], provenance={"source": "demo"})
    row = audit["partitions"][0]
    assert row["positive_forecast_examples"] == 5
    assert row["distinct_observed_flare_events"] == 2
    assert row["distinct_observed_mx_events"] == 2
    assert row["m_class_events"] == row["x_class_events"] == 1
    assert row["event_observation_status"] == "synthetic_fixture"
    assert row["rare_event_evaluation_status"] == "synthetic_fixture_not_scientific_evaluation"


def test_event_onset_catalog_does_not_double_count_flare_crossing_boundary():
    frame = pd.DataFrame({
        "timestamp": _times(4), "state": [0, 2, 2, 2],
        "flare_class": ["-", "M", "M", "M"],
        "forecast_1h": [0, 0, 0, 0],
    })
    catalog = [{"timestamp": _times(4)[1], "flare_class_group": "M"}]
    audit = build_split_audit(
        {"train": frame.iloc[:2], "test": frame.iloc[2:]}, ["forecast_1h"],
        provenance={"source": "demo"}, event_catalog=catalog,
    )
    assert [row["distinct_observed_flare_events"] for row in audit["partitions"]] == [1, 0]
    assert [row["m_class_events"] for row in audit["partitions"]] == [1, 0]


def test_end_to_end_split_labels_context_windows_and_audit():
    frame = pd.DataFrame({
        "timestamp": _times(),
        "solexs_flux_scaled": np.arange(12, dtype=float),
        "helios_flux_scaled": np.arange(12, dtype=float) + 10,
        "state": [0] * 10 + [2, 0],
        "flare_class": ["-"] * 10 + ["M"] + ["-"],
    })
    train, validation, test = chronological_split(frame, train_frac=0.5, val_frac=0.25)
    from app.services.preprocess_service import generate_split_scoped_labels
    from ml.features.labels import LabelConfig
    labeled, purged = generate_split_scoped_labels(
        frame, {"train": train, "validation": validation, "test": test},
        LabelConfig(cadence_seconds=3600, horizons_hours=[1]),
    )
    config = WindowConfig(sequence_length=2,
                          channel_cols=["solexs_flux_scaled", "helios_flux_scaled"],
                          horizon_cols=["forecast_1h"])
    test_rows = labeled[labeled.timestamp.isin(test.timestamp)]
    context = labeled[labeled.timestamp < test.timestamp.min()].tail(1)
    X, _, targets, origins = build_windows(test_rows, config, context_df=context)
    assert list(origins) == list(test.timestamp)
    assert targets[0, 0] == 1  # event is in the next hour; event row itself is not positive
    assert targets[-1, 0] != 0 or np.isnan(targets[-1, 0])  # test horizon tail was purged
    assert len(X) == len(test)
    report = build_split_audit(
        {"train": labeled[labeled.timestamp.isin(train.timestamp)],
         "validation": labeled[labeled.timestamp.isin(validation.timestamp)],
         "test": test_rows}, ["forecast_1h"],
        {"train_end": train.timestamp.max().isoformat(), "validation_end": validation.timestamp.max().isoformat()},
        {"source": "demo", "synthetic": True, "purged": purged}, sequence_length=2,
    )
    test_audit = report["partitions"][-1]
    assert test_audit["positive_forecast_examples"] == 1
    assert test_audit["distinct_observed_flare_events"] == 1
    assert test_audit["windows_excluded_by_purge"] == 1
    assert test_audit["event_observation_status"] == "synthetic_fixture"


def test_split_audit_counts_cross_partition_history_context_as_valid_windows():
    frame = pd.DataFrame({
        "timestamp": _times(8),
        "x": np.arange(8, dtype=float),
        "state": [0] * 8,
        "forecast_1h": [0.0] * 6 + [np.nan, np.nan],
    })
    audit = build_split_audit(
        {"validation": frame.iloc[4:].copy()}, ["forecast_1h"],
        sequence_length=3, channel_cols=["x"],
        context_frames={"validation": frame.iloc[1:4].copy()},
    )
    row = audit["partitions"][0]
    assert row["historical_context_observations"] == 3
    assert row["valid_supervised_windows"] == 4
    assert row["forecast_window_counts"]["forecast_1h"]["known"] == 2


def test_baseline_preparation_does_not_label_training_from_a_test_event():
    from ml.evaluation.baseline_benchmark import prepare

    timestamps = pd.date_range("2026-01-01", periods=4 * 24 * 60, freq="min", tz="UTC")
    values = 1.0 + np.sin(np.arange(len(timestamps)) / 30.0) * 0.05
    state = np.zeros(len(timestamps), dtype=int)
    flare_class = np.full(len(timestamps), "-", dtype=object)
    event_idx = 3 * 24 * 60 + 18 * 60
    state[event_idx] = 2
    flare_class[event_idx] = "M"
    raw = pd.DataFrame({
        "timestamp": timestamps,
        "solexs_flux": values,
        "helios_flux": values * 1.1,
        "state": state,
        "flare_class": flare_class,
    })
    parts, _ = prepare(raw)
    train_known = parts[0].forecast_1h.dropna()
    assert train_known.eq(0).all()
    assert parts[0].forecast_1h.isna().any()
    assert parts[2].forecast_1h.eq(1).any()
    assert parts[2].forecast_24h.isna().all()
