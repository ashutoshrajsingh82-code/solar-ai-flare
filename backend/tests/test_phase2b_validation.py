import numpy as np
import pandas as pd
import pytest
from app.services.model_service import _compute_full_metrics
from ml.models.baseline import BaselineMultiTaskModel, build_logistic_regression

from ml.features.engineering import FeatureConfig, add_rolling_features, add_stl_features
from ml.features.engineering import get_feature_columns
from ml.inference.input_validation import normalize_prediction_rows
from ml.features.windowing import WindowConfig, build_windows
from ml.training.validation import validate_classical_training_data, validate_training_data
from app.services.data_sources.goes_labels import (
    label_observations_from_goes,
    read_goes_flare_report_csv,
)


def test_feature_engineering_rejects_duplicate_or_out_of_order_observations():
    base = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="min", tz="UTC"),
        "solexs_flux": [1.0, 2.0, 3.0],
    })
    duplicate = base.copy()
    duplicate.loc[2, "timestamp"] = duplicate.loc[1, "timestamp"]
    with pytest.raises(ValueError, match="duplicates"):
        add_rolling_features(duplicate)

    out_of_order = base.iloc[[0, 2, 1]].reset_index(drop=True)
    with pytest.raises(ValueError, match="ordered by timestamp"):
        add_rolling_features(out_of_order)


def test_model_features_exclude_forecast_and_label_columns():
    frame = pd.DataFrame({
        "solexs_flux": [1.0], "solexs_diff1": [0.0], "forecast_1h": [1],
        "target_event": [1], "label_quality": [1], "nowcast_score": [1],
    })
    assert get_feature_columns(frame) == ["solexs_diff1"]


def test_feature_engineering_normalizes_utc_without_mutating_input():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=2, freq="min"),
        "solexs_flux": [1.0, 2.0],
    })
    features = add_rolling_features(frame, FeatureConfig(windows_minutes=[1]))
    assert str(features["timestamp"].dt.tz) == "UTC"
    assert frame["timestamp"].dt.tz is None


def test_stl_features_at_a_time_do_not_depend_on_future_observations():
    timestamps = pd.date_range("2026-01-01", periods=24, freq="min", tz="UTC")
    values = np.linspace(1.0, 4.0, len(timestamps)) + np.sin(np.arange(len(timestamps))) ** 2
    original = pd.DataFrame({"timestamp": timestamps, "solexs_flux": values})
    changed_future = original.copy()
    changed_future.loc[12:, "solexs_flux"] *= 100
    config = FeatureConfig(cadence_seconds=60, windows_minutes=[3], compute_stl=False, stl_period_minutes=3)

    original_features = add_rolling_features(original, config)
    changed_features = add_rolling_features(changed_future, config)
    original_stl = add_stl_features(original, config)
    changed_stl = add_stl_features(changed_future, config)

    assert np.allclose(
        original_stl.loc[:11, "solexs_flux_stl_trend"],
        changed_stl.loc[:11, "solexs_flux_stl_trend"],
        equal_nan=True,
    )
    assert np.allclose(
        original_features.loc[:11, "solexs_mean_3m"],
        changed_features.loc[:11, "solexs_mean_3m"],
        equal_nan=True,
    )

def _valid_training_arrays():
    X_train = np.ones((4, 3, 2), dtype=float)
    y_now_train = np.array([0, 1, 0, 1])
    y_fc_train = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    X_val = np.ones((2, 3, 2), dtype=float)
    y_now_val = np.array([0, 1])
    y_fc_val = np.array([[0, 1], [1, 0]])
    return X_train, y_now_train, y_fc_train, X_val, y_now_val, y_fc_val


def test_training_validation_rejects_unknown_or_single_class_targets():
    validate_training_data(*_valid_training_arrays())

    unknown = list(_valid_training_arrays())
    unknown[2] = unknown[2].astype(float)
    unknown[2][0, 0] = np.nan
    validate_training_data(*unknown)
    unknown_everywhere = list(_valid_training_arrays())
    unknown_everywhere[2] = unknown_everywhere[2].astype(float)
    unknown_everywhere[2][:, 0] = np.nan
    with pytest.raises(ValueError, match="both positive and negative"):
        validate_training_data(*unknown_everywhere)

    single_class = list(_valid_training_arrays())
    single_class[1] = np.zeros_like(single_class[1])
    with pytest.raises(ValueError, match="two nowcast classes"):
        validate_training_data(*single_class)

    constant_horizon = list(_valid_training_arrays())
    constant_horizon[2][:, 0] = 0
    with pytest.raises(ValueError, match="both positive and negative"):
        validate_training_data(*constant_horizon)


def test_training_validation_rejects_nonfinite_features_and_misaligned_lengths():
    invalid_features = list(_valid_training_arrays())
    invalid_features[0][0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="only finite values"):
        validate_training_data(*invalid_features)

    misaligned = list(_valid_training_arrays())
    misaligned[1] = misaligned[1][:-1]
    with pytest.raises(ValueError, match="sample counts must match"):
        validate_training_data(*misaligned)


def test_classical_training_validation_checks_data_and_label_diversity():
    X_train = np.ones((4, 3))
    y_now_train = np.array([0, 1, 0, 1])
    y_fc_train = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    X_test = np.ones((2, 3))
    y_now_test = np.array([0, 1])
    y_fc_test = np.array([[0, 1], [1, 0]])
    validate_classical_training_data(
        X_train, y_now_train, y_fc_train, X_test, y_now_test, y_fc_test,
    )

    invalid = y_fc_train.astype(float)
    invalid[0, 0] = np.nan
    validate_classical_training_data(
        X_train, y_now_train, invalid, X_test, y_now_test, y_fc_test,
    )

    invalid_nowcast = y_now_train.astype(float)
    invalid_nowcast[0] = 1.5
    with pytest.raises(ValueError, match="finite class indices"):
        validate_classical_training_data(
            X_train, invalid_nowcast, y_fc_train, X_test, y_now_test, y_fc_test,
        )


def test_windowing_requires_targets_and_preserves_unknown_forecast_labels():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=4, freq="min", tz="UTC"),
        "x": [1.0, 2.0, 3.0, 4.0],
        "state": [0, 1, 0, 1],
        "forecast_1h": [0.0, np.nan, 1.0, 0.0],
    })
    config = WindowConfig(sequence_length=1, channel_cols=["x"], horizon_cols=["forecast_1h"])
    X, y_now, y_forecast, end_timestamps = build_windows(frame, config)
    assert X.shape == (4, 1, 1)
    assert np.array_equal(y_now, [0, 1, 0, 1])
    assert np.array_equal(y_forecast[[0, 2, 3], 0], [0.0, 1.0, 0.0])
    assert np.isnan(y_forecast[1, 0])
    assert str(end_timestamps[0].tzinfo) == "UTC"

    with pytest.raises(ValueError, match="Missing required targets"):
        build_windows(frame.drop(columns="forecast_1h"), config)

    duplicate_timestamps = frame.copy()
    duplicate_timestamps.loc[2, "timestamp"] = duplicate_timestamps.loc[1, "timestamp"]
    with pytest.raises(ValueError, match="duplicates"):
        build_windows(duplicate_timestamps, config)


def test_prediction_input_rejects_duplicate_timestamps():
    rows = [
        {"timestamp": "2026-01-01T00:00:00Z", "solexs_flux": 1.0, "helios_flux": 2.0},
        {"timestamp": "2026-01-01T00:00:00Z", "solexs_flux": 1.1, "helios_flux": 2.1},
    ]
    with pytest.raises(ValueError, match="duplicate timestamps"):
        normalize_prediction_rows(rows)


def _goes_events(starts, classes=None):
    classes = classes or ["M"] * len(starts)
    return pd.DataFrame({
        "start_time": pd.to_datetime(starts, utc=True),
        "flare_class_group": classes,
    })


def _label_one(observation, event, coverage_end="2026-01-02T00:00:00Z"):
    events = _goes_events([event])
    labels, _ = label_observations_from_goes(
        pd.DataFrame({"timestamp": [observation]}), events,
        "2026-01-01T00:00:00Z", coverage_end, horizons_hours=(1,),
    )
    return labels["forecast_1h"].iloc[0]


def test_goes_forecast_window_includes_right_boundary_but_excludes_observation_time():
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    assert _label_one(start, start) == 0
    assert _label_one(start, start - pd.Timedelta(1, unit="ns")) == 0
    assert _label_one(start, start + pd.Timedelta(1, unit="ns")) == 1
    assert _label_one(start, start + pd.Timedelta(hours=1)) == 1
    assert _label_one(start, start + pd.Timedelta(hours=1, nanoseconds=1)) == 0
    assert _label_one(start, "2026-01-01T01:00:00+01:00") == 0  # same instant as t


def test_goes_reader_preserves_mixed_utc_offsets_and_subsecond_precision(tmp_path):
    path = tmp_path / "mixed-times.csv"
    pd.DataFrame([
        {"start_time": "2026-01-01T00:00:00.000000001Z", "time": "2026-01-01 00:01:00",
         "end_time": "2026-01-01T00:02:00Z", "flare_class": "M1.0"},
        {"start_time": "2026-01-01T01:00:00+01:00", "time": "2026-01-01T01:01:00+01:00",
         "end_time": "2026-01-01T01:02:00+01:00", "flare_class": "X1.0"},
    ]).to_csv(path, index=False)

    parsed = read_goes_flare_report_csv(
        path, "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z",
    )
    first, second = parsed.data["start_time"]
    assert first.value - second.value == 1
    assert str(first.tz) == "UTC"


def test_goes_coverage_empty_events_mixed_classes_and_duplicate_observations():
    starts = [
        pd.Timestamp("2026-01-01T00:01:00Z"),
        pd.Timestamp("2026-01-01T00:02:00Z"),
        pd.Timestamp("2026-01-01T00:03:00Z"),
    ]
    events = _goes_events(starts, ["C", "M", "X"])
    obs = pd.DataFrame({"timestamp": [
        "2026-01-01T00:00:00Z", "2026-01-01T00:02:00Z", "2026-01-01T00:03:00Z",
    ]})
    labels, _ = label_observations_from_goes(
        obs, events, "2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z", horizons_hours=(1,),
    )
    assert labels["forecast_1h"].tolist() == [1.0, 1.0, 0.0]

    empty_events = _goes_events([])
    boundary_obs = pd.DataFrame({"timestamp": [
        "2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T01:00:00.000000001Z",
        "2026-01-01T02:00:00Z", "2026-01-01T00:00:00Z",
    ]})
    covered, _ = label_observations_from_goes(
        boundary_obs, empty_events, "2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z", horizons_hours=(1,),
    )
    assert covered["forecast_1h"].iloc[:2].tolist() == [0.0, 0.0]
    assert covered["forecast_1h"].iloc[2:4].isna().all()
    assert covered["forecast_1h"].iloc[0] == covered["forecast_1h"].iloc[4]


def test_classical_model_fits_each_forecast_horizon_on_known_rows_only():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 3))
    y_now = np.array([0, 1] * 15)
    y_fc = np.column_stack([
        np.array([0, 1] * 15, dtype=float),
        np.array([1, 0] * 15, dtype=float),
    ])
    y_fc[:10, 0] = np.nan
    y_fc[10:20, 1] = np.nan
    model = BaselineMultiTaskModel(build_logistic_regression, "test")
    model.fit(X, y_now, y_fc)
    probabilities = model.predict_forecast_proba(X)
    assert probabilities.shape == (30, 2)
    assert np.isfinite(probabilities).all()
    assert len(model.forecast_models_) == 2


def test_evaluation_excludes_unknown_targets_independently_per_horizon():
    y_now = np.array([0, 1, 2])
    y_now_pred = np.array([0, 1, 1])
    y_fc = np.array([
        [np.nan, 0, 1, 0, 1],
        [np.nan, 1, 0, 1, 0],
        [np.nan, np.nan, 1, 0, 1],
    ], dtype=float)
    y_prob = np.array([
        [0.2, 0.1, 0.9, 0.1, 0.9],
        [0.3, 0.8, 0.2, 0.7, 0.2],
        [0.4, 0.5, 0.7, 0.2, 0.8],
    ])
    metrics = _compute_full_metrics(y_now, y_now_pred, y_fc, y_prob)
    assert metrics["forecast_per_horizon"]["1h"]["sample_count"] == 0
    assert metrics["forecast_per_horizon"]["1h"]["tss"] is None
    assert "No known labels" in metrics["forecast_per_horizon"]["1h"]["undefined_metric_reasons"]["all"]
    assert metrics["forecast_per_horizon"]["3h"]["sample_count"] == 2
    assert metrics["forecast_per_horizon"]["3h"]["positive_count"] == 1


