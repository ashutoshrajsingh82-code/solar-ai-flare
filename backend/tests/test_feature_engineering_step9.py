import numpy as np
import pandas as pd
import pytest

from ml.features.engineering import (
    FeatureConfig, add_rolling_features, get_feature_columns, select_feature_columns,
)
from ml.features.manifest import (
    create_feature_manifest, load_feature_manifest, save_feature_manifest,
    validate_feature_manifest,
)
from ml.preprocessing.transform import FluxTransformer
from ml.preprocessing.alignment import AlignmentConfig, align_and_resample


def _frame(n=35):
    t = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
    x = 1.0 + (np.arange(n) % 5) * 0.04
    if n > 20:
        x[20] = 4.0
    return pd.DataFrame({
        "timestamp": t,
        "solexs_flux": x,
        "helios_flux": x * 2,
    })


def _ratio_config(**kwargs):
    metadata = {
        "helios_flux": {"units": "W m-2", "energy_band_kev": [10, 20],
                        "physical_quantity": "energy_flux", "measurement_convention": "integrated_band",
                        "ratio_compatible": True},
        "solexs_flux": {"units": "W m-2", "energy_band_kev": [1, 5],
                        "physical_quantity": "energy_flux", "measurement_convention": "integrated_band",
                        "ratio_compatible": True},
    }
    return FeatureConfig(windows_minutes=[5], spike_min_samples=3,
                         enable_hard_soft_ratio=True, channel_metadata=metadata, **kwargs)


def test_future_perturbation_preserves_every_feature_at_forecast_origin():
    raw = _frame()
    cfg = _ratio_config()
    baseline = add_rolling_features(raw, cfg)
    origin = 24
    changed = raw.copy()
    changed.loc[origin + 1:, ["solexs_flux", "helios_flux"]] *= 1000
    changed.loc[origin + 1:, "solexs_flux"] = np.nan
    recomputed = add_rolling_features(changed, cfg)
    numeric = [c for c in baseline if c not in {"timestamp"} and pd.api.types.is_numeric_dtype(baseline[c])]
    np.testing.assert_allclose(
        baseline.loc[origin, numeric].to_numpy(dtype=float),
        recomputed.loc[origin, numeric].to_numpy(dtype=float),
        rtol=0, atol=0, equal_nan=True,
    )
    assert baseline.loc[origin, "solexs_spike_flag"] == recomputed.loc[origin, "solexs_spike_flag"]


def test_causal_spikes_gaps_duplicates_and_insufficient_history():
    raw = _frame()
    regular = add_rolling_features(raw, FeatureConfig(windows_minutes=[5], spike_min_samples=3))
    assert regular.loc[5, "sampling_interval_seconds"] == 60
    assert regular.loc[5, "irregular_sampling"] == 0
    assert regular.loc[20, "solexs_spike_flag"] == 1
    consecutive = raw.copy()
    consecutive.loc[21, "solexs_flux"] = 4.0
    consecutive.loc[21, "helios_flux"] = 8.0
    sustained = add_rolling_features(consecutive, FeatureConfig(windows_minutes=[5], spike_min_samples=3))
    assert sustained.loc[20, "solexs_spike_flag"] == 1
    assert sustained.loc[21, "solexs_spike_flag"] == 1

    raw.loc[10, "timestamp"] += pd.Timedelta(seconds=25)
    features = add_rolling_features(raw, FeatureConfig(windows_minutes=[5], spike_min_samples=3))
    assert features.loc[0, "solexs_spike_flag"] != features.loc[0, "solexs_spike_flag"]
    assert features.loc[10, "irregular_sampling"] == 1
    raw.loc[11:, "timestamp"] += pd.Timedelta(minutes=10)
    features = add_rolling_features(raw, FeatureConfig(windows_minutes=[5], spike_min_samples=3))
    assert features.loc[11, "long_observation_gap"] == 1
    duplicate = pd.concat([raw.iloc[:2], raw.iloc[[1]], raw.iloc[2:]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicates"):
        add_rolling_features(duplicate)


def test_missingness_and_time_since_valid_are_explicit_and_causal():
    raw = _frame()
    raw.loc[8, "solexs_flux"] = np.nan
    features = add_rolling_features(raw, FeatureConfig(windows_minutes=[5]))
    assert features.loc[8, "solexs_missing"] == 1
    assert features.loc[8, "solexs_time_since_valid_min"] > 0
    assert np.isnan(features.loc[8, "solexs_flux_diff_5m"])


def test_registry_selection_isolates_each_instrument_and_global_time_quality():
    features = add_rolling_features(_frame(), FeatureConfig(windows_minutes=[5]))
    solexs = select_feature_columns(features, "solexs_only")
    helios = select_feature_columns(features, "helios_only")
    assert "sampling_interval_seconds" in solexs and "sampling_interval_seconds" in helios
    assert all(not name.startswith("helios") for name in solexs)
    assert all(not name.startswith("solexs") for name in helios)
    assert not any("hard_soft_ratio" in name for name in get_feature_columns(features))


def test_ratio_requires_compatible_metadata_and_counts_invalid_inputs():
    raw = _frame()
    raw.loc[5, "solexs_flux"] = 0
    raw.loc[6, "helios_flux"] = np.nan
    valid = add_rolling_features(raw, _ratio_config())
    assert valid.attrs["feature_quality_report"]["enabled"] is True
    assert valid.attrs["feature_quality_report"]["invalid_ratio_count"] == 2
    assert np.isnan(valid.loc[5, "hard_soft_ratio"])
    incompatible = _ratio_config()
    incompatible.channel_metadata["helios_flux"]["units"] = "counts/s"
    disabled = add_rolling_features(raw, incompatible)
    assert "hard_soft_ratio" not in disabled
    assert not disabled.attrs["feature_quality_report"]["enabled"]


def test_manifest_round_trip_order_schema_and_instrument_compatibility(tmp_path):
    cfg = FeatureConfig(windows_minutes=[5])
    features = add_rolling_features(_frame(), cfg)
    names = select_feature_columns(features, "solexs_only", cfg)
    manifest = create_feature_manifest(
        features, names, "solexs_only", cfg, source_frame=_frame(),
        split_frames={"train": _frame(20), "validation": _frame(8), "test": _frame(7)},
        preprocessing={"fit_on": "train_split_only"}, random_seed=7,
    )
    path = tmp_path / "feature_manifest.json"
    save_feature_manifest(path, manifest)
    loaded = load_feature_manifest(path)
    selected = validate_feature_manifest(loaded, features, "solexs_only")
    assert list(selected.columns) == names
    with pytest.raises(ValueError, match="configuration"):
        validate_feature_manifest(loaded, features, "fusion")


def test_feature_generation_is_reproducible_and_duplicate_timestamps_fail():
    raw, cfg = _frame(), FeatureConfig(windows_minutes=[5])
    first, second = add_rolling_features(raw, cfg), add_rolling_features(raw.copy(), cfg)
    pd.testing.assert_frame_equal(first, second)
    raw.loc[1, "timestamp"] = raw.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="duplicates"):
        add_rolling_features(raw, cfg)


def test_chronological_training_features_manifest_and_inference_round_trip():
    raw = _frame(100)
    train, validation, test = raw.iloc[:70], raw.iloc[70:85], raw.iloc[85:]
    transformer = FluxTransformer().fit(train)
    transformed = pd.concat(
        [transformer.transform(part) for part in (train, validation, test)],
        ignore_index=True,
    )
    cfg = FeatureConfig()
    features = add_rolling_features(transformed, cfg)
    names = select_feature_columns(features, "solexs_only", cfg)
    manifest = create_feature_manifest(
        features, names, "solexs_only", cfg, source_frame=raw,
        split_frames={"train": train, "validation": validation, "test": test},
        preprocessing={"fit_on": "train_split_only", "transformer": transformer.config.scaler_type},
    )

    # Simulate an inference replay using the saved fitted transformer/config.
    replay = add_rolling_features(transformer.transform(raw), cfg)
    ordered = validate_feature_manifest(manifest, replay, "solexs_only")
    expected = features.loc[:, names]
    pd.testing.assert_frame_equal(ordered, expected)


def test_resampling_labels_aggregates_at_the_right_edge_when_available():
    raw = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01T00:00:20Z", "2026-01-01T00:00:50Z", "2026-01-01T00:01:10Z"
        ]),
        "solexs_flux": [1.0, 3.0, 100.0],
        "helios_flux": [2.0, 4.0, 200.0],
    })
    aligned, _ = align_and_resample(raw, AlignmentConfig(target_cadence="1min"))
    first_edge = aligned.loc[aligned.timestamp == pd.Timestamp("2026-01-01T00:01:00Z")].iloc[0]
    assert first_edge.solexs_flux == 2.0
    assert first_edge.solexs_flux != 100.0
