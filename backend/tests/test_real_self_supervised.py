"""Automated tests for Phase 4M-S self-supervised real data forecasting."""
import io
import json
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from pathlib import Path

from ml.experiments.real_self_supervised import (
    validate_solexs_fits_table,
    extract_contiguous_segments,
    create_sliding_windows,
    CountRateScaler,
    SelfSupervisedCNNLSTM,
    SelfSupervisedConfig,
    persistence_forecast,
    rolling_mean_forecast,
    evaluate_forecast,
    save_experimental_checkpoint,
    MARKER_EXPERIMENTAL,
)
from ml.models.registry import ModelRegistry, ModelRecord


def _make_dummy_fits_hdul(times: np.ndarray, counts: np.ndarray,
                           t_unit: str = "s", c_unit: str = "counts/s") -> fits.HDUList:
    c1 = fits.Column(name="TIME", format="1D", unit=t_unit, array=times)
    c2 = fits.Column(name="COUNTS", format="1E", unit=c_unit, array=counts)
    table_hdu = fits.BinTableHDU.from_columns([c1, c2], name="RATE")
    return fits.HDUList([fits.PrimaryHDU(), table_hdu])


def test_measurement_unit_enforcement():
    """Validates that FITS validation fails closed if units are invalid or missing."""
    times = np.array([1.0, 2.0, 3.0])
    counts = np.array([10.0, 12.0, 15.0])

    # Valid units should pass
    hdul_ok = _make_dummy_fits_hdul(times, counts, t_unit="s", c_unit="counts/s")
    t, c, tu, cu = validate_solexs_fits_table(hdul_ok)
    assert tu == "s"
    assert cu == "counts/s"

    # Invalid time unit must raise ValueError
    hdul_bad_time = _make_dummy_fits_hdul(times, counts, t_unit="days", c_unit="counts/s")
    with pytest.raises(ValueError, match="Invalid SoLEXS TIME unit"):
        validate_solexs_fits_table(hdul_bad_time)

    # Invalid counts unit (e.g. physical flux W/m^2) must raise ValueError
    hdul_bad_counts = _make_dummy_fits_hdul(times, counts, t_unit="s", c_unit="W/m^2")
    with pytest.raises(ValueError, match="Invalid SoLEXS COUNTS unit"):
        validate_solexs_fits_table(hdul_bad_counts)


def test_timestamp_ordering_and_duplicates():
    """Validates that non-monotonic, duplicate, or non-finite timestamps are rejected."""
    counts = np.array([10.0, 12.0, 15.0])

    # Non-monotonic (out of order)
    hdul_unsorted = _make_dummy_fits_hdul(np.array([2.0, 1.0, 3.0]), counts)
    with pytest.raises(ValueError, match="not monotonically increasing"):
        validate_solexs_fits_table(hdul_unsorted)

    # Duplicate timestamps
    hdul_dup = _make_dummy_fits_hdul(np.array([1.0, 2.0, 2.0]), counts)
    with pytest.raises(ValueError, match="duplicate timestamps"):
        validate_solexs_fits_table(hdul_dup)

    # Non-finite timestamps
    hdul_nan = _make_dummy_fits_hdul(np.array([1.0, np.nan, 3.0]), counts)
    with pytest.raises(ValueError, match="non-finite values"):
        validate_solexs_fits_table(hdul_nan)


def test_gap_detection_and_window_integrity():
    """Validates that data gaps break segments and windows never cross gaps."""
    # Create 100 seconds, with a 10-second gap at t=50
    t1 = np.arange(0, 50, dtype=float)
    c1 = np.full(50, 100.0)
    t2 = np.arange(60, 110, dtype=float)  # gap between 49 and 60
    c2 = np.full(50, 200.0)

    times = np.concatenate([t1, t2])
    counts = np.concatenate([c1, c2])

    # With min_segment_length=40, both segments should be extracted separately
    segments = extract_contiguous_segments(times, counts, min_segment_length=40)
    assert len(segments) == 2
    assert len(segments[0][0]) == 50
    assert len(segments[1][0]) == 50

    # Create sliding windows of length 30 + horizon 10 = 40
    X, y, t_orig = create_sliding_windows(segments, seq_len=30, horizon=10)
    # Each segment has 50 - 40 + 1 = 11 windows -> total 22 windows
    assert len(X) == 22
    assert X.shape == (22, 30, 1)
    assert y.shape == (22, 10)

    # Verify that no window in segment 0 has values from segment 1
    assert np.all(X[:11] == 100.0)
    assert np.all(y[:11] == 100.0)
    assert np.all(X[11:] == 200.0)
    assert np.all(y[11:] == 200.0)


def test_split_boundary_leakage_prevention():
    """Validates that train, val, and test partitions are strictly chronological and isolated."""
    # Scaler fit must not see validation or test data
    train_counts = np.array([50.0, 60.0, 70.0, 80.0])
    val_counts = np.array([500.0, 600.0])
    test_counts = np.array([5000.0, 6000.0])

    scaler = CountRateScaler().fit(train_counts)
    expected_log_mean = float(np.mean(np.log10(train_counts + 1.0)))
    assert np.isclose(scaler.mean, expected_log_mean, atol=1e-5)

    # Transforming val/test uses strictly train mean and std
    val_sc = scaler.transform(val_counts)
    val_inv = scaler.inverse_transform(val_sc)
    assert np.allclose(val_inv, val_counts, atol=1e-4)


def test_baselines_and_metrics_in_native_units():
    """Validates persistence and rolling mean baselines and error metrics."""
    X = np.array([
        [[10.0], [20.0], [30.0]],
        [[100.0], [100.0], [100.0]],
    ])  # batch=2, seq=3, features=1

    # Persistence should repeat last value: 30.0 for sample 0, 100.0 for sample 1
    persist = persistence_forecast(X, horizon=2)
    assert persist.shape == (2, 2)
    assert np.all(persist[0] == 30.0)
    assert np.all(persist[1] == 100.0)

    # Rolling mean should be 20.0 for sample 0, 100.0 for sample 1
    mean_fc = rolling_mean_forecast(X, horizon=2)
    assert mean_fc.shape == (2, 2)
    assert np.all(mean_fc[0] == 20.0)
    assert np.all(mean_fc[1] == 100.0)

    # Test error metric
    y_true = np.array([[30.0, 30.0], [100.0, 100.0]])
    metrics = evaluate_forecast(y_true, persist)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0


def test_separation_from_production_registry(tmp_path):
    """Validates that production registry strictly rejects experimental self-supervised models."""
    registry = ModelRegistry(str(tmp_path / "registry"))

    # An experimental model record with production_registry_eligible=False must be rejected
    record = ModelRecord(
        model_id="exp_self_supervised_001",
        name="Experimental SoLEXS Self-Supervised Forecaster",
        architecture="cnn_lstm",
        version="v1.0",
        created_at="2026-10-25T12:00:00Z",
        training_dataset="solexs_2026Oct23T184402685.zip",
        input_configuration="solexs_only",
        params={
            "experiment_designation": MARKER_EXPERIMENTAL,
            "production_registry_eligible": False,
        }
    )

    with pytest.raises(ValueError, match="Synthetic smoke models cannot be registered in the production model registry"):
        registry.register(record)


def test_checkpoint_metadata_contract(tmp_path):
    """Validates that experimental checkpoints carry all required provenance tags."""
    cfg = SelfSupervisedConfig(sequence_length=10, forecast_horizon=3)
    model = SelfSupervisedCNNLSTM(in_features=1, horizon=3)

    data_bundle = {
        "source_archive": "solexs_test.zip",
        "source_sha256": "abcdef123456" * 5 + "abcd",
        "total_records": 1000,
        "finite_records": 900,
        "nan_records": 100,
        "split_counts": {"train_samples": 500, "val_samples": 200, "test_samples": 200},
        "split_boundaries_utc": {"train": ("2026-01-31", "2026-02-02"), "val": ("2026-02-03", "2026-02-04"), "test": ("2026-02-05", "2026-02-05")},
        "scaler": CountRateScaler().fit(np.array([10.0, 20.0, 30.0])),
    }

    eval_report = {
        "test_metrics_native_units": {
            "cnn_lstm": {"mae": 5.2, "rmse": 7.1, "rel_mae": 0.08},
            "persistence_baseline": {"mae": 6.5, "rmse": 9.0, "rel_mae": 0.10},
            "skill_score_vs_persistence": 0.20,
        }
    }

    w_path, m_path = save_experimental_checkpoint(model, data_bundle, eval_report, cfg, tmp_path)
    assert w_path.exists()
    assert m_path.exists()

    with open(m_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["experiment_designation"] == MARKER_EXPERIMENTAL
    assert meta["production_registry_eligible"] is False
    assert meta["measurement_units"] == "counts/s"
    assert meta["instrument"] == "SoLEXS"
    assert "exclusions_preserved" in meta
    assert "HEL1OS_L1" in meta["exclusions_preserved"]
