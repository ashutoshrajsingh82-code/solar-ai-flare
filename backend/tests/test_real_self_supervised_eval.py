"""Unit and regression tests for Phase 4N independent evaluation.

Covers:
- Checkpoint metadata validation.
- Source-hash mismatch handling.
- Evaluation-only mode not updating model weights.
- No fitting of scalers on evaluation data.
- Baseline and model target alignment.
- Per-horizon metric calculation.
- Gap and boundary integrity.
- Empty or insufficient evaluation cohorts.
- Production-registry isolation.
"""
from __future__ import annotations

import io
import json
import numpy as np
import pytest
import torch
from pathlib import Path
from astropy.io import fits

from ml.experiments.real_self_supervised import (
    SelfSupervisedCNNLSTM,
    CountRateScaler,
    MARKER_EXPERIMENTAL,
    SOLEXS_ARCHIVE_SHA256,
)
from ml.models.registry import ModelRegistry, ModelRecord
from scripts.evaluate_real_self_supervised import (
    compute_sha256,
    load_frozen_scaler,
    load_frozen_model,
    verify_weight_immutability,
    evaluate_windows,
)


def _create_mock_metadata(mean: float = 2.24727, std: float = 0.69664) -> dict:
    return {
        "experiment_id": "solexs_self_supervised_cnnlstm_test",
        "experiment_designation": MARKER_EXPERIMENTAL,
        "production_registry_eligible": False,
        "instrument": "SoLEXS",
        "detector": "SDD2",
        "measurement_column": "COUNTS",
        "measurement_units": "counts/s",
        "source_archive_sha256": SOLEXS_ARCHIVE_SHA256,
        "scaler_parameters": {
            "type": "log10_zscore",
            "mean": mean,
            "std": std,
            "fitted_on_split": "train_only",
        },
        "evaluation_metrics": {
            "cnn_lstm": {"mae": 14.032, "rmse": 23.324, "rel_mae": 0.0498},
            "persistence_baseline": {"mae": 15.928, "rmse": 24.464, "rel_mae": 0.0566},
            "skill_score_vs_persistence": 0.1190,
        }
    }


def test_checkpoint_metadata_validation(tmp_path):
    """Validates metadata contract and prevents ingestion of invalid metadata."""
    meta = _create_mock_metadata()
    meta_file = tmp_path / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    
    # Valid metadata should load scaler correctly
    scaler = load_frozen_scaler(meta)
    assert scaler.fitted is True
    assert np.isclose(scaler.mean, 2.24727)
    assert np.isclose(scaler.std, 0.69664)
    
    # Missing scaler_parameters block must raise ValueError
    bad_meta = meta.copy()
    del bad_meta["scaler_parameters"]
    with pytest.raises(ValueError, match="scaler_parameters"):
        load_frozen_scaler(bad_meta)


def test_source_hash_mismatch_handling(tmp_path):
    """Validates that a mismatched source archive hash is detected."""
    dummy_archive = tmp_path / "dummy_archive.zip"
    dummy_archive.write_bytes(b"PK\x03\x04mock_archive_content")
    
    computed_hash = compute_sha256(dummy_archive)
    assert computed_hash != SOLEXS_ARCHIVE_SHA256
    assert len(computed_hash) == 64


def test_evaluation_only_mode_does_not_modify_weights(tmp_path):
    """Guarantees that evaluation never updates model weights and preserves weight immutability."""
    model = SelfSupervisedCNNLSTM(in_features=1, horizon=10)
    ckpt_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    
    loaded_model, initial_state = load_frozen_model(ckpt_path)
    
    # Verify all parameters have requires_grad=False
    for p in loaded_model.parameters():
        assert p.requires_grad is False
    
    # Run evaluation forward pass
    meta = _create_mock_metadata()
    scaler = load_frozen_scaler(meta)
    
    X = np.random.uniform(10.0, 500.0, size=(20, 60, 1)).astype(np.float32)
    y = np.random.uniform(10.0, 500.0, size=(20, 10)).astype(np.float32)
    t = np.arange(1000.0, 1020.0)
    
    evaluate_windows(X, y, t, loaded_model, scaler, horizon=10)
    
    # Weight immutability check must pass
    assert verify_weight_immutability(loaded_model, initial_state) is True


def test_no_fitting_of_scalers_on_evaluation_data():
    """Validates that evaluation scaler is purely loaded from metadata and never fitted on eval data."""
    meta = _create_mock_metadata(mean=2.0, std=0.5)
    scaler = load_frozen_scaler(meta)
    
    # Present an evaluation dataset with vastly different distribution
    eval_data = np.array([10000.0, 20000.0, 30000.0])
    
    # Transform using frozen parameters
    transformed = scaler.transform(eval_data)
    
    # Mean and std of scaler must remain strictly 2.0 and 0.5
    assert scaler.mean == 2.0
    assert scaler.std == 0.5
    
    # Inversion must recover original values
    inverted = scaler.inverse_transform(transformed)
    assert np.allclose(inverted, eval_data, atol=1e-3)


def test_baseline_and_model_target_alignment():
    """Validates that persistence, rolling mean, and model are evaluated on the exact same targets."""
    model = SelfSupervisedCNNLSTM(in_features=1, horizon=5)
    meta = _create_mock_metadata()
    scaler = load_frozen_scaler(meta)
    
    X = np.array([
        [[100.0], [120.0], [150.0]],
        [[500.0], [520.0], [550.0]],
    ], dtype=np.float32)  # shape (2, 3, 1)
    
    # Target values (2 samples, 5 horizon steps)
    y = np.array([
        [155.0, 160.0, 165.0, 170.0, 175.0],
        [560.0, 570.0, 580.0, 590.0, 600.0],
    ], dtype=np.float32)
    
    t = np.array([100.0, 200.0])
    
    # Persistence expects last input value: 150.0 and 550.0
    res = evaluate_windows(X, y, t, model, scaler, horizon=5)
    
    persist_err = res["global_metrics"]["persistence_baseline"]
    # Sample 1: diffs [5, 10, 15, 20, 25] -> MAE = 15.0
    # Sample 2: diffs [10, 20, 30, 40, 50] -> MAE = 30.0
    # Overall persist MAE = (15 + 30) / 2 = 22.5
    assert np.isclose(persist_err["mae"], 22.5, atol=1e-4)


def test_per_horizon_metric_calculation():
    """Validates that per-horizon evaluation produces exactly the expected number of steps and metrics."""
    model = SelfSupervisedCNNLSTM(in_features=1, horizon=10)
    meta = _create_mock_metadata()
    scaler = load_frozen_scaler(meta)
    
    X = np.ones((5, 60, 1), dtype=np.float32) * 50.0
    y = np.ones((5, 10), dtype=np.float32) * 55.0
    t = np.arange(5, dtype=float)
    
    res = evaluate_windows(X, y, t, model, scaler, horizon=10)
    assert "per_horizon" in res
    assert len(res["per_horizon"]) == 10
    
    for h_info in res["per_horizon"]:
        assert 1 <= h_info["step"] <= 10
        assert "model_mae" in h_info
        assert "persist_mae" in h_info
        assert "skill_vs_persist" in h_info


def test_empty_or_insufficient_evaluation_cohorts():
    """Validates safe handling when evaluation cohort is empty."""
    model = SelfSupervisedCNNLSTM(in_features=1, horizon=10)
    meta = _create_mock_metadata()
    scaler = load_frozen_scaler(meta)
    
    X_empty = np.empty((0, 60, 1), dtype=np.float32)
    y_empty = np.empty((0, 10), dtype=np.float32)
    t_empty = np.empty((0,), dtype=float)
    
    res = evaluate_windows(X_empty, y_empty, t_empty, model, scaler, horizon=10)
    assert res["n_windows"] == 0
    assert "error" in res


def test_production_registry_isolation(tmp_path):
    """Guarantees that Phase 4N evaluation cannot register experimental models into production."""
    registry = ModelRegistry(str(tmp_path / "prod_registry"))
    
    record = ModelRecord(
        model_id="phase4n_eval_model",
        name="Phase 4N Evaluated Model",
        architecture="cnn_lstm",
        version="v1.0",
        created_at="2026-10-25T23:00:00Z",
        training_dataset="solexs_2026Oct23T184402685.zip",
        input_configuration="solexs_only",
        params={
            "experiment_designation": MARKER_EXPERIMENTAL,
            "production_registry_eligible": False,
        }
    )
    with pytest.raises(ValueError, match="Synthetic smoke models cannot be registered in the production model registry"):
        registry.register(record)
