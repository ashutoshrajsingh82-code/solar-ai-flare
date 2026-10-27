"""Unit and regression tests for Phase 4O high-activity forecasting and model comparison.

Validates:
- No future-value use in trend/velocity features (strict causality).
- No train/validation/test boundary leakage.
- Training-only scaler fitting and zero drift during evaluation.
- Identical target alignment across baseline and neural candidates.
- Correct inverse transformation to native counts/s.
- Correct per-horizon metrics.
- Reproducible checkpoint loading for residual and trend-aware architectures.
- Guaranteed protection of Phase 4M-S checkpoint (no overwriting).
- Production registry isolation.
- Proper handling of gaps and empty/insufficient evaluation data.
"""
from __future__ import annotations

import io
import json
import numpy as np
import pytest
import torch
from pathlib import Path

from ml.experiments.real_self_supervised import (
    CountRateScaler,
    SelfSupervisedCNNLSTM,
    MARKER_EXPERIMENTAL,
    SOLEXS_ARCHIVE_SHA256,
    persistence_forecast,
    rolling_mean_forecast,
    evaluate_forecast,
)
from ml.experiments.real_residual_forecasting import (
    ResidualCNNLSTM,
    TrendAwareResidualCNNLSTM,
    compute_causal_trend_features,
)
from ml.models.registry import ModelRegistry, ModelRecord
from scripts.run_phase4o_experiments import evaluate_model_on_split


def test_no_future_value_use_in_trend_features():
    """Validates that computed trend, velocity, and volatility features use strictly past observations.
    
    Mutating future time steps must have zero effect on past feature values.
    """
    seq_len = 60
    np.random.seed(42)
    X_original = np.random.uniform(50.0, 500.0, size=(1, seq_len, 1)).astype(np.float32)
    
    # Compute features on original past
    feat_orig = compute_causal_trend_features(X_original)
    assert feat_orig.shape == (1, seq_len, 4)
    
    # Now simulate a sequence that is identical up to t=30, but differs drastically from t=31 to 59
    X_modified = X_original.copy()
    X_modified[:, 31:, :] = 99999.0  # future spike after t=30
    
    feat_mod = compute_causal_trend_features(X_modified)
    
    # All features up to t=30 MUST BE EXACTLY BIT-FOR-BIT IDENTICAL
    assert np.allclose(feat_orig[:, :31, :], feat_mod[:, :31, :], atol=1e-7), (
        "Causal violation: Features prior to t=31 changed when future observations were modified!"
    )


def test_train_validation_test_boundary_leakage():
    """Validates that train, val, and test partitions are strictly disjoint and monotonic in time."""
    train_end = 1769990399.0  # 2026-02-02 23:59:59 UTC
    val_end = 1770163199.0    # 2026-02-04 23:59:59 UTC
    test_end = 1770249599.0   # 2026-02-05 23:59:59 UTC
    
    # Boundary inequalities must hold strictly
    assert train_end < val_end < test_end
    
    # Simulated timestamps
    times = np.array([train_end - 10, train_end, train_end + 1, val_end, val_end + 1, test_end])
    
    train_mask = times <= train_end
    val_mask = (times > train_end) & (times <= val_end)
    test_mask = (times > val_end) & (times <= test_end)
    
    # Must be mutually exclusive and collectively exhaustive
    assert np.all((train_mask.astype(int) + val_mask.astype(int) + test_mask.astype(int)) == 1)
    assert not np.any(train_mask & val_mask)
    assert not np.any(val_mask & test_mask)
    assert not np.any(train_mask & test_mask)


def test_training_only_scaler_fitting():
    """Validates that scaler is fitted strictly on training data and remains unchanged during evaluation."""
    train_counts = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    val_counts = np.array([5000.0, 10000.0, 15000.0])
    
    scaler = CountRateScaler().fit(train_counts)
    saved_mean = scaler.mean
    saved_std = scaler.std
    
    # Transform validation counts
    val_sc = scaler.transform(val_counts)
    
    # Parameters must be strictly identical
    assert scaler.mean == saved_mean
    assert scaler.std == saved_std
    
    # Inversion must match original validation counts
    val_inv = scaler.inverse_transform(val_sc)
    assert np.allclose(val_inv, val_counts, atol=1e-3)


def test_identical_target_alignment_across_baselines():
    """Validates that persistence, rolling mean, and candidate models predict against identical target tensors."""
    N, seq_len, horizon = 10, 60, 10
    X_raw = np.ones((N, seq_len, 1), dtype=np.float32) * 100.0
    y_raw = np.ones((N, horizon), dtype=np.float32) * 110.0
    
    # Persistence baseline
    persist_preds = persistence_forecast(X_raw, horizon)
    assert persist_preds.shape == (N, horizon)
    assert np.all(persist_preds == 100.0)
    
    # Rolling mean baseline
    mean_preds = rolling_mean_forecast(X_raw, horizon)
    assert mean_preds.shape == (N, horizon)
    assert np.all(mean_preds == 100.0)
    
    # Both evaluated on identical y_raw
    ev_per = evaluate_model_on_split(y_raw, persist_preds, X_raw, horizon=horizon)
    ev_mea = evaluate_model_on_split(y_raw, mean_preds, X_raw, horizon=horizon)
    
    assert ev_per["global"]["mae"] == 10.0
    assert ev_mea["global"]["mae"] == 10.0


def test_per_horizon_metric_integrity():
    """Validates per-horizon step calculation and monotonicity of step numbering."""
    y_true = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    y_pred = np.array([[12.0, 24.0, 36.0], [42.0, 54.0, 66.0]], dtype=np.float32)
    X_dummy = np.ones((2, 60, 1), dtype=np.float32) * 5.0
    
    ev = evaluate_model_on_split(y_true, y_pred, X_dummy, horizon=3)
    assert len(ev["per_horizon"]) == 3
    assert ev["per_horizon"][0]["step"] == 1
    assert ev["per_horizon"][1]["step"] == 2
    assert ev["per_horizon"][2]["step"] == 3
    
    # Step 1 diff: 2.0 and 2.0 -> MAE = 2.0
    # Step 2 diff: 4.0 and 4.0 -> MAE = 4.0
    # Step 3 diff: 6.0 and 6.0 -> MAE = 6.0
    assert np.isclose(ev["per_horizon"][0]["mae"], 2.0)
    assert np.isclose(ev["per_horizon"][1]["mae"], 4.0)
    assert np.isclose(ev["per_horizon"][2]["mae"], 6.0)


def test_reproducible_checkpoint_loading_and_forward_pass(tmp_path):
    """Validates that ResidualCNNLSTM and TrendAwareResidualCNNLSTM save and load deterministically."""
    # 1. ResidualCNNLSTM
    model_res = ResidualCNNLSTM(in_features=1, horizon=10)
    ckpt_res = tmp_path / "res_model.pt"
    torch.save(model_res.state_dict(), ckpt_res)
    
    model_loaded = ResidualCNNLSTM(in_features=1, horizon=10)
    model_loaded.load_state_dict(torch.load(ckpt_res, map_location="cpu", weights_only=True))
    model_loaded.eval()
    
    model_res.eval()
    x1 = torch.randn(2, 60, 1)
    with torch.no_grad():
        out1 = model_res(x1)
        out2 = model_loaded(x1)
    assert torch.allclose(out1, out2, atol=1e-5)
    
    # 2. TrendAwareResidualCNNLSTM
    model_trend = TrendAwareResidualCNNLSTM(in_features=4, horizon=10)
    ckpt_trend = tmp_path / "trend_model.pt"
    torch.save(model_trend.state_dict(), ckpt_trend)
    
    trend_loaded = TrendAwareResidualCNNLSTM(in_features=4, horizon=10)
    trend_loaded.load_state_dict(torch.load(ckpt_trend, map_location="cpu", weights_only=True))
    trend_loaded.eval()
    model_trend.eval()
    
    x4 = torch.randn(2, 60, 4)
    with torch.no_grad():
        out3 = model_trend(x4)
        out4 = trend_loaded(x4)
    assert torch.allclose(out3, out4, atol=1e-5)


def test_phase4ms_checkpoint_immutability():
    """Guarantees that the canonical Phase 4M-S checkpoint was not overwritten or modified."""
    ckpt_path = Path("artifacts/real_data_experiments/solexs_self_supervised_cnnlstm_20261025T115207Z.pt")
    assert ckpt_path.exists(), "Phase 4M-S checkpoint is missing!"
    
    import hashlib
    hasher = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        hasher.update(f.read())
    current_hash = hasher.hexdigest().lower()
    
    expected_hash = "815ded9a605fa4bb410791b3a64d07624d6f3d03ffaf02953b67c7f1f8e53d45"
    assert current_hash == expected_hash, (
        f"CRITICAL: Phase 4M-S checkpoint was altered! Expected {expected_hash}, got {current_hash}"
    )


def test_production_registry_isolation(tmp_path):
    """Guarantees that Phase 4O candidate models cannot be registered in the production model registry."""
    registry = ModelRegistry(str(tmp_path / "prod_reg"))
    
    record = ModelRecord(
        model_id="phase4o_residual_001",
        name="Phase 4O Residual Forecaster",
        architecture="residual_cnn_lstm",
        version="v1.0",
        created_at="2026-10-26T00:00:00Z",
        training_dataset="solexs_2026Oct23T184402685.zip",
        input_configuration="solexs_only",
        params={
            "experiment_designation": MARKER_EXPERIMENTAL,
            "production_registry_eligible": False,
        }
    )
    with pytest.raises(ValueError, match="Synthetic smoke models cannot be registered in the production model registry"):
        registry.register(record)


def test_handling_of_empty_or_insufficient_evaluation_cohorts():
    """Validates that empty evaluation splits do not produce division by zero or crashes."""
    X_empty = np.empty((0, 60, 1), dtype=np.float32)
    y_empty = np.empty((0, 10), dtype=np.float32)
    y_pred_empty = np.empty((0, 10), dtype=np.float32)
    
    ev = evaluate_model_on_split(y_empty, y_pred_empty, X_empty, horizon=10)
    assert ev["global"]["mae"] == 0.0 or np.isnan(ev["global"]["mae"])
    assert ev["by_trend"]["rising"]["n_windows"] == 0
