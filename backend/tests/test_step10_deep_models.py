import numpy as np
import pandas as pd
import pytest
import torch

from ml.features.decomposition import (
    DecompositionConfig,
    HistoricalDecompositionCache,
    decompose_historical_window,
)
from ml.models.deep_models import (
    DEEP_MODEL_BUILDERS,
    DecompositionLSTM,
    SinusoidalPositionalEncoding,
)
from ml.training.train import TrainConfig, train_deep_model


def test_sinusoidal_positions_follow_time_axis_and_support_lengths():
    pe = SinusoidalPositionalEncoding(7, max_length=16)
    for length in (3, 8, 16):
        encoded = pe(torch.zeros(2, length, 7, dtype=torch.float64))
        assert encoded.shape == (2, length, 7)
        assert encoded.dtype == torch.float64
        assert not torch.equal(encoded[:, 0], encoded[:, -1])
    with pytest.raises(ValueError, match="exceeds configured maximum"):
        pe(torch.zeros(1, 17, 7))


def test_cnn_transformer_output_is_deterministic_and_order_sensitive():
    model = DEEP_MODEL_BUILDERS["cnn_transformer"](
        n_channels=1, dropout=0.0, max_sequence_length=32
    ).eval()
    # A structured ramp avoids relying on random input to exercise temporal order.
    x = torch.linspace(0.1, 3.2, 32).reshape(1, 32, 1)
    reversed_x = x.flip(1)
    with torch.no_grad():
        first = model(x)
        repeated = model(x)
        reordered = model(reversed_x)
    assert first[0].shape == (1, 4) and first[1].shape == (1, 5)
    assert torch.equal(first[0], repeated[0])
    assert torch.equal(first[1], repeated[1])
    assert not torch.allclose(first[0], reordered[0]) or not torch.allclose(first[1], reordered[1])


def test_dlstm_contract_factory_and_validation():
    model = DEEP_MODEL_BUILDERS["dlstm"](n_channels=6, hidden_size=12, num_layers=1, min_history=4).eval()
    with torch.no_grad():
        for batch in (1, 3):
            now, forecast = model(torch.ones(batch, 7, 6))
            assert now.shape == (batch, 4)
            assert forecast.shape == (batch, 5)
    with pytest.raises(ValueError, match="at least 4"):
        model(torch.ones(1, 3, 6))
    with pytest.raises(ValueError, match="invalid decomposition"):
        model(torch.full((1, 4, 6), float("nan")))
    with pytest.raises(ValueError, match="expects"):
        model(torch.ones(1, 4, 5))


def test_stl_uses_only_given_historical_window_and_cache_is_origin_scoped():
    cfg = DecompositionConfig(period_samples=4)
    timeline = (10 + np.sin(np.arange(24) * np.pi / 2))[:, None]
    origin_index = 15
    raw = timeline[origin_index - 15:origin_index + 1]
    scales = {"solexs_flux": 10.0}
    base = decompose_historical_window(raw, ["solexs_flux"], scales, cfg)
    perturbed_timeline = timeline.copy()
    perturbed_timeline[origin_index + 1:] *= 100.0
    future_window = perturbed_timeline[origin_index - 15:origin_index + 1]
    future_perturbed = decompose_historical_window(future_window, ["solexs_flux"], scales, cfg)
    np.testing.assert_array_equal(base, future_perturbed)

    cache = HistoricalDecompositionCache()
    origin = pd.Timestamp("2026-01-01T00:15:00Z")
    cached, meta = cache.get_or_compute(raw, origin, ["solexs_flux"], scales, cfg, "source-a")
    again, _ = cache.get_or_compute(raw, origin, ["solexs_flux"], scales, cfg, "source-a")
    np.testing.assert_array_equal(cached, again)
    assert len(cache) == 1
    assert meta["forecast_origin_utc"].endswith("+00:00")
    cache.get_or_compute(raw, origin + pd.Timedelta(minutes=1), ["solexs_flux"], scales, cfg, "source-a")
    cache.get_or_compute(raw, origin, ["solexs_flux"], scales, DecompositionConfig(period_samples=5), "source-a")
    cache.get_or_compute(raw, origin, ["solexs_flux"], scales, cfg, "source-b")
    assert len(cache) == 4


def test_stl_rejects_short_invalid_negative_or_unscaled_windows():
    cfg = DecompositionConfig(period_samples=4)
    scales = {"solexs_flux": 1.0}
    with pytest.raises(ValueError, match="at least 8"):
        decompose_historical_window(np.ones((7, 1)), ["solexs_flux"], scales, cfg)
    for bad in (np.nan, -1.0, np.inf):
        values = np.ones((8, 1))
        values[0, 0] = bad
        with pytest.raises(ValueError):
            decompose_historical_window(values, ["solexs_flux"], scales, cfg)
    with pytest.raises(ValueError, match="reference scale"):
        decompose_historical_window(np.ones((8, 1)), ["solexs_flux"], {}, cfg)


def test_validation_threshold_helper_uses_validation_predictions_only(monkeypatch):
    from app.services import model_service

    monkeypatch.setattr(model_service.pipeline_state, "val_df", pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    }))
    labels = np.array([[0.0], [0.0], [1.0], [1.0]])
    probs = np.array([[0.1], [0.2], [0.8], [0.9]])
    thresholds, report = model_service._select_validation_thresholds(labels, probs)
    assert thresholds == [pytest.approx(0.8)]
    assert report["1h"]["method"] == "validation_max_tss"
    assert report["1h"]["validation_positive_windows"] == 2
    assert report["1h"]["selection_period"]["start_utc"].startswith("2026-01-01")
    fallback, uncertain = model_service._select_validation_thresholds(labels[:2], probs[:2])
    assert fallback == [0.5]
    assert uncertain["1h"]["method"] == "uncertain_fallback_0.5"


def test_training_restores_best_validation_epoch_and_records_scheduler_seed(tmp_path):
    rng = np.random.default_rng(5)
    x_train = rng.normal(size=(12, 8, 2)).astype(np.float32)
    x_val = rng.normal(size=(6, 8, 2)).astype(np.float32)
    y_now_train = np.arange(12, dtype=np.int64) % 4
    y_now_val = np.arange(6, dtype=np.int64) % 4
    y_fc_train = np.tile((np.arange(12)[:, None] % 2).astype(np.float32), (1, 5))
    y_fc_val = np.tile((np.arange(6)[:, None] % 2).astype(np.float32), (1, 5))
    cfg = TrainConfig(model_type="lstm", sequence_length=8, batch_size=6, epochs=3,
                      n_channels=2, device="cpu", random_seed=19,
                      early_stopping_patience=1, scheduler_patience=0)
    model, history = train_deep_model(
        x_train, y_now_train, y_fc_train, x_val, y_now_val, y_fc_val, config=cfg
    )
    metadata = model._training_metadata
    assert metadata["best_epoch"] == min(range(1, len(history) + 1), key=lambda i: history[i - 1]["val_loss"])
    assert metadata["scheduler"] == "ReduceLROnPlateau"
    assert metadata["random_seed"] == 19
    assert metadata["class_distribution"]["forecast_positive"] == [6, 6, 6, 6, 6]
    assert "optimizer_state_dict" in metadata and "scheduler_state_dict" in metadata
    checkpoint = tmp_path / "training.pt"
    torch.save({"state_dict": model.state_dict(), "training_metadata": metadata}, checkpoint)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert restored["training_metadata"]["best_epoch"] == metadata["best_epoch"]


def test_dlstm_end_to_end_decomposition_training_checkpoint_and_reload(tmp_path):
    rng = np.random.default_rng(17)
    scales = {"solexs_flux": 5.0}
    raw_train = rng.uniform(1.0, 8.0, size=(10, 8, 1))
    raw_val = rng.uniform(1.0, 8.0, size=(6, 8, 1))
    cfg = DecompositionConfig(period_samples=4)
    train_x = np.stack([
        decompose_historical_window(window, ["solexs_flux"], scales, cfg)
        for window in raw_train
    ])
    val_x = np.stack([
        decompose_historical_window(window, ["solexs_flux"], scales, cfg)
        for window in raw_val
    ])
    y_now_train = np.arange(10, dtype=np.int64) % 4
    y_now_val = np.arange(6, dtype=np.int64) % 4
    y_fc_train = np.tile((np.arange(10)[:, None] % 2).astype(np.float32), (1, 5))
    y_fc_val = np.tile((np.arange(6)[:, None] % 2).astype(np.float32), (1, 5))
    trained, _ = train_deep_model(
        train_x, y_now_train, y_fc_train, val_x, y_now_val, y_fc_val,
        config=TrainConfig(model_type="dlstm", sequence_length=8, batch_size=5, epochs=1,
                           n_channels=3, dlstm_hidden_size=8, dlstm_num_layers=1,
                           early_stopping_patience=0, device="cpu"),
    )
    checkpoint = tmp_path / "dlstm.pt"
    torch.save({"state_dict": trained.state_dict(), "model_type": "dlstm", "n_channels": 3,
                "architecture_version": 2,
                "model_config": {"hidden_size": 8, "num_layers": 1}}, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    loaded = DEEP_MODEL_BUILDERS[saved["model_type"]](
        n_channels=saved["n_channels"], **saved["model_config"]
    )
    loaded.load_state_dict(saved["state_dict"])
    loaded.eval()
    with torch.no_grad():
        nowcast, forecast = loaded(torch.as_tensor(val_x[:2]))
    assert nowcast.shape == (2, 4)
    assert forecast.shape == (2, 5)
