import numpy as np
import pytest
import torch

from ml.features.windowing import WindowConfig, build_windows, chronological_split
from ml.models.registry import ModelRecord, ModelRegistry
from ml.preprocessing.transform import FluxTransformer
from ml.training.synthetic_smoke import (
    SYNTHETIC_TAG,
    _targets_within_partition,
    make_synthetic_observations,
    HORIZON_COLS,
    require_observational_readiness,
    run_synthetic_smoke,
)


def test_generated_dataset_is_reproducible_and_partition_tails_remain_unknown():
    first = make_synthetic_observations(seed=23, rows=4200)
    second = make_synthetic_observations(seed=23, rows=4200)
    assert first.equals(second)
    train, validation, test = chronological_split(first)
    labels = _targets_within_partition(train)
    assert labels.iloc[-24:][["forecast_24h"]].isna().all().all()
    assert labels.iloc[-1][HORIZON_COLS].isna().all()
    assert train.timestamp.max() < validation.timestamp.min() < test.timestamp.min()
    assert labels["forecast_24h"].isna().sum() == 24


def test_future_perturbations_cannot_change_prior_window_features():
    frame = make_synthetic_observations(seed=7, rows=300)
    transformer = FluxTransformer().fit(frame.iloc[:200])
    base = transformer.transform(_targets_within_partition(frame.iloc[:100]))
    changed_raw = _targets_within_partition(frame.iloc[:100])
    changed_raw.loc[24:, ["solexs_flux", "helios_flux"]] *= 100
    changed = transformer.transform(changed_raw)
    cfg = WindowConfig(sequence_length=24, channel_cols=["solexs_flux_scaled", "helios_flux_scaled"],
                       horizon_cols=HORIZON_COLS)
    base_x, *_ = build_windows(base, cfg)
    changed_x, *_ = build_windows(changed, cfg)
    np.testing.assert_array_equal(base_x[0], changed_x[0])


def test_real_observational_training_stays_blocked_without_phase3c_readiness():
    with pytest.raises(ValueError, match="Real observational training is blocked") as exc:
        require_observational_readiness(
            {"source": "verified_observations"},
            {"status": "BLOCKED", "training_allowed": False,
             "blockers": ["Phase 3C instrument semantics and labels are unverified."]},
        )
    assert "Phase 3C instrument semantics" in str(exc.value)


def test_synthetic_model_cannot_enter_production_registry(tmp_path):
    registry = ModelRegistry(str(tmp_path))
    checkpoint = tmp_path / "synthetic.pt"
    checkpoint.write_bytes(b"synthetic")
    record = ModelRecord(
        model_id="synthetic", name="synthetic", architecture="lstm", version="smoke",
        created_at="now", training_dataset=SYNTHETIC_TAG, input_configuration="fusion",
        checkpoint_path=str(checkpoint), model_kind="deep", status="trained",
        transformer_required=False,
        params={"dataset_status": SYNTHETIC_TAG, "production_registry_eligible": False},
    )
    with pytest.raises(ValueError, match="cannot be registered"):
        registry.register(record)
    assert registry.list_all() == []


def test_smoke_training_metrics_checkpoint_reload_and_repeatability(tmp_path):
    one = run_synthetic_smoke(tmp_path / "one.pt", seed=19, rows=4000, epochs=1)
    two = run_synthetic_smoke(tmp_path / "two.pt", seed=19, rows=4000, epochs=1)
    assert one["metadata"]["dataset_status"] == SYNTHETIC_TAG
    assert one["metadata"]["model_status"] == SYNTHETIC_TAG
    assert one["metadata"]["production_registry_eligible"] is False
    assert one["metadata"]["partition_windows"]["test"] > 0
    assert set(one["metrics"]) == {"1h", "3h", "6h", "12h", "24h"}
    assert one["metadata"]["partition_time_bounds"]["train"][1] < one["metadata"]["partition_time_bounds"]["validation"][0]
    assert one["metadata"]["partition_time_bounds"]["validation"][1] < one["metadata"]["partition_time_bounds"]["test"][0]
    saved_one = torch.load(one["checkpoint"], map_location="cpu", weights_only=True)
    saved_two = torch.load(two["checkpoint"], map_location="cpu", weights_only=True)
    assert saved_one["metadata"]["dataset_status"] == SYNTHETIC_TAG
    assert saved_one["metadata"]["production_registry_eligible"] is False
    for name, tensor in saved_one["state_dict"].items():
        assert torch.equal(tensor, saved_two["state_dict"][name])
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_synthetic_smoke(tmp_path / "one.pt", seed=19, rows=4000, epochs=1)
