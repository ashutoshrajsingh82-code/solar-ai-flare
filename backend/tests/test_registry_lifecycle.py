import json

import joblib
import pytest

from ml.models.registry import ModelRecord, ModelRegistry


class FittedTransformer:
    fitted = True

    def transform(self, values):
        return values


def record(model_id, checkpoint, transformer, status="trained"):
    return ModelRecord(
        model_id=model_id, name=model_id, architecture="random_forest", version="v1",
        created_at="now", training_dataset="test", input_configuration="fusion",
        checkpoint_path=str(checkpoint), transformer_path=str(transformer),
        scaler_path=str(transformer), model_kind="classical", status=status,
    )


def test_untrained_and_incomplete_models_cannot_activate(tmp_path):
    registry = ModelRegistry(str(tmp_path))
    checkpoint, transformer = tmp_path / "model.joblib", tmp_path / "scaler.joblib"
    checkpoint.write_text("checkpoint")
    transformer.write_text("transformer")
    registry.register(record("placeholder", checkpoint, transformer, "untrained"))
    with pytest.raises(ValueError, match="untrained"):
        registry.set_active("placeholder")
    registry.register(record("no-checkpoint", tmp_path / "missing", transformer))
    with pytest.raises(ValueError, match="checkpoint is missing"):
        registry.set_active("no-checkpoint")


def test_missing_transformer_cannot_activate(tmp_path):
    registry = ModelRegistry(str(tmp_path))
    checkpoint = tmp_path / "model.joblib"
    joblib.dump({"model": object(), "feature_cols": ["flux"]}, checkpoint)
    registry.register(record("no-transformer", checkpoint, tmp_path / "missing"))
    with pytest.raises(ValueError, match="transformer is missing"):
        registry.set_active("no-transformer")


def test_valid_artifacts_can_activate(tmp_path):
    registry = ModelRegistry(str(tmp_path))
    checkpoint, transformer = tmp_path / "model.joblib", tmp_path / "scaler.joblib"
    joblib.dump({"model": object(), "feature_cols": ["flux"]}, checkpoint)
    joblib.dump(FittedTransformer(), transformer)
    registry.register(record("valid", checkpoint, transformer))
    registry.set_active("valid")
    assert registry.get_active()["model_id"] == "valid"


def test_legacy_records_are_untrained_and_not_active(tmp_path):
    registry = ModelRegistry(str(tmp_path))
    (tmp_path / "registry_index.json").write_text(json.dumps({
        "legacy": {"model_id": "legacy", "active": True, "checkpoint_path": "missing"}
    }))
    assert registry.get("legacy")["status"] == "untrained"
    assert registry.get_active() is None
