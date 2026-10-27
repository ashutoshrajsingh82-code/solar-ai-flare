from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from app.schemas.schemas import TrainRequest
from app.services import model_service
from ml.features.engineering import FeatureConfig
from ml.models.registry import ModelRegistry


def test_classical_training_uses_validation_threshold_and_persists_it(tmp_path, monkeypatch):
    timestamps = pd.date_range("2026-01-01", periods=60, freq="min", tz="UTC")
    labels = (np.arange(60) % 2).astype(float)
    feature = labels + (np.arange(60) % 7) * 0.01
    state_labels = (np.arange(60) % 4).astype(int)
    labeled = pd.DataFrame({
        "timestamp": timestamps,
        "solexs_flux": np.ones(60),
        "helios_flux": np.ones(60),
        "feature": feature,
        "state": state_labels,
        **{f"forecast_{h}h": labels.copy() for h in (1, 3, 6, 12, 24)},
    })
    local_state = SimpleNamespace(
        train_df=labeled.iloc[:40][["timestamp"]].copy(),
        val_df=labeled.iloc[40:50][["timestamp"]].copy(),
        test_df=labeled.iloc[50:][["timestamp"]].copy(),
        labeled_df=labeled,
        feature_columns=["feature"],
        raw_df=labeled[["timestamp", "solexs_flux", "helios_flux"]].copy(),
        dataset_meta={"source": "demo", "filename": "fixture.csv"},
        transformer=SimpleNamespace(config=SimpleNamespace(scaler_type="test", log_transform=None)),
        stage_reports={},
    )
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    registry = ModelRegistry(str(model_dir))
    monkeypatch.setattr(model_service, "pipeline_state", local_state)
    monkeypatch.setattr(model_service, "registry", registry)
    monkeypatch.setattr(model_service.settings, "model_dir", str(model_dir))
    monkeypatch.setattr(model_service, "_training_status", lambda: "demo_only")
    monkeypatch.setattr(model_service, "_pipeline_feature_config", lambda: FeatureConfig())
    monkeypatch.setattr(model_service, "select_feature_columns", lambda *args, **kwargs: ["feature"])

    def write_manifest(model_id, *args, **kwargs):
        path = model_dir / f"{model_id}.feature_manifest.json"
        path.write_text("{}", encoding="utf-8")
        return str(path)

    monkeypatch.setattr(model_service, "_save_model_feature_manifest", write_manifest)
    monkeypatch.setattr(model_service, "_preprocessor_manifest_metadata", lambda *_: {"fit_on": "train_split_only"})
    monkeypatch.setattr(model_service, "_persist_evaluation_report", lambda *args, **kwargs: str(model_dir / "eval.json"))
    monkeypatch.setattr(model_service, "catalog_model_artifacts", lambda *args, **kwargs: [])

    result = model_service._train_classical(
        TrainRequest(model_type="logistic_regression", input_configuration="fusion"),
        ["solexs_flux_scaled", "helios_flux_scaled"], "integration-model",
    )
    record = registry.get("integration-model")
    checkpoint = joblib.load(record["checkpoint_path"])
    assert result["model_id"] == "integration-model"
    assert all(item["method"] == "validation_max_tss" for item in record["params"]["threshold_selection"].values())
    assert checkpoint["thresholds"] == record["params"]["thresholds"]
    assert checkpoint["threshold_selection"] == record["params"]["threshold_selection"]
    assert record["metrics"]["evaluation_split"] == "chronological_test"
