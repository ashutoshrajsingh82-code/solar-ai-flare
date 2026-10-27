import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api import predict as predict_api
from app.api import experiments as experiments_api
from app.api import models as models_api
from app.api import health as health_api
from app.services import alert_service
from app.core.config import settings
from app.database.db import Base
from app.database.models import InferenceRecord
from app.services import artifact_catalog, model_service, predict_service
from ml.features.engineering import FeatureConfig, add_rolling_features
from ml.features.manifest import create_feature_manifest, save_feature_manifest
from ml.models.baseline import MODEL_BUILDERS
from ml.models.registry import ModelRecord, ModelRegistry
from ml.preprocessing.transform import FluxTransformer, TransformConfig
from ml.inference.input_validation import normalize_prediction_rows
from app.schemas.schemas import PredictRequest, PredictResponse


@pytest.fixture
def trained_artifact(tmp_path, monkeypatch):
    """Create a real, small CPU sklearn artifact and its integrity metadata."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    db_path = tmp_path / "artifacts.sqlite"
    engine = create_engine(f"sqlite:///{db_path.as_posix()}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(artifact_catalog, "SessionLocal", factory)
    monkeypatch.setattr(settings, "model_dir", str(model_dir))
    registry = ModelRegistry(str(model_dir))
    monkeypatch.setattr(model_service, "registry", registry)
    monkeypatch.setattr(predict_service, "registry", registry)

    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=180, freq="min")
    phase = np.arange(len(timestamps), dtype=float)
    raw = pd.DataFrame({
        "timestamp": timestamps,
        "solexs_flux": 2.0e-6 + (1.0 + np.sin(phase / 8)) * 1.0e-6,
        "helios_flux": 1.0e-6 + (1.0 + np.cos(phase / 11)) * 0.6e-6,
    })
    transformer = FluxTransformer(TransformConfig()).fit(raw.iloc[:120])
    transformed = transformer.transform(raw)
    config = FeatureConfig(cadence_seconds=60)
    featured = add_rolling_features(transformed, config)
    feature_names = ["solexs_mean_5m"]
    x = featured[feature_names].to_numpy(dtype=float)
    y_now = (np.arange(len(x)) % 4).astype(int)
    y_forecast = np.column_stack([(np.arange(len(x)) + i) % 2 for i in range(5)]).astype(int)
    valid_train = np.flatnonzero(np.isfinite(x[:, 0]))[:120]
    model = MODEL_BUILDERS["logistic_regression"]().fit(
        x[valid_train], y_now[valid_train], y_forecast[valid_train], feature_names
    )

    model_id = "test_logistic_v1"
    transformer_path = model_dir / f"{model_id}.transformer.joblib"
    checkpoint_path = model_dir / f"{model_id}.joblib"
    manifest_path = model_dir / f"{model_id}.feature_manifest.json"
    joblib.dump(transformer, transformer_path)
    thresholds = [0.31, 0.41, 0.51, 0.61, 0.71]
    joblib.dump({"model": model, "feature_cols": feature_names, "thresholds": thresholds}, checkpoint_path)
    manifest = create_feature_manifest(
        featured, feature_names, "solexs_only", config, source_frame=raw,
        split_frames={"train": raw.iloc[:120], "validation": raw.iloc[120:150], "test": raw.iloc[150:]},
        preprocessing={"fit_on": "train_split_only"}, random_seed=42,
    )
    save_feature_manifest(str(manifest_path), manifest)
    compatibility = {"target_definition": "pipeline_forecast_targets_v1",
                     "forecast_horizons": ["1h", "3h", "6h", "12h", "24h"],
                     "feature_schema_version": manifest["feature_schema_version"]}
    labels = [0, 1, 0, 1, 0, 1]
    probs = [0.2, 0.8, 0.7, 0.4, 0.1, 0.9]
    horizon_pairs = {h: {"timestamps_utc": [t.isoformat() for t in timestamps[-6:]],
                         "labels": labels, "probabilities": probs, "unknown_label_count": 2}
                     for h in ["1h", "3h", "6h", "12h", "24h"]}
    evaluation_path = _write_json(model_dir / "evaluation.json", {
        "report_version": 2, "model_id": model_id, "evaluation_split": "chronological_test",
        "decision_thresholds": dict(zip(["1h", "3h", "6h", "12h", "24h"], thresholds)),
        "threshold_selection": {h: {"method": "validation_tss"} for h in horizon_pairs},
        "data_provenance": {"source": "fixture", "filename": "deterministic-test-fixture.csv"},
        "scientific_limitations": [],
        "test_predictions": {"available": True, "sample_count": 8, "evaluation_horizons": horizon_pairs},
    })
    refs = artifact_catalog.catalog_model_artifacts(model_id, model_dir, [
        ("model", checkpoint_path, "joblib"),
        ("preprocessor", transformer_path, "joblib"),
        ("evaluation", evaluation_path, "json"),
        ("feature_manifest", manifest_path, "json"),
    ], compatibility)
    record = ModelRecord(
        model_id=model_id, name="Test logistic regression", architecture="logistic_regression", version="v1",
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(), training_dataset="deterministic-test-fixture",
        input_configuration="solexs_only", metrics={"evaluation_split": "chronological_test"},
        checkpoint_path=str(checkpoint_path), scaler_path=str(transformer_path),
        transformer_path=str(transformer_path), model_kind="classical", status="trained",
        params={**compatibility, "n_features": len(feature_names), "thresholds": thresholds,
                "sequence_length": 60, "feature_manifest_path": str(manifest_path),
                "evaluation_report_path": str(evaluation_path),
                "feature_manifest_artifact_id": f"{model_id}:feature_manifest",
                "artifact_ids": [ref["artifact_id"] for ref in refs]},
    )
    registry.register(record)
    registry.set_active(model_id)
    yield {"record": registry.get(model_id), "raw": raw, "engine": engine, "factory": factory}
    engine.dispose()


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_real_supported_artifact_predicts_and_persists_result(trained_artifact, monkeypatch):
    raw = trained_artifact["raw"].iloc[:150]
    rows = raw.assign(timestamp=raw.timestamp.astype(str)).to_dict(orient="records")
    result = predict_service.predict_from_raw_window(rows, "test_logistic_v1")

    assert result["model_id"] == "test_logistic_v1"
    assert result["model_version"] == "v1"
    assert result["forecast_thresholds"] == dict(zip(["1h", "3h", "6h", "12h", "24h"], [0.31, 0.41, 0.51, 0.61, 0.71]))
    assert result["forecast_decisions"] == {
        horizon: result["forecast"][horizon] >= threshold
        for horizon, threshold in result["forecast_thresholds"].items()
    }
    assert result["score_semantics"].startswith("Uncalibrated")
    assert result["forecast_origin_timestamp_utc"] == raw.timestamp.iloc[-1].isoformat()
    assert result["input_window"]["row_count"] == len(raw)
    models = model_service.list_models()
    public_record = next(item for item in models if item["model_id"] == "test_logistic_v1")
    assert public_record["model_ready"] is True
    assert "checkpoint_path" not in public_record
    from fastapi import Response
    assert health_api.ml_readiness(Response())["model_ready"] is True
    engine = trained_artifact["engine"]
    from sqlalchemy.orm import Session
    with Session(engine) as db:
        db.add(InferenceRecord(
            model_id=result["model_id"], nowcast_state=result["nowcast"]["state"],
            nowcast_confidence=result["nowcast"]["confidence"], forecast=result["forecast"], risk=result["risk"],
            input_snapshot={key: value for key, value in result.items() if key != "input"},
        ))
        db.commit()
        record = db.query(InferenceRecord).one()
        assert record.input_snapshot["model_version"] == "v1"
        assert "solexs_flux" not in record.input_snapshot


def test_model_and_preprocessor_reload_in_a_new_process(trained_artifact, tmp_path):
    rows = trained_artifact["raw"].iloc[:150].assign(
        timestamp=lambda frame: frame.timestamp.astype(str)
    ).to_dict(orient="records")
    expected = predict_service.predict_from_raw_window(rows, "test_logistic_v1")
    request_path = tmp_path / "inference_rows.json"
    request_path.write_text(json.dumps(rows), encoding="utf-8")
    script = (
        "import json, sys; "
        "from app.services.predict_service import predict_from_raw_window; "
        "rows=json.load(open(sys.argv[1], encoding='utf-8')); "
        "result=predict_from_raw_window(rows, 'test_logistic_v1'); "
        "print(json.dumps({'forecast': result['forecast'], 'nowcast': result['nowcast']}))"
    )
    env = os.environ.copy()
    env["MODEL_DIR"] = str(Path(trained_artifact["record"]["checkpoint_path"]).parent)
    database_path = Path(trained_artifact["engine"].url.database).resolve().as_posix()
    env["DATABASE_URL"] = f"sqlite:///{database_path}"
    env["APP_ENV"] = "test"
    env["AUTO_MIGRATE"] = "false"
    completed = subprocess.run(
        [sys.executable, "-c", script, str(request_path)],
        cwd=Path(__file__).resolve().parents[1], env=env, check=True,
        capture_output=True, text=True, timeout=120,
    )
    restored = json.loads(completed.stdout.strip().splitlines()[-1])
    for horizon, value in expected["forecast"].items():
        assert restored["forecast"][horizon] == pytest.approx(value, rel=1e-6, abs=1e-7)
    for label, value in expected["nowcast"].items():
        if isinstance(value, (int, float)):
            assert restored["nowcast"][label] == pytest.approx(value, rel=1e-6, abs=1e-7)
        else:
            assert restored["nowcast"][label] == value


def test_liveness_is_independent_of_database_and_model(monkeypatch):
    monkeypatch.setattr(health_api, "engine", None)
    monkeypatch.setattr(health_api, "resolve_model", lambda: (_ for _ in ()).throw(AssertionError("model touched")))
    assert health_api.liveness() == {"status": "alive"}


def test_production_configuration_requires_explicit_database_url():
    from app.core.config import Settings

    production = Settings(_env_file=None, app_env="production", auto_migrate=False)
    with pytest.raises(ValueError, match="DATABASE_URL must be explicitly set"):
        production.validate_runtime()


def test_prediction_api_returns_typed_result_and_retrieves_persisted_prediction(trained_artifact):
    from sqlalchemy.orm import Session

    rows = trained_artifact["raw"].iloc[:150].assign(
        timestamp=lambda frame: frame.timestamp.astype(str)
    ).to_dict(orient="records")
    with Session(trained_artifact["engine"]) as db:
        body = predict_api.predict(PredictRequest(model_id="test_logistic_v1", rows=rows), db)
        validated = PredictResponse.model_validate(body)
        assert validated.model_version == "v1"
        assert validated.forecast_periods_utc["1h"]["start_exclusive_utc"] == validated.forecast_origin_timestamp_utc
        fetched = predict_api.get_prediction(validated.prediction_id, db)
        assert fetched["forecast"] == body["forecast"]
        assert db.query(InferenceRecord).count() == 1
        with pytest.raises(HTTPException) as missing:
            predict_api.get_prediction("not-a-prediction", db)
        assert missing.value.status_code == 404

    with pytest.raises(HTTPException) as unavailable:
        predict_api.predict(PredictRequest(model_id="missing-model", rows=rows), db=None)
    assert unavailable.value.status_code == 404


@pytest.mark.parametrize("rows, message", [
    ([{"timestamp": "not-a-time", "solexs_flux": 1, "helios_flux": 2}], "timestamps must be valid"),
    ([{"timestamp": "2026-01-01T00:00:00Z", "solexs_flux": "bad", "helios_flux": 2}], "non-numeric"),
    ([{"timestamp": "2026-01-01T00:00:00Z", "solexs_flux": float("inf"), "helios_flux": 2}], "infinite"),
    ([{"timestamp": "2026-01-01T00:00:00Z", "solexs_flux": -1, "helios_flux": 2}], "negative"),
    ([{"timestamp": "2026-01-01T00:00:00Z", "solexs_flux": float("nan"), "helios_flux": float("nan")}], "at least one finite"),
    ([{"timestamp": "2026-01-01T00:01:00Z", "solexs_flux": 1, "helios_flux": 2},
      {"timestamp": "2026-01-01T00:00:00Z", "solexs_flux": 1, "helios_flux": 2}], "strictly increasing"),
    ([{"timestamp": "2026-01-01T00:00:00Z", "solexs_flux": 1, "helios_flux": 2},
      {"timestamp": "2026-01-01T00:00:00Z", "solexs_flux": 1, "helios_flux": 2}], "duplicate"),
])
def test_prediction_input_validation(rows, message):
    with pytest.raises(ValueError, match=message):
        normalize_prediction_rows(rows, expected_cadence_seconds=60)


def test_irregular_cadence_is_reported_without_resampling():
    frame = normalize_prediction_rows([
        {"timestamp": "2026-01-01T00:00:00Z", "solexs_flux": 1, "helios_flux": 2},
        {"timestamp": "2026-01-01T00:02:00Z", "solexs_flux": 2, "helios_flux": 3},
    ], expected_cadence_seconds=60)
    assert "1 timestamp interval" in frame.attrs["data_quality_warnings"][0]
    assert frame.timestamp.iloc[1] - frame.timestamp.iloc[0] == pd.Timedelta(minutes=2)


def test_insufficient_feature_history_fails_without_imputation(trained_artifact):
    rows = trained_artifact["raw"].iloc[:1].assign(
        timestamp=lambda frame: frame.timestamp.astype(str)
    ).to_dict(orient="records")
    with pytest.raises(ValueError, match="sufficient valid history"):
        predict_service.predict_from_raw_window(rows, "test_logistic_v1")


def test_alerts_honor_saved_negative_decisions_and_configured_trigger(monkeypatch):
    from sqlalchemy.pool import StaticPool
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, expire_on_commit=False)()
    monkeypatch.setattr(settings, "alert_trigger_score_threshold", 0.4)
    monkeypatch.setattr(settings, "alert_clear_score_threshold", 0.3)
    try:
        assert alert_service.maybe_create_alert(
            db, "quiet", {"1h": 0.99}, "test-model", forecast_decisions={"1h": False}
        ) is None
        created = alert_service.maybe_create_alert(
            db, "quiet", {"1h": 0.42}, "test-model", is_demo=False,
            forecast_decisions={"1h": True}, forecast_thresholds={"1h": 0.40},
        )
        assert created is not None
        assert "0.420" in created.message and "trigger 0.400" in created.message
        assert "not a calibrated probability" in created.message
    finally:
        db.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_future_values_do_not_change_past_features_or_prediction(trained_artifact):
    raw = trained_artifact["raw"]
    cut = 140
    config = FeatureConfig(cadence_seconds=60)
    transformer = joblib.load(trained_artifact["record"]["transformer_path"])
    prefix = raw.iloc[:cut].copy()
    future = raw.iloc[cut:].copy()
    altered_future = future.copy()
    altered_future[["solexs_flux", "helios_flux"]] *= 1000

    left = add_rolling_features(transformer.transform(pd.concat([prefix, future])), config)
    right = add_rolling_features(transformer.transform(pd.concat([prefix, altered_future])), config)
    columns = ["solexs_flux_scaled", "helios_flux_scaled", "solexs_mean_5m", "solexs_spike_score"]
    np.testing.assert_allclose(left.iloc[:cut][columns], right.iloc[:cut][columns], equal_nan=True)
    before = predict_service._predict_with_record(trained_artifact["record"], left.iloc[:cut].copy())
    after = predict_service._predict_with_record(trained_artifact["record"], right.iloc[:cut].copy())
    assert before["forecast"] == after["forecast"]


def test_missing_or_corrupt_artifact_is_reported_unavailable(trained_artifact):
    checkpoint = trained_artifact["record"]["checkpoint_path"]
    with open(checkpoint, "ab") as stream:
        stream.write(b"corruption")
    rows = trained_artifact["raw"].iloc[:100].assign(
        timestamp=lambda frame: frame.timestamp.astype(str)
    ).to_dict(orient="records")
    with pytest.raises(predict_service.ModelUnavailableError, match="artifacts"):
        predict_service.predict_from_raw_window(rows, "test_logistic_v1")


def test_model_evaluation_recomputes_from_stored_holdout_pairs(trained_artifact):
    initial = model_service.get_model_evaluation("test_logistic_v1", "1h")
    explored = model_service.get_model_evaluation("test_logistic_v1", "1h", threshold=0.75)
    assert initial["threshold_used_for_test"] == 0.31
    assert initial["threshold_exploratory"] is False
    assert initial["metrics"]["tp"] == 3
    assert explored["threshold_exploratory"] is True
    assert explored["metrics"]["tp"] == 2
    assert initial["metrics"]["brier_score"] == explored["metrics"]["brier_score"]
    assert initial["reliability"]["sample_count"] == 6
    assert "labels" not in initial and "probabilities" not in initial


def test_evaluation_api_uses_service_payload_and_threshold_override(trained_artifact):
    default = models_api.model_evaluation("test_logistic_v1", "1h", None)
    exploratory = models_api.model_evaluation("test_logistic_v1", "1h", 0.75)
    assert default["metrics"]["brier_score"] == exploratory["metrics"]["brier_score"]
    assert exploratory["threshold_exploratory"] is True


def test_fusion_experiment_api_returns_blocked_status_not_fake_scores(monkeypatch):
    from app.services import pipeline_state

    monkeypatch.setattr(pipeline_state.pipeline_state, "dataset_meta", {"source": "demo"})
    payload = experiments_api.fusion_experiment_readiness()
    assert payload["status"] == "blocked"
    assert payload["results"] == {}
    assert payload["completed_seed_count"] == 0


def test_fusion_readiness_is_blocked_without_paired_scientific_data(monkeypatch):
    from app.services import pipeline_state

    monkeypatch.setattr(pipeline_state.pipeline_state, "dataset_meta", {"source": "demo"})
    result = model_service.run_fusion_experiment(None)
    assert result["status"] == "blocked"
    assert result["completed"] is False
    assert result["scientific_conclusion"] == "cannot_yet_be_determined"
    assert result["completed_seed_count"] == 0
    assert result["results"] == {}
    assert any("SoLEXS" in blocker for blocker in result["blockers"])
