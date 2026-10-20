import hashlib

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.db import Base
from app.services import artifact_catalog


def test_catalog_publishes_metadata_and_rejects_corruption(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{(tmp_path / 'artifact.sqlite').as_posix()}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(artifact_catalog, "SessionLocal", factory)
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"validated model artifact")
    refs = artifact_catalog.catalog_model_artifacts(
        "model-v1", tmp_path, [("model", artifact, "binary")],
        {"target_definition": "test-v1", "horizon": "1h"}, job_id="job-1",
    )
    assert refs[0]["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    record = artifact_catalog.verify_catalog_artifact("model-v1:model", tmp_path)
    assert record.job_id == "job-1"
    artifact.write_bytes(b"x" * len(b"validated model artifact"))
    with pytest.raises(ValueError, match="checksum"):
        artifact_catalog.verify_catalog_artifact("model-v1:model", tmp_path)
    engine.dispose()


def test_catalog_rejects_paths_outside_configured_root(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{(tmp_path / 'artifact.sqlite').as_posix()}")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(artifact_catalog, "SessionLocal", sessionmaker(bind=engine))
    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
    outside.write_bytes(b"data")
    with pytest.raises(ValueError, match="out-of-root"):
        artifact_catalog.catalog_model_artifacts("m", tmp_path, [("model", outside, "binary")], {})
    outside.unlink(missing_ok=True)
    engine.dispose()


def test_catalog_pipeline_artifacts_requires_manifest_hashes(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{(tmp_path / 'pipeline.sqlite').as_posix()}")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(artifact_catalog, "SessionLocal", sessionmaker(bind=engine))
    frame = tmp_path / "raw.csv"
    frame.write_bytes(b"timestamp,value\n2026-01-01,1\n")
    valid_hash = hashlib.sha256(frame.read_bytes()).hexdigest()
    refs = artifact_catalog.catalog_pipeline_artifacts(
        "job-pipe", tmp_path, {"raw": {"file": frame.name, "format": "csv", "sha256": valid_hash}}
    )
    assert refs[0]["artifact_id"] == "job-pipe:pipeline:raw"
    with pytest.raises(ValueError, match="integrity"):
        artifact_catalog.catalog_pipeline_artifacts(
            "job-bad", tmp_path, {"raw": {"file": frame.name, "format": "csv", "sha256": "0" * 64}}
        )
    engine.dispose()


def test_model_activation_contract_requires_all_compatible_artifacts(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{(tmp_path / 'model-set.sqlite').as_posix()}")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(artifact_catalog, "SessionLocal", sessionmaker(bind=engine))
    compatibility = {"target_definition": "target-v1", "forecast_horizons": ["1h"],
                     "feature_schema_version": "features-v1"}
    artifacts = []
    for name in ("model", "preprocessor", "evaluation"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("ascii"))
        artifacts.append((name, path, "binary"))
    refs = artifact_catalog.catalog_model_artifacts("model-1", tmp_path, artifacts, compatibility)
    record = {"model_id": "model-1", "params": {**compatibility,
               "artifact_ids": [item["artifact_id"] for item in refs]}}
    artifact_catalog.verify_model_artifact_set(record, tmp_path)
    record["params"]["forecast_horizons"] = ["3h"]
    with pytest.raises(ValueError, match="inconsistent"):
        artifact_catalog.verify_model_artifact_set(record, tmp_path)
    engine.dispose()
