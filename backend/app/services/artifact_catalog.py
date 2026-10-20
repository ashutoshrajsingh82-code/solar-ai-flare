"""Checksum-backed metadata catalog for artifacts in the configured model directory."""
import hashlib
from pathlib import Path

from app.database.db import SessionLocal
from app.database.models import ArtifactRecord


def _digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def catalog_model_artifacts(model_id, model_dir, artifacts, compatibility, job_id=None):
    """Persist artifact metadata after files have been completely written and validated."""
    root = Path(model_dir).resolve()
    rows = []
    for artifact_type, file_path, file_format in artifacts:
        path = Path(file_path).resolve()
        if path.parent != root or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Refusing to catalog missing, empty, or out-of-root {artifact_type} artifact")
        rows.append(ArtifactRecord(
            artifact_id=f"{model_id}:{artifact_type}", job_id=job_id, model_id=model_id,
            artifact_type=artifact_type, format=file_format, version="1",
            storage_key=path.name, sha256=_digest(path), size_bytes=path.stat().st_size,
            compatibility=dict(compatibility),
        ))
    refs = [{"artifact_id": row.artifact_id, "type": row.artifact_type,
             "storage_key": row.storage_key, "sha256": row.sha256} for row in rows]
    db = SessionLocal()
    try:
        for row in rows:
            existing = db.query(ArtifactRecord).filter_by(artifact_id=row.artifact_id).one_or_none()
            if existing:
                if existing.sha256 != row.sha256:
                    raise ValueError(f"Artifact identity already exists with different content: {row.artifact_id}")
                continue
            db.add(row)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    return refs


def verify_catalog_artifact(artifact_id, model_dir):
    db = SessionLocal()
    try:
        record = db.query(ArtifactRecord).filter_by(artifact_id=artifact_id).one_or_none()
        if record is None:
            raise ValueError(f"Artifact metadata is missing: {artifact_id}")
        root = Path(model_dir).resolve()
        path = (root / record.storage_key).resolve()
        if path.parent != root or not path.is_file() or path.stat().st_size != record.size_bytes:
            raise ValueError(f"Artifact file is missing or unsafe: {artifact_id}")
        if _digest(path) != record.sha256:
            raise ValueError(f"Artifact checksum validation failed: {artifact_id}")
        return record
    finally:
        db.close()


def verify_model_artifact_set(model_record, model_dir):
    """Require integrity-checked model, transformer, and evaluation records."""
    model_id = model_record["model_id"]
    artifact_ids = set(model_record.get("params", {}).get("artifact_ids", []))
    required = {f"{model_id}:model", f"{model_id}:preprocessor", f"{model_id}:evaluation"}
    if not required.issubset(artifact_ids):
        raise ValueError(f"Model {model_id} is missing required artifact catalog references")
    for artifact_id in sorted(required):
        item = verify_catalog_artifact(artifact_id, model_dir)
        compatibility = item.compatibility or {}
        for key in ("target_definition", "forecast_horizons", "feature_schema_version"):
            expected = model_record.get("params", {}).get(key)
            if compatibility.get(key) != expected:
                raise ValueError(f"Model artifact compatibility metadata is inconsistent: {artifact_id}")


def verify_model_artifact_set(model_record, model_dir):
    """Require integrity-checked model, transformer, and evaluation records."""
    model_id = model_record["model_id"]
    artifact_ids = set(model_record.get("params", {}).get("artifact_ids", []))
    required = {f"{model_id}:model", f"{model_id}:preprocessor", f"{model_id}:evaluation"}
    if not required.issubset(artifact_ids):
        raise ValueError(f"Model {model_id} is missing required artifact catalog references")
    for artifact_id in sorted(required):
        item = verify_catalog_artifact(artifact_id, model_dir)
        compatibility = item.compatibility or {}
        for key in ("target_definition", "forecast_horizons", "feature_schema_version"):
            expected = model_record.get("params", {}).get(key)
            if compatibility.get(key) != expected:
                raise ValueError(f"Model artifact compatibility metadata is inconsistent: {artifact_id}")


def catalog_pipeline_artifacts(job_id, artifact_dir, manifest_entries):
    """Catalog manifest-referenced pipeline outputs only after their hashes verify."""
    root = Path(artifact_dir).resolve()
    artifacts = []
    for stage, entry in manifest_entries.items():
        name = entry.get("file")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"Unsafe pipeline artifact filename in stage {stage}")
        path = (root / name).resolve()
        if path.parent != root or not path.is_file() or _digest(path) != entry.get("sha256"):
            raise ValueError(f"Pipeline artifact failed integrity validation: {stage}")
        artifacts.append(ArtifactRecord(
            artifact_id=f"{job_id}:pipeline:{stage}", job_id=job_id, artifact_type=f"pipeline_{stage}",
            format=entry.get("format", path.suffix.lstrip(".")), version="1", storage_key=name,
            sha256=entry["sha256"], size_bytes=path.stat().st_size,
            compatibility={"pipeline_manifest_version": 1},
        ))
    manifest_path = root / "pipeline_manifest.json"
    if manifest_path.is_file():
        artifacts.append(ArtifactRecord(
            artifact_id=f"{job_id}:pipeline:manifest", job_id=job_id,
            artifact_type="pipeline_manifest", format="json", version="1",
            storage_key=manifest_path.name, sha256=_digest(manifest_path),
            size_bytes=manifest_path.stat().st_size,
            compatibility={"pipeline_manifest_version": 1},
        ))
    refs = [{"artifact_id": a.artifact_id, "type": a.artifact_type,
             "storage_key": a.storage_key, "sha256": a.sha256} for a in artifacts]
    db = SessionLocal()
    try:
        for record in artifacts:
            if db.query(ArtifactRecord).filter_by(artifact_id=record.artifact_id).one_or_none() is None:
                db.add(record)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    return refs
