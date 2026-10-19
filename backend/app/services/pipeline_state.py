"""Pipeline state with versioned, local-disk artifact persistence.

Parquet is preferred. If no Parquet engine is installed, CSV is used for
DataFrames. The manifest only references files created beneath the configured
artifact root and verifies their SHA-256 before loading.
"""
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd

from app.core.config import settings
from ml.preprocessing.transform import FluxTransformer

logger = logging.getLogger("solar-flare-ai")
STAGES = ("raw", "cleaned", "aligned", "transformed", "featured", "labeled")
STATE_FRAMES = (*STAGES, "train", "val", "test")
MANIFEST_NAME = "pipeline_manifest.json"
MANIFEST_VERSION = 1


def _json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_error(exc):
    message = str(exc).lower()
    if "integrity" in message or "digest" in message:
        return "artifact integrity check failed"
    if "outside" in message or "invalid pipeline artifact filename" in message:
        return "artifact path rejected"
    if "missing" in message or isinstance(exc, FileNotFoundError):
        return "artifact file missing or inaccessible"
    return "artifact unreadable or incompatible"


class PipelineState:
    def __init__(self, artifact_dir=None, restore=True):
        self.artifact_dir = Path(artifact_dir or settings.pipeline_artifact_dir)
        if not self.artifact_dir.is_absolute():
            backend_root = Path(__file__).resolve().parents[2]
            self.artifact_dir = backend_root / self.artifact_dir
        self.artifact_dir = self.artifact_dir.resolve()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        for name in STATE_FRAMES:
            setattr(self, f"{name}_df", None)
        self.dataset_meta: Dict[str, Any] = {}
        self.stage_reports: Dict[str, Any] = {}
        self.feature_columns: list = []
        self.transformer = None
        self.cadence_seconds: int = 60
        self._artifacts: Dict[str, dict] = {}
        self._artifact_errors: Dict[str, str] = {}
        self._lock = threading.RLock()
        self.operation_lock = threading.RLock()
        if restore:
            self.restore()

    @property
    def manifest_path(self):
        return self.artifact_dir / MANIFEST_NAME

    def _safe_artifact_path(self, filename):
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("Invalid pipeline artifact filename")
        path = (self.artifact_dir / filename).resolve()
        if path.parent != self.artifact_dir:
            raise ValueError("Pipeline artifact path is outside the configured directory")
        return path

    def _write_frame(self, key, df):
        if df is None:
            return None
        token = uuid.uuid4().hex
        # Try Parquet first, falling back only when the optional engine is absent.
        try:
            filename = f"{key}-{token}.parquet"
            target = self._safe_artifact_path(filename)
            fd, temp_name = tempfile.mkstemp(prefix=".pipeline-", suffix=".parquet", dir=self.artifact_dir)
            os.close(fd)
            try:
                df.to_parquet(temp_name, index=False)
                os.replace(temp_name, target)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
            fmt = "parquet"
        except ImportError:
            filename = f"{key}-{token}.csv"
            target = self._safe_artifact_path(filename)
            fd, temp_name = tempfile.mkstemp(prefix=".pipeline-", suffix=".csv", dir=self.artifact_dir)
            os.close(fd)
            try:
                df.to_csv(temp_name, index=False)
                os.replace(temp_name, target)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
            fmt = "csv"
            logger.info("Parquet engine unavailable; persisted pipeline frame %s as CSV", key)
        digest = _sha256_file(target)
        return {"file": filename, "format": fmt, "sha256": digest, "rows": int(len(df))}

    def _write_transformer(self):
        if self.transformer is None:
            return None
        if not isinstance(self.transformer, FluxTransformer) or not self.transformer.fitted:
            raise ValueError("Refusing to persist an incompatible or unfitted pipeline transformer")
        filename = f"transformer-{uuid.uuid4().hex}.joblib"
        target = self._safe_artifact_path(filename)
        fd, temp_name = tempfile.mkstemp(prefix=".pipeline-", suffix=".joblib", dir=self.artifact_dir)
        os.close(fd)
        try:
            joblib.dump(self.transformer, temp_name)
            os.replace(temp_name, target)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return {"file": filename, "format": "joblib", "sha256": _sha256_file(target)}

    def _write_manifest(self):
        manifest = {
            "version": MANIFEST_VERSION,
            "dataset_meta": _json_safe(self.dataset_meta),
            "stage_reports": _json_safe(self.stage_reports),
            "feature_columns": list(self.feature_columns),
            "cadence_seconds": int(self.cadence_seconds),
            "artifacts": self._artifacts,
        }
        fd, temp_name = tempfile.mkstemp(prefix=".pipeline-manifest-", suffix=".json", dir=self.artifact_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.manifest_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def _cleanup_unreferenced(self):
        referenced = {item.get("file") for item in self._artifacts.values() if isinstance(item, dict)}
        prefixes = (*STATE_FRAMES, "transformer")
        for prefix in prefixes:
            for path in self.artifact_dir.glob(f"{prefix}-*"):
                if path.name not in referenced and path.is_file():
                    try:
                        path.unlink()
                    except OSError:
                        logger.warning("Could not remove stale pipeline artifact %s", path.name)

    def persist(self, include_frames=True):
        with self._lock:
            return self._persist_locked(include_frames)

    def _persist_locked(self, include_frames=True):
        """Persist train/validation/test frames, transformer and manifest."""
        # Write all new artifacts first; publish the manifest only after they succeed.
        new_artifacts = dict(self._artifacts)
        if include_frames:
            for key in ("train", "val", "test"):
                df = getattr(self, f"{key}_df")
                if df is not None:
                    new_artifacts[key] = self._write_frame(key, df)
            if self.transformer is not None:
                new_artifacts["transformer"] = self._write_transformer()
        old = self._artifacts
        self._artifacts = new_artifacts
        try:
            self._write_manifest()
        except Exception:
            self._artifacts = old
            raise
        self._cleanup_unreferenced()
        logger.info("Persisted pipeline manifest and artifacts under %s", self.artifact_dir)

    def _read_frame(self, key, entry):
        path = self._safe_artifact_path(entry["file"])
        if not path.is_file():
            raise FileNotFoundError(f"Pipeline artifact for {key} is missing")
        if _sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"Pipeline artifact for {key} failed integrity validation")
        if entry.get("format") == "parquet" and path.suffix == ".parquet":
            return pd.read_parquet(path)
        if entry.get("format") == "csv" and path.suffix == ".csv":
            columns = pd.read_csv(path, nrows=0).columns
            return pd.read_csv(path, parse_dates=["timestamp"] if "timestamp" in columns else None)
        raise ValueError(f"Unsupported pipeline artifact format for {key}")

    def restore(self):
        if not self.manifest_path.exists():
            return
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") != MANIFEST_VERSION or not isinstance(manifest.get("artifacts"), dict):
                raise ValueError("Unsupported or malformed pipeline manifest")
            self.dataset_meta = manifest.get("dataset_meta") or {}
            self.stage_reports = manifest.get("stage_reports") or {}
            self.feature_columns = manifest.get("feature_columns") or []
            self.cadence_seconds = int(manifest.get("cadence_seconds", 60))
            self._artifacts = manifest["artifacts"]
        except Exception as exc:
            logger.exception("Pipeline manifest could not be restored")
            self._artifact_errors["manifest"] = "manifest is invalid or unreadable"
            return

        for key, entry in self._artifacts.items():
            if key not in STATE_FRAMES and key != "transformer":
                self._artifact_errors[key] = "Unknown artifact entry"
                continue
            try:
                if key == "transformer":
                    path = self._safe_artifact_path(entry["file"])
                    if not path.is_file() or _sha256_file(path) != entry.get("sha256"):
                        raise ValueError("Transformer artifact is missing or failed integrity validation")
                    transformer = joblib.load(path)
                    if not isinstance(transformer, FluxTransformer) or not transformer.fitted:
                        raise ValueError("Transformer artifact is incompatible or unfitted")
                    self.transformer = transformer
                else:
                    setattr(self, f"{key}_df", self._read_frame(key, entry))
                self._artifact_errors.pop(key, None)
                logger.info("Restored pipeline artifact %s", key)
            except Exception as exc:
                logger.exception("Pipeline artifact %s could not be restored", key)
                self._artifact_errors[key] = _artifact_error(exc)

    def reset_for_dataset(self):
        """Clear prior derived state before loading a different source dataset."""
        with self._lock:
            old = {name: getattr(self, f"{name}_df") for name in STATE_FRAMES}
            old_state = (self._artifacts, self.dataset_meta, self.stage_reports, self.feature_columns,
                         self.transformer, self.cadence_seconds, self._artifact_errors)
            for name in STATE_FRAMES:
                setattr(self, f"{name}_df", None)
            self._artifacts, self.dataset_meta, self.stage_reports = {}, {}, {}
            self.feature_columns, self.transformer, self.cadence_seconds = [], None, 60
            self._artifact_errors = {}
            try:
                self._write_manifest()
            except Exception:
                for name, value in old.items():
                    setattr(self, f"{name}_df", value)
                (self._artifacts, self.dataset_meta, self.stage_reports, self.feature_columns,
                 self.transformer, self.cadence_seconds, self._artifact_errors) = old_state
                raise

    def set_stage(self, stage_name: str, df: pd.DataFrame, report: dict, duration_sec: float):
        if stage_name not in STAGES:
            raise ValueError(f"Unknown pipeline stage: {stage_name}")
        enriched = {
            **report,
            "duration_sec": round(duration_sec, 4),
            "status": "complete",
            "updated_at": time.time(),
        }
        # Artifact file is committed before manifest/reference and in-memory state.
        with self._lock:
            self._set_stage_locked(stage_name, df, enriched)

    def _set_stage_locked(self, stage_name, df, enriched):
        entry = self._write_frame(stage_name, df)
        old_artifacts = self._artifacts
        new_artifacts = {**old_artifacts, stage_name: entry}
        if stage_name == "transformed":
            for split in ("train", "val", "test"):
                split_df = getattr(self, f"{split}_df")
                if split_df is not None:
                    new_artifacts[split] = self._write_frame(split, split_df)
            if self.transformer is not None:
                new_artifacts["transformer"] = self._write_transformer()
        old_report = self.stage_reports.get(stage_name)
        self._artifacts = new_artifacts
        self.stage_reports[stage_name] = enriched
        try:
            self._write_manifest()
        except Exception:
            self._artifacts = old_artifacts
            if old_report is None:
                self.stage_reports.pop(stage_name, None)
            else:
                self.stage_reports[stage_name] = old_report
            raise
        setattr(self, f"{stage_name}_df", df)
        self._artifact_errors.pop(stage_name, None)
        self._cleanup_unreferenced()

    def pipeline_status(self):
        out = []
        for stage in STAGES:
            report = self.stage_reports.get(stage)
            df = getattr(self, f"{stage}_df")
            warnings = list(report.get("warnings", [])) if report else []
            if stage in self._artifact_errors:
                warnings.append(f"Artifact unavailable: {self._artifact_errors[stage]}")
            if stage == "raw" and "manifest" in self._artifact_errors:
                warnings.append(f"Pipeline manifest unavailable: {self._artifact_errors['manifest']}")
            if stage == "labeled":
                warnings.extend(
                    f"{split} split unavailable: {self._artifact_errors[split]}"
                    for split in ("train", "val", "test") if split in self._artifact_errors
                )
                if self.transformer is None:
                    warnings.append("Fitted preprocessing transformer unavailable")
            report_without_artifact = bool(report and stage not in self._artifacts)
            if report_without_artifact:
                warnings.append("Stage has metadata but no persisted artifact reference")
            status = "unavailable" if stage in self._artifact_errors or (report and df is None) or report_without_artifact else ("complete" if report and df is not None else "pending")
            out.append({
                "stage": stage,
                "status": status,
                "input_rows": report.get("input_rows") if report else None,
                "output_rows": len(df) if df is not None else report.get("output_rows") if report else None,
                "duration_sec": report.get("duration_sec") if report else None,
                "warnings": warnings,
                "missing_values": report.get("nan_count_final", report.get("missing_after_resample")) if report else None,
                "details": report or {},
            })
        return out

    def is_ready_for_features(self) -> bool:
        return self.transformed_df is not None

    def is_ready_for_training(self) -> bool:
        return (
            self.labeled_df is not None
            and len(self.feature_columns) > 0
            and all(getattr(self, f"{name}_df") is not None for name in ("train", "val", "test"))
            and self.transformer is not None
            and getattr(self.transformer, "fitted", False)
        )


pipeline_state = PipelineState()
