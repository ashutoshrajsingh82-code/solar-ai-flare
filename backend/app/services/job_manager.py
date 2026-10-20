"""SQL-backed job history for the existing single-process thread workers."""
import datetime as dt
import logging
import math
import re
import threading
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.exc import IntegrityError

from app.database.db import SessionLocal
from app.database.models import BackgroundJobRecord

logger = logging.getLogger("solar-flare-ai")


def _json_safe(value: Any):
    if value is pd.NA or value is pd.NaT:
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (dt.datetime, dt.date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        timestamp = pd.Timestamp(value)
        return None if pd.isna(timestamp) else timestamp.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if pd.isna(value):
        return None
    raise TypeError(f"Job value of type {type(value).__name__} is not JSON serializable")


def _utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def _public_error(value):
    if not value:
        return value
    value = re.sub(r"[A-Za-z]:\\[^\s\"']+", "[path]", str(value))
    value = re.sub(r"(?<!:)\/(?:[^\s/]+\/)+[^\s/]*", "[path]", value)
    value = re.sub(r"(?:sqlite|postgres(?:ql)?|mysql)://[^\s]+", "[database-url]", value, flags=re.I)
    return value


def _epoch(value):
    if value is None:
        return None
    return value.replace(tzinfo=dt.timezone.utc).timestamp()


class Job:
    def __init__(self, job_id: str, status="pending", result=None, error=None, stage="", progress=None,
                 created_at=None, updated_at=None, started_at=None, finished_at=None,
                 input_config=None, artifact_refs=None, error_category=None, retry_count=0, worker_id=None):
        self.job_id = job_id
        self.status = status
        self.result = result
        self.error = error
        self.stage = stage or ""
        self.progress: List[dict] = progress or []
        now = dt.datetime.now(dt.timezone.utc).timestamp()
        self.created_at = created_at if created_at is not None else now
        self.updated_at = updated_at if updated_at is not None else self.created_at
        self.started_at = started_at
        self.finished_at = finished_at
        self.input_config = input_config or {}
        self.artifact_refs = artifact_refs or []
        self.error_category = error_category
        self.retry_count = retry_count
        self.worker_id = worker_id


class JobManager:
    def __init__(self, job_type: str = "generic", session_factory=None):
        self.job_type = job_type
        self._session_factory = session_factory or SessionLocal
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()

    def _to_job(self, rec):
        return Job(rec.job_id, rec.status, rec.result, rec.error, rec.stage, rec.progress or [],
                   _epoch(rec.created_at), _epoch(rec.updated_at), _epoch(rec.started_at), _epoch(rec.finished_at),
                   rec.input_config, rec.artifact_refs, rec.error_category, rec.retry_count, rec.worker_id)

    def _db_update(self, job_id: str, update):
        db = self._session_factory()
        try:
            rec = db.query(BackgroundJobRecord).filter_by(job_id=job_id, job_type=self.job_type).with_for_update().one_or_none()
            if rec is None:
                raise KeyError(f"{self.job_type} job {job_id} not found")
            update(rec)
            rec.updated_at = _utc_now()
            db.commit()
            db.refresh(rec)
            return rec
        except Exception:
            db.rollback()
            logger.exception("Could not persist %s job %s update", self.job_type, job_id)
            raise
        finally:
            db.close()

    def create(self, input_config=None, idempotency_key=None, return_created=False):
        with self._lock:
            db = self._session_factory()
            try:
                if idempotency_key:
                    if len(idempotency_key) > 128:
                        raise ValueError("Idempotency key must be at most 128 characters")
                    existing = db.query(BackgroundJobRecord).filter_by(
                        job_type=self.job_type, idempotency_key=idempotency_key
                    ).one_or_none()
                    if existing:
                        normalized = _json_safe(input_config or {})
                        if existing.input_config != normalized:
                            raise ValueError("Idempotency key was already used with different job configuration")
                        return (existing.job_id, False) if return_created else existing.job_id
                for _ in range(5):
                    job_id = uuid.uuid4().hex
                    if db.query(BackgroundJobRecord.id).filter_by(job_id=job_id).first() is None:
                        break
                else:
                    raise RuntimeError("Could not allocate a unique background job ID")
                now = _utc_now()
                rec = BackgroundJobRecord(job_id=job_id, job_type=self.job_type, status="pending", stage="",
                                          progress=[], created_at=now, updated_at=now, schema_version=1,
                                          input_config=_json_safe(input_config or {}), artifact_refs=[],
                                          retry_count=0, idempotency_key=idempotency_key)
                db.add(rec)
                db.commit()
                db.refresh(rec)
                self._jobs[job_id] = self._to_job(rec)
                logger.info("Created %s job %s", self.job_type, job_id)
                return (job_id, True) if return_created else job_id
            except IntegrityError:
                db.rollback()
                existing = None
                if idempotency_key:
                    existing = db.query(BackgroundJobRecord).filter_by(
                        job_type=self.job_type, idempotency_key=idempotency_key
                    ).one_or_none()
                if existing and existing.input_config == _json_safe(input_config or {}):
                    return (existing.job_id, False) if return_created else existing.job_id
                logger.exception("Could not persist creation of %s job", self.job_type)
                raise
            except Exception:
                db.rollback()
                logger.exception("Could not persist creation of %s job", self.job_type)
                raise
            finally:
                db.close()

    def _update(self, job_id: str, **kwargs):
        with self._lock:
            safe = _json_safe(kwargs)
            now = _utc_now()
            def apply(rec):
                allowed = {"pending": {"running", "failed"}, "running": {"done", "failed"},
                           "done": set(), "failed": set()}
                if "status" in safe and safe["status"] not in allowed.get(rec.status, set()):
                    raise ValueError(f"Invalid {self.job_type} job transition: {rec.status} -> {safe['status']}")
                if "status" not in safe and rec.status in {"done", "failed"}:
                    raise ValueError(f"Cannot update terminal {self.job_type} job {job_id}")
                for key, value in safe.items():
                    setattr(rec, key, value)
                if safe.get("status") in {"done", "failed"}:
                    rec.finished_at = now
                if safe.get("status") == "running" and rec.started_at is None:
                    rec.started_at = now
            rec = self._db_update(job_id, apply)
            if rec is not None:
                self._jobs[job_id] = self._to_job(rec)
                if "status" in safe:
                    logger.info("%s job %s status changed to %s", self.job_type, job_id, safe["status"])
                if "stage" in safe:
                    logger.info("%s job %s entered stage %s", self.job_type, job_id, safe["stage"])
            return rec

    def set_running(self, job_id: str):
        self._update(job_id, status="running", worker_id=f"pid-{__import__('os').getpid()}-{threading.get_ident()}")

    def set_done(self, job_id: str, result: dict, artifact_refs=None):
        self._update(job_id, status="done", result=_json_safe(result), error=None, error_category=None,
                     artifact_refs=artifact_refs or [])

    def set_failed(self, job_id: str, error: str, category="permanent"):
        self._update(job_id, status="failed", error=_public_error(error), error_category=category)

    def set_stage(self, job_id: str, stage: str):
        self._update(job_id, stage=str(stage))

    def add_progress(self, job_id: str, record: dict):
        with self._lock:
            safe = _json_safe(record)
            def apply(rec):
                if rec.status in {"done", "failed"}:
                    raise ValueError(f"Cannot update terminal {self.job_type} job {job_id}")
                rec.progress = list(rec.progress or []) + [safe]
            result = self._db_update(job_id, apply)
            if result is not None:
                self._jobs[job_id] = self._to_job(result)
                logger.debug("Persisted progress for %s job %s", self.job_type, job_id)

    def get(self, job_id: str) -> Optional[Job]:
        db = self._session_factory()
        try:
            rec = db.query(BackgroundJobRecord).filter_by(job_id=job_id, job_type=self.job_type).one_or_none()
            if rec is None:
                with self._lock:
                    self._jobs.pop(job_id, None)
                return None
            job = self._to_job(rec)
            with self._lock:
                self._jobs[job_id] = job
            return job
        except Exception:
            db.rollback()
            logger.exception("Could not retrieve %s job %s", self.job_type, job_id)
            raise
        finally:
            db.close()

    def to_dict(self, job_id: str) -> Optional[dict]:
        job = self.get(job_id)
        if not job:
            return None
        lifecycle = {"pending": "queued", "running": "running", "done": "succeeded", "failed": "failed"}
        return {
            "job_id": job.job_id, "status": job.status, "result": job.result,
            "lifecycle_state": lifecycle.get(job.status, job.status),
            "error": _public_error(job.error), "stage": job.stage, "progress": list(job.progress),
            "created_at": job.created_at, "updated_at": job.updated_at,
            "started_at": job.started_at, "finished_at": job.finished_at,
            "input_config": job.input_config, "artifact_refs": list(job.artifact_refs),
            "error_category": job.error_category, "retry_count": job.retry_count,
            "worker_id": job.worker_id,
        }

    def list(self, status=None, offset=0, limit=50):
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("offset must be nonnegative and limit must be between 1 and 100")
        if status is not None and status not in {"pending", "running", "done", "failed"}:
            raise ValueError("Unknown job status")
        db = self._session_factory()
        try:
            query = db.query(BackgroundJobRecord).filter_by(job_type=self.job_type)
            if status:
                query = query.filter_by(status=status)
            rows = query.order_by(BackgroundJobRecord.created_at.desc()).offset(offset).limit(limit).all()
            return [self.to_dict(row.job_id) for row in rows]
        finally:
            db.close()

    def claim_next(self, worker_id):
        """Atomically claim the oldest queued job; safe for concurrent DB workers."""
        db = self._session_factory()
        now = _utc_now()
        try:
            query = db.query(BackgroundJobRecord).filter_by(job_type=self.job_type, status="pending")
            rec = query.order_by(BackgroundJobRecord.created_at.asc()).first()
            if rec is None:
                return None
            changed = db.query(BackgroundJobRecord).filter_by(
                id=rec.id, status="pending"
            ).update({"status": "running", "worker_id": worker_id,
                      "started_at": now, "updated_at": now}, synchronize_session=False)
            if changed != 1:
                db.rollback()
                return None
            db.commit()
            return self.to_dict(rec.job_id)
        except Exception:
            db.rollback()
            logger.exception("Could not claim queued %s job", self.job_type)
            raise
        finally:
            db.close()

    def recover_interrupted(self):
        """Fail jobs left running after restart; queued jobs remain available to a worker."""
        db = self._session_factory()
        now = _utc_now()
        try:
            records = db.query(BackgroundJobRecord).filter(
                BackgroundJobRecord.job_type == self.job_type,
                BackgroundJobRecord.status == "running",
            ).all()
            for rec in records:
                old_status = rec.status
                rec.status = "failed"
                rec.error = f"Process restarted before {self.job_type} job completed (was {old_status}); computation was not resumed."
                rec.error_category = "interrupted"
                rec.stage = "interrupted"
                rec.finished_at = now
                rec.updated_at = now
            db.commit()
            with self._lock:
                for rec in records:
                    self._jobs[rec.job_id] = self._to_job(rec)
            if records:
                logger.warning("Recovered %d interrupted %s jobs", len(records), self.job_type)
            return len(records)
        except Exception:
            db.rollback()
            logger.exception("Could not recover interrupted %s jobs", self.job_type)
            raise
        finally:
            db.close()


training_jobs = JobManager("training")
preprocess_jobs = JobManager("preprocessing")
