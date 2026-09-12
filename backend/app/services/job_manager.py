"""
Simple in-memory job registry for long-running background tasks (e.g. model
training). Matches the existing single-process, in-memory pattern already
used by PipelineState in this prototype -- this is not a distributed job
queue, just enough to let an HTTP request return immediately while the real
work continues in a background thread, with a job_id the frontend can poll.

Note: jobs live only in this process's memory. If the process restarts
(e.g. a Render free-tier spin-down), any jobs in flight are lost and the
frontend's poll will start getting 404s for that job_id -- which the
frontend should treat as "failed, please retry."
"""
import threading
import time
import uuid
from typing import Dict, Any, Optional


class Job:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = "pending"  # pending | running | done | failed
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.updated_at = self.created_at


class JobManager:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = Job(job_id)
        return job_id

    def set_running(self, job_id: str):
        self._update(job_id, status="running")

    def set_done(self, job_id: str, result: dict):
        self._update(job_id, status="done", result=result)

    def set_failed(self, job_id: str, error: str):
        self._update(job_id, status="failed", error=error)

    def _update(self, job_id: str, **kwargs):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for k, v in kwargs.items():
                setattr(job, k, v)
            job.updated_at = time.time()

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def to_dict(self, job_id: str) -> Optional[dict]:
        job = self.get(job_id)
        if not job:
            return None
        return {
            "job_id": job.job_id,
            "status": job.status,
            "result": job.result,
            "error": job.error,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }


# Module-level singleton -- mirrors how pipeline_state is used elsewhere.
training_jobs = JobManager()