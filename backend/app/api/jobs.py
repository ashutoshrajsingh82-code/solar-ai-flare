"""Read-only inspection endpoints for durable preprocessing and training jobs."""
from fastapi import APIRouter, HTTPException, Query

from app.services.job_manager import preprocess_jobs, training_jobs

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
def list_jobs(
    job_type: str | None = Query(default=None, pattern="^(training|preprocessing)$"),
    status: str | None = Query(default=None, pattern="^(pending|running|done|failed)$"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
):
    managers = {"training": training_jobs, "preprocessing": preprocess_jobs}
    selected = [managers[job_type]] if job_type else list(managers.values())
    rows = [job for manager in selected for job in manager.list(status=status, offset=offset, limit=limit)]
    rows.sort(key=lambda job: job["created_at"], reverse=True)
    return {"items": rows[:limit], "offset": offset, "limit": limit}


@router.get("/{job_id}")
def get_job(job_id: str):
    for manager in (training_jobs, preprocess_jobs):
        job = manager.to_dict(job_id)
        if job is not None:
            return job
    raise HTTPException(status_code=404, detail="Background job not found")
