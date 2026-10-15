import logging
import traceback

from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.schemas.schemas import TrainRequest, ModelSelectRequest
from app.services import model_service
from app.services.job_manager import training_jobs

logger = logging.getLogger("solar-flare-ai")

router = APIRouter(prefix="/api/models", tags=["models"])


def _run_training_job(job_id: str, req: TrainRequest):
    """Runs in a background thread after the POST /train response is sent."""
    training_jobs.set_running(job_id)
    try:
        result = model_service.train_model(
            req,
            progress_callback=lambda rec: training_jobs.add_progress(job_id, rec),
            stage_callback=lambda s: training_jobs.set_stage(job_id, s),
        )
        training_jobs.set_done(job_id, result)
    except ValueError as e:
        training_jobs.set_failed(job_id, str(e))
    except BaseException as e:  # never leave a job stuck in "running"
        logger.error("Training job %s crashed:\n%s", job_id, traceback.format_exc())
        training_jobs.set_failed(job_id, f"Training failed: {e}")


@router.post("/train")
def train(req: TrainRequest, background_tasks: BackgroundTasks):
    """Kicks off training in the background and returns immediately with a
    job_id. Poll GET /api/models/train/{job_id}/status for progress/result.
    This avoids the request blocking (and the frontend's axios timeout
    firing) while a CNN-LSTM etc. trains on CPU for multiple minutes."""
    job_id = training_jobs.create()
    background_tasks.add_task(_run_training_job, job_id, req)
    return {"job_id": job_id, "status": "pending"}


@router.get("/train/{job_id}/status")
def train_status(job_id: str):
    job = training_jobs.to_dict(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    return job


@router.get("")
def list_models():
    return model_service.list_models()


@router.post("/select")
def select_model(req: ModelSelectRequest):
    try:
        return model_service.set_active_model(req.model_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{model_id}/metrics")
def model_metrics(model_id: str):
    try:
        return model_service.get_model_metrics(model_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))