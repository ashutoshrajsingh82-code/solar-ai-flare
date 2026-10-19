
import logging
import traceback

from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.schemas.schemas import TrainRequest, ModelSelectRequest
from app.services import model_service
from app.services.job_manager import training_jobs

logger = logging.getLogger("solar-flare-ai")

router = APIRouter(prefix="/api/models", tags=["models"])


def _run_training_job(job_id: str, req: TrainRequest):
    """Run model training as a background task."""
    training_jobs.set_running(job_id)

    try:
        result = model_service.train_model(
            req,
            progress_callback=lambda record: training_jobs.add_progress(job_id, record),
            stage_callback=lambda stage: training_jobs.set_stage(job_id, stage),
        )

        training_jobs.set_done(job_id, result)

    except ValueError as e:
        training_jobs.set_failed(job_id, str(e))

    except Exception as e:
        logger.error(
            "Training job %s crashed:\n%s",
            job_id,
            traceback.format_exc(),
        )
        training_jobs.set_failed(job_id, f"Training failed: {e}")


@router.post("/train")
def train(
    req: TrainRequest,
    background_tasks: BackgroundTasks,
):
    """Start model training in the background."""
    job_id = training_jobs.create()

    background_tasks.add_task(
        _run_training_job,
        job_id,
        req,
    )

    return {
        "job_id": job_id,
        "status": "pending",
    }


@router.get("/train/{job_id}/status")
def train_status(job_id: str):
    """Return the status of a training job."""
    job = training_jobs.to_dict(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Training job {job_id} not found",
        )

    return job


@router.get("")
def list_models():
    """List registered models."""
    return model_service.list_models()


@router.post("/select")
def select_model(req: ModelSelectRequest):
    """Select the active model."""
    try:
        return model_service.set_active_model(req.model_id)

    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/{model_id}/metrics")
def model_metrics(model_id: str):
    """Return metrics for a registered model."""
    try:
        return model_service.get_model_metrics(model_id)

    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
