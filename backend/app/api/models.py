
from typing import Annotated

from fastapi import APIRouter, HTTPException, BackgroundTasks, Header

from app.schemas.schemas import TrainRequest, ModelSelectRequest
from app.services import model_service
from app.services.job_manager import training_jobs
from app.services.pipeline_state import pipeline_state

router = APIRouter(prefix="/api/models", tags=["models"])


@router.post("/train")
def train(
    req: TrainRequest,
    background_tasks: BackgroundTasks,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key", max_length=128)] = None,
):
    """Start model training in the background."""
    try:
        input_config = {**req.model_dump(), "_pipeline_fingerprint": pipeline_state.snapshot_fingerprint()}
        job_id, created = training_jobs.create(input_config=input_config, idempotency_key=idempotency_key,
                                               return_created=True)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    # Execution is performed by app.worker, not by the API process. Jobs remain
    # durable and queued if the worker is temporarily unavailable.
    job_status = training_jobs.to_dict(job_id)["status"]

    return {
        "job_id": job_id,
        "status": job_status,
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
