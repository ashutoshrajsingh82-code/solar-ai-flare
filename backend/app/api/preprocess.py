from typing import Annotated

from fastapi import APIRouter, HTTPException, BackgroundTasks, Header
import pandas as pd
from app.schemas.schemas import PreprocessRequest
from app.services.pipeline_state import pipeline_state
from app.services.job_manager import preprocess_jobs
from app.core.config import settings

router = APIRouter(prefix="/api", tags=["preprocess"])


@router.post("/preprocess")
def run_preprocess(req: PreprocessRequest, background_tasks: BackgroundTasks,
                   idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key", max_length=128)] = None):
    """Kicks off the full clean -> align -> transform -> feature -> label
    pipeline in the background and returns immediately with a job_id. Poll
    GET /api/preprocess/{job_id}/status for progress/result. Cleaning +
    resampling + STL decomposition + feature engineering over the full
    dataset can take well over 2 minutes on Render's free-tier CPU,
    especially with compute_stl enabled -- that used to blow past the
    frontend's 120s axios timeout."""
    if pipeline_state.raw_df is None:
        detail = "No dataset loaded. Upload a local real-data product; real mode does not fall back to demo data." \
            if settings.data_mode == "real" else "No dataset loaded. Load demo data or upload a dataset first."
        raise HTTPException(status_code=422, detail=detail)
    try:
        input_config = {**req.model_dump(), "_pipeline_fingerprint": pipeline_state.snapshot_fingerprint()}
        job_id, created = preprocess_jobs.create(input_config=input_config, idempotency_key=idempotency_key,
                                                 return_created=True)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    # The separate database-polling worker claims queued work. The API request
    # only persists the job and returns its stable identifier.
    return {"job_id": job_id, "status": preprocess_jobs.to_dict(job_id)["status"]}


@router.get("/preprocess/{job_id}/status")
def preprocess_status(job_id: str):
    job = preprocess_jobs.to_dict(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Preprocessing job {job_id} not found")
    return job


@router.get("/pipeline/status")
def pipeline_status():
    return {"stages": pipeline_state.pipeline_status()}


@router.get("/features")
def list_features():
    if not pipeline_state.feature_columns:
        raise HTTPException(status_code=404, detail="No features generated yet. Run /api/preprocess first.")
    df = pipeline_state.featured_df
    sample = df[pipeline_state.feature_columns].tail(200)
    recent_values = sample.tail(50).astype(object).where(pd.notna(sample.tail(50)), None)
    ratio_values = df["hard_soft_ratio"].tail(200).astype(object).where(
        pd.notna(df["hard_soft_ratio"].tail(200)), None
    ).tolist() if "hard_soft_ratio" in df.columns else []
    return {
        "feature_columns": pipeline_state.feature_columns,
        "correlation_matrix": sample.corr().round(3).to_dict(),
        "recent_values": recent_values.to_dict(orient="records"),
        "hard_soft_ratio_recent": ratio_values,
        "timestamps_recent": df["timestamp"].tail(200).astype(str).tolist(),
    }
