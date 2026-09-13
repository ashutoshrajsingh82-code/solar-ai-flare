from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.schemas.schemas import PreprocessRequest
from app.services import preprocess_service
from app.services.pipeline_state import pipeline_state
from app.services.job_manager import preprocess_jobs

router = APIRouter(prefix="/api", tags=["preprocess"])


def _run_preprocess_job(job_id: str, req: PreprocessRequest):
    """Runs in a background thread after the POST /preprocess response is sent."""
    preprocess_jobs.set_running(job_id)
    try:
        result = preprocess_service.run_full_pipeline(req)
        preprocess_jobs.set_done(job_id, result)
    except ValueError as e:
        preprocess_jobs.set_failed(job_id, str(e))
    except Exception as e:
        preprocess_jobs.set_failed(job_id, f"Preprocessing failed: {e}")


@router.post("/preprocess")
def run_preprocess(req: PreprocessRequest, background_tasks: BackgroundTasks):
    """Kicks off the full clean -> align -> transform -> feature -> label
    pipeline in the background and returns immediately with a job_id. Poll
    GET /api/preprocess/{job_id}/status for progress/result. Cleaning +
    resampling + STL decomposition + feature engineering over the full
    dataset can take well over 2 minutes on Render's free-tier CPU,
    especially with compute_stl enabled -- that used to blow past the
    frontend's 120s axios timeout."""
    if pipeline_state.raw_df is None:
        raise HTTPException(status_code=422, detail="No dataset loaded. Load demo data or upload a dataset first.")
    job_id = preprocess_jobs.create()
    background_tasks.add_task(_run_preprocess_job, job_id, req)
    return {"job_id": job_id, "status": "pending"}


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
    sample = df[pipeline_state.feature_columns].tail(200).fillna(0)
    return {
        "feature_columns": pipeline_state.feature_columns,
        "correlation_matrix": sample.corr().round(3).to_dict(),
        "recent_values": sample.tail(50).to_dict(orient="records"),
        "hard_soft_ratio_recent": df["hard_soft_ratio"].tail(200).fillna(0).tolist() if "hard_soft_ratio" in df.columns else [],
        "timestamps_recent": df["timestamp"].tail(200).astype(str).tolist(),
    }