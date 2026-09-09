from fastapi import APIRouter, HTTPException
from app.schemas.schemas import PreprocessRequest
from app.services import preprocess_service
from app.services.pipeline_state import pipeline_state

router = APIRouter(prefix="/api", tags=["preprocess"])


@router.post("/preprocess")
def run_preprocess(req: PreprocessRequest):
    try:
        return preprocess_service.run_full_pipeline(req)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")


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
