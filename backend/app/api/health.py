from fastapi import APIRouter
import torch
from app.core.config import settings
from app.services.pipeline_state import pipeline_state
from app.services.model_service import registry

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health():
    active = registry.get_active()
    return {
        "status": "ok",
        "data_mode": settings.data_mode,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_loaded": active is not None or len(registry.list_all()) > 0,
        "dataset_loaded": pipeline_state.raw_df is not None,
    }
