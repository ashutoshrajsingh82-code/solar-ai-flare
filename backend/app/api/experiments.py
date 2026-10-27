from fastapi import APIRouter
from app.services import model_service

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.get("/fusion")
def fusion_experiment_readiness():
    """Read-only preflight for a rigorously paired, multi-instrument evaluation."""
    return model_service.run_fusion_experiment(None)
