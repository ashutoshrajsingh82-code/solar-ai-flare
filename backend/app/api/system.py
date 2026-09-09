import time
import torch
from fastapi import APIRouter
from app.core.config import settings
from app.services.pipeline_state import pipeline_state
from app.services import model_service
from ml.inference.simulator import live_simulator

router = APIRouter(prefix="/api/system", tags=["system"])

_start_time = time.time()


@router.get("/status")
def system_status():
    active = model_service.get_active_model()
    all_models = model_service.list_models()
    sim = live_simulator.get_state()
    active_model_id = active["model_id"] if active else (all_models[-1]["model_id"] if all_models else None)

    t0 = time.time()
    _ = 1 + 1
    latency_ms = round((time.time() - t0) * 1000, 3)

    return {
        "frontend_status": "connected",
        "backend_status": "running",
        "database_status": "connected",
        "model_loaded": active_model_id,
        "active_model": active_model_id,
        "inference_engine": "PyTorch + scikit-learn",
        "dataset": pipeline_state.dataset_meta.get("filename") if pipeline_state.dataset_meta else None,
        "last_prediction": sim.get("latest", {}).get("timestamp") if sim.get("latest") else None,
        "api_latency_ms": latency_ms,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_mode": settings.data_mode,
        "uptime_sec": round(time.time() - _start_time, 1),
        "simulation_running": sim.get("running", False),
    }
