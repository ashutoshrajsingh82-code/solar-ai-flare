import threading
import time

from fastapi import APIRouter, Response
from sqlalchemy import text

from app.core.config import settings
from app.database.db import engine
from app.services.pipeline_state import pipeline_state
from app.services.model_service import registry
from app.services.predict_service import resolve_model, ModelUnavailableError

router = APIRouter(tags=["health"])
_ML_CACHE_SECONDS = 30
_ml_cache_lock = threading.Lock()
_ml_cache = {"checked_at": 0.0, "model_id": None, "ready": False}


def _ml_status():
    active = registry.get_active()
    model_id = active.get("model_id") if active else None
    now = time.monotonic()
    with _ml_cache_lock:
        if model_id == _ml_cache["model_id"] and now - _ml_cache["checked_at"] < _ML_CACHE_SECONDS:
            return active, _ml_cache["ready"]
        try:
            ready = resolve_model() is not None
        except ModelUnavailableError:
            ready = False
        _ml_cache.update(checked_at=now, model_id=model_id, ready=ready)
        return active, ready


@router.get("/api/health/live")
def liveness():
    """Process liveness; deliberately does not touch the database or model files."""
    return {"status": "alive"}


@router.get("/api/health/ready")
def readiness(response: Response):
    """Bounded database check; model validation is a separate endpoint."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        response.status_code = 503
        return {"status": "not_ready", "database_ready": False}
    return {"status": "ready", "database_ready": True}


@router.get("/api/health/ml")
def ml_readiness(response: Response):
    active, ready = _ml_status()
    if not ready:
        response.status_code = 503
    return {"status": "ready" if ready else "not_ready", "model_ready": ready,
            "active_model_id": active.get("model_id") if active else None,
            "message": None if ready else "No compatible active model artifacts are ready."}


@router.get("/api/health")
def health():
    # Keep the legacy endpoint cheap; explicit artifact validation is /api/health/ml.
    active = registry.get_active()
    return {
        "status": "ok",
        "data_mode": settings.data_mode,
        # This process does not initialize or require a GPU for readiness.
        "device": "cpu",
        "model_loaded": None,
        "model_ready": None,
        "active_model_id": active.get("model_id") if active else None,
        "readiness_error": "Model readiness is checked separately at /api/health/ml.",
        "dataset_loaded": pipeline_state.raw_df is not None,
    }
