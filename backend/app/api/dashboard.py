from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.services.pipeline_state import pipeline_state
from app.services import model_service, alert_service
from ml.inference.simulator import live_simulator

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary")
def summary(db: Session = Depends(get_db)):
    sim_state = live_simulator.get_state()
    latest = sim_state.get("latest")

    active_model = model_service.get_active_model()
    if not active_model:
        all_models = model_service.list_models()
        active_model = all_models[-1] if all_models else None

    recent_alerts = alert_service.list_alerts(db, limit=5)

    current_state = latest["nowcast"]["state"] if latest else "quiet"
    risk = latest.get("risk", "LOW") if latest else "LOW"
    forecast_1h = latest["forecast"].get("1h") if latest else None
    hard_soft_ratio = latest.get("hard_soft_ratio") if latest else None

    stages = pipeline_state.pipeline_status()
    pipeline_health = {
        "stages_complete": sum(1 for s in stages if s["status"] == "complete"),
        "total_stages": len(stages),
        "ready_for_training": pipeline_state.is_ready_for_training(),
    }

    return {
        "current_state": current_state,
        "risk_level": risk,
        "forecast_1h": forecast_1h,
        "active_model": active_model["name"] if active_model else None,
        "system_status": "MONITORING" if sim_state.get("running") else "IDLE",
        "recent_alerts": [
            {"id": a.id, "severity": a.severity, "message": a.message, "timestamp": str(a.timestamp),
             "acknowledged": a.acknowledged}
            for a in recent_alerts
        ],
        "pipeline_health": pipeline_health,
        "latest_inference": latest,
        "hard_soft_ratio": hard_soft_ratio,
        "dataset_meta": pipeline_state.dataset_meta,
    }
