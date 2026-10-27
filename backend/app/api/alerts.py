from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional

from app.database.db import get_db
from app.database.models import AlertEventRecord
from app.schemas.schemas import AlertAckRequest
from app.services import alert_service

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


def _serialize(a):
    return {
        "id": a.id, "timestamp": str(a.timestamp), "severity": a.severity,
        "state": a.state, "deduplication_key": a.deduplication_key,
        "nowcast_state": a.nowcast_state, "forecast_horizon": a.forecast_horizon,
        # Keep the legacy key for API compatibility, but expose the honest score meaning.
        "probability": a.probability, "model_score": a.probability,
        "score_semantics": "Illustrative demo score" if a.is_demo else "Uncalibrated model classification score",
        "message": a.message, "model": a.model,
        "acknowledged": a.acknowledged, "is_demo": a.is_demo,
        "source_classification": a.source_classification,
        "model_version": a.model_version, "prediction_id": a.prediction_id,
        "raised_at": str(a.timestamp), "last_updated_at": str(a.last_updated_at) if a.last_updated_at else None,
        "latest_observation_timestamp": str(a.latest_observation_timestamp) if a.latest_observation_timestamp else None,
        "latest_forecast_issue_timestamp": str(a.latest_forecast_issue_timestamp) if a.latest_forecast_issue_timestamp else None,
        "trigger_threshold": a.trigger_threshold, "clear_threshold": a.clear_threshold,
        "notification_count": a.notification_count, "notification_status": a.notification_status,
        "last_notification_at": str(a.last_notification_at) if a.last_notification_at else None,
        "resolved_at": str(a.resolved_at) if a.resolved_at else None,
        "resolution_reason": a.resolution_reason, "configuration_version": a.configuration_version,
    }


@router.get("")
def get_alerts(severity: Optional[str] = None, limit: int = 100, active_only: bool = False, db: Session = Depends(get_db)):
    alerts = alert_service.list_alerts(db, severity=severity, limit=limit, active_only=active_only)
    return [_serialize(a) for a in alerts]


@router.get("/config")
def alert_configuration():
    from app.core.config import settings
    return {"trigger_score_threshold": settings.alert_trigger_score_threshold,
            "clear_score_threshold": settings.alert_clear_score_threshold,
            "cooldown_seconds": settings.alert_cooldown_seconds,
            "expiry_seconds": settings.alert_expiry_seconds,
            "material_score_change": settings.alert_material_score_change,
            "configuration_version": settings.alert_config_version,
            "status": "provisional_validation_blocked",
            "note": "Score bands and trigger/clear settings are provisional; validation support and score calibration have not established production operating thresholds."}


@router.get("/{alert_id}/events")
def alert_events(alert_id: int, db: Session = Depends(get_db)):
    record = db.query(alert_service.AlertRecord).filter_by(id=alert_id).one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="Alert not found.")
    rows = db.query(AlertEventRecord).filter_by(alert_id=alert_id).order_by(AlertEventRecord.created_at.asc()).all()
    return [{"transition": row.transition, "prediction_id": row.prediction_id,
             "forecast_horizon": row.forecast_horizon, "score": row.score,
             "risk_band": row.risk_band, "timestamp": row.created_at.isoformat()} for row in rows]


@router.patch("/{alert_id}/acknowledge")
def acknowledge(alert_id: int, req: AlertAckRequest, db: Session = Depends(get_db)):
    try:
        record = alert_service.acknowledge_alert(db, alert_id, req.acknowledged)
        return _serialize(record)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
