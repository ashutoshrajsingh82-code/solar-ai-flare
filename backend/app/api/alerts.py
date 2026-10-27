from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional

from app.database.db import get_db
from app.schemas.schemas import AlertAckRequest
from app.services import alert_service

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


def _serialize(a):
    return {
        "id": a.id, "timestamp": str(a.timestamp), "severity": a.severity,
        "nowcast_state": a.nowcast_state, "forecast_horizon": a.forecast_horizon,
        "probability": a.probability, "message": a.message, "model": a.model,
        "acknowledged": a.acknowledged, "is_demo": a.is_demo,
    }


@router.get("")
def get_alerts(severity: Optional[str] = None, limit: int = 100, db: Session = Depends(get_db)):
    alerts = alert_service.list_alerts(db, severity=severity, limit=limit)
    return [_serialize(a) for a in alerts]


@router.patch("/{alert_id}/acknowledge")
def acknowledge(alert_id: int, req: AlertAckRequest, db: Session = Depends(get_db)):
    try:
        record = alert_service.acknowledge_alert(db, alert_id, req.acknowledged)
        return _serialize(record)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
