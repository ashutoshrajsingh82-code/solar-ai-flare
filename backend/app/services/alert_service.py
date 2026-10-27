"""Persistent, idempotent alert state transitions with provisional hysteresis."""
import datetime as dt
import math
import uuid

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.models import AlertEventRecord, AlertRecord
from ml.inference.risk import risk_band_for_score


def _utc_naive(value=None):
    value = value or dt.datetime.now(dt.timezone.utc)
    if isinstance(value, str):
        value = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, dt.datetime):
        raise ValueError("Alert provenance timestamps must be ISO strings or datetime values.")
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).replace(tzinfo=None)


def _validated_settings():
    trigger = float(settings.alert_trigger_score_threshold)
    clear = float(settings.alert_clear_score_threshold)
    if not (math.isfinite(trigger) and math.isfinite(clear) and 0 <= clear < trigger <= 1):
        raise ValueError("Alert thresholds must satisfy 0 <= clear < trigger <= 1.")
    if settings.alert_cooldown_seconds < 0 or settings.alert_expiry_seconds <= 0:
        raise ValueError("Alert cooldown and expiry settings are invalid.")
    return trigger, clear


def severity_for_score(p: float, thresholds=None) -> str:
    """Compatibility alias for the canonical provisional risk-band mapping."""
    return risk_band_for_score(p)


def build_alert_message(severity: str, horizon: str, score: float, nowcast_state: str,
                        decision_threshold=None, is_demo=False) -> str:
    source = "Synthetic/demo replay score" if is_demo else "Uncalibrated model score"
    threshold_text = (f"; provisional trigger {decision_threshold:.3f} was crossed"
                      if decision_threshold is not None else "")
    return (f"{severity} SOLAR FLARE SCORE ADVISORY — M/X onset classifier score for {horizon}: {score:.3f}"
            f"{threshold_text}. {source}; not a calibrated probability. "
            f"Current nowcast state: {str(nowcast_state).replace('_', ' ').title()}.")


def _alert_key(model_id, horizon, source_classification):
    return f"solar-flare|{model_id}|{horizon}|{source_classification}"


def _risk_rank(band):
    return {"LOW": 0, "WATCH": 1, "MODERATE": 2, "HIGH": 3, "CRITICAL": 4}.get(band, -1)


def _event_exists(db, key, prediction_id, horizon):
    return db.query(AlertEventRecord).filter_by(
        deduplication_key=key, prediction_id=prediction_id, forecast_horizon=horizon
    ).one_or_none()


def _process_one(db, *, horizon, score, model_id, model_name, model_version, nowcast_state,
                 prediction_id, source_classification, is_demo, observation_timestamp,
                 forecast_issue_timestamp, now, notification_sender=None, allow_raise=True):
    key = _alert_key(model_id, horizon, source_classification)
    # SQLite lacks SELECT ... FOR UPDATE. Acquire its write reservation before
    # checking the active key; PostgreSQL uses row locks plus the partial index.
    if db.get_bind().dialect.name == "sqlite" and not db.in_transaction():
        db.connection().exec_driver_sql("BEGIN IMMEDIATE")
    # A missing/invalid score records no normal/clear transition. The caller must
    # not turn invalid model output into zero risk or clear an active alert.
    if score is None or isinstance(score, bool):
        return {"transition": "UNCHANGED_INVALID", "alert": db.query(AlertRecord).filter_by(
            deduplication_key=key, state="ACTIVE").one_or_none(), "reason": "score_missing_or_invalid"}
    try:
        numeric = float(score)
    except (TypeError, ValueError):
        numeric = math.nan
    if not math.isfinite(numeric) or not 0 <= numeric <= 1:
        return {"transition": "UNCHANGED_INVALID", "alert": db.query(AlertRecord).filter_by(
            deduplication_key=key, state="ACTIVE").one_or_none(), "reason": "score_missing_or_invalid"}

    previous_event = _event_exists(db, key, prediction_id, horizon)
    if previous_event:
        existing = db.get(AlertRecord, previous_event.alert_id) if previous_event.alert_id else None
        return {"transition": previous_event.transition, "alert": existing, "idempotent": True}

    trigger, clear = _validated_settings()
    band = risk_band_for_score(numeric)
    active = db.query(AlertRecord).filter_by(deduplication_key=key, state="ACTIVE").with_for_update().one_or_none()
    prior_state = None
    if active and active.last_updated_at and (now - active.last_updated_at).total_seconds() >= settings.alert_expiry_seconds:
        active.state = "CLEARED"
        active.resolved_at = now
        active.resolution_reason = "expired"
        active.last_updated_at = now
        prior_state = "EXPIRED"
        active = None

    transition = "NORMAL"
    notification_due = False
    if active is None and numeric >= trigger and allow_raise:
        transition = "REARMED" if prior_state == "EXPIRED" else "RAISED"
        active = AlertRecord(
            timestamp=now, severity=band, nowcast_state=nowcast_state,
            forecast_horizon=horizon, probability=numeric,
            message=build_alert_message(band, horizon, numeric, nowcast_state, trigger, is_demo),
            model=model_name or model_id, acknowledged=False, is_demo=is_demo,
            deduplication_key=key, state="ACTIVE", source_classification=source_classification,
            model_version=model_version, prediction_id=prediction_id,
            latest_observation_timestamp=observation_timestamp,
            latest_forecast_issue_timestamp=forecast_issue_timestamp, last_updated_at=now,
            trigger_threshold=trigger, clear_threshold=clear, notification_count=0,
            notification_status="not_configured", configuration_version=settings.alert_config_version,
        )
        db.add(active)
        db.flush()
        notification_due = True
    elif active is not None and numeric < clear:
        active.probability = numeric
        active.severity = band
        active.prediction_id = prediction_id
        active.latest_observation_timestamp = observation_timestamp
        active.latest_forecast_issue_timestamp = forecast_issue_timestamp
        active.last_updated_at = now
        active.state = "CLEARED"
        active.resolved_at = now
        active.resolution_reason = "score_below_clear_threshold"
        active.message = build_alert_message(band, horizon, numeric, nowcast_state, clear, is_demo)
        transition = "CLEARED"
    elif active is not None:
        previous_band, previous_score = active.severity, float(active.probability or 0.0)
        active.probability = numeric
        active.severity = band
        active.prediction_id = prediction_id
        active.latest_observation_timestamp = observation_timestamp
        active.latest_forecast_issue_timestamp = forecast_issue_timestamp
        active.last_updated_at = now
        active.message = build_alert_message(band, horizon, numeric, nowcast_state, trigger, is_demo)
        transition = "UPDATED"
        # Notifications are intentionally disabled until a delivery integration
        # is configured. State changes are still persisted; no false success is recorded.
        meaningful = (_risk_rank(band) > _risk_rank(previous_band) or
                      abs(numeric - previous_score) >= settings.alert_material_score_change)
        if meaningful:
            last_sent = active.last_notification_at
            cooldown_passed = last_sent is None or (now - last_sent).total_seconds() >= settings.alert_cooldown_seconds
            notification_due = bool(cooldown_passed)
            if notification_sender is None:
                active.notification_status = "not_configured"
            elif not cooldown_passed:
                active.notification_status = "cooldown_suppressed"
    else:
        active = None

    db.add(AlertEventRecord(
        alert_id=active.id if active is not None else None,
        deduplication_key=key, prediction_id=prediction_id,
        forecast_horizon=horizon, transition=transition,
        score=numeric, risk_band=band, created_at=now,
    ))
    try:
        db.commit()
    except IntegrityError:
        # Unique active-alert and event-ledger indexes serialize duplicate work.
        db.rollback()
        previous_event = _event_exists(db, key, prediction_id, horizon)
        if previous_event:
            existing = db.get(AlertRecord, previous_event.alert_id) if previous_event.alert_id else None
            return {"transition": previous_event.transition, "alert": existing, "idempotent": True}
        active = db.query(AlertRecord).filter_by(deduplication_key=key, state="ACTIVE").one_or_none()
        if active is None:
            raise
        return _process_existing_after_race(
            db, active, key, prediction_id, horizon, numeric, band, nowcast_state,
            model_name, model_version, source_classification, is_demo,
            observation_timestamp, forecast_issue_timestamp, now, trigger, clear,
        )
    if active is not None:
        db.refresh(active)
    if active is not None and notification_due and notification_sender is not None:
        _deliver_notification(db, active.id, notification_sender, now)
    return {"transition": transition, "alert": active, "idempotent": False}


def _deliver_notification(db, alert_id, sender, now):
    """Deliver after the alert transaction commits; only confirmed sends count."""
    record = db.get(AlertRecord, alert_id)
    try:
        if sender(record) is not True:
            raise RuntimeError("Notification adapter did not confirm delivery.")
    except Exception:
        record.notification_status = "delivery_failed"
        db.commit()
        return
    record.last_notification_at = now
    record.notification_count = int(record.notification_count or 0) + 1
    record.notification_status = "delivered"
    db.commit()


def _process_existing_after_race(db, active, key, prediction_id, horizon, score, band, nowcast_state,
                                 model_name, model_version, source_classification, is_demo,
                                 observation_timestamp, forecast_issue_timestamp, now, trigger, clear):
    transition = "CLEARED" if score < clear else "UPDATED"
    active.probability = score
    active.severity = band
    active.prediction_id = prediction_id
    active.latest_observation_timestamp = observation_timestamp
    active.latest_forecast_issue_timestamp = forecast_issue_timestamp
    active.last_updated_at = now
    if score < clear:
        active.state = "CLEARED"
        active.resolved_at = now
        active.resolution_reason = "score_below_clear_threshold"
    db.add(AlertEventRecord(alert_id=active.id, deduplication_key=key,
                            prediction_id=prediction_id, forecast_horizon=horizon,
                            transition=transition, score=score, risk_band=band, created_at=now))
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        previous_event = _event_exists(db, key, prediction_id, horizon)
        return {"transition": previous_event.transition if previous_event else "UPDATED",
                "alert": db.get(AlertRecord, previous_event.alert_id) if previous_event and previous_event.alert_id else active,
                "idempotent": True}
    return {"transition": transition, "alert": active, "idempotent": False}


def process_forecast_prediction(db: Session, nowcast_state: str, forecast: dict, model: str,
                                is_demo: bool = False, forecast_decisions=None, forecast_thresholds=None,
                                *, model_id=None, model_version=None, prediction_id=None,
                                source_classification="api_prediction", observation_timestamp=None,
                                forecast_issue_timestamp=None, now=None, notification_sender=None):
    """Update the active alert per model/horizon and return transition summaries."""
    _validated_settings()
    model_id = model_id or model or "unknown-model"
    model_name = model or model_id
    prediction_id = str(prediction_id or uuid.uuid4())
    now = _utc_naive(now)
    obs = _utc_naive(observation_timestamp) if observation_timestamp else None
    issue = _utc_naive(forecast_issue_timestamp) if forecast_issue_timestamp else None
    results = []
    for horizon in ("1h", "3h", "6h", "12h", "24h"):
        if horizon not in (forecast or {}):
            continue
        results.append(_process_one(
            db, horizon=horizon, score=forecast.get(horizon), model_id=model_id,
            model_name=model_name, model_version=model_version, nowcast_state=nowcast_state,
            prediction_id=prediction_id, source_classification=source_classification,
            is_demo=is_demo, observation_timestamp=obs, forecast_issue_timestamp=issue, now=now,
            notification_sender=notification_sender,
            allow_raise=(forecast_decisions or {}).get(horizon) is not False,
        ))
    return results


def maybe_create_alert(db: Session, nowcast_state: str, forecast: dict, model: str,
                       is_demo: bool = True, forecast_decisions=None, forecast_thresholds=None,
                       **provenance):
    """Backward-compatible name; now performs stateful score-based transitions."""
    results = process_forecast_prediction(
        db, nowcast_state, forecast, model, is_demo, forecast_decisions,
        forecast_thresholds, **provenance,
    )
    return next((result["alert"] for result in results if result.get("alert") is not None), None)


def list_alerts(db: Session, severity: str = None, limit: int = 100, active_only: bool = False):
    q = db.query(AlertRecord).order_by(AlertRecord.timestamp.desc())
    if severity:
        q = q.filter(AlertRecord.severity == severity)
    if active_only:
        q = q.filter(AlertRecord.state == "ACTIVE")
    return q.limit(limit).all()


def acknowledge_alert(db: Session, alert_id: int, acknowledged: bool = True):
    record = db.query(AlertRecord).filter(AlertRecord.id == alert_id).first()
    if not record:
        raise KeyError(f"Alert {alert_id} not found")
    record.acknowledged = acknowledged
    db.commit()
    db.refresh(record)
    return record


def seed_demo_alerts(db: Session):
    """Legacy illustrative entries remain explicitly cleared demo history."""
    existing = db.query(AlertRecord).count()
    if existing > 0:
        return
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    samples = [(0.42, "WATCH", "pre_flare", "6h"), (0.61, "MODERATE", "pre_flare", "3h"),
               (0.78, "HIGH", "flare", "1h"), (0.33, "WATCH", "quiet", "12h")]
    for i, (score, band, state, horizon) in enumerate(samples):
        db.add(AlertRecord(
            timestamp=now - dt.timedelta(hours=(len(samples) - i) * 3), severity=band,
            nowcast_state=state, forecast_horizon=horizon, probability=score,
            message=build_alert_message(band, horizon, score, state, is_demo=True),
            model="demo illustration", acknowledged=(i == 0), is_demo=True,
            state="CLEARED", source_classification="demo_seed", resolved_at=now,
            resolution_reason="illustrative_seed_record", notification_status="not_configured",
        ))
    db.commit()
