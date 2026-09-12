import datetime as dt
from sqlalchemy.orm import Session
from app.database.models import AlertRecord

DEFAULT_THRESHOLDS = [
    (0.85, "CRITICAL"),
    (0.70, "HIGH"),
    (0.50, "MODERATE"),
    (0.30, "WATCH"),
]


def severity_for_probability(p: float, thresholds=None) -> str:
    thr = thresholds or DEFAULT_THRESHOLDS
    for cutoff, label in sorted(thr, key=lambda x: -x[0]):
        if p >= cutoff:
            return label
    return "LOW"


def build_alert_message(severity: str, horizon: str, probability: float, nowcast_state: str) -> str:
    pct = round(probability * 100, 1)
    if severity in ("CRITICAL", "HIGH"):
        return (f"{severity} SOLAR FLARE WARNING — M/X flare probability within {horizon}: {pct}%. "
                f"Current nowcast state: {nowcast_state.replace('_', ' ').title()}.")
    return f"{severity} advisory — M/X flare probability within {horizon}: {pct}%. Nowcast: {nowcast_state.replace('_', ' ').title()}."


def maybe_create_alert(db: Session, nowcast_state: str, forecast: dict, model: str, is_demo: bool = True):
    """Given a prediction result, creates an alert if probability crosses WATCH threshold or higher."""
    horizon, probability = max(forecast.items(), key=lambda kv: kv[1]) if forecast else ("1h", 0.0)
    severity = severity_for_probability(probability)
    if severity == "LOW":
        return None

    record = AlertRecord(
        timestamp=dt.datetime.utcnow(),
        severity=severity,
        nowcast_state=nowcast_state,
        forecast_horizon=horizon,
        probability=float(probability),
        message=build_alert_message(severity, horizon, probability, nowcast_state),
        model=model,
        acknowledged=False,
        is_demo=is_demo,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def list_alerts(db: Session, severity: str = None, limit: int = 100):
    q = db.query(AlertRecord).order_by(AlertRecord.timestamp.desc())
    if severity:
        q = q.filter(AlertRecord.severity == severity)
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
    """Populate a few illustrative demo alerts so the Alerts page isn't empty on first run."""
    existing = db.query(AlertRecord).count()
    if existing > 0:
        return
    samples = [
        (0.42, "WATCH", "pre_flare", "6h"),
        (0.61, "MODERATE", "pre_flare", "3h"),
        (0.78, "HIGH", "flare", "1h"),
        (0.33, "WATCH", "quiet", "12h"),
    ]
    now = dt.datetime.utcnow()
    for i, (prob, sev, state, horizon) in enumerate(samples):
        record = AlertRecord(
            timestamp=now - dt.timedelta(hours=(len(samples) - i) * 3),
            severity=sev, nowcast_state=state, forecast_horizon=horizon,
            probability=prob, message=build_alert_message(sev, horizon, prob, state),
            model="cnn_lstm_fusion (demo)", acknowledged=(i == 0), is_demo=True,
        )
        db.add(record)
    db.commit()
