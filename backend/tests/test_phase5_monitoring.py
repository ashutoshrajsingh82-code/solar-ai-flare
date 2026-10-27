import datetime as dt
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.database.db import Base
from app.database.models import AlertEventRecord, AlertRecord
from app.services import alert_service
from ml.inference.simulator import LiveSimulator
from ml.inference.risk import risk_band_for_score


@pytest.fixture
def alert_db(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False)()
    monkeypatch.setattr(settings, "alert_trigger_score_threshold", 0.5)
    monkeypatch.setattr(settings, "alert_clear_score_threshold", 0.4)
    monkeypatch.setattr(settings, "alert_cooldown_seconds", 900)
    monkeypatch.setattr(settings, "alert_expiry_seconds", 21600)
    monkeypatch.setattr(settings, "alert_material_score_change", 0.1)
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def _score(db, score, prediction_id, when):
    return alert_service.process_forecast_prediction(
        db, "quiet", {"1h": score}, "model-1", model_id="model-1",
        prediction_id=prediction_id, source_classification="test_replay", now=when,
    )[0]


def test_alert_raise_hysteresis_clear_and_rearm(alert_db):
    now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    raised = _score(alert_db, 0.55, "p1", now)
    assert raised["transition"] == "RAISED"
    assert raised["alert"].state == "ACTIVE"

    held = _score(alert_db, 0.45, "p2", now + dt.timedelta(minutes=1))
    assert held["transition"] == "UPDATED"
    assert held["alert"].state == "ACTIVE"

    cleared = _score(alert_db, 0.39, "p3", now + dt.timedelta(minutes=2))
    assert cleared["transition"] == "CLEARED"
    assert cleared["alert"].state == "CLEARED"

    rearmed = _score(alert_db, 0.6, "p4", now + dt.timedelta(minutes=3))
    assert rearmed["transition"] == "RAISED"
    assert alert_db.query(AlertRecord).filter_by(deduplication_key=raised["alert"].deduplication_key).count() == 2
    assert [event.transition for event in alert_db.query(AlertEventRecord).order_by(AlertEventRecord.id)] == [
        "RAISED", "UPDATED", "CLEARED", "RAISED"
    ]


def test_low_risk_is_normal_and_threshold_boundaries_are_shared(alert_db):
    for score, expected in [(0.29, "LOW"), (0.30, "WATCH"), (0.499999, "WATCH"),
                            (0.50, "MODERATE"), (0.70, "HIGH"), (0.85, "CRITICAL")]:
        assert risk_band_for_score(score) == expected
    result = _score(alert_db, 0.30, "low", dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc))
    assert result["transition"] == "NORMAL"
    assert alert_db.query(AlertRecord).count() == 0


def test_expiration_records_resolution_and_allows_rearm(alert_db):
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    first = _score(alert_db, 0.8, "expires", start)["alert"]
    second = _score(alert_db, 0.8, "after-expiry", start + dt.timedelta(seconds=21601))
    assert first.state == "CLEARED"
    assert first.resolution_reason == "expired"
    assert second["transition"] == "REARMED"
    assert second["alert"].id != first.id


def test_cooldown_suppresses_notice_but_keeps_alert_current(alert_db):
    now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    sender_calls = []
    send = lambda alert: sender_calls.append(alert.id) or True
    initial = alert_service.process_forecast_prediction(
        alert_db, "quiet", {"1h": 0.55}, "model-1", model_id="model-1",
        prediction_id="notice-1", source_classification="test_replay", now=now,
        notification_sender=send,
    )[0]["alert"]
    update = alert_service.process_forecast_prediction(
        alert_db, "quiet", {"1h": 0.72}, "model-1", model_id="model-1",
        prediction_id="notice-2", source_classification="test_replay", now=now + dt.timedelta(seconds=30),
        notification_sender=send,
    )[0]
    assert update["alert"].id == initial.id
    assert update["alert"].probability == 0.72
    assert update["alert"].notification_status == "cooldown_suppressed"
    assert sender_calls == [initial.id]


def test_notification_failure_is_not_marked_delivered(alert_db):
    now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    result = alert_service.process_forecast_prediction(
        alert_db, "quiet", {"1h": 0.8}, "model-1", model_id="model-1",
        prediction_id="notify-fail", source_classification="test_replay", now=now,
        notification_sender=lambda _alert: False,
    )[0]
    assert result["alert"].notification_status == "delivery_failed"
    assert result["alert"].notification_count == 0
    assert result["alert"].last_notification_at is None


def test_alert_state_survives_session_recreation(alert_db):
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    result = _score(alert_db, 0.66, "persisted", start)
    alert_id = result["alert"].id
    engine = alert_db.get_bind()
    alert_db.close()
    with sessionmaker(bind=engine)() as reopened:
        persisted = reopened.get(AlertRecord, alert_id)
        assert persisted.state == "ACTIVE"
        assert persisted.probability == 0.66
        assert reopened.query(AlertEventRecord).filter_by(alert_id=alert_id).count() == 1


def test_alert_duplicate_is_idempotent_and_invalid_score_does_not_clear(alert_db):
    now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    first = _score(alert_db, 0.8, "same", now)
    duplicate = _score(alert_db, 0.8, "same", now + dt.timedelta(minutes=1))
    assert first["transition"] == "RAISED"
    assert duplicate["idempotent"] is True
    assert alert_db.query(AlertEventRecord).count() == 1

    invalid = _score(alert_db, float("nan"), "bad", now + dt.timedelta(minutes=2))
    assert invalid["transition"] == "UNCHANGED_INVALID"
    assert first["alert"].state == "ACTIVE"


def test_concurrent_duplicate_alert_processing_keeps_one_active_alert(tmp_path, monkeypatch):
    """Database uniqueness constraints serialize simultaneous duplicate predictions."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'concurrent-alerts.db'}",
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr(settings, "alert_trigger_score_threshold", 0.5)
    monkeypatch.setattr(settings, "alert_clear_score_threshold", 0.4)
    barrier = threading.Barrier(2)
    when = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)

    def process():
        with sessions() as db:
            barrier.wait(timeout=5)
            return _score(db, 0.8, "same-concurrent-prediction", when)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: process(), range(2)))
        with sessions() as db:
            assert db.query(AlertRecord).filter_by(state="ACTIVE").count() == 1
            assert db.query(AlertEventRecord).count() == 1
            assert sum(bool(result.get("idempotent")) for result in results) == 1
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_notification_status_only_counts_confirmed_delivery(alert_db):
    now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    result = alert_service.process_forecast_prediction(
        alert_db, "quiet", {"1h": 0.8}, "model-1", model_id="model-1",
        prediction_id="notify", source_classification="test_replay", now=now,
        notification_sender=lambda _alert: True,
    )[0]
    assert result["alert"].notification_status == "delivered"
    assert result["alert"].notification_count == 1
    assert result["alert"].last_notification_at == now.replace(tzinfo=None)


def test_replay_predictor_only_gets_causal_raw_prefix_and_labels_remain_separate(monkeypatch):
    monkeypatch.setattr("ml.inference.simulator.BASE_TICK_SECONDS", 0.01)
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="min", tz="UTC"),
        "solexs_flux": [1.0, 2.0, 3.0], "helios_flux": [4.0, 5.0, 6.0],
    })
    outcomes = pd.DataFrame({"timestamp": frame.timestamp, "forecast_1h": [1, 0, 1]})
    seen = []
    sim = LiveSimulator()
    sim.configure(frame, frame.timestamp.iloc[0], frame.timestamp.iloc[-1],
                  lambda prefix: seen.append(prefix.copy()) or {"forecast": {"1h": 0.2}},
                  outcome_df=outcomes, expected_cadence_seconds=60)
    sim.set_speed(50)
    sim.start()
    sim._thread.join(timeout=2)
    assert sim.get_state()["status"] == "completed"
    assert [len(prefix) for prefix in seen] == [1, 2, 3]
    assert all(list(prefix.columns) == ["timestamp", "solexs_flux", "helios_flux"] for prefix in seen)
    assert "outcomes" not in seen[0].columns


def test_replay_rejects_unordered_or_duplicate_observation_times():
    sim = LiveSimulator()
    frame = pd.DataFrame({"timestamp": ["2026-01-01T00:01:00Z", "2026-01-01T00:00:00Z"],
                          "solexs_flux": [1, 2], "helios_flux": [3, 4]})
    with pytest.raises(ValueError, match="chronologically ordered"):
        sim.configure(frame, frame.timestamp.iloc[0], frame.timestamp.iloc[1], lambda _prefix: {})
