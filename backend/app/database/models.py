import datetime as dt
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON, UniqueConstraint, Index, text
from app.database.db import Base


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    source = Column(String, default="upload")   # "demo" | "upload" | "real_adapter"
    n_rows = Column(Integer)
    date_range_start = Column(String)
    date_range_end = Column(String)
    cadence_seconds = Column(Integer)
    missing_values = Column(Integer)
    data_quality_score = Column(Float)
    channels_detected = Column(JSON)
    source_metadata = Column(JSON, default=dict)
    uploaded_at = Column(DateTime, default=dt.datetime.utcnow)
    filepath = Column(String)
    content_sha256 = Column(String(64), nullable=True, index=True)
    file_format = Column(String(16), nullable=True)
    processing_status = Column(String(24), nullable=False, default="complete", server_default="complete")
    schema_metadata = Column(JSON, nullable=True)


class ExperimentRecord(Base):
    __tablename__ = "experiments"
    id = Column(Integer, primary_key=True, index=True)
    experiment_name = Column(String)
    input_configuration = Column(String)   # "solexs_only" | "helios_only" | "fusion"
    model_architecture = Column(String)
    metrics = Column(JSON)
    dataset_id = Column(Integer)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    is_demo = Column(Boolean, default=True)


class ModelMetricRecord(Base):
    __tablename__ = "model_metrics"
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, index=True)
    architecture = Column(String)
    metrics = Column(JSON)
    training_history = Column(JSON)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class AlertRecord(Base):
    __tablename__ = "alerts"
    __table_args__ = (
        Index("uq_alert_active_deduplication_key", "deduplication_key", unique=True,
              sqlite_where=text("state = 'ACTIVE'"), postgresql_where=text("state = 'ACTIVE'")),
    )
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=dt.datetime.utcnow)
    severity = Column(String)          # LOW | WATCH | MODERATE | HIGH | CRITICAL
    nowcast_state = Column(String)
    forecast_horizon = Column(String)
    probability = Column(Float)  # Legacy column stores an uncalibrated classifier score.
    message = Column(Text)
    model = Column(String)
    acknowledged = Column(Boolean, default=False)
    is_demo = Column(Boolean, default=True)
    deduplication_key = Column(String(256), nullable=True, index=True)
    state = Column(String(24), nullable=False, default="ACTIVE", server_default="ACTIVE", index=True)
    source_classification = Column(String(32), nullable=False, default="api_prediction", server_default="api_prediction")
    model_version = Column(String(64), nullable=True)
    prediction_id = Column(String(64), nullable=True)
    latest_observation_timestamp = Column(DateTime, nullable=True)
    latest_forecast_issue_timestamp = Column(DateTime, nullable=True)
    last_updated_at = Column(DateTime, nullable=True)
    trigger_threshold = Column(Float, nullable=True)
    clear_threshold = Column(Float, nullable=True)
    last_notification_at = Column(DateTime, nullable=True)
    notification_count = Column(Integer, nullable=False, default=0, server_default="0")
    notification_status = Column(String(32), nullable=False, default="not_configured", server_default="not_configured")
    resolved_at = Column(DateTime, nullable=True)
    resolution_reason = Column(String(128), nullable=True)
    configuration_version = Column(String(64), nullable=True)


class AlertEventRecord(Base):
    """Idempotency ledger and append-only state-transition audit for alert scoring."""
    __tablename__ = "alert_events"
    __table_args__ = (UniqueConstraint("deduplication_key", "prediction_id", "forecast_horizon",
                                       name="uq_alert_event_prediction_horizon"),)
    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(Integer, nullable=True, index=True)
    deduplication_key = Column(String(256), nullable=False, index=True)
    prediction_id = Column(String(64), nullable=False)
    forecast_horizon = Column(String(8), nullable=False)
    transition = Column(String(24), nullable=False)
    score = Column(Float, nullable=True)
    risk_band = Column(String(16), nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)


class InferenceRecord(Base):
    __tablename__ = "inference_history"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=dt.datetime.utcnow)
    model_id = Column(String)
    nowcast_state = Column(String)
    nowcast_confidence = Column(Float)
    forecast = Column(JSON)
    risk = Column(String)
    input_snapshot = Column(JSON)
    source_classification = Column(String(32), nullable=False, default="api_prediction", server_default="api_prediction")
    replay_id = Column(String(64), nullable=True, index=True)
    observation_timestamp = Column(DateTime, nullable=True)
    forecast_issue_timestamp = Column(DateTime, nullable=True)
    input_window_start = Column(DateTime, nullable=True)
    input_window_end = Column(DateTime, nullable=True)
    data_quality = Column(JSON, nullable=True)
    processing_status = Column(String(24), nullable=False, default="complete", server_default="complete")
    outcomes = Column(JSON, nullable=True)


class SimulationReplayRecord(Base):
    __tablename__ = "simulation_replays"
    id = Column(String(64), primary_key=True)
    dataset_id = Column(String(128), nullable=False)
    model_id = Column(String(128), nullable=False)
    model_version = Column(String(64), nullable=False)
    source_classification = Column(String(32), nullable=False, default="historical_test_replay")
    status = Column(String(24), nullable=False, index=True)
    period_start_utc = Column(DateTime(timezone=True), nullable=False)
    period_end_utc = Column(DateTime(timezone=True), nullable=False)
    configuration = Column(JSON, nullable=False, default=dict)
    preprocessing_fingerprint = Column(String(64), nullable=True)
    last_observation_timestamp = Column(DateTime, nullable=True)
    last_prediction_id = Column(String(64), nullable=True)
    resume_policy = Column(String(48), nullable=False, default="restart_from_beginning_idempotent",
                           server_default="restart_from_beginning_idempotent")
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    finished_at = Column(DateTime, nullable=True)


class BackgroundJobRecord(Base):
    __tablename__ = "background_jobs"
    __table_args__ = (UniqueConstraint("job_type", "idempotency_key", name="uq_background_job_type_idempotency_key"),)
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(64), unique=True, nullable=False, index=True)
    job_type = Column(String(32), nullable=False, index=True)
    status = Column(String(24), nullable=False, index=True)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    stage = Column(Text, default="", nullable=False)
    progress = Column(JSON, default=list, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    schema_version = Column(Integer, nullable=False, default=1)
    input_config = Column(JSON, nullable=True)
    artifact_refs = Column(JSON, nullable=False, default=list)
    error_category = Column(String(64), nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    worker_id = Column(String(128), nullable=True)
    idempotency_key = Column(String(128), nullable=True)


class ArtifactRecord(Base):
    """Metadata catalog for locally published model and evaluation artifacts."""
    __tablename__ = "artifacts"
    id = Column(Integer, primary_key=True, index=True)
    artifact_id = Column(String(128), unique=True, nullable=False, index=True)
    job_id = Column(String(64), nullable=True, index=True)
    model_id = Column(String(128), nullable=True, index=True)
    artifact_type = Column(String(64), nullable=False)
    format = Column(String(32), nullable=False)
    version = Column(String(64), nullable=False)
    storage_key = Column(String(512), nullable=False)
    sha256 = Column(String(64), nullable=False)
    size_bytes = Column(Integer, nullable=False)
    compatibility = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
