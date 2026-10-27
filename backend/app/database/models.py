import datetime as dt
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON, UniqueConstraint
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
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=dt.datetime.utcnow)
    severity = Column(String)          # LOW | WATCH | MODERATE | HIGH | CRITICAL
    nowcast_state = Column(String)
    forecast_horizon = Column(String)
    probability = Column(Float)
    message = Column(Text)
    model = Column(String)
    acknowledged = Column(Boolean, default=False)
    is_demo = Column(Boolean, default=True)


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
