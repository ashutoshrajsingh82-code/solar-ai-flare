import datetime as dt
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON
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
