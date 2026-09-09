from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class HealthResponse(BaseModel):
    status: str
    data_mode: str
    device: str
    model_loaded: bool
    dataset_loaded: bool


class DashboardSummary(BaseModel):
    current_state: str
    risk_level: str
    forecast_1h: Optional[float]
    active_model: Optional[str]
    system_status: str
    recent_alerts: List[Dict[str, Any]]
    pipeline_health: Dict[str, Any]
    latest_inference: Optional[Dict[str, Any]]
    hard_soft_ratio: Optional[float]


class DataInfoResponse(BaseModel):
    filename: str
    n_rows: int
    date_range_start: Optional[str]
    date_range_end: Optional[str]
    cadence_seconds: Optional[int]
    channels_detected: List[str]
    missing_values: int
    data_quality_score: float
    source: str


class PreprocessRequest(BaseModel):
    dataset_id: Optional[int] = None
    target_cadence: str = "1min"
    scaler_type: str = "standard"
    preflare_minutes: int = 30
    spike_zscore_threshold: float = 6.0
    compute_stl: bool = False


class TrainRequest(BaseModel):
    model_type: str                         # "logistic_regression" | "random_forest" | "svm" | "cnn_1d" | "lstm" | "cnn_lstm_fusion" | "cnn_transformer"
    input_configuration: str = "fusion"      # "solexs_only" | "helios_only" | "fusion"
    sequence_length: int = 60
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    dropout: float = 0.3
    class_weighting: bool = True
    alpha: float = 1.0
    beta: float = 1.0


class PredictRequest(BaseModel):
    model_id: Optional[str] = None
    rows: List[Dict[str, Any]]   # a window of raw rows: timestamp, solexs_flux, helios_flux


class AlertAckRequest(BaseModel):
    acknowledged: bool = True


class SimulationStartRequest(BaseModel):
    model_id: Optional[str] = None
    speed: int = 5
    sequence_length: int = 60


class ModelSelectRequest(BaseModel):
    model_id: str


class ThresholdRequest(BaseModel):
    model_id: str
    horizon: str = "1h"
    threshold: float = 0.5
