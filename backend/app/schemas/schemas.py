import datetime as dt
from pydantic import BaseModel, Field
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
    train_fraction: float = Field(default=0.70, gt=0, lt=1)
    validation_fraction: float = Field(default=0.15, gt=0, lt=1)
    train_end_utc: Optional[str] = None
    validation_end_utc: Optional[str] = None


class TrainRequest(BaseModel):
    model_type: str                         # baselines: "climatology" | "flux_rule" | "hard_soft_ratio_rule" | "two_feature_logreg" | "logistic_regression" | "random_forest" | "svm" | "gradient_boosting"; deep: "cnn_1d" | "lstm" | "cnn_lstm_fusion" | "cnn_transformer" | "dlstm"
    input_configuration: str = "fusion"      # "solexs_only" | "helios_only" | "fusion"
    sequence_length: int = 60
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    dropout: float = 0.3
    class_weighting: bool = True
    alpha: float = 1.0
    beta: float = 1.0
    random_seed: int = 42
    early_stopping_patience: int = 5
    minimum_improvement: float = 0.0001
    scheduler_patience: int = 2
    deterministic: bool = True
    decomposition_period_samples: int = 12
    decomposition_robust: bool = False
    dlstm_hidden_size: int = 64
    dlstm_num_layers: int = 2
    device: str = "auto"  # "auto" | "cpu" | "cuda"


class PredictRequest(BaseModel):
    model_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    rows: List[Dict[str, Any]] = Field(min_length=1, max_length=10000)  # timestamp and raw instrument fluxes


class PredictResponse(BaseModel):
    prediction_id: str
    model_id: str
    model_version: str
    model: str
    score_semantics: str
    target_definition: str
    forecast_horizons: List[str]
    forecast_periods_utc: Dict[str, Dict[str, str]]
    prediction_timestamp_utc: str
    forecast_origin_timestamp_utc: str
    input_window: Dict[str, Any]
    data_quality: Dict[str, Any]
    nowcast: Dict[str, Any]
    forecast: Dict[str, float]
    forecast_thresholds: Dict[str, float] = Field(default_factory=dict)
    forecast_decisions: Dict[str, bool] = Field(default_factory=dict)
    input: Dict[str, Any] = Field(default_factory=dict)
    risk: str


class AlertAckRequest(BaseModel):
    acknowledged: bool = True


class SimulationStartRequest(BaseModel):
    model_id: Optional[str] = None
    speed: int = 5
    sequence_length: int = 60
    dataset_id: Optional[str] = None
    replay_start_utc: Optional[dt.datetime] = None
    replay_end_utc: Optional[dt.datetime] = None


class ModelSelectRequest(BaseModel):
    model_id: str


class ThresholdRequest(BaseModel):
    model_id: str
    horizon: str = "1h"
    threshold: float = 0.5
