import os
from urllib.parse import urlsplit
from sqlalchemy.engine import make_url
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    app_env: str = os.getenv("APP_ENV", "development")
    data_mode: str = os.getenv("DATA_MODE", "demo")   # "demo" | "real"
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./solarflare.db")
    model_dir: str = os.getenv("MODEL_DIR", "./ml/saved_models")
    sample_data_dir: str = os.getenv("SAMPLE_DATA_DIR", "./sample_data")
    pipeline_artifact_dir: str = "./artifacts/pipeline"
    upload_storage_dir: str = "./data/uploads"
    max_upload_bytes: int = 50 * 1024 * 1024
    max_upload_rows: int = 1_000_000
    max_upload_columns: int = 64
    cors_origins: str = ""
    auto_migrate: bool | None = None
    alert_trigger_score_threshold: float = 0.50
    alert_clear_score_threshold: float = 0.40
    alert_cooldown_seconds: int = 900
    alert_expiry_seconds: int = 21600
    alert_material_score_change: float = 0.10
    alert_config_version: str = "provisional-v1"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        return value or ""

    @field_validator("cors_origins")
    @classmethod
    def reject_wildcard_cors(cls, origins):
        if "*" in origins:
            raise ValueError("CORS_ORIGINS cannot contain wildcard access.")
        for origin in (item.strip() for item in origins.split(",") if item.strip()):
            parsed = urlsplit(origin)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname or
                    parsed.username or parsed.password or parsed.path not in {"", "/"} or
                    parsed.query or parsed.fragment):
                raise ValueError("Each CORS origin must be an exact http(s) origin with no path, credentials, query, or fragment.")
        return origins

    @field_validator("max_upload_bytes", "max_upload_rows", "max_upload_columns")
    @classmethod
    def positive_upload_limits(cls, value):
        if value <= 0:
            raise ValueError("Upload size and dimension limits must be positive.")
        return value

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def should_auto_migrate(self) -> bool:
        return self.auto_migrate if self.auto_migrate is not None else self.app_env.lower() in {"development", "dev", "test"}

    def validate_runtime(self) -> None:
        mode = self.app_env.lower()
        if mode not in {"development", "dev", "test", "production", "prod"}:
            raise ValueError("APP_ENV must be development, test, or production.")
        if self.data_mode not in {"demo", "real"}:
            raise ValueError("DATA_MODE must be demo or real.")
        if mode in {"production", "prod"}:
            if not os.getenv("DATABASE_URL") and "database_url" not in self.model_fields_set:
                raise ValueError("DATABASE_URL must be explicitly set in production to a persistent database location.")
            database = make_url(self.database_url)
            if database.drivername.startswith("sqlite") and database.database in {None, ":memory:"}:
                raise ValueError("Production DATABASE_URL cannot use an in-memory SQLite database.")
            if self.should_auto_migrate:
                raise ValueError("AUTO_MIGRATE must be false in production; run `python -m app.migrate` before startup.")
        if any(origin.strip() == "*" for origin in self.allowed_origins):
            raise ValueError("Wildcard CORS origins are not permitted.")
        if not 0 <= self.alert_clear_score_threshold < self.alert_trigger_score_threshold <= 1:
            raise ValueError("Alert thresholds must satisfy 0 <= clear < trigger <= 1.")
        if self.alert_cooldown_seconds < 0 or self.alert_expiry_seconds <= 0 or self.alert_material_score_change <= 0:
            raise ValueError("Alert cooldown, expiry, and material-change settings are invalid.")

    class Config:
        env_file = ".env"


settings = Settings()
