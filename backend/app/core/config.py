import os
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    app_env: str = os.getenv("APP_ENV", "development")
    data_mode: str = os.getenv("DATA_MODE", "demo")   # "demo" | "real"
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./solarflare.db")
    model_dir: str = os.getenv("MODEL_DIR", "./ml/saved_models")
    sample_data_dir: str = os.getenv("SAMPLE_DATA_DIR", "./sample_data")
    cors_origins: str = ""

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        return value or ""

    @field_validator("cors_origins")
    @classmethod
    def reject_wildcard_cors(cls, origins):
        if "*" in origins.split(","):
            raise ValueError("CORS_ORIGINS cannot contain wildcard access.")
        return origins

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    class Config:
        env_file = ".env"


settings = Settings()
