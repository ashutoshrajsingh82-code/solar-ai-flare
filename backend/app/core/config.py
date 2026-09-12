import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_env: str = os.getenv("APP_ENV", "development")
    data_mode: str = os.getenv("DATA_MODE", "demo")   # "demo" | "real"
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./solarflare.db")
    model_dir: str = os.getenv("MODEL_DIR", "./ml/saved_models")
    sample_data_dir: str = os.getenv("SAMPLE_DATA_DIR", "./sample_data")
    cors_origins: list = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "*"
    ]

    class Config:
        env_file = ".env"


settings = Settings()
