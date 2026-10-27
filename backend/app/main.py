import sys
import os

# Ensure the backend root (containing both `app` and `ml` packages) is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.core.config import settings
from app.database.db import engine, init_db, SessionLocal
from app.api import health, data, preprocess, models, predict, alerts, simulation, experiments, dashboard, system, jobs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("solar-flare-ai")

app = FastAPI(
    title="Aditya-L1 Solar Flare Intelligence API",
    description="Dual X-Ray (SoLEXS + HEL1OS) nowcasting & forecasting backend. "
                "DEMO MODE uses synthetic data unless DATA_MODE=real is configured.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "X-Request-ID"],
)

for router in [health.router, data.router, preprocess.router, models.router, predict.router,
               alerts.router, simulation.router, experiments.router, dashboard.router, system.router, jobs.router]:
    app.include_router(router)


@app.on_event("startup")
def on_startup():
    try:
        settings.validate_runtime()
        if settings.should_auto_migrate:
            init_db()
            logger.info("database_migrations_completed")
        else:
            logger.info("database_migrations_skipped; run python -m app.migrate before startup")
        from app.services.simulation_service import recover_interrupted_replays
        recovered = recover_interrupted_replays()
        if recovered:
            logger.warning("Marked %d in-process replays interrupted for idempotent restart", recovered)
    except Exception:
        logger.exception("application_startup_configuration_or_database_failed")
        raise
    # DEMO MODE bootstrap: auto-load synthetic data + seed alerts so the app is usable immediately.
    if settings.data_mode == "demo" and settings.app_env.lower() not in {"production", "prod"}:
        try:
            db = SessionLocal()
            from app.services import data_service
            try:
                data_service.load_demo_data(db)
            finally:
                db.close()
            logger.info("Demo dataset loaded.")
        except Exception as e:
            logger.warning(f"Demo bootstrap failed (app will still run, load data manually): {e}")


@app.on_event("shutdown")
def on_shutdown():
    try:
        from ml.inference.simulator import live_simulator
        live_simulator.stop()
    finally:
        engine.dispose()


@app.get("/")
def root():
    return {
        "name": "ADITYA-L1 Solar Flare Intelligence API",
        "docs": "/docs",
        "data_mode": settings.data_mode,
    }
