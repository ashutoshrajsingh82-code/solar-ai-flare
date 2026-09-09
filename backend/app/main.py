import sys
import os

# Ensure the backend root (containing both `app` and `ml` packages) is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.core.config import settings
from app.database.db import init_db, SessionLocal
from app.api import health, data, preprocess, models, predict, alerts, simulation, experiments, dashboard, system

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
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in [health.router, data.router, preprocess.router, models.router, predict.router,
               alerts.router, simulation.router, experiments.router, dashboard.router, system.router]:
    app.include_router(router)


@app.on_event("startup")
def on_startup():
    init_db()
    logger.info(f"Database initialized at {settings.database_url}")

    # DEMO MODE bootstrap: auto-load synthetic data + seed alerts so the app is usable immediately.
    if settings.data_mode == "demo":
        try:
            db = SessionLocal()
            from app.services import data_service, alert_service
            data_service.load_demo_data(db)
            alert_service.seed_demo_alerts(db)
            db.close()
            logger.info("Demo dataset loaded and demo alerts seeded.")
        except Exception as e:
            logger.warning(f"Demo bootstrap failed (app will still run, load data manually): {e}")


@app.get("/")
def root():
    return {
        "name": "ADITYA-L1 Solar Flare Intelligence API",
        "docs": "/docs",
        "data_mode": settings.data_mode,
    }
