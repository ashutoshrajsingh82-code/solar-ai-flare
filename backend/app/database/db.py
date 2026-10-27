from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker, declarative_base
from pathlib import Path
from app.core.config import settings

# Resolve relative SQLite paths against backend/, not the process working directory.
database_url = make_url(settings.database_url)
if database_url.drivername.startswith("sqlite") and database_url.database not in (None, ":memory:"):
    sqlite_path = Path(database_url.database)
    if not sqlite_path.is_absolute():
        backend_root = Path(__file__).resolve().parents[2]
        database_url = database_url.set(database=str((backend_root / sqlite_path).resolve()))

# Modular: swap DATABASE_URL to a postgres:// DSN to move off SQLite later.
connect_args = {"check_same_thread": False, "timeout": 5} if database_url.drivername.startswith("sqlite") else {}
engine = create_engine(database_url, connect_args=connect_args, pool_pre_ping=True, pool_timeout=5)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Apply the repository's ordered, idempotent schema migrations.

    Migrations are additive: they preserve historical rows and only add columns,
    indexes and tables required by newer application versions.
    """
    from app.database import models  # noqa: ensure models are registered
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"))
        applied = {row[0] for row in connection.execute(text("SELECT version FROM schema_migrations"))}
        if 1 not in applied:
            Base.metadata.create_all(bind=connection)
            connection.execute(text("INSERT INTO schema_migrations (version) VALUES (1)"))
    # create_all does not add columns to existing SQLite/PostgreSQL tables.
    # This migration is additive and preserves existing records.
    if engine.dialect.name in {"sqlite", "postgresql"}:
        additions = {
            "datasets": {"source_metadata": "JSON"},
            "background_jobs": {
                "schema_version": "INTEGER NOT NULL DEFAULT 1",
                "input_config": "JSON",
                "artifact_refs": "JSON NOT NULL DEFAULT '[]'",
                "error_category": "VARCHAR(64)",
                "retry_count": "INTEGER NOT NULL DEFAULT 0",
                "worker_id": "VARCHAR(128)",
                "idempotency_key": "VARCHAR(128)",
            },
        }
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"))
            applied = {row[0] for row in connection.execute(text("SELECT version FROM schema_migrations"))}
            if 2 not in applied:
                inspector = inspect(connection)
                for table, columns_to_add in additions.items():
                    if not inspector.has_table(table):
                        continue
                    existing = {column["name"] for column in inspector.get_columns(table)}
                    for name, sql_type in columns_to_add.items():
                        if name not in existing:
                            connection.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {sql_type}'))
                if inspector.has_table("background_jobs"):
                    indexes = {item["name"]: item.get("column_names", [])
                               for item in inspector.get_indexes("background_jobs")}
                    old_index = indexes.get("uq_background_job_idempotency_key")
                    if old_index is not None:
                        connection.execute(text("DROP INDEX uq_background_job_idempotency_key"))
                    if indexes.get("uq_background_job_type_idempotency_key") != ["job_type", "idempotency_key"]:
                        connection.execute(text(
                            "CREATE UNIQUE INDEX uq_background_job_type_idempotency_key "
                            "ON background_jobs (job_type, idempotency_key) WHERE idempotency_key IS NOT NULL"
                        ))
                connection.execute(text("INSERT INTO schema_migrations (version) VALUES (2)"))
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"))
            applied = {row[0] for row in connection.execute(text("SELECT version FROM schema_migrations"))}
            if 3 not in applied:
                additions = {
                    "alerts": {
                        "deduplication_key": "VARCHAR(256)", "state": "VARCHAR(24) NOT NULL DEFAULT 'CLEARED'",
                        "source_classification": "VARCHAR(32) NOT NULL DEFAULT 'legacy'", "model_version": "VARCHAR(64)",
                        "prediction_id": "VARCHAR(64)", "latest_observation_timestamp": "TIMESTAMP",
                        "latest_forecast_issue_timestamp": "TIMESTAMP", "last_updated_at": "TIMESTAMP",
                        "trigger_threshold": "FLOAT", "clear_threshold": "FLOAT",
                        "last_notification_at": "TIMESTAMP", "notification_count": "INTEGER NOT NULL DEFAULT 0",
                        "notification_status": "VARCHAR(32) NOT NULL DEFAULT 'not_configured'",
                        "resolved_at": "TIMESTAMP", "resolution_reason": "VARCHAR(128)",
                        "configuration_version": "VARCHAR(64)",
                    },
                    "inference_history": {
                        "source_classification": "VARCHAR(32) NOT NULL DEFAULT 'api_prediction'",
                        "replay_id": "VARCHAR(64)", "observation_timestamp": "TIMESTAMP",
                        "forecast_issue_timestamp": "TIMESTAMP", "input_window_start": "TIMESTAMP",
                        "input_window_end": "TIMESTAMP", "data_quality": "JSON",
                        "processing_status": "VARCHAR(24) NOT NULL DEFAULT 'complete'", "outcomes": "JSON",
                    },
                }
                inspector = inspect(connection)
                for table, columns_to_add in additions.items():
                    if not inspector.has_table(table):
                        continue
                    existing = {column["name"] for column in inspector.get_columns(table)}
                    for name, sql_type in columns_to_add.items():
                        if name not in existing:
                            connection.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {sql_type}'))
                Base.metadata.create_all(bind=connection)
                if engine.dialect.name in {"sqlite", "postgresql"}:
                    connection.execute(text(
                        "CREATE UNIQUE INDEX IF NOT EXISTS uq_alert_active_deduplication_key "
                        "ON alerts (deduplication_key) WHERE state = 'ACTIVE' AND deduplication_key IS NOT NULL"
                    ))
                    connection.execute(text(
                        "CREATE UNIQUE INDEX IF NOT EXISTS uq_alert_event_prediction_horizon "
                        "ON alert_events (deduplication_key, prediction_id, forecast_horizon)"
                    ))
                    connection.execute(text(
                        "CREATE UNIQUE INDEX IF NOT EXISTS uq_replay_inference_issue "
                        "ON inference_history (replay_id, model_id, forecast_issue_timestamp) "
                        "WHERE replay_id IS NOT NULL"
                    ))
                connection.execute(text("INSERT INTO schema_migrations (version) VALUES (3)"))
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"))
            applied = {row[0] for row in connection.execute(text("SELECT version FROM schema_migrations"))}
            if 4 not in applied:
                additions = {
                    "datasets": {
                        "content_sha256": "VARCHAR(64)", "file_format": "VARCHAR(16)",
                        "processing_status": "VARCHAR(24) NOT NULL DEFAULT 'complete'",
                        "schema_metadata": "JSON",
                    },
                    "simulation_replays": {
                        "preprocessing_fingerprint": "VARCHAR(64)", "last_observation_timestamp": "TIMESTAMP",
                        "last_prediction_id": "VARCHAR(64)",
                        "resume_policy": "VARCHAR(48) NOT NULL DEFAULT 'restart_from_beginning_idempotent'",
                    },
                }
                inspector = inspect(connection)
                for table, columns_to_add in additions.items():
                    if not inspector.has_table(table):
                        continue
                    existing = {column["name"] for column in inspector.get_columns(table)}
                    for name, sql_type in columns_to_add.items():
                        if name not in existing:
                            connection.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {sql_type}'))
                Base.metadata.create_all(bind=connection)
                connection.execute(text(
                    "CREATE INDEX IF NOT EXISTS ix_datasets_content_sha256 ON datasets (content_sha256)"
                ))
                connection.execute(text("INSERT INTO schema_migrations (version) VALUES (4)"))
