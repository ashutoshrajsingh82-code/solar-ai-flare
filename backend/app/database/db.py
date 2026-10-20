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
connect_args = {"check_same_thread": False} if database_url.drivername.startswith("sqlite") else {}
engine = create_engine(database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from app.database import models  # noqa: ensure models are registered
    Base.metadata.create_all(bind=engine)
    # create_all does not add columns to existing SQLite/PostgreSQL tables.
    # These additive migrations preserve existing records and are idempotent.
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
