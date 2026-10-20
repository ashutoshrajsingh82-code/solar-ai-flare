from sqlalchemy import create_engine, inspect, text

from app.database import db


def test_init_db_adds_job_metadata_columns_without_dropping_records(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{(tmp_path / 'legacy.sqlite').as_posix()}")
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE background_jobs (
                id INTEGER PRIMARY KEY, job_id VARCHAR(64) UNIQUE NOT NULL,
                job_type VARCHAR(32) NOT NULL, status VARCHAR(24) NOT NULL,
                result JSON, error TEXT, stage TEXT NOT NULL DEFAULT '', progress JSON NOT NULL DEFAULT '[]',
                created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL,
                started_at DATETIME, finished_at DATETIME
            )
        """))
        connection.execute(text("""INSERT INTO background_jobs
            (job_id, job_type, status, created_at, updated_at) VALUES
            ('legacy-job', 'training', 'done', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"""))
    monkeypatch.setattr(db, "engine", engine)
    db.init_db()
    db.init_db()
    columns = {item["name"] for item in inspect(engine).get_columns("background_jobs")}
    assert {"schema_version", "input_config", "artifact_refs", "error_category",
            "retry_count", "worker_id", "idempotency_key"} <= columns
    indexes = {item["name"]: item.get("column_names") for item in inspect(engine).get_indexes("background_jobs")}
    assert indexes["uq_background_job_type_idempotency_key"] == ["job_type", "idempotency_key"]
    with engine.connect() as connection:
        assert connection.execute(text("SELECT status FROM background_jobs WHERE job_id='legacy-job'")).scalar_one() == "done"
    engine.dispose()
