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
    # create_all does not add columns to an existing SQLite database.
    if engine.dialect.name == "sqlite":
        columns = {column["name"] for column in inspect(engine).get_columns("datasets")}
        if "source_metadata" not in columns:
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE datasets ADD COLUMN source_metadata JSON"))
