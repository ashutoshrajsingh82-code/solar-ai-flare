"""Run the application's ordered, additive database migrations."""

import logging

from app.core.config import settings
from app.database.db import engine, init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("solar-flare-ai.migrations")


def main() -> None:
    try:
        settings.validate_runtime()
        init_db()
        logger.info("database_migrations_completed database_dialect=%s", engine.dialect.name)
    except Exception:
        logger.exception("database_migrations_failed")
        raise


if __name__ == "__main__":
    main()
