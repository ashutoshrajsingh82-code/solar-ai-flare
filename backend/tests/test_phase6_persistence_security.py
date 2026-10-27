import asyncio
import io
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.datastructures import UploadFile

from app.api import data as data_api
from app.core.config import Settings, settings
from app.database.db import Base
from app.database.models import Dataset, SimulationReplayRecord
from app.services import data_service, simulation_service
from app.services.pipeline_state import PipelineState


def test_upload_is_generated_scoped_and_dataset_metadata_is_persistent(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_storage_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(data_service, "pipeline_state", PipelineState(tmp_path / "pipeline", restore=False))
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    body = ("timestamp,solexs_flux,helios_flux\n"
            "2026-01-01T00:00:00Z,1.0,2.0\n"
            "2026-01-01T00:01:00Z,1.1,2.1\n").encode()
    try:
        stored = asyncio.run(data_api._store_upload(
            UploadFile(filename="..\\..\\observations.csv", file=io.BytesIO(body)), {".csv"}
        ))
        assert Path(stored["path"]).parent == tmp_path / "uploads" / "raw"
        assert stored["original_filename"] == "observations.csv"
        assert Path(stored["path"]).name != stored["original_filename"]

        with factory() as db:
            result = data_service.load_uploaded_csv(
                db, stored["path"], stored["original_filename"], upload_metadata=stored,
            )
            record = db.get(Dataset, result["dataset_id"])
            assert record.processing_status == "complete"
            assert record.filepath == stored["storage_key"]
            assert record.content_sha256 == stored["content_sha256"]
            assert record.file_format == "csv"
            assert record.schema_metadata["rows"] == 2
            assert (tmp_path / "uploads" / record.filepath).is_file()
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_upload_rejects_binary_oversized_and_empty_files_without_temps(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_storage_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "max_upload_bytes", 8)
    with pytest.raises(ValueError, match="binary"):
        asyncio.run(data_api._store_upload(
            UploadFile(filename="renamed.csv", file=io.BytesIO(b"\x00\x01binary")), {".csv"}
        ))
    with pytest.raises(ValueError, match="maximum file size"):
        asyncio.run(data_api._store_upload(
            UploadFile(filename="large.csv", file=io.BytesIO(b"timestamp,solexs_flux,helios_flux")), {".csv"}
        ))
    with pytest.raises(ValueError, match="empty"):
        asyncio.run(data_api._store_upload(
            UploadFile(filename="empty.csv", file=io.BytesIO(b"")), {".csv"}
        ))
    root = tmp_path / "uploads"
    assert not list(root.glob(".upload-*.tmp"))
    assert not list((root / "raw").glob("*"))


def test_csv_parser_rejects_duplicate_headers_invalid_timestamps_and_non_numeric_flux(tmp_path):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as db:
        cases = {
            "duplicate.csv": "timestamp,solexs_flux,solexs_flux,helios_flux\n2026-01-01,1,2,3\n",
            "timestamp.csv": "timestamp,solexs_flux,helios_flux\nnot-a-time,1,2\n",
            "flux.csv": "timestamp,solexs_flux,helios_flux\n2026-01-01T00:00:00Z,unknown,2\n",
            "empty.csv": "timestamp,solexs_flux,helios_flux\n",
        }
        for filename, text in cases.items():
            path = tmp_path / filename
            path.write_text(text, encoding="utf-8")
            with pytest.raises(ValueError):
                data_service.load_uploaded_csv(db, str(path), filename)
        assert db.query(Dataset).count() == 0
    engine.dispose()


def test_cors_origin_parser_rejects_wildcards_and_non_origins():
    with pytest.raises(Exception):
        Settings(cors_origins="https://*.example.com")
    with pytest.raises(Exception):
        Settings(cors_origins="https://example.com/private/path")
    assert Settings(cors_origins="http://localhost:5173").allowed_origins == ["http://localhost:5173"]


def test_running_replay_is_marked_interrupted_for_deterministic_retry(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(simulation_service, "SessionLocal", factory)
    with factory() as db:
        db.add(SimulationReplayRecord(
            id="replay-1", dataset_id="dataset-1", model_id="model-1", model_version="v1",
            status="running", period_start_utc=pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime(),
            period_end_utc=pd.Timestamp("2026-01-02", tz="UTC").to_pydatetime(), configuration={},
        ))
        db.commit()
    try:
        assert simulation_service.recover_interrupted_replays() == 1
        with factory() as db:
            replay = db.get(SimulationReplayRecord, "replay-1")
            assert replay.status == "interrupted"
            assert replay.resume_policy == "restart_from_beginning_idempotent"
            assert "restarts from the beginning" in replay.error
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()
