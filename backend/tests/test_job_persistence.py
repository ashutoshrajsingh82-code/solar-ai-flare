import numpy as np
import pytest
from fastapi import BackgroundTasks, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.db import Base
from app.services.job_manager import JobManager
from app.schemas.schemas import TrainRequest


def make_manager(tmp_path, job_type="training"):
    engine = create_engine(f"sqlite:///{(tmp_path / 'jobs.sqlite').as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return JobManager(job_type, factory), engine


def test_job_history_survives_manager_recreation_and_progress_is_json_safe(tmp_path):
    manager, engine = make_manager(tmp_path)
    job_id = manager.create()
    manager.set_running(job_id)
    manager.set_stage(job_id, "training")
    manager.add_progress(job_id, {"epoch": np.int64(2), "loss": np.float32(0.25)})
    manager.set_done(job_id, {"finished": True})

    recreated = JobManager("training", manager._session_factory)
    recreated._jobs.clear()
    result = recreated.to_dict(job_id)
    assert result["status"] == "done"
    assert result["stage"] == "training"
    assert result["progress"] == [{"epoch": 2, "loss": 0.25}]
    assert result["result"] == {"finished": True}
    assert result["started_at"] is not None and result["finished_at"] is not None
    assert JobManager("training", manager._session_factory).to_dict(job_id)["status"] == "done"
    engine.dispose()


def test_job_types_are_separated_and_interrupted_jobs_recover_idempotently(tmp_path):
    manager, engine = make_manager(tmp_path, "preprocessing")
    job_id = manager.create()
    manager.set_running(job_id)
    pending_id = manager.create()
    other_type = JobManager("training", manager._session_factory)
    assert other_type.to_dict(job_id) is None

    assert manager.recover_interrupted() == 2
    recovered = manager.to_dict(job_id)
    assert recovered["status"] == "failed"
    assert recovered["stage"] == "interrupted"
    assert "not resumed" in recovered["error"]
    assert manager.to_dict(pending_id)["status"] == "failed"
    assert manager.recover_interrupted() == 0
    engine.dispose()


def test_manager_persists_stage_and_completion_timestamps(tmp_path):
    manager, engine = make_manager(tmp_path)
    job_id = manager.create()
    manager.set_stage(job_id, "fitting_preprocessor")
    assert manager.to_dict(job_id)["stage"] == "fitting_preprocessor"
    manager.set_running(job_id)
    manager.set_done(job_id, {})
    result = manager.to_dict(job_id)
    assert result["status"] == "done"
    assert result["started_at"] is not None
    assert result["finished_at"] is not None
    engine.dispose()


def test_failed_jobs_keep_error_and_unknown_jobs_remain_missing(tmp_path):
    manager, engine = make_manager(tmp_path)
    job_id = manager.create()
    manager.set_failed(job_id, "input validation failed")
    assert manager.to_dict(job_id)["error"] == "input validation failed"
    assert manager.to_dict("missing") is None
    engine.dispose()


def test_existing_training_job_routes_retain_response_contract(tmp_path, monkeypatch):
    from app.api import models as models_api

    manager, engine = make_manager(tmp_path)
    monkeypatch.setattr(models_api, "training_jobs", manager)
    response = models_api.train(TrainRequest(model_type="random_forest"), BackgroundTasks())
    assert set(response) == {"job_id", "status"}
    assert response["status"] == "pending"
    assert models_api.train_status(response["job_id"])["status"] == "pending"
    manager.set_running(response["job_id"])
    assert models_api.train_status(response["job_id"])["status"] == "running"
    manager.set_done(response["job_id"], {"metrics": {"score": 0.7}})
    manager._jobs.clear()
    assert models_api.train_status(response["job_id"])["status"] == "done"
    failed_id = manager.create()
    manager.set_failed(failed_id, "input validation failed")
    assert models_api.train_status(failed_id)["status"] == "failed"
    with pytest.raises(HTTPException) as exc:
        models_api.train_status("unknown-job")
    assert exc.value.status_code == 404
    engine.dispose()
