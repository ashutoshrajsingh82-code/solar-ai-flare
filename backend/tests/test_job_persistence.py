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
    assert result["lifecycle_state"] == "succeeded"
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

    assert manager.recover_interrupted() == 1
    recovered = manager.to_dict(job_id)
    assert recovered["status"] == "failed"
    assert recovered["stage"] == "interrupted"
    assert "not resumed" in recovered["error"]
    assert manager.to_dict(pending_id)["status"] == "pending"
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


def test_job_transitions_are_validated_and_terminal_records_are_immutable(tmp_path):
    manager, engine = make_manager(tmp_path)
    job_id = manager.create()
    with pytest.raises(ValueError, match="pending -> done"):
        manager.set_done(job_id, {})
    manager.set_running(job_id)
    manager.set_done(job_id, {"ok": True}, artifact_refs=[{"artifact_id": "m:evaluation"}])
    data = manager.to_dict(job_id)
    assert data["artifact_refs"] == [{"artifact_id": "m:evaluation"}]
    assert data["worker_id"]
    with pytest.raises(ValueError, match="terminal"):
        manager.add_progress(job_id, {"step": 2})
    with pytest.raises(ValueError, match="done -> failed"):
        manager.set_failed(job_id, "late failure")
    engine.dispose()


def test_idempotency_key_reuses_same_job_and_rejects_changed_configuration(tmp_path):
    manager, engine = make_manager(tmp_path)
    job_id, created = manager.create({"model_type": "lr"}, "client-1", return_created=True)
    same_id, duplicate_created = manager.create({"model_type": "lr"}, "client-1", return_created=True)
    assert same_id == job_id and created and not duplicate_created
    other_type = JobManager("preprocessing", manager._session_factory)
    other_job_id = other_type.create({"model_type": "lr"}, "client-1")
    assert other_job_id != job_id
    with pytest.raises(ValueError, match="different job configuration"):
        manager.create({"model_type": "rf"}, "client-1")
    engine.dispose()


def test_job_listing_is_filtered_and_bounded(tmp_path):
    manager, engine = make_manager(tmp_path)
    one = manager.create({"n": 1})
    two = manager.create({"n": 2})
    manager.set_running(two)
    assert len(manager.list(status="pending", limit=1)) == 1
    assert manager.list(status="running")[0]["job_id"] == two
    with pytest.raises(ValueError, match="limit"):
        manager.list(limit=101)
    engine.dispose()


def test_queued_jobs_survive_recreation_and_only_one_worker_claims(tmp_path):
    manager, engine = make_manager(tmp_path)
    job_id = manager.create({"model_type": "random_forest"})
    restarted = JobManager("training", manager._session_factory)
    worker = "worker-test-1"
    claimed = restarted.claim_next(worker)
    assert claimed["job_id"] == job_id
    assert claimed["lifecycle_state"] == "running"
    assert claimed["worker_id"] == worker
    assert JobManager("training", manager._session_factory).claim_next("worker-test-2") is None
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


def test_training_route_idempotency_does_not_enqueue_a_duplicate(tmp_path, monkeypatch):
    from app.api import models as models_api

    manager, engine = make_manager(tmp_path)
    monkeypatch.setattr(models_api, "training_jobs", manager)
    req = TrainRequest(model_type="random_forest")
    first = models_api.train(req, BackgroundTasks(), idempotency_key="request-42")
    second = models_api.train(req, BackgroundTasks(), idempotency_key="request-42")
    assert first == second
    assert len(manager.list()) == 1
    with pytest.raises(HTTPException) as exc:
        models_api.train(TrainRequest(model_type="logistic_regression"), BackgroundTasks(), idempotency_key="request-42")
    assert exc.value.status_code == 409
    engine.dispose()


def test_worker_executes_claimed_training_and_persists_artifact_refs(tmp_path, monkeypatch):
    from app import worker

    manager, engine = make_manager(tmp_path)
    request = TrainRequest(model_type="random_forest")
    job_id = manager.create(request.model_dump())
    claimed = manager.claim_next("worker-fixture")
    expected = {"artifact_refs": [{"artifact_id": "model-1:evaluation"}], "model_id": "model-1"}
    monkeypatch.setattr(worker.model_service, "train_model", lambda *args, **kwargs: expected)
    worker._execute(manager, claimed, "worker-fixture")
    result = manager.to_dict(job_id)
    assert result["status"] == "done"
    assert result["result"]["model_id"] == "model-1"
    assert result["artifact_refs"] == expected["artifact_refs"]
    engine.dispose()


def test_worker_validation_failure_is_persisted_as_failed(tmp_path, monkeypatch):
    from app import worker

    manager, engine = make_manager(tmp_path)
    job_id = manager.create(TrainRequest(model_type="random_forest").model_dump())
    claimed = manager.claim_next("worker-fixture")

    def reject(*args, **kwargs):
        raise ValueError("real-data evidence missing")

    monkeypatch.setattr(worker.model_service, "train_model", reject)
    worker._execute(manager, claimed, "worker-fixture")
    result = manager.to_dict(job_id)
    assert result["status"] == "failed"
    assert result["error_category"] == "validation"
    assert "real-data evidence missing" in result["error"]
    engine.dispose()


def test_interrupted_worker_recovery_fails_running_but_preserves_queue(tmp_path):
    manager, engine = make_manager(tmp_path)
    running = manager.create()
    queued = manager.create()
    manager.set_running(running)
    assert manager.recover_interrupted() == 1
    assert manager.to_dict(running)["error_category"] == "interrupted"
    assert manager.to_dict(queued)["lifecycle_state"] == "queued"
    engine.dispose()
