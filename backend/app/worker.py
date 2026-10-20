"""Single-process durable database-polling worker for supported long jobs.

Run from backend/: python -m app.worker
"""
import argparse
import logging
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from app.database.db import init_db
from app.schemas.schemas import PreprocessRequest, TrainRequest
from app.services import model_service, preprocess_service
from app.services.artifact_catalog import catalog_pipeline_artifacts
from app.services.job_manager import preprocess_jobs, training_jobs
from app.services.pipeline_state import pipeline_state

logger = logging.getLogger("solar-flare-ai.worker")


@contextmanager
def _single_worker_lock(lock_path):
    """Acquire a cross-process advisory lock for the configured local artifact volume."""
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = open(path, "a+b")
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another worker already holds the configured artifact-volume lock") from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Another worker already holds the configured artifact-volume lock") from exc
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        stream.close()


def _execute(manager, job, worker_id):
    job_id = job["job_id"]
    try:
        manager.set_stage(job_id, "validating_configuration")
        config = job.get("input_config") or {}
        pipeline_state.restore()
        expected_fingerprint = config.pop("_pipeline_fingerprint", None)
        if expected_fingerprint and pipeline_state.snapshot_fingerprint() != expected_fingerprint:
            raise ValueError("The persisted pipeline dataset changed after this job was queued; submit a new job.")
        if manager.job_type == "training":
            request = TrainRequest(**config)
            result = model_service.train_model(
                request,
                progress_callback=lambda item: manager.add_progress(job_id, item),
                stage_callback=lambda stage: manager.set_stage(job_id, stage),
                job_id=job_id,
            )
            refs = result.get("artifact_refs", [])
        else:
            request = PreprocessRequest(**config)
            result = preprocess_service.run_full_pipeline(
                request, stage_callback=lambda stage: manager.set_stage(job_id, stage)
            )
            refs = catalog_pipeline_artifacts(job_id, pipeline_state.artifact_dir, pipeline_state._artifacts)
            result["artifact_refs"] = refs
        manager.set_done(job_id, result, artifact_refs=refs)
    except ValueError as exc:
        manager.set_failed(job_id, str(exc), category="validation")
    except Exception as exc:
        logger.exception("Worker %s failed %s job %s", worker_id, manager.job_type, job_id)
        manager.set_failed(job_id, f"{manager.job_type.capitalize()} failed: {exc}", category="internal")


def run_worker(poll_interval=1.0, stop_event=None):
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")
    init_db()
    worker_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    lock_path = pipeline_state.artifact_dir / ".worker.lock"
    with _single_worker_lock(lock_path):
        training_jobs.recover_interrupted()
        preprocess_jobs.recover_interrupted()
        logger.info("Worker %s started; interrupted jobs fail safely and queued jobs are retained", worker_id)
        managers = (training_jobs, preprocess_jobs)
        while stop_event is None or not stop_event.is_set():
            did_work = False
            for manager in managers:
                job = manager.claim_next(worker_id)
                if job is not None:
                    did_work = True
                    _execute(manager, job, worker_id)
            if not did_work:
                if stop_event is None:
                    time.sleep(poll_interval)
                else:
                    stop_event.wait(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="Run the Solar AI persistent job worker")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_worker(args.poll_interval)


if __name__ == "__main__":
    main()
