# Phase 2D: persistent jobs and artifact storage

## Audit and scope

This phase extends the existing SQLAlchemy database, `JobManager`, pipeline manifest, model registry, and existing `/api/preprocess` and `/api/models/train` operations. It does not claim Phase 2C scientific verification. The current data contract still lacks verified paired mission observations and does not feed independent GOES labels into the existing model training target; see [Phase 2C readiness](PHASE2C_IMPLEMENTATION.md).

The database schema is initialized with SQLAlchemy `create_all()` and additive SQLite/PostgreSQL migrations in `app.database.db.init_db()`; there is no Alembic migration framework in this repository. Existing job rows are retained while job metadata columns are added. An artifact catalog stores metadata in the same configured database. Pipeline binaries remain in the configured local artifact directory; trained models remain under `MODEL_DIR`.

## Execution and job lifecycle

The API only validates and durably records existing preprocessing/training requests. A separate single worker process polls the `background_jobs` table and atomically claims a queued record. Worker commands:

```powershell
cd C:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main\backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second PowerShell window using the same environment variables and persistent volumes:

```powershell
cd C:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main\backend
python -m app.worker --poll-interval 1
```

Set `DATABASE_URL`, `MODEL_DIR`, and `PIPELINE_ARTIFACT_DIR` consistently for API and worker. Both processes must use the same database and durable local artifact volume. The worker holds an advisory lock in the pipeline artifact directory to prevent a second worker from running against that volume. This is a single-host/single-worker design; it is not a distributed queue and does not support multiple hosts with independent local volumes. SQLite and PostgreSQL additive schema updates are implemented; other database engines need a reviewed migration before use.

Internally, existing status strings remain API compatible: `pending`, `running`, `done`, and `failed`. The job response also supplies the normalized lifecycle state `queued`, `running`, `succeeded`, or `failed`. Transitions are validated. Queued jobs survive API/worker restarts and are picked up when a worker starts. Running jobs found by a newly started worker are marked failed with category `interrupted`; work is not automatically resumed because training may have passed a non-idempotent boundary. Retries are disabled (maximum retry count 0) because current operations are local upload/preprocess/train work without a classified transient source operation. Validation errors are categorized separately from internal worker errors. API routes accept an optional `Idempotency-Key`; reusing a key with the same operation configuration returns the existing job, while a different configuration returns 409.

Each queued operation records a fingerprint of its persisted pipeline snapshot. The worker reloads the current snapshot before execution and fails the job as a validation error if the dataset or derived pipeline changed after submission. This prevents a queued request from silently running against a different upload or train/test split.

Inspection routes are `GET /api/jobs?job_type=training|preprocessing&status=...&offset=0&limit=50` and `GET /api/jobs/{job_id}`. Existing job status routes remain supported. No cancellation API is provided.

## Artifacts, activation, and failure handling

Pipeline stages continue to use versioned, atomically replaced manifest references and SHA-256 validation in `PIPELINE_ARTIFACT_DIR`. After preprocessing, manifest-referenced files are integrity-checked and cataloged against their producing job before success is committed. Model, fitted preprocessor, and evaluation JSON files are written via temporary files and atomic replacement. Their artifact metadata, hashes, sizes, compatibility contract, and producing job ID are stored in the existing SQL database. Job success includes the output artifact references.

Models are registered inactive. Activation remains an explicit `/api/models/select` request. It validates checkpoint/transformer compatibility, target/horizon/feature schema metadata, and catalog artifact hashes. Inference repeats catalog integrity checks before loading. Failed training cannot auto-activate a model; unique model IDs and atomic file writes leave previously active models intact. Rollback consists of explicitly selecting a prior valid model still present in the registry/catalog. There is no automatic promotion or fallback to an untrained model.

The model registry itself is still the existing atomic JSON index rather than a database reference. The artifact catalog and registry therefore cannot be committed in one cross-store transaction; a crash can leave unreferenced catalog entries/files, but cannot mark a partially written checkpoint active. Local filesystem storage needs a persistent volume. Job records contain safe request parameters, not credentials or raw datasets. Existing application authentication behavior is unchanged.

## Verification status and limitations

The supported job types are the existing preprocessing and model-training operations only. There is no dedicated data-download job, automatic evaluation-only job, cancellation, external transient-retry mechanism, multi-worker scheduling, object storage, or automatic resume. Evaluation remains the existing training pipeline's chronological test evaluation. Phase 2C's real-data prerequisites remain unmet, so no real observation training run or operational model claim is made.

`C:\Users\ASHUTOSH\anaconda3\python.exe -m pytest -q -p no:cacheprovider --basetemp=.pytest-phase2d-verify --ignore=tests/test_api.py` completed with 99 passed. The full suite was also attempted; verbose execution collected all tests and stalled at `tests/test_api.py::test_health_endpoint` during TestClient startup/request in this environment, so the full suite has no passing result. Focused job, artifact, migration, and registry tests also passed. `compileall`, `git diff --check`, and `python -m app.worker --help` passed. Do not use demo artifacts as evidence of real solar forecasting performance.
