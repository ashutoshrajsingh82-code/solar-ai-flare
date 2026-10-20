# Phase 2 implementation report

## Persistence design

- Job state is stored in the existing SQLAlchemy database in `background_jobs`. Each training and preprocessing manager filters by its own job type. Updates use short-lived sessions, commit before updating memory, and serialize NumPy/Pandas values into JSON-safe structures. Database failures are logged and raised rather than reported as successful updates.
- Startup recovery marks `pending` and `running` jobs failed with stage `interrupted` and a restart explanation. Recovery is idempotent. It does not resume training or preprocessing.
- Pipeline stages use versioned artifact filenames and an atomically replaced manifest. Parquet is preferred; CSV is the local fallback when no Parquet engine is installed. The fitted `FluxTransformer`, train/validation/test splits, metadata, feature columns, cadence, and stage reports are persisted. Artifact paths are manifest basenames constrained to the configured directory and data is SHA-256 checked before restore. The joblib transformer is only loaded after its recorded digest matches.
- A matching dataset reloaded during demo startup retains its restored derived stages; a changed dataset clears stale stages before persisting the new raw data.

## API and deployment behavior

The existing job creation and polling routes retain their route names and response keys. Completed and failed job records remain queryable from a new manager instance. SQLite `create_all()` adds the new table without dropping existing tables. Thread workers, the live simulator, and the in-process pipeline object remain single-process; database persistence does not resume interrupted computation or coordinate multiple workers.

Local artifact files survive process restart if the same disk remains attached. Deployments using ephemeral storage must attach a persistent volume. This implementation does not claim object-storage or cross-instance persistence.

## Verification

The requested commands were attempted in the current environment:

- `python -m compileall -q app ml tests` — syntax compilation completed successfully.
- `python -m pytest -q` — blocked because the active Python 3.13 interpreter does not have `pytest` installed. Import checks also show `pandas` is missing, so the backend test suite cannot run in this environment. New tests are present but unverified.
- `npx vite build --configLoader runner` — not required for this backend-only phase; the prior Phase 1 build passed. No Phase 2 frontend changes were needed.

The repository prompt reported a 48-test baseline, but that baseline could not be reproduced in this interpreter. Install dependencies from `backend/requirements.txt` before claiming test results.

## Manual Windows CMD check

Start the backend from `backend` so the default SQLite database path is stable:

```cmd
cd /d C:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main\backend
set PIPELINE_ARTIFACT_DIR=./artifacts/pipeline
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second CMD window, create a preprocessing job and copy its returned `job_id`:

```cmd
curl -X POST http://127.0.0.1:8000/api/preprocess -H "Content-Type: application/json" -d "{\"target_cadence\":\"1min\",\"compute_stl\":false}"
curl http://127.0.0.1:8000/api/preprocess/JOB_ID/status
```

Poll until `status` is `done` or `failed`. Stop and restart the backend using the first command, then poll the same ID again. A completed job should remain `done`; an in-flight job is marked `failed` with stage `interrupted`. The pipeline status route can confirm restored stage metadata:

```cmd
curl http://127.0.0.1:8000/api/pipeline/status
```

## Remaining limitations

Install the pinned backend dependencies and run the test suite before deployment. Local filesystem artifacts need a persistent volume across host replacement. The current thread workers and locks are not distributed, and interrupted work is marked failed rather than resumed. CSV fallback does not preserve every possible pandas dtype as exactly as Parquet; use a Parquet engine for richer datasets.
