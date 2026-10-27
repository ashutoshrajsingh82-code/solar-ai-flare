# Deployment and acceptance workflow

## Current deployment boundary

Docker Compose builds a CPU-capable FastAPI backend, a static React/Nginx frontend, a one-shot migration service, and the existing durable job worker. Compose stores SQLite data, the model registry/artifacts, and pipeline artifacts in separate named volumes. The backend is reachable through the frontend's same-origin `/api` proxy; only the frontend port is published. No Redis, PostgreSQL, cloud object store, external feed, GPU, credentials, or implicit model downloads are required.

The tracked model directory contains legacy demo artifacts, not a compatible, observationally validated operational model. `/api/health/ml` reports whether a compatible active artifact set is actually usable. A 503 there is an explicit inference blocker; Docker health and database readiness do not imply model readiness. Synthetic demo data is visibly marked and must not be interpreted as real solar observations. The repository remains an integration-tested prototype until a supported artifact is trained/validated and deployment/security requirements are met.

## Prerequisites and configuration

- Windows 10/11 with Docker Desktop and WSL2, or Docker Engine with the Compose plugin.
- Python 3.11 for the pinned backend image (the source test suite can run in a compatible local Python environment).
- Node.js 20 for local frontend development; Docker uses Node 20 and Nginx.
- Copy `.env.example` to `.env` at the repository root. Supported Compose variables are `DATA_MODE` (`demo` by default), `FRONTEND_PORT` (`5173`), and `CORS_ORIGINS` (empty for same-origin browser access). `CORS_ORIGINS` is only needed for direct cross-origin API access; wildcard origins are rejected.
- For direct backend execution, copy `backend/.env.example` to `backend/.env`. `APP_ENV`, `DATABASE_URL`, `MODEL_DIR`, `DATA_MODE`, `SAMPLE_DATA_DIR`, `PIPELINE_ARTIFACT_DIR`, `CORS_ORIGINS`, and `AUTO_MIGRATE` reflect the settings read by the backend. Development defaults use local SQLite and demo mode. Production requires an explicit persistent `DATABASE_URL`; set `AUTO_MIGRATE=false` and run migrations as a separate deployment step. Do not commit `.env` files or place secrets in the image.

The current deployment database is SQLite. There is no configured Redis, object store, or PostgreSQL service. The model directory and persistent job artifacts are local files, so production requires a single writable shared volume and one worker process. The worker serializes work with an artifact-volume lock. Scale-out, authentication/authorization, backup/restore, and database failover are not provided by this compose file.

## Docker Compose acceptance (PowerShell)

From the repository root:

```powershell
Copy-Item .env.example .env
docker compose config --quiet
docker compose build
docker compose run --rm migrate
docker compose up -d
./scripts/acceptance.ps1
```

If `.env` already exists, keep it and review the three supported values instead of overwriting it. The script builds and starts the existing services, waits for backend/frontend container health, calls backend liveness and database readiness through the frontend proxy, loads the explicitly synthetic demo dataset, and checks model readiness. If a compatible model and valid fixture are available, complete the end-to-end inference and persisted retrieval check with:

```powershell
./scripts/acceptance.ps1 -ModelId '<registered-model-id>' -PredictionRowsPath '<path-to-valid-rows.json>'
```

The JSON file must contain the API's chronological raw observation row array (`timestamp`, `solexs_flux`, `helios_flux`); use only observations the selected model supports. The script verifies the returned model version and schema, then retrieves the persisted prediction by ID. It never treats an absent model as a successful inference.

Health checks can also be run individually:

```powershell
Invoke-RestMethod http://localhost:5173/api/health/live
Invoke-RestMethod http://localhost:5173/api/health/ready
Invoke-RestMethod http://localhost:5173/api/health/ml
docker compose ps
docker compose logs --tail 100 migrate backend worker frontend
```

Liveness is process-only and does not access the database or model files. Readiness performs a bounded database query and reports model readiness separately. ML readiness checks active model compatibility and caches the result for 30 seconds to avoid repeatedly validating/loading large artifacts.

To stop while retaining all data and models:

```powershell
docker compose down
```

This intentionally does not pass `--volumes`. Persistent data lives in `solarflare-data`, `solarflare-models`, and `solarflare-artifacts`; removing named volumes is a separate, destructive maintenance action and is not part of acceptance.

## Database migrations

Schema migrations are ordered in `app.database.db.init_db()` and recorded in the `schema_migrations` table. Migration 1 creates missing current tables; migration 2 adds job metadata/indexes additively, preserving existing rows. Applying the migration command repeatedly is supported and does not drop the database.

Compose runs the one-shot migration service before the API and worker. For a local production-mode backend, run from `backend/` with the intended persistent connection configured:

```powershell
$env:APP_ENV = 'production'
$env:AUTO_MIGRATE = 'false'
$env:DATABASE_URL = 'sqlite:///./solarflare.db'
python -m app.migrate
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In development, `AUTO_MIGRATE` defaults to true so a new local SQLite database remains convenient. Production startup does not apply schema changes; missing database connectivity fails the readiness check and migration failures stop the migration step with an error.

## Frontend and backend local development

Install frontend dependencies and build the static bundle:

```powershell
cd frontend
npm ci
npm run build
```

Run backend and frontend in separate PowerShell windows:

```powershell
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```powershell
cd frontend
npm ci
npm run dev
```

URLs are `http://localhost:5173` for the frontend and `http://127.0.0.1:8000/docs` for local API docs. Vite proxies `/api` to port 8000; the Docker Nginx build proxies `/api` to the Compose service name `backend:8000`. No localhost API URL is compiled into production frontend assets.

## ML and API verification

Run focused deterministic CPU inference tests from `backend/`:

```powershell
python -m pytest tests/test_production_inference.py tests/test_job_schema_migration.py -q
```

The inference fixture fits a small CPU logistic-regression artifact, writes its preprocessing and feature manifest, registers checksummed artifacts, calls the same prediction service and API functions, and verifies thresholds, response validation, persistence, retrieval, invalid input behavior, and unavailable/corrupt-model behavior. The fixture is synthetic and verifies software behavior, not forecast skill.

For the complete backend suite:

```powershell
python -m pytest -q
```

On this Windows/Python 3.13 host, the non-HTTP suite passed 145 tests with `python -m pytest -q --ignore=tests/test_api.py`. The existing synchronous `TestClient` suite stalls at its first health request even after health was made dependency-light; it was interrupted, so the complete suite is not reported as passing. The Docker daemon was unavailable for container verification (see the acceptance run report).

There is no frontend test or lint script in `package.json`; the production bundle is checked with `npm run build`. There is no automatic model training on app startup. Demo data can be loaded explicitly with `POST /api/data/load-demo`; it is synthetic and does not make a legacy artifact compatible.

## Troubleshooting and recovery

- **Migration container fails:** inspect `docker compose logs migrate`; confirm the volume is writable and the configured SQLite file is not locked by another deployment. Do not delete the volume to repair a migration.
- **Backend is unhealthy:** inspect `docker compose logs backend`; confirm migration completed and `/api/health/ready` reports `database_ready: true`.
- **ML readiness returns 503:** inspect `/api/models` and `/api/models/{model_id}/readiness`. Retrain/register a compatible model with its checkpoint, train-fitted preprocessor, evaluation record, ordered feature manifest, and catalog metadata. Missing or incompatible weights are not replaced with fabricated predictions.
- **Frontend cannot reach the API:** use the Compose frontend URL and keep `/api` same-origin; inspect `docker compose logs frontend backend`. For direct browser access, set exact origins in `CORS_ORIGINS` and recreate services.
- **Database unavailable:** readiness returns 503 after bounded connection/pool timeouts; liveness remains available. Restore the configured persistent database and retry. No destructive recovery runs automatically.
- **Job remains queued:** verify the `worker` service is healthy/running and uses the same database, model volume, and artifact volume. Restarting the worker retains queued jobs; interrupted running jobs are recorded failed by the existing recovery behavior.
- **Frontend/API input errors:** malformed prediction rows return 422, an unknown model ID returns 404, and unavailable/incompatible model artifacts return 503. These failures are surfaced rather than converted to demo success.

## Security and known limitations

Containers run as non-root users, backend/internal ports are not published, CORS wildcard access is rejected, and error logs omit raw inference payloads. The app currently has no authentication/authorization layer; do not expose this prototype to an untrusted network. The SQLite registry is single-instance oriented. No independent operational model, real-observation acceptance fixture, calibrated probability, live data feed, or authentication configuration is bundled. Thus a passing software acceptance test is not evidence for operational forecasting or production readiness.
