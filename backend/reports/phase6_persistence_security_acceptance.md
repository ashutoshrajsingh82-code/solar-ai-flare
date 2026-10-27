# Phase 6 — Persistence, Security, and Acceptance Evidence

## Audit and implementation

- Raw uploads are written under the configured upload root using generated names, staged files, a byte limit, and atomic replacement. Database dataset records retain the original name as metadata, a relative storage key, SHA-256, file format, schema metadata, and processing status. Raw files are separated from pipeline artifacts. CSV parsing rejects malformed schema, invalid timestamps, duplicate headers, and non-numeric/infinite flux values; SoLEXS ingestion checks FITS dimensions before reading.
- Cleaned/split pipeline frames, the fitted transformer, provenance, and stage reports are stored as checksummed files referenced by an atomically replaced manifest under `PIPELINE_ARTIFACT_DIR`. The manifest is the durable authority and is restored when the service starts. This pipeline stage history is file-backed, not stored in relational stage rows.
- Background job status/progress is database-backed. Startup recovery identifies queued/running jobs left by a prior process; the worker owns queued work. Model artifacts are written under the controlled model directory and registered with checksums; load paths and the preprocessing contract are verified. Existing model versions are retained.
- Historical replay records are database-backed. Process restart marks an active replay interrupted. Retry is explicitly restart-from-beginning and idempotent: replay identity includes data and preprocessing fingerprints; prediction rows and alert events have uniqueness constraints; the replay cursor advances only after its prediction transaction commits. This is historical held-out replay, not a live satellite feed.
- SQLite is shared by API and one worker through the persistent `solarflare-data` volume; model and pipeline artifacts use their own persistent volumes. Compose keeps the API private behind the frontend proxy. Production CORS accepts exact configured HTTP(S) origins only; an empty allowlist disables cross-origin browser requests. Credentials are disabled and allowed methods/headers are explicit.
- Upload size, rows, and columns are configurable and bounded. User filenames do not form storage paths. Failed ingestion rolls back its dataset record and removes staged/final files. The project does not currently implement retention/garbage collection for successful raw uploads, cleaned pipeline artifacts, or model artifacts; operators must manage the persistent volumes.
- `ExperimentRecord` exists as a legacy schema, but `/api/experiments/fusion` is currently a read-only readiness preflight. Experiment execution records, per-seed prediction persistence, and aggregated uncertainty results are **not implemented**. No synthetic experiment records are fabricated.

## Acceptance checklist

| Checklist | Status | Evidence / limitation |
|---|---|---|
| Real-format ingestion and invalid-file rejection | PASS | `tests/test_phase6_persistence_security.py` verifies bounded/scoped CSV storage, generated names, invalid/empty/oversized rejection, cleanup, and metadata persistence; `tests/test_real_data_sources.py` exercises valid FITS fixtures and fail-closed instrument readers. |
| Cleaning, cadence alignment, and missing-data handling | PASS | `tests/test_pipeline.py`, `tests/test_real_data_sources.py`, and `tests/test_feature_engineering_step9.py` cover ordering, gaps, cadence, invalid observations, and no interpolation of missing instrument data. |
| GOES-derived labels at known event timestamps | PASS | `tests/test_real_data_sources.py` covers UTC timestamps, event windows, coverage-aware positives/negatives/unknowns, and independent reference provenance. |
| Chronological splits and forecast-leakage prevention | PASS | `tests/test_chronological_audit.py`, `tests/test_phase3_integration.py`, and `tests/test_phase5_monitoring.py` cover chronological boundaries, train-only fitting/validation thresholds, split-scoped labels, and causal replay inputs. |
| Training job progress, failure, and checkpoint creation | PASS | `tests/test_job_persistence.py`, `tests/test_job_schema_migration.py`, and `tests/test_phase3_integration.py` exercise durable jobs, idempotency, failures, and artifact references. |
| Model and scaler reload after backend restart | PASS | `tests/test_production_inference.py::test_model_and_preprocessor_reload_in_a_new_process` creates artifacts in one process and verifies the same output in a separate Python process. |
| Metric calculations against known example predictions | PASS | `tests/test_metrics.py` checks hand-computed scores, FAR vs POFD, undefined denominators, invalid probabilities, reliability bins, and threshold-independent Brier score. |
| Single-instrument versus fusion input isolation | BLOCKED | Manifests and model feature contracts are enforced; baseline tests confirm the hard/soft ratio requires fusion and a one-channel model selects its own columns. No end-to-end paired-cohort test proves both instrument-specific inference isolation and common-cohort fairness. |
| Live inference with insufficient history | PASS | `tests/test_production_inference.py` checks explicit insufficient-history failure, model verification, causal windows, and health separation. Actual real-time satellite feed is unavailable and is not claimed. |
| Alert deduplication and threshold transitions | PASS | `tests/test_phase5_monitoring.py` covers thresholds, hysteresis, cooldown, expiry/re-arm, idempotency, DB session recreation, invalid score handling, and concurrent duplicate processing. |
| Frontend API integration and Docker health checks | BLOCKED | Frontend production build and `docker compose config --quiet` pass. No browser E2E test script exists. Docker image build/start could not run because Docker Desktop’s Linux engine pipe was unavailable. Readiness and liveness endpoints are distinct; backend and frontend images define health checks. |

## Verification executed

- Focused Phase 6 persistence/security/restart tests: **29 passed** before adding the concurrency test.
- Full backend suite before adding the concurrency test: **174 passed**.
- Concurrent duplicate-alert test after addition: **1 passed**.
- Frontend: `npm run build` — passed; Vite reports the existing generated JavaScript chunk exceeds 500 kB.
- Compose: `docker compose config --quiet` — passed.
- Compose images: `docker compose build` — blocked; Docker Desktop reported that `//./pipe/dockerDesktopLinuxEngine` does not exist.

Final backend suite after the concurrency test: `python -m pytest -q --basetemp=.pytest-phase6-tmp` — **175 passed, 620 warnings**.

## Remaining gaps

- Experiment execution and durable experiment result/seed prediction records are not implemented.
- End-to-end, paired-cohort single-instrument/fusion comparison is not proven by the current tests.
- Browser-based frontend E2E validation and Docker container startup/recreation verification require available browser test tooling and a running Docker engine.
- Successful upload/artifact retention and cleanup policy needs an operational retention job before long-running production use.
- SQLite shared-volume deployment is intended for the current single-worker local deployment, not horizontally scaled concurrent workers.
