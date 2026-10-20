# Phase 2B implementation report

## Repository audit

The project root is this directory and the backend root is `backend/`. The current branch is `main`. Phase 2A's real-data contract is described in `docs/real_data_ingestion.md`: SoLEXS Level-1 light-curve counts are retained in native counts/s, HEL1OS parsing remains gated on its registered manual and a representative product, and GOES supplies independent M/X onset references. The active forecast pipeline still expects normalized dual-instrument `solexs_flux` and `helios_flux` plus a valid nowcast `state`; it correctly refuses to infer those values or labels from native SoLEXS counts.

The existing architecture was retained: the data-source adapters and provenance path, pipeline persistence, chronological 70/15/15 split, train-only fitted transformer, model registry, model-specific preprocessing artifacts, metric functions, and current FastAPI routes. No database schema or public response contract change was needed.

## Phase 2B changes

- `backend/ml/features/engineering.py` validates timestamp order, duplicate timestamps, finite numerical ranges, and positive window configuration before feature generation. Numeric `forecast_*`, `target_*`, `label_*`, and `nowcast_*` columns are excluded from model feature discovery.
- Optional STL features now use fixed trailing windows, with each row estimated from data available up through that row. Missing values in a window remain unknown; the implementation no longer interpolates across gaps or fits a two-sided decomposition over future observations.
- `backend/ml/features/windowing.py` requires the timestamp, state, and configured forecast targets. It drops sequences whose endpoint has an unknown target and rejects invalid class indices or non-binary forecast values; missing labels are never fabricated as quiet/negative examples.
- `backend/ml/training/validation.py` validates deep and classical training inputs for compatible shapes, finite features, valid targets, and sufficient training class diversity. The deep trainer uses a fixed default random seed (`42`) for model initialization and sample ordering.
- `backend/ml/inference/input_validation.py` centralizes raw inference validation, UTC timestamp normalization, numeric finite flux checks, duplicate-time rejection, and deterministic chronological ordering. `backend/app/services/predict_service.py` uses this validator and fills remaining historical feature gaps from past values only; it no longer backfills from later rows.
- `backend/tests/test_phase2b_validation.py` covers causal features, feature/target separation, unknown-label handling, training-array validation, and duplicate inference timestamps.

## Data flow and scientific rules

Local SoLEXS FITS observations pass through the existing schema validator and keep their native COUNTS/s identity. The Phase 2B changes do not reinterpret counts as calibrated flux. HEL1OS remains disabled pending primary product documentation and sample files. GOES CSV events keep their separate source provenance and use `start_time` as onset; M/X events are positive within `(observation_time, observation_time + horizon]`. A window becomes negative only when the full horizon has declared event coverage; otherwise it remains unknown. UTC is used at ingestion, feature, label, and inference boundaries. Mixed timezone offsets and subsecond precision are preserved while parsing.

Rolling differences, ratios, and window statistics use current and prior observations. Flux changes use one-row differences and percent changes; rolling windows default to 5, 10, 30, and 60 minutes (with available windows requiring at least `max(2, window_steps // 3)` observations). Hard-to-soft ratio summaries use 5, 10, and 30 minutes. Optional STL uses the last two configured seasonal periods as a trailing fit window and emits no value until that window is complete and finite. Training continues to split chronologically, fit the transformer on training only, and generate/purge forecast labels within each split. Training now rejects unknown or malformed labels and training partitions without both classes for each forecast horizon. Test evaluation remains held out; metrics include precision, recall, F1, TSS, HSS, ROC-AUC and PR-AUC when defined, Brier score, and confusion counts. Demo metrics are not real-observation validation.

Inference loads the selected model and its registered transformer after registry artifact checks. It returns forecast probabilities and the existing risk categories; forecast evaluation uses the existing 0.5 metric threshold. These are model estimates, not confirmed events. Model/version metadata and artifact hashes continue to use the existing registry and pipeline persistence formats.

## Configuration and Windows commands

Configuration continues through the existing environment/settings system: `DATA_MODE`, `DATABASE_URL`, `MODEL_DIR`, `SAMPLE_DATA_DIR`, `PIPELINE_ARTIFACT_DIR`, and `CORS_ORIGINS`. Dependency pins are in `backend/requirements.txt`. From PowerShell or CMD, run the backend checks from `backend`:

```powershell
python -m pytest -q
python -m compileall -q app ml tests
```

The API can be started with the repository's existing `uvicorn app.main:app` entry point from the backend directory. No new endpoint or environment variable was introduced.

## Verification and limitations

On 2026-10-19, verification produced:

- `python -m pytest -q` — could not start; `C:\Python313\python.exe` has no pytest installed.
- `C:\Users\ASHUTOSH\anaconda3\python.exe -m pytest -q` — collection blocked because FastAPI is missing for `tests/test_api.py` and `tests/test_job_persistence.py`.
- Focused tests (`tests/test_phase2b_validation.py tests/test_pipeline.py tests/test_real_data_sources.py`) — **32 passed**.
- Available suite with the two FastAPI-dependent files excluded — **72 passed**.
- `python -m compileall -q app ml tests` using Anaconda Python — passed.

PyTorch is also absent from the available environment, so deep-model training and actual model inference could not be executed here. The Phase 2B changes to those paths have unit coverage through dependency-light validators, but runtime integration remains unverified.

No real SoLEXS or HEL1OS product was available for validation. No observational training, calibration, or operational performance claim is made. HEL1OS product parsing, calibrated cross-instrument fusion, model training and prediction on mission data, API integration tests, deep-training runtime checks, distributed job execution, and production deployment verification remain deferred. Existing uncommitted Phase 2 work was preserved; no commit or push was made.
