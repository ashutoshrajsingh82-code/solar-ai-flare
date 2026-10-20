# Phase 2C audit and readiness

## Status

Phase 2C is **not scientifically complete**. No observational training run was made and no holdout performance is claimed. The only local CSVs are demo SoLEXS/HEL1OS rows and demo flare events. The documented SoLEXS reader preserves native counts/s, HEL1OS parsing is gated pending authenticated product documentation and a representative product, and GOES reference labels currently attach independently rather than replacing the pipeline's generated `state` and forecast targets. Thus the available artifacts do not constitute a verified training dataset for the existing model.

The real-observation ingestion contract and source limitations are documented in [docs/real_data_ingestion.md](docs/real_data_ingestion.md). Required prerequisites are real, provenance-bearing SoLEXS and HEL1OS observations with validated calibration/time semantics, a verified event reference covering the observation intervals, and an explicit scientifically valid target/feature contract. A chronological holdout and actual real-data run remain outstanding.

## Changes made for Phase 2C readiness

- Non-demo training now fails closed unless dataset metadata contains observation and label source identifiers and explicit Phase 2C evidence for provenance, UTC timestamps, target definition, prediction-time features, unknown-label exclusion, and chronological embargo. The gate does not certify the claims; it prevents an unverified upload from being called trained.
- Binary forecast metrics now report sample counts and positive prevalence. Undefined POD, FAR, TSS, HSS, precision, recall, ROC-AUC, or PR-AUC are returned as null with a reason. Brier score remains computed from probabilities. FAR is explicitly exposed as `FP / (TP + FP)`.
- Existing model artifact storage now also receives a JSON evaluation report (`<model_id>.evaluation.json`) with test metrics, split periods, feature schema, provenance metadata, threshold, and scientific limitations. Demo reports are visibly marked `demo_only` and are not observational evidence.

## Verification and blockers

Historical baseline supplied in the prompt: 71 passed, 0 failed. Current pre-change environment baseline did not collect: default `python` has no pytest; Anaconda pytest stopped at collection because FastAPI is missing for `tests/test_api.py` and `tests/test_job_persistence.py`.

The environment can run focused tests with `C:\Users\ASHUTOSH\anaconda3\python.exe`. The complete requested suite remains blocked until the backend dependencies, including FastAPI, are installed. No real-data training command can be run successfully because eligible real paired observations and verified target labels are not present; the documented project entry point is the existing `/api/models/train` workflow, now gated by the evidence contract.
