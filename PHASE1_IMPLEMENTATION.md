# Phase 1 implementation report

## Implemented

- Added conservative lifecycle metadata and atomic registry writes. Activation checks lifecycle state, checkpoint readability, and required transformer readability.
- Removed the normal path that seeded random deep weights and hard-coded model metrics as trained models. Empty registries now remain empty.
- Training stores a unique fitted transformer alongside each run and marks demo-mode training `demo_only`.
- Inference requires a selected or active model, loads its recorded transformer, validates required columns, timestamps, and finite numeric fluxes, and rejects missing artifacts. Deep inference no longer initializes random weights when checkpoints are missing.
- Simulator predictions apply the selected model's saved transformer.
- Removed the evaluation page's guessed metrics, threshold slider calculations, default Brier score, and sample calibration curve. Missing calibration data now has an explicit empty state.
- Removed wildcard CORS and added comma-separated exact `CORS_ORIGINS` configuration.
- Added registry lifecycle tests and updated local setup and scientific limitation documentation.

## Verification status

`python -m compileall -q app ml tests` passed. `npx vite build --configLoader runner` passed (Vite warned that the minified JS chunk exceeds 500 kB). `python -m pytest tests -q` could not start because `pytest` is not installed (`No module named pytest`), so the new and existing backend tests have not executed.

## Known limitations

- The historical baseline metric file and legacy registry records are preserved. The baseline entries are reference-only; legacy records are conservatively treated as untrained when loaded, but existing stored metrics are not deleted.
- The pipeline transformer currently lives in process memory until training. Restarting the server between preprocessing and training loses pipeline state.
- Status `trained` means a run completed on configured non-demo data; it does not mean validated on genuine solar observations. Dataset provenance must be strengthened before making that claim.
- Deep model architecture parameters (for example dropout) are not all persisted as complete reconstruction metadata yet.
- Calibration bins and stored raw evaluation predictions are not currently generated, so reliability curves and threshold recalculation remain unavailable.
- Demo simulator input remains a replay, not live telemetry. Operational alert semantics require further review.
