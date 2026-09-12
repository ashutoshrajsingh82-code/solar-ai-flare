# BUILD STATUS — Aditya-L1 Solar Flare Intelligence System

This zip is a **work-in-progress checkpoint**, not the final deliverable.
It contains the fully working data + ML backend core. The FastAPI wiring,
frontend, tests, README, and Docker setup are either partially built or
not yet started. Continue the build in a follow-up message and I will
keep extending this same structure.

## ✅ Built AND execution-verified in this sandbox (numpy/pandas/sklearn available, no network)

- `ml/data/synthetic_generator.py` — synthetic SoLEXS + HEL1OS generator
  (Neupert-style hard-leads-soft physics, B/C/M/X classes, noise/gaps/spikes).
  Demo CSV pre-generated at `backend/sample_data/demo_aditya_l1_solexs_helios.csv`
  (10 days, 1-minute cadence, 14 flare events) so the pipeline has data to run on immediately.
- `ml/preprocessing/cleaning.py` — duplicate/spike/gap handling
- `ml/preprocessing/alignment.py` — resampling to configurable cadence
- `ml/preprocessing/transform.py` — log10 + scaler, fit-on-train-only
- `ml/features/engineering.py` — rolling stats + hard-to-soft ratio + optional STL
- `ml/features/labels.py` — nowcast states + 5 forecast horizon labels
- `ml/features/windowing.py` — chronological split + leak-free sliding windows
- `ml/evaluation/metrics.py` — TSS, HSS, POD, FAR, ROC/PR-AUC, Brier, reliability
- `ml/models/baseline.py` — Logistic Regression / Random Forest / SVM multi-task wrapper

Ran a full offline test: generate → clean → align → transform (train-only fit) →
features (97 columns incl. hard/soft ratio) → labels → chronological split →
windows (8401 windows, shape (30, 2)) → RandomForest multi-task train/predict →
TSS/HSS computed correctly. All green.

## 🟡 Built but NOT execution-tested (torch/fastapi unavailable, no network in this sandbox)

- `ml/models/deep_models.py` — CNN1D, LSTM, CNN-LSTM hybrid, optional CNN-Transformer (PyTorch)
- `ml/training/train.py` — multi-task training loop, class-weighted losses
- `ml/models/registry.py`, `ml/inference/predictor.py`, `ml/inference/simulator.py`
- Full FastAPI app: `app/main.py` + routers for health/data/preprocess/models/
  predict/alerts/simulation/experiments/dashboard/system, SQLAlchemy models,
  and the services layer connecting them all together.

These were written carefully and cross-checked for import/route consistency,
but I have not been able to `pip install fastapi torch statsmodels` or actually
boot `uvicorn` here to catch runtime bugs — I'll do that verification pass
(and fix anything that breaks) once we continue.

## ⬜ Not yet started

- React/Vite/Tailwind frontend (all pages, charts, components)
- `README.md`, `docker-compose.yml`, `.gitignore`, `tests/`
- `.env.example` for frontend

## How to pick this back up

Just say "continue" and I'll keep building directly on this structure —
frontend next, then a full install + boot verification pass (which will work
once I have network/npm/pip access in a fresh session), then README/tests/Docker.
# Project status

The end-to-end application is wired for demo use:

- FastAPI exposes data ingestion, preprocessing, model, prediction,
  simulation, alert, dashboard, and system-status APIs.
- React provides the dashboard and proxies same-origin API calls in both Vite
  development and Docker/nginx deployments.
- The demo bootstrap loads synthetic telemetry and seeds example alerts.

Before a production deployment, replace demo telemetry with validated mission
data and add authentication, persistence for pipeline artifacts, background
training jobs, observability, and domain validation.
