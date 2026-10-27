# Phase 4M training readiness

Pipeline implementation readiness is **PASS**: ingestion, chronological splitting, train-only transforms, feature generation, label-tail purging, training-array validation, and registry lifecycle checks exist.  Phase 4M adds a standalone fail-closed dataset eligibility assessment and a no-fit dry run.

Dependency/environment readiness is **BLOCKED**: the selected Python 3.13.9 global Anaconda environment lacks `torch`, `netCDF4`, FastAPI, and Starlette and is not aligned to the pinned requirements.  A dedicated Python 3.11 environment remains required.

Scientific semantics readiness is **BLOCKED**: Phase 3C remains `BLOCKED_SCIENTIFIC_SEMANTICS_UNRESOLVED` (7 verified, 6 partial, 1 unverified).  Dataset/label readiness and chronological split feasibility are **BLOCKED** because no eligible real observation/label population or final verified split boundaries exist.  Production activation eligibility is **BLOCKED**: no eligible real-data checkpoint exists and the synthetic smoke checkpoint remains `SYNTHETIC_NOT_FOR_PRODUCTION`.

Run the non-training preflight as `python scripts/training_readiness_dry_run.py` from `backend`.  It returns exit 0 only for an eligible dataset and never calls a trainer, fits parameters, evaluates, or writes a checkpoint.
