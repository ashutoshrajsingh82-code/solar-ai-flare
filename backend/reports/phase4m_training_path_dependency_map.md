# Phase 4M training-path dependency map

`app/api/data.py` accepts uploads and calls `app/services/data_service.py`; real FITS ingestion is implemented by `app/services/data_sources/solexs.py`, while GOES reference labels are attached through `goes_labels.py`.  Provenance is retained in `pipeline_state.dataset_meta`.

`app/services/preprocess_service.py` is the orchestration boundary.  It rejects missing dual physical-flux columns and unknown labels, cleans/alines observations, calls `ml/features/windowing.chronological_split`, fits `FluxTransformer` on train only, calls `ml/features/engineering.py`, then uses split-scoped label creation.  Its split audit uses past-only context; no test/validation future rows are inputs to a training origin.

`ml/features/windowing.py` constructs sequence inputs and validates unique ordered timestamps.  `ml/features/labels.py` defines forecast targets; `generate_split_scoped_labels` in preprocessing purges unsupported horizon tails rather than calling them quiet.

`app/services/model_service.py` is the training entry point.  It calls `scientific_readiness.generate_scientific_readiness_report`, checks pipeline state and Phase 2C evidence, selects classical/deep training, and delegates validation to `ml/training/validation.py`.  Threshold selection and final metrics are downstream evaluation functions; checkpoint registration flows through `ml/models/registry.py`, which rejects synthetic/non-production records and activation validates artifact contracts.

Phase 4M adds `ml/training/dataset_eligibility.py` before this path.  It is a non-mutating evidence/split validator and `scripts/training_readiness_dry_run.py` invokes no trainer.  The smallest engineering change is therefore an auditable eligibility assessment, not a duplicate ingestion, feature, or model implementation.
