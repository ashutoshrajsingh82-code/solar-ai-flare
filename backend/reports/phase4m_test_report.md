# Phase 4M test report

Interpreter: `C:\Users\ASHUTOSH\anaconda3\python.exe` (Python 3.13.9).

Focused command:

`python -m pytest tests\\test_dataset_eligibility.py tests\\test_phase2c_eligibility.py tests\\test_chronological_audit.py tests\\test_feature_engineering_step9.py tests\\test_fits_semantics_audit.py tests\\test_phase3c_semantics_audit.py tests\\test_phase4h_data_readiness.py -q --basetemp=%TEMP%\\solar-ai-flare-phase4m-focused-20261025`

Exit 0: 47 passed, 25 warnings.  Warnings were existing Pydantic/SQLAlchemy deprecations, Pandas fragmentation performance warnings, and a denied pytest-cache path.

Relevant non-ML command:

`python -m pytest tests\\test_dataset_eligibility.py tests\\test_blocker_register.py tests\\test_fits_semantics_audit.py tests\\test_phase3b_authoritative_audit.py tests\\test_phase3c_semantics_audit.py tests\\test_phase4h_data_readiness.py tests\\test_real_data_sources.py tests\\test_chronological_audit.py tests\\test_feature_engineering_step9.py tests\\test_phase2c_eligibility.py -q --basetemp=%TEMP%\\solar-ai-flare-phase4m-nonml-20261025`

Exit 0: 76 passed, 27 warnings of the same existing categories.

No-fit dry run:

`python scripts\\training_readiness_dry_run.py`

Exit 2 as designed for the current demo/ineligible pipeline state.  Its output records `trainer_invoked=false`, `parameters_fitted=false`, and `checkpoint_created=false`.

`python -m py_compile ml\\training\\dataset_eligibility.py scripts\\training_readiness_dry_run.py app\\services\\model_service.py` exited 0.  The existing import-dependent real-trainer refusal test could not be collected in this environment because `model_service.py` imports missing `torch`; it is deferred to the dedicated environment.

Full suite was not run.  The Phase 4L collection assessment remains applicable: the selected environment lacks FastAPI, Starlette, `torch`, and `netCDF4`; it collected 144 tests with 10 dependency-related collection errors.  A compatible Python 3.11 project environment is required before a valid full-suite run.
