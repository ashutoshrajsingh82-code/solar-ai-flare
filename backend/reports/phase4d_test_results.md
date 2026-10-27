# Phase 4D focused regression results

Command (from `backend`):

```powershell
python -m pytest tests\test_phase3a_data_audit.py tests\test_phase3b_authoritative_audit.py tests\test_phase3c_semantics_audit.py tests\test_real_data_sources.py tests\test_chronological_audit.py tests\test_feature_engineering_step9.py tests\test_scientific_readiness.py tests\test_phase2c_eligibility.py -q --basetemp=.pytest-phase4d-readiness-20261024
```

Result: **63 passed, 25 warnings, 0 failed** (13.63 s). Warnings: upstream Pydantic/SQLAlchemy deprecations and pandas DataFrame fragmentation performance warnings. Existing tests cover label unknowns, GOES candidate matching and ambiguity, cross-catalog class discrepancies, horizon-tail purging, and fail-closed product readers. No new tests were added for instrument rules because no such rule was verified by authoritative documentation; asserting them would encode unsupported semantics.
