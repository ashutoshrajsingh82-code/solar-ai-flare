# Phase 4M-R: Test and Environment Report

**Audit Date**: 2026-10-25  
**Active Python Interpreter**: `C:\Python313\python.exe` (Python 3.13.4, 64-bit Windows)  
**Full Test Suite Result**: **235 PASSED, 0 FAILED** (Exit 0)  

---

## 1. Environment Assessment

### 1.1 Python Interpreters and Runtimes
An inspection of installed interpreters on the host system identified:
- `C:\Python313\python.exe` (Python 3.13.4) — Primary active interpreter on system PATH
- `C:\Users\ASHUTOSH\anaconda3\python.exe` (Python 3.13.9) — Global Conda base (previously noted in Phase 4L/4M as lacking PyTorch/FastAPI)
- `C:\Users\ASHUTOSH\AppData\Local\Programs\Python\Python311\python.exe` (Python 3.11.9)
- `C:\Users\ASHUTOSH\AppData\Local\Programs\Python\Python312\python.exe` (Python 3.12.7)

### 1.2 Dependency Manifest Alignment
The project manifest (`backend/requirements.txt`) specifies:
- `fastapi==0.115.6`, `pydantic==2.10.3`, `sqlalchemy==2.0.36`, `numpy==1.26.4`, `pandas==2.2.3`, `scikit-learn==1.5.2`, `torch==2.4.1`, `astropy==6.1.7`, `pytest==8.3.4`, `httpx==0.27.2`

The active environment (`C:\Python313\python.exe`) contains an installed, fully operational scientific ML stack:
- `fastapi 0.115.12`, `starlette 0.46.2`, `torch 2.14.0`, `scikit-learn 1.9.0`, `astropy 8.0.1`, `netCDF4 1.7.4`, `scipy 1.18.0`, `statsmodels 0.15.0`, `pandas 3.0.5`, `numpy 2.5.2`, `pytest 8.3.5`.
Because all dependencies are present and functional, no new package installations were required, preventing disk churn and avoiding risk to existing environments.

### 1.3 Disk Space Verification
Storage capacity was measured prior to test execution:
- **C: Drive**: 5.83 GB Free
- **D: Drive**: 109.64 GB Free
- **E: Drive (Real Data Root)**: 107.31 GB Free

---

## 2. Test Executions and Results

### 2.1 Focused Test Suite
- **Command**:
  ```powershell
  python -m pytest tests\test_blocker_register.py tests\test_fits_semantics_audit.py tests\test_phase3c_semantics_audit.py tests\test_phase4h_data_readiness.py tests\test_dataset_eligibility.py -v
  ```
- **Interpreter**: `C:\Python313\python.exe`
- **Working Directory**: `c:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main\backend`
- **Exit Code**: `0`
- **Summary**: **31 passed in 1.78s**
- **Warnings**: 0 warnings.
- **Coverage**: Verifies Phase 4K blocker register integrity, read-only FITS structural inspection, Phase 3C semantic audit fail-closed behavior, Phase 4H data readiness report generation, and Phase 4M fail-closed dataset eligibility checks.

### 2.2 Relevant Non-ML Regression Suite
- **Command**:
  ```powershell
  python -m pytest tests\test_blocker_register.py tests\test_fits_semantics_audit.py tests\test_phase3b_authoritative_audit.py tests\test_phase3c_semantics_audit.py tests\test_phase4h_data_readiness.py tests\test_real_data_sources.py tests\test_chronological_audit.py tests\test_feature_engineering_step9.py tests\test_phase2c_eligibility.py tests\test_scientific_readiness.py tests\test_dataset_eligibility.py -q
  ```
- **Interpreter**: `C:\Python313\python.exe`
- **Working Directory**: `backend`
- **Exit Code**: `0`
- **Summary**: **80 passed in 9.33s**
- **Warnings**: 25 warnings (standard Pydantic v2 `ConfigDict` deprecation notices, SQLAlchemy `utcnow` deprecations, and Pandas dataframe fragmentation performance warnings).
- **Coverage**: Covers authoritative data sources, fail-closed HEL1OS adapter, SoLEXS lightcurve reader, chronological split isolation, feature engineering manifest integrity, and scientific readiness report persistence.

### 2.3 Full Backend Test Suite
- **Command**:
  ```powershell
  python -m pytest tests -q
  ```
- **Interpreter**: `C:\Python313\python.exe`
- **Working Directory**: `backend`
- **Exit Code**: `0`
- **Summary**: **235 passed, 0 failed in 37.44s**
- **Warnings**: 620 warnings (comprising Pydantic class-based config deprecations, FastAPI lifespan event handler notices, and NumPy 2.5 shape assignment deprecations in joblib).
- **Result**: Complete test suite passed cleanly.

### 2.4 Training Readiness Dry Run
- **Command**:
  ```powershell
  python scripts\training_readiness_dry_run.py
  ```
- **Interpreter**: `C:\Python313\python.exe`
- **Working Directory**: `backend`
- **Exit Code**: `1` (as designed: returns non-zero when dataset eligibility status is `INELIGIBLE` or `UNRESOLVED`)
- **Safety Verification**:
  - `trainer_invoked`: `false`
  - `parameters_fitted`: `false`
  - `checkpoint_created`: `false`
