# ARUN-X
## Automated Radiation Understanding Network for Solar Flare Forecasting

# Project Status Report

**Project:** Aditya-L1 Solar Flare Intelligence System  
**Status:** Development and Integration  
**Purpose:** Track implementation progress, verification status, and remaining development requirements.

## 1. Project Overview

ARUN-X is a research prototype for solar flare nowcasting and probabilistic forecasting using combined soft and hard X-ray observations from the SoLEXS and HEL1OS instruments aboard Aditya-L1.

The project integrates synthetic data generation, preprocessing, feature engineering, machine learning, deep learning, inference, alert management, and a web-based mission control interface.

The current implementation includes a data-processing and machine-learning core, along with application components intended to support end-to-end demonstration workflows. Verification status varies across individual components.

## 2. Implemented and Execution-Verified Components

The following components have been executed and verified in an offline development environment with NumPy, Pandas, and Scikit-learn available.

### 2.1 Synthetic Data Generation

- `ml/data/synthetic_generator.py`
  - Generates synthetic SoLEXS and HEL1OS time-series observations.
  - Simulates flare events, measurement noise, data gaps, and anomalous spikes.
  - Incorporates a Neupert-effect-inspired temporal relationship between hard and soft X-ray signals.
  - Supports simulated flare classes B, C, M, and X.

- `backend/sample_data/demo_aditya_l1_solexs_helios.csv`
  - Contains a pre-generated demonstration dataset.
  - Covers 10 days at a one-minute cadence.
  - Includes 14 simulated flare events.

### 2.2 Data Preprocessing

- `ml/preprocessing/cleaning.py`
  - Handles duplicate observations, anomalous spikes, and missing data.

- `ml/preprocessing/alignment.py`
  - Aligns observations and resamples them to a configurable temporal cadence.

- `ml/preprocessing/transform.py`
  - Implements logarithmic transformation and feature scaling.
  - Supports fitting scaling parameters on training data only.

### 2.3 Feature Engineering and Label Generation

- `ml/features/engineering.py`
  - Generates rolling statistical features and hard-to-soft X-ray ratios.
  - Supports optional seasonal-trend decomposition.

- `ml/features/labels.py`
  - Generates four-class nowcasting labels.
  - Generates binary forecast labels for five prediction horizons.

- `ml/features/windowing.py`
  - Implements chronological data partitioning and sliding-window sequence generation.
  - Supports leakage-aware preparation of time-series training samples.

### 2.4 Evaluation Metrics

- `ml/evaluation/metrics.py`
  - Implements True Skill Statistic (TSS).
  - Implements Heidke Skill Score (HSS).
  - Implements Probability of Detection (POD) and False Alarm Ratio (FAR).
  - Supports ROC-AUC, precision-recall AUC, Brier Score, and reliability analysis.

### 2.5 Classical Machine Learning

- `ml/models/baseline.py`
  - Provides a multi-task baseline interface for Logistic Regression, Random Forest, and Support Vector Machine models.

### 2.6 Verified End-to-End Workflow

The following offline workflow was executed successfully:

1. Synthetic observation generation.
2. Data cleaning.
3. Temporal alignment and resampling.
4. Transformation and training-only scaler fitting.
5. Feature engineering, producing 97 columns, including the hard-to-soft ratio.
6. Nowcasting and forecasting label generation.
7. Chronological data splitting.
8. Sliding-window generation, producing 8,401 windows with a sequence shape of `(30, 2)`.
9. Random Forest model training and prediction.
10. Computation of TSS and HSS.

**Verification result:** The reported workflow completed successfully in the available environment. This verification applies to the tested synthetic-data pipeline and does not establish forecasting performance on real Aditya-L1 observations.

## 3. Implemented but Not Execution-Verified Components

The following components have been developed but have not been fully executed and validated in the reported development environment.

### 3.1 Deep Learning Models

- `ml/models/deep_models.py`
  - One-dimensional Convolutional Neural Network.
  - Long Short-Term Memory network.
  - CNN-LSTM hybrid architecture.
  - Optional CNN-Transformer architecture.

These models use PyTorch and require dependency installation and runtime verification.

### 3.2 Training and Model Management

- `ml/training/train.py`
  - Multi-task training workflow.
  - Class-weighted loss handling.

- `ml/models/registry.py`
  - Model registration and selection.

- `ml/inference/predictor.py`
  - Prediction and inference logic.

- `ml/inference/simulator.py`
  - Sequential historical telemetry playback.

### 3.3 Backend Application

The FastAPI application and supporting services have been developed to provide:

- Health and system-status endpoints.
- Data ingestion and preprocessing.
- Model listing, selection, and training.
- Nowcasting and multi-horizon forecasting.
- Alert creation and acknowledgement.
- Historical telemetry simulation.
- Dashboard summaries and fusion experiments.
- SQLAlchemy-based database models.

The backend is intended to connect the machine-learning components to the web application through REST APIs.

**Verification limitation:** The listed components have not been fully boot-tested in the reported environment because the required FastAPI, PyTorch, and other dependencies were unavailable. Successful imports, endpoint execution, database operations, and complete application startup must therefore be verified separately.

## 4. Frontend and Application Integration

The React application and its API integration are intended to provide a mission control interface for data inspection, preprocessing, model management, prediction results, historical playback, and alert monitoring.

The application architecture includes same-origin API routing for local Vite development and Docker/nginx deployment.

The reported implementation includes frontend components and backend API integration intended for demonstration use. A complete, independently verified end-to-end deployment has not been established by the offline machine-learning test alone.

## 5. Remaining Development and Verification Tasks

The following tasks remain outstanding or require further verification.

### 5.1 Dependency and Runtime Verification

- Install the complete backend dependency set.
- Install frontend dependencies.
- Verify Python package compatibility.
- Run the FastAPI application using Uvicorn.
- Resolve import errors, runtime exceptions, and API integration issues.
- Verify database initialization and persistence.
- Confirm frontend-to-backend communication.

### 5.2 Automated Testing

- Add and execute unit tests for preprocessing and feature engineering.
- Test model training and prediction interfaces.
- Validate forecast-label generation and chronological split boundaries.
- Test all REST endpoints.
- Test alert persistence and acknowledgement.
- Verify historical telemetry playback.
- Run integration tests across the full application.

### 5.3 Frontend Verification

- Validate dashboard rendering and data loading.
- Test charts, tables, and prediction visualizations.
- Verify upload workflows and error handling.
- Test model selection, training, and inference interfaces.
- Verify alert management and simulation controls.

### 5.4 Documentation and Repository Configuration

- Complete `README.md` with installation, configuration, and usage instructions.
- Add `docker-compose.yml` and verify the containerized deployment.
- Add `.gitignore`.
- Add `.env.example` for supported environment variables.
- Document configuration options and required dependencies.
- Confirm that the `LICENSE` file matches the license declared in the documentation.

## 6. Production Readiness Requirements

The current system should be treated as a research and demonstration prototype until additional validation is completed.

Before production deployment, the following requirements should be addressed:

- Replace synthetic demonstration telemetry with appropriately validated observational data.
- Verify instrument-specific calibration, timestamp alignment, and data quality.
- Establish reliable flare labels and scientifically appropriate forecast targets.
- Evaluate models on independent chronological test periods.
- Measure probability calibration, false alarms, missed events, and uncertainty.
- Validate forecasting performance across different solar activity conditions.
- Implement authentication and authorization where required.
- Add durable storage for uploaded datasets, trained models, and pipeline artifacts.
- Introduce background jobs for long-running training and preprocessing tasks.
- Add structured logging, monitoring, and error reporting.
- Document operational limitations and establish a model-validation procedure.

## 7. Current Assessment

| Area | Status |
|---|---|
| Synthetic data generation | Execution-verified |
| Data preprocessing | Execution-verified in the tested pipeline |
| Feature engineering and labels | Execution-verified in the tested pipeline |
| Classical machine-learning baseline | Execution-verified in the tested pipeline |
| Evaluation metrics | Selected metrics executed successfully |
| Deep learning models | Implemented; runtime verification pending |
| Training and inference services | Implemented; full verification pending |
| FastAPI backend | Developed; full startup and endpoint verification pending |
| React frontend | Integration and end-to-end verification required |
| Automated test suite | Comprehensive verification pending |
| Docker deployment | Build and runtime verification pending |
| Real-data scientific validation | Required |

## 8. Verification Scope and Disclaimer

The execution-verified results refer to the synthetic-data workflow completed in the available offline development environment. They do not establish the correctness of every application component or the scientific validity of the forecasting system.

Synthetic observations, demonstration alerts, and unvalidated predictions must not be used for operational space-weather decisions.

ARUN-X is an independent educational and research prototype. It is not an official ISRO operational forecasting system and is not endorsed by ISRO.

---

**Last updated:** 17 October 2026  
**Overall status:** Development and integration; selected machine-learning pipeline components execution-verified, with application-wide verification and real-data scientific validation outstanding.
