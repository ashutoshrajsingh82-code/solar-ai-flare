# Phase 4W: Pre-Registered Holdout Evaluation Protocol for October 2024 Epoch

**Document ID:** `phase4w_holdout_evaluation_protocol.md`  
**Phase:** Phase 4W — Final Development Benchmark, Error Analysis, and Model Freeze  
**Execution Timestamp:** 2026-10-26T16:51:29Z  
**Repository:** `solar-ai-flare`  
**Target Epoch:** October 2024 (Reserved Holdout Observation Epoch)  
**Holdout Status:** `LOCKED_NOT_ACQUIRED` — Strictly preserved, zero bytes evaluated prior to formal protocol registration.

---

## 1. Protocol Objective and Scientific Rationale

To maintain the highest scientific rigor and prevent inadvertent data leakage, data snooping, or post-hoc p-hacking, this document establishes the binding, pre-registered evaluation protocol for the frozen Phase 4V neural forecasting model against the independent **October 2024** solar observation epoch.

Under this protocol:
1. The model weights, architecture, scaler constants, and inference scripts are immutably **frozen**.
2. Evaluation metrics and subgroup stratifications are pre-declared prior to unsealing the holdout data.
3. No hyperparameter adjustment, fine-tuning, retraining, or cherry-picking of test segments is permitted.
4. Evaluation is executed as a strictly single-pass pipeline.

---

## 2. Frozen Artifact Registry and Cryptographic Signatures

Evaluation on the October 2024 holdout must execute exclusively with the following cryptographically validated artifacts:

### 2.1 Model Checkpoint
- **Filename:** `solexs_phase4v_cnn_lstm_20261026T163620Z.pt`
- **Path:** `backend/artifacts/real_data_experiments/phase4v/solexs_phase4v_cnn_lstm_20261026T163620Z.pt`
- **SHA256 Checksum:** `e67662368c49f0d110b15e2378c04dceb8df8fbd9936d2b882a028d4fb0bd26d`
- **Architecture:** `SelfSupervisedCNNLSTM` (`ml.experiments.real_self_supervised`)
  - Input Features: 1 (SDD2 Count Rate in counts/s)
  - CNN Channels: 32 (Kernel Size: 3, ReLU)
  - LSTM Hidden Dimension: 64 (2 Layers, Dropout: 0.2)
  - Direct Multi-Horizon Linear Head: 10 output steps ($H \in [1, 10]\text{ s}$)

### 2.2 Frozen Preprocessing and Normalizer
- **Scaler Type:** `log10_zscore`
- **Frozen Normalization Parameters:**
  $$\mu_{\log_{10}} = 2.5978752099759608$$
  $$\sigma_{\log_{10}} = 0.582402100097115$$
  *(Fitted strictly on development training partition; MUST NOT be refitted on October 2024 data).*
- **Sequence Parameters:**
  - Input Context Length ($L$): 60 seconds (1-second cadence)
  - Forecast Horizon ($H$): 10 seconds (1-second cadence)
  - Segment Continuity Gate: Contiguous segments with minimum length $\ge 70\text{ seconds}$ ($\Delta t = 1.0\text{ s} \pm 0.05\text{ s}$).

### 2.3 Frozen Code Modules
- **Pipeline Implementation:** `backend/scripts/phase4v_pipeline.py` (SHA256: `06a97149db0667f8adfb82cab36b9e5aa32ea9a2f39cdc21aaac72775ae1362f`)
- **Neural Architecture Definition:** `backend/ml/experiments/real_self_supervised.py` (SHA256: `6c5081e333d1a9684a7600110a4e52df3167c3a13df5d6eafc1bfeb00a2957ee`)

---

## 3. Pre-Declared Evaluation Methodology and Metrics

The holdout evaluation must compute the following metrics across all forecast horizons $h \in \{1, 2, \dots, 10\}\text{ seconds}$:

### 3.1 Mathematical Metric Formulations
1. **Mean Absolute Error (MAE):**
   $$\text{MAE}(h) = \frac{1}{N} \sum_{i=1}^{N} |y_{i, t+h} - \hat{y}_{i, t+h}|$$
2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE}(h) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_{i, t+h} - \hat{y}_{i, t+h})^2}$$
3. **Relative Mean Absolute Error (RelMAE):**
   $$\text{RelMAE}(h) = \frac{\text{MAE}(h)}{\bar{y}_{t+h}} \times 100\%$$
4. **Skill Score vs. Persistence:**
   $$\text{Skill}_{\text{Pers}}(h) = \left( 1 - \frac{\text{MAE}_{\text{Model}}(h)}{\text{MAE}_{\text{Pers}}(h)} \right) \times 100\%$$

### 3.2 Pre-Declared Subgroup Cohorts
All metrics must be reported for the full dataset and stratified causally by conditions at forecast origin ($t=0$):
1. **`ALL`:** Unrestricted holdout cohort.
2. **`Causal_REG_QUIET`:** Origin flux $y_t \le 200\text{ counts/s}$.
3. **`Causal_REG_MOD`:** Origin flux $200 < y_t \le 1000\text{ counts/s}$.
4. **`Causal_REG_HIGH`:** Origin flux $y_t > 1000\text{ counts/s}$.
5. **`Causal_SURGE_50`:** Rapid change $|y_{t+10} - y_t| \ge 50\text{ counts/s}$.

### 3.3 Benchmark Baselines
The frozen CNN-LSTM must be compared directly against:
- **Persistence Baseline:** $\hat{y}_{t+h} = y_t$
- **Rolling Mean Baseline:** $\hat{y}_{t+h} = \frac{1}{60} \sum_{k=0}^{59} y_{t-k}$

---

## 4. Pre-Declared Success Criteria for Holdout Evaluation

The holdout evaluation in the subsequent phase (Phase 4X) will be adjudicated according to the following pre-registered scientific criteria:

| Criterion | Target Metric | Required Threshold | Scientific Significance |
|---|---|---|---|
| **Primary Defined-Regime Skill** | Skill vs Persistence in Quiet Regime ($H=1\text{ s}$) | $\text{Skill} \ge +15.0\%$ | Validates neural representation of background solar variability |
| **Primary Defined-Regime Skill** | Skill vs Persistence in Moderate Regime ($H=1\text{ s}$) | $\text{Skill} \ge +15.0\%$ | Validates flare precursor and moderate active region forecasting |
| **Multi-Step Stability** | Skill vs Persistence in Quiet Regime ($H=10\text{ s}$) | $\text{Skill} \ge +15.0\%$ | Validates forecast stability over extended lead times |
| **Residual Well-Behavedness** | RMSE / MAE ratio in Quiet and Moderate Regimes | $\text{Ratio} \le 1.60$ | Confirms absence of erratic numerical blowups or wild oscillations |
| **Baseline Superiority** | Model MAE vs Rolling Mean MAE in ALL | $\text{MAE}_{\text{Model}} \le \text{MAE}_{\text{RollingMean}}$ | Confirms model outperforms simple historical moving averages |

---

## 5. Execution Safeguards and Governance Rules

1. **Single-Pass Evaluation:** The holdout evaluation script must be executed once. Results are saved immediately to `reports/phase4x_holdout_metrics.csv`.
2. **Zero In-Sample Fitting:** Neither weights, scaler parameters, nor threshold boundaries may be adjusted after inspecting holdout metrics.
3. **Quarantine Assurance:** As of this phase (Phase 4W), no file matching `*202410*.fits` or any October 2024 Level-1 data has been parsed, loaded, or utilized by the model pipeline.
4. **Reproducibility Guarantee:** Any independent researcher possessing `backend/scripts/phase4w_benchmark.py` and the frozen checkpoint `solexs_phase4v_cnn_lstm_20261026T163620Z.pt` will reproduce identical evaluation results.
