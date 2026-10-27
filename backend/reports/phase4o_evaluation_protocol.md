# Phase 4O: Leakage-Safe Evaluation Protocol & Chronological Split Specification

**Document Version:** 1.0.0  
**Audit Date:** 2026-10-26  
**Canonical Designation:** `PHASE4O_EVALUATION_PROTOCOL_SPECIFICATION`  
**Parent Phase:** Phase 4O (High-Activity Forecasting Improvement & Leakage-Safe Model Comparison)  

---

## 1. Principles of the Evaluation Protocol

When working with short observational time series (such as the 6-day Aditya-L1 SoLEXS archive), there is an acute danger of **evaluation contamination**: if researchers use held-out test data to iterate on architectures or discover failure modes, that test set becomes contaminated by implicit model selection.

Phase 4O establishes a strict, non-negotiable protocol:

1. **Quarantine of February 5:** The held-out test day (`2026-02-05`) was evaluated and inspected in Phase 4M-S and Phase 4N. It is treated as an **immutable historical benchmark**. Under zero circumstances may February 5 be used to select architectures, tune learning rates, optimize dropout, or establish operational thresholds for Phase 4O experiments.
2. **Validation-Only Model Selection:** All candidate model comparisons, architecture selections, and early stopping decisions must be conducted **strictly on the validation partition (`2026-02-03` to `2026-02-04`)**.
3. **Causal Rolling-Origin Integrity:** Every evaluated forecast window $(X_t, Y_t)$ is evaluated strictly at its forecast origin $T$, where $X_t$ contains past history $[T-59 \dots T]$ and $Y_t$ contains future targets $[T+1 \dots T+10]$.
4. **Zero Multi-Month Generalization Claims:** Six calendar days cannot certify generalization across the 11-year solar cycle. Any claim of broad space-weather validity is explicitly prohibited until multi-month observations are acquired.

---

## 2. Chronological Split Specification Table

| Partition Cohort | Calendar Range (UTC) | Start Unix Timestamp | End Unix Timestamp | Total Raw Seconds | Finite Valid Records | Valid Contiguous Windows ($60\text{s} \to 10\text{s}$) | Source FITS Products | Protocol Role in Phase 4O |
|---|---|---|---|---|---|---|---|---|
| **Training Partition** | 2026-01-31 00:00:00 to 2026-02-02 23:59:59 | 1,769,817,600.0 | 1,769,990,399.0 | 259,200 | 176,488 (68.09%) | 174,334 | `AL1_SOLEXS_20260131_SDD2_L1.lc.gz`<br>`AL1_SOLEXS_20260201_SDD2_L1.lc.gz`<br>`AL1_SOLEXS_20260202_SDD2_L1.lc.gz` | **Model training, loss computation, gradient updates, scaler fitting, and threshold discovery.** |
| **Validation Partition** | 2026-02-03 00:00:00 to 2026-02-04 23:59:59 | 1,769,990,400.0 | 1,770,163,199.0 | 172,800 | 50,426 (29.18%) | 48,523 | `AL1_SOLEXS_20260203_SDD2_L1.lc.gz`<br>`AL1_SOLEXS_20260204_SDD2_L1.lc.gz` | **Candidate model comparison, validation loss tracking, and candidate selection decision.** |
| **Historical Test Partition** | 2026-02-05 00:00:00 to 2026-02-05 23:59:59 | 1,770,163,200.0 | 1,770,249,599.0 | 86,400 | 45,309 (52.44%) | 44,221 | `AL1_SOLEXS_20260205_SDD2_L1.lc.gz` | **Quarantined. Preserved as immutable benchmark for historical comparison only.** |

- **Source Archive SHA-256:** `34c39624194ee2da95bbf8e3dfed5ac002c55a517a31bc4091bfdd0d2b4671c2`
- **Cadence:** Uniform 1.000 s strictly enforced. Zero duplicate or out-of-order timestamps permitted.
- **Detector / Energy Band:** SoLEXS SDD2 Total Rate (`RATE`), 2.0–22.0 keV.

---

## 3. Strict Preprocessing and Transformation Protocol

1. **Scaler Parameters:**
   The normalization transform is strictly $\log_{10}(x+1)$ followed by standard z-score standardization:
   $$\tilde{x} = \frac{\log_{10}(x+1) - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
   Parameters $\mu = 2.2472749$ and $\sigma = 0.6966392$ are fitted **exclusively on the 176,488 finite records of the training partition**. No refitting or statistical updating is allowed on validation or test data.
2. **Inverse Transform Requirement:**
   All loss values and optimization gradients occur in scaled log-space, but **every reported evaluation metric (MSE, RMSE, MAE, Relative MAE) must be calculated on inverse-transformed predictions in native detector units (`counts/s`)**:
   $$\hat{y} = \max\left(0, 10^{(\hat{\tilde{y}} \cdot \sigma + \mu)} - 1\right)$$
3. **Gap Handling:**
   Data gaps ($\Delta t > 1.0\text{ s}$) or NaN values break time-series continuity. No windows are constructed across gap boundaries.

---

## 4. Candidate Model Selection Decision Rule

Under Requirement 4, candidate models are selected according to an explicit, objective decision rule evaluated strictly on the 48,523 validation windows:

$$\text{Select Candidate } M^* \iff \text{MAE}_{\text{val}}(M^*) < \text{MAE}_{\text{val}}(\text{CNN-LSTM}_{\text{direct}}) \quad \land \quad \text{Skill}_{\text{persist}}(M^*) > \text{Skill}_{\text{persist}}(\text{CNN-LSTM}_{\text{direct}})$$

If no candidate satisfies both conditions on the validation partition, the protocol mandates:
- Retain the reference Phase 4M-S model (`cnn_lstm_direct`).
- Issue the evidence-based outcome: `NO_RELIABLE_IMPROVEMENT_ESTABLISHED`.
- Strictly prohibit tuning hyperparameters against February 5.
