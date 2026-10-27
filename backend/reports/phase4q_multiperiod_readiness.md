# Phase 4Q: Multi-Period Evaluation Readiness & Retraining Policy Audit

**Document ID:** `PHASE4Q-EVAL-READY-001`  
**Phase:** Phase 4Q — Authorized Observational Data Acquisition, Scientific Eligibility Gate, and Multi-Period Readiness  
**Repository:** `C:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main`  
**Backend:** `C:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main\backend`  
**Authoritative Reference Checkpoint:** `artifacts/real_data_experiments/phase4o/solexs_phase4o_cnn_lstm_direct_20261025T184851Z.pt`  
(SHA-256: `d6076ab20c99fdc852d338c47f9b7248250a333c4708916a8ced8de3a6336514`)  
**Execution Timestamp:** 2026-10-26T01:25:00 UTC  

---

## 1. Executive Summary & Retraining Gate Determination

In accordance with Phase 4Q Requirement 7 ("Establish multi-period evaluation readiness") and Requirement 8 ("Retraining policy"), this document conducts a formal readiness audit of candidate evaluation partitions and evaluates the mandatory conditions required to authorize model retraining.

### Formal Retraining Policy Gate Audit

| Gate Requirement | Condition Description | Audit Finding | Gate Status |
|---|---|---|---|
| **Condition 1** | New observational data exist locally and pass byte count / SHA-256 integrity checks. | No new observation dates have been downloaded into the workspace; local data is strictly limited to the original 6-day epoch (2026-01-31 to 2026-02-05). | **FAILED (BLOCKED)** |
| **Condition 2** | Scientific eligibility is established for the intended forecasting task. | Self-supervised SDD2 count-rate forecasting is eligible, but HEL1OS and physical flux conversion remain blocked. | **CONDITIONALLY SATISFIED** |
| **Condition 3** | Expanded manifest is deterministic, bit-verified, and reproducible. | `phase4q_expanded_dataset_manifest.json` generated and bit-verified with automated tests. | **PASSED** |
| **Condition 4** | Chronological evaluation partitions defined prior to model selection. | Rolling-origin chronological splits defined; strict temporal monotonicity enforced. | **PASSED** |
| **Condition 5** | Final-test period is strictly reserved, isolated, and untouched. | Held-out test period (2026-02-05) remains completely isolated; no tuning or scaler fitting allowed. | **PASSED** |

### Retraining Decision: **MODEL RETRAINING HALTED (DISALLOWED)**
Because **Condition 1 is not satisfied** (no new observational periods exist locally), model retraining is strictly prohibited. Re-running training loops on the unchanged 6-day epoch would represent data snooping without answering the core scientific question of multi-month generalization across solar activity regimes.

The frozen Phase 4M-S/4O reference model checkpoint remains the authoritative benchmark.

---

## 2. Chronological Multi-Period Evaluation Partitioning

To ensure zero temporal leakage across training, validation, and test sets, evaluation must follow a strict walk-forward design:

```
+─────────────────────────────────────────────────────────────────────────────+
|                          CHRONOLOGICAL TIMELINE                             |
+─────────────────────────────────────────────────────────────────────────────+
| Day 1 (Jan 31): Baseline Quiet (86.4k bins) ──> [ TRAINING PARTITION ]     |
| Day 2 (Feb 01): Flare Eruptions (45.1k bins) ─> [ TRAINING PARTITION ]     |
| Day 3 (Feb 02): High Activity (45.0k bins) ───> [ TRAINING PARTITION ]     |
+─────────────────────────────────────────────────────────────────────────────+
                                      │
                                      ▼
+─────────────────────────────────────────────────────────────────────────────+
| Day 4 (Feb 03): Telemetry Gap (5.1k bins) ────> [ VALIDATION PARTITION ]   |
| Day 5 (Feb 04): Flare Decay (45.3k bins) ─────> [ VALIDATION PARTITION ]   |
+─────────────────────────────────────────────────────────────────────────────+
                                      │
                                      ▼
+─────────────────────────────────────────────────────────────────────────────+
| Day 6 (Feb 05): Moderate Quiet (45.3k bins) ──> [ HELD-OUT TEST (FROZEN) ] |
+─────────────────────────────────────────────────────────────────────────────+
```

### Partitioning Safeguards
1. **Strict Temporal Directionality:** $\max(\text{Time}_{\text{train}}) < \min(\text{Time}_{\text{val}}) < \min(\text{Time}_{\text{test}})$. Future observations never inform past states.
2. **Scaler Isolation:** Scalers (e.g. `CountRateScaler`) must be fit strictly on the training partition:
   $$\mu_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{i \in \text{train}} c_i, \quad \sigma_{\text{train}} = \sqrt{\frac{1}{N_{\text{train}}} \sum_{i \in \text{train}} (c_i - \mu_{\text{train}})^2}$$
   The validation and test partitions must be transformed using $(\mu_{\text{train}}, \sigma_{\text{train}})$ without updating parameters.
3. **Contiguity Enforcement:** Input windows ($T_{\text{in}} = 60\,\text{s}$) and forecast targets ($T_{\text{out}} = 10\,\text{s}$) can only be formed within unbroken valid observation spans. Windows are strictly prohibited from crossing split boundaries or telemetry gaps.
4. **Partition Independence:** Duplicates or version counterpart packages (such as HEL1OS V111 vs V211) cannot appear across splits.

---

## 3. Standardized Metric Framework & Baselines

All evaluations must be performed in native observational units ($\text{counts/s}$, 2–22 keV). Performance must be reported for each independent observation period and compared against two non-parametric baselines evaluated on identical target ground truth:

### 3.1 Non-Parametric Baselines
1. **Persistence Forecast:**
   $$\hat{y}_{t+h}^{\text{persist}} = c_t, \quad \forall h \in \{1, \dots, 10\}$$
2. **Rolling Mean Forecast:**
   $$\hat{y}_{t+h}^{\text{mean}} = \frac{1}{10} \sum_{k=0}^{9} c_{t-k}, \quad \forall h \in \{1, \dots, 10\}$$

### 3.2 Evaluation Metrics
- **Mean Absolute Error (MAE):** $\text{MAE}(h) = \frac{1}{N} \sum_{i=1}^N |y_{i, t+h} - \hat{y}_{i, t+h}|$
- **Root Mean Squared Error (RMSE):** $\text{RMSE}(h) = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_{i, t+h} - \hat{y}_{i, t+h})^2}$
- **Forecast Skill Score:**
  $$S_{\text{persist}}(h) = \left( 1 - \frac{\text{MAE}_{\text{model}}(h)}{\text{MAE}_{\text{persist}}(h)} \right) \times 100\%$$

---

## 4. Multi-Month Generalization Assessment

### Scientific Reality of 6-Day Observations
The held-out test day (2026-02-05) exhibits moderate quiet background conditions (mean count rate $275.1\,\text{counts/s}$). In Phase 4N and Phase 4O, out-of-cohort evaluations revealed:
- Overall test day skill score is positive ($+6.20\%$ at $t+1$, $+7.11\%$ at $t+10$).
- In high-flux regimes ($> 1,000\,\text{counts/s}$), the skill score becomes negative ($-2.76\%$), demonstrating that the model suffers from phase lag during rapid count-rate rises.
- **Scientific Determination:** Six days of data collected during a single solar rotation cannot prove generalization across the 11-year solar cycle. Multi-period generalization remains **unestablished** until multi-month observations spanning diverse activity regimes (such as the May 2024 X-class solar flare epoch) are acquired and evaluated.

---

## 5. Frozen Reference Benchmark Metrics (Phase 4O Baseline)

The reference CNN-LSTM direct-forecast checkpoint (`solexs_phase4o_cnn_lstm_direct_20261025T184851Z.pt`) remains the frozen benchmark:

| Forecast Horizon | Target Metric | Reference CNN-LSTM | Persistence Baseline | Rolling Mean Baseline | Skill Score ($S_{\text{persist}}$) |
|---|---|---|---|---|---|
| **$t+1$ second** | MAE (counts/s) | **1.99** | 2.12 | 2.45 | **+6.13%** |
| **$t+1$ second** | RMSE (counts/s) | **15.22** | 16.10 | 17.80 | **+5.47%** |
| **$t+5$ seconds** | MAE (counts/s) | **7.55** | 8.11 | 9.32 | **+6.91%** |
| **$t+5$ seconds** | RMSE (counts/s) | **48.10** | 50.85 | 54.20 | **+5.41%** |
| **$t+10$ seconds** | MAE (counts/s) | **12.33** | 13.28 | 15.10 | **+7.15%** |
| **$t+10$ seconds** | RMSE (counts/s) | **74.50** | 79.10 | 83.45 | **+5.82%** |

This benchmark must remain unperturbed until multi-month expansion is executed.
