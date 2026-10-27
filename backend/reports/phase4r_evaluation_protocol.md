# Phase 4R: Chronological Multi-Period Evaluation Protocol

**Document ID:** `PHASE4R-PROTO-EVAL-001`  
**Phase:** Phase 4R — Real Observational Data Expansion, Gated Training, and Independent Evaluation  
**Repository:** `C:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main`  
**Backend:** `C:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main\backend`  
**Target Architecture:** Direct-Forecasting CNN-LSTM Sequence-to-Sequence Model  
**Execution Timestamp:** 2026-10-26T01:36:00 UTC  

---

## 1. Protocol Architecture & Anti-Leakage Safeguards

This protocol defines the formal, leakage-safe evaluation standard governing all self-supervised count-rate forecasting experiments. In compliance with Stage 5 of the Phase 4R Mission, evaluation enforces strictly chronological walk-forward partitions, quarantine of the held-out test period, training-only scaler fitting, and gap-preserving window segmentation.

```
+─────────────────────────────────────────────────────────────────────────────+
|                         WALK-FORWARD PARTITIONING                           |
+─────────────────────────────────────────────────────────────────────────────+
| Training Partition:   2026-01-31 to 2026-02-02 (175,798 windows, 10 segs)  |
|                       Fit CountRateScaler (mu_train, sigma_train) strictly. |
+─────────────────────────────────────────────────────────────────────────────+
                                      │
                                      ▼
+─────────────────────────────────────────────────────────────────────────────+
| Validation Partition: 2026-02-03 to 2026-02-04 (50,150 windows, 4 segs)    |
|                       Transform using training statistics. Early stopping.  |
+─────────────────────────────────────────────────────────────────────────────+
                                      │
                                      ▼
+─────────────────────────────────────────────────────────────────────────────+
| Final Test Partition: 2026-02-05 (45,102 windows, 3 segs) [FROZEN]         |
|                       Evaluated ONCE for final benchmark metrics. Zero tune.|
+─────────────────────────────────────────────────────────────────────────────+
```

### Non-Negotiable Anti-Leakage Rules:
1. **Temporal Monotonicity:** $\text{Date}(\text{Train}) < \text{Date}(\text{Validation}) < \text{Date}(\text{Test})$. Observations from future dates never inform earlier splits.
2. **Scaler Isolation:** Scalers (e.g. `CountRateScaler`) must be fit strictly on training windows:
   $$\mu_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{i \in \text{train}} c_i, \quad \sigma_{\text{train}} = \sqrt{\frac{1}{N_{\text{train}}} \sum_{i \in \text{train}} (c_i - \mu_{\text{train}})^2}$$
   Fitting scalers on validation or test sets constitutes fatal pipeline leakage.
3. **Quarantine of Held-Out Test Set:** The February 5, 2026 test set is strictly quarantined. It is never used for architecture selection, hyperparameter tuning, early stopping, or threshold selection.
4. **Zero Gap Crossing:** Windows require 70 consecutive 1-second records ($T_{\text{in}} = 60\,\text{s}$, $T_{\text{out}} = 10\,\text{s}$) where $\Delta t = 1.0\,\text{s}$ and $c_i \ge 0$. No window may cross a missing value or partition boundary.

---

## 2. Partition Breakdown & Window Accounting

| Split Role | Date Range | Solar Activity Conditions | Finite Bins | Contiguous Segments | Valid Windows ($60\text{s} \to 10\text{s}$) | Scaler Role |
|---|---|---|---|---|---|---|
| **Training** | 2026-01-31 to 2026-02-02 | Quiet background to major flare eruptions (peak 27,465 c/s) | 176,488 | 10 | 175,798 | **Fit & Transform** |
| **Validation** | 2026-02-03 to 2026-02-04 | Telemetry gap recovery and moderate flare decay (peak 4,019 c/s) | 50,426 | 4 | 50,150 | **Transform Only** |
| **Final Test** | 2026-02-05 | Moderate quiet background (peak 2,030 c/s; mean 275.1 c/s) | 45,309 | 3 | 45,102 | **Transform Only (Frozen)** |
| **Total** | **6 Days** | **Multi-Regime Epoch** | **272,223** | **17** | **271,050** | **Leakage-Safe** |

---

## 3. Solar Activity Regime Stratification

To prevent aggregate metric dilution where calm baseline hours conceal flare-peak prediction errors, all evaluation windows must be stratified by target peak magnitude:

$$\text{Regime Tag} = \begin{cases} 
\mathbf{REG\_HIGH} & \text{if } \max_{h \in [1,10]} y_{t+h} > 1,000\,\text{counts/s} \\ 
\mathbf{REG\_MOD} & \text{if } 200 \le \max_{h \in [1,10]} y_{t+h} \le 1,000\,\text{counts/s} \\ 
\mathbf{REG\_QUIET} & \text{if } \max_{h \in [1,10]} y_{t+h} < 200\,\text{counts/s} 
\end{cases}$$

### Scientific Purpose of Regime Stratification:
In Phase 4N, evaluating aggregate test-day metrics showed a misleading $+6.20\%$ skill score, while stratified evaluation revealed negative skill ($-2.76\%$) in the high-flux regime (`REG_HIGH`). Every candidate model must report metrics separately across all three regimes.

---

## 4. Benchmark Baselines & Metric Definitions

All models must be benchmarked against two non-parametric baselines evaluated on identical target vectors $\mathbf{y}_t = [c_{t+1}, \dots, c_{t+10}]^T$:

### 4.1 Persistence Baseline
Assumes the solar count rate remains frozen at the final observed second $c_t$:
$$\hat{\mathbf{y}}_t^{\text{persist}} = [c_t, c_t, \dots, c_t]^T \in \mathbb{R}^{10}$$

### 4.2 Rolling Mean Baseline
Projects the moving average of the preceding $W = 10\,\text{s}$ input window across the horizon:
$$\hat{y}_{t+h}^{\text{mean}} = \frac{1}{10} \sum_{k=0}^{9} c_{t-k}, \quad \forall h \in \{1, \dots, 10\}$$

### 4.3 Mathematical Metrics (Computed in Native `counts/s` Units)
1. **Mean Absolute Error (MAE):**
   $$\text{MAE}(h) = \frac{1}{N} \sum_{i=1}^N |y_{i, t+h} - \hat{y}_{i, t+h}|$$
2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE}(h) = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_{i, t+h} - \hat{y}_{i, t+h})^2}$$
3. **Forecast Skill Score relative to Persistence ($S_{\text{persist}}$):**
   $$S_{\text{persist}}(h) = \left( 1 - \frac{\text{MAE}_{\text{model}}(h)}{\text{MAE}_{\text{persistence}}(h)} \right) \times 100\%$$
   - $S_{\text{persist}} > 0.0\%$: Model demonstrates genuine autoregressive predictive skill.
   - $S_{\text{persist}} \le 0.0\%$: Model performs worse than trivially repeating the latest observation.

---

## 5. Model Promotion & Production Quarantine

Under non-negotiable scientific safeguards:
1. **Model Registry Quarantine:** All experimental checkpoints are marked with:
   ```json
   {
     "production_registry_eligible": false,
     "experimental_marker": "PHASE4R_RESEARCH_ONLY"
   }
   ```
2. **Promotion Threshold:** A model is eligible for platform advancement only when it demonstrates positive skill ($S_{\text{persist}} > 0.0\%$) across all three activity regimes on an independent out-of-epoch evaluation block.
