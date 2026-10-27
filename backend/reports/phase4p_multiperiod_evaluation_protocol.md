# Phase 4P: Multi-Period Observational Generalization Protocol

**Document ID:** `PHASE4P-PROTO-EVAL-001`  
**Phase:** Phase 4P — Real Observational Data Expansion & Multi-Period Generalization  
**Repository:** `C:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main`  
**Backend:** `C:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main\backend`  
**Target Application:** Multi-Month Solar Activity Regime Generalization for SoLEXS Autoregressive Forecasting  
**Execution Timestamp:** 2026-10-26T01:05:00 UTC  

---

## 1. Executive Summary & Scientific Motivation

In Phase 4N, bit-for-bit reproducibility of the Phase 4M-S self-supervised model was successfully established ($\Delta = 0.00\,\text{counts/s}$), but rigorous out-of-cohort evaluation revealed critical generalization constraints:
1. **Regime Inversion:** In the high-flux flaring regime ($> 1,000\,\text{counts/s}$), the reference CNN-LSTM model exhibited negative skill against persistence ($-2.76\%$), caused by phase lag and under-prediction of rapid count-rate surges.
2. **Evaluation Window Bias:** Evaluating exclusively on a single held-out test day (2026-02-05) reflects only a narrow solar condition (moderate quiet background, mean rate $275.1\,\text{counts/s}$). Six days of observation cannot validate generalization across multi-month solar cycle variations.

This protocol establishes a standardized, leakage-safe **Multi-Period Observational Generalization Protocol**. It defines regime stratification, rolling-origin walk-forward splits, scaler isolation, and rigorous multi-horizon benchmarks for future model assessment once additional observation epochs are acquired.

---

## 2. Solar Activity Regime Stratification

To prevent aggregate metric dilution where thousands of quiet-Sun points mask poor performance during solar flares, future multi-period datasets must be stratified into three distinct physical regimes:

| Regime ID | Regime Classification | SoLEXS SDD2 Count-Rate Range | Physical Characteristic | Scientific Evaluation Priority |
|---|---|---|---|---|
| **`REG-QUIET`** | Quiet-Sun Background | $c(t) < 200\,\text{counts/s}$ | Low-noise, stationary Poisson background | Variance reduction, baseline stability |
| **`REG-MOD`** | Moderate Active Region | $200 \le c(t) \le 1,000\,\text{counts/s}$ | Plage heating, microflare variability | Trend tracking, non-lagged response |
| **`REG-HIGH`** | High-Flux Eruptive Flaring | $c(t) > 1,000\,\text{counts/s}$ | Major flare impulsive peaks and decay phases | Peak amplitude capture, surge responsiveness |

Every evaluation window $[t+1, \dots, t+10]$ must be tagged by the maximum true target value:
$$\text{Regime} = \begin{cases} 
\text{REG-HIGH} & \text{if } \max_{h \in [1,10]} y_{t+h} > 1,000\,\text{counts/s} \\ 
\text{REG-MOD} & \text{if } 200 \le \max_{h \in [1,10]} y_{t+h} \le 1,000\,\text{counts/s} \\ 
\text{REG-QUIET} & \text{if } \max_{h \in [1,10]} y_{t+h} < 200\,\text{counts/s} 
\end{cases}$$

---

## 3. Leakage-Safe Rolling-Origin Partitioning

Cross-validation with random shuffling is strictly prohibited for solar time series due to high autocorrelation. Instead, a **Rolling-Origin Walk-Forward Split** must be enforced across multi-period observation blocks:

```
Block 1 (Month 1): [====== Train ======] [= Val =] [= Test =]
Block 2 (Month 2): [============= Train =============] [= Val =] [= Test =]
Block 3 (Month 3): [==================== Train ====================] [= Val =] [= Test =]
```

### Partitioning Rules
1. **Temporal Monotonicity:** $\text{Date}(\text{Train}) < \text{Date}(\text{Validation}) < \text{Date}(\text{Test})$. No window in the training set may contain timestamps greater than or equal to the earliest timestamp in the validation set.
2. **Gap Buffering:** To prevent boundary leakage across contiguous segments, an explicit buffer equal to the total window length ($T_{\text{in}} + T_{\text{out}} = 70\,\text{s}$) must be discarded between adjacent splits if they occur within the same observation file.
3. **Scaler Fitting Isolation:** 
   $$\mu_{\text{train}}, \sigma_{\text{train}} = \text{Fit}(\text{Train Set})$$
   The validation and test sets must be transformed strictly using the training statistics $(\mu_{\text{train}}, \sigma_{\text{train}})$. Fitting scalers on validation or test partitions is treated as fatal pipeline leakage.

---

## 4. Evaluation Window & Target Specification

- **Input History Window ($T_{\text{in}}$):** 60 seconds of consecutive 1 Hz observations:
  $$\mathbf{x}_t = [c_{t-59}, c_{t-58}, \dots, c_t]^T \in \mathbb{R}^{60}$$
- **Forecast Horizon ($T_{\text{out}}$):** 10 seconds ahead:
  $$\mathbf{y}_t = [c_{t+1}, c_{t+2}, \dots, c_{t+10}]^T \in \mathbb{R}^{10}$$
- **Contiguity Requirement:** No window $(\mathbf{x}_t, \mathbf{y}_t)$ may span a missing value (NaN) or a telemetry discontinuity ($\Delta t > 1.0\,\text{s}$).

---

## 5. Standardized Benchmark Baselines

Any candidate forecasting model must be benchmarked against two non-parametric baselines evaluated on identical target vectors $\mathbf{y}_t$:

### 5.1 Persistence Baseline
Assumes the solar count rate remains frozen at the final observed second $c_t$:
$$\hat{\mathbf{y}}_t^{\text{persist}} = [c_t, c_t, \dots, c_t]^T \in \mathbb{R}^{10}$$

### 5.2 Rolling Mean Baseline
Projects the moving average of the preceding $W = 10\,\text{s}$ input window across the horizon:
$$\hat{y}_{t+h}^{\text{mean}} = \frac{1}{10} \sum_{k=0}^{9} c_{t-k}, \quad \forall h \in \{1, \dots, 10\}$$

---

## 6. Mathematical Evaluation Metrics

All metrics must be computed in native observational units ($\text{counts/s}$). Converting predictions to log-space or physical flux prior to metric evaluation is prohibited.

1. **Mean Absolute Error (MAE):**
   $$\text{MAE}(h) = \frac{1}{N} \sum_{i=1}^{N} |y_{i, t+h} - \hat{y}_{i, t+h}|$$
2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE}(h) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_{i, t+h} - \hat{y}_{i, t+h})^2}$$
3. **Forecast Skill Score relative to Persistence ($S_{\text{persist}}$):**
   $$S_{\text{persist}}(h) = \left( 1 - \frac{\text{MAE}_{\text{model}}(h)}{\text{MAE}_{\text{persistence}}(h)} \right) \times 100\%$$
   - A model only demonstrates true forecasting skill when $S_{\text{persist}} > 0.0\%$.
   - A model with $S_{\text{persist}} \le 0.0\%$ performs worse than trivially repeating the latest observation.

---

## 7. Model Comparison & Production Eligibility Protocol

Before any model checkpoint is considered for deployment or promotion:
1. **Multi-Regime Superiority:** The model must demonstrate positive skill ($S_{\text{persist}} > 0.0\%$) across all three stratified regimes (`REG-QUIET`, `REG-MOD`, and `REG-HIGH`).
2. **Production Registry Isolation:** Under scientific safeguard protocols, all experimental models remain isolated with:
   ```json
   {
     "production_registry_eligible": false,
     "experimental_marker": "PHASE4P_RESEARCH_ONLY"
   }
   ```
3. **Zero Synthetic Negative Assertions:** In accordance with blocker `B-LBL-NEG-007`, evaluation must never compute classification metrics (F1-score, precision, recall) against synthetic quiet-Sun negatives.
