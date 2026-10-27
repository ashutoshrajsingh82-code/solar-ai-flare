# Phase 4O: Controlled Model Comparison & Residual Architecture Analysis

**Document Version:** 1.0.0  
**Audit Date:** 2026-10-26  
**Canonical Designation:** `PHASE4O_EXPERIMENT_COMPARISON_REPORT`  
**Parent Phase:** Phase 4O (High-Activity Forecasting Improvement & Leakage-Safe Model Comparison)  
**Evaluated Cohort:** Validation Partition (`2026-02-03` to `2026-02-04`, $N = 48,523$ windows)  

---

## 1. Executive Summary

Phase 4O conducted a controlled model comparison to test whether residual forecasting heads or causal trend/volatility features could mitigate prediction lag and overcome the high-flux forecasting degradation identified in Phase 4N.

All models were evaluated under strict experimental controls:
- **Identical Ingestion:** Evaluated on the exact same 48,523 validation windows spanning February 3 and 4.
- **Identical Targets:** Evaluated against the exact same ground-truth 10-second target matrix in native detector units (`counts/s`, 2.0–22.0 keV).
- **Identical Normalization:** Scaler parameters ($\mu = 2.2472749, \sigma = 0.6966392$) fitted strictly on the training partition (`Jan 31 – Feb 2`).

### Key Finding
**Neither the residual CNN-LSTM nor the trend-aware residual CNN-LSTM improved upon the reference direct CNN-LSTM model on the validation cohort.** In accordance with the protocol decision rule (§4 of [`phase4o_evaluation_protocol.md`](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/reports/phase4o_evaluation_protocol.md)), the reference Phase 4M-S direct model (`cnn_lstm_direct`) was retained, and the final decision is certified as **`NO_RELIABLE_IMPROVEMENT_ESTABLISHED`**.

---

## 2. Validation Set Model Comparison Table

| Candidate Model | Architecture / Method | Validation MAE (`counts/s`) | Validation RMSE (`counts/s`) | Relative MAE (%) | Skill vs Persistence | Skill vs Rolling Mean | MAE at High Flux ($>1,000\text{ c/s}$) | MAE during Rising Trends ($>+10\%$) | MAE during Stable Trends ($\pm 10\%$) | Model Selection Outcome |
|---|---|---|---|---|---|---|---|---|---|---|
| **Persistence** | Physical baseline: $\hat{y}_{T+k} = r_T$ | 20.284 | **33.346** | 4.55% | 0.00% | +21.87% | **53.115** | 14.928 | **24.051** | Standard benchmark |
| **Rolling Mean** | Historical baseline: $\hat{y}_{T+k} = \bar{r}_{60}$ | 25.962 | 67.412 | 5.83% | -27.99% | 0.00% | 91.253 | 22.450 | 31.164 | Suboptimal on trends |
| **CNN-LSTM Direct** | Phase 4M-S direct sequence forecaster | **20.275** | 38.403 | **4.55%** | **+0.05%** | **+21.91%** | 64.287 | **12.200** | 25.870 | **SELECTED (Retained Reference)** |
| **CNN-LSTM Residual** | Residual head in scaled space: $\tilde{x}_T + \hat{\Delta}$ | 375.511 | 703.812 | 84.28% | -1,751.27% | -1,346.40% | 1,714.172 | 134.738 | 510.220 | Rejected (Multiplicative log error) |
| **Trend-Aware Residual** | 4-channel causal trend + residual head | 374.950 | 704.142 | 84.16% | -1,748.50% | -1,344.23% | 1,715.293 | 131.538 | 511.373 | Rejected (Multiplicative log error) |

*Companion CSV file:* [`phase4o_metrics.csv`](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/reports/phase4o_metrics.csv).

---

## 3. Mathematical Diagnosis: The Multiplicative Log-Space Trap

The controlled experiment revealed a critical mathematical pathology that explains why residual heads formulated in scaled log-space suffered catastrophic errors ($\text{MAE} \approx 375\text{ counts/s}$):

### 3.1 The Multiplicative Exponentiation Hazard
Let $x$ be the raw count rate and $\tilde{x} = \frac{\log_{10}(x+1) - \mu}{\sigma}$ be the scaled input.
In the residual model, the network predicted a delta $\hat{\Delta}$ in scaled space:
$$\hat{\tilde{y}} = \tilde{x}_T + \hat{\Delta}$$

When this is inverted back into physical detector units:
$$\hat{y} + 1 = 10^{(\hat{\tilde{y}} \cdot \sigma + \mu)} = 10^{(\tilde{x}_T \cdot \sigma + \mu + \sigma \hat{\Delta})} = (r_T + 1) \cdot 10^{\sigma \hat{\Delta}}$$
$$\hat{y} = (r_T + 1) \cdot 10^{\sigma \hat{\Delta}} - 1$$

Notice the mathematical consequence:
> **An additive delta in logarithmic space acts as a MULTIPLICATIVE SCALE FACTOR in physical count space!**

Because $\sigma = 0.6966$, a minor error in predicted delta of just $\hat{\Delta} = 0.50$ produces:
$$10^{\sigma \hat{\Delta}} = 10^{0.6966 \times 0.50} = 10^{0.3483} \approx 2.23$$
The forecast multiplies the current count rate by **2.23**:
- When $r_T = 100\text{ counts/s}$, $\hat{y} \approx 223$ (error: $+123\text{ counts/s}$).
- When $r_T = 2,000\text{ counts/s}$ (active region on Feb 3), $\hat{y} \approx 4,460$ (error: $+2,460\text{ counts/s}$)!

Even tiny neural fluctuations around zero delta exponentiate into massive physical count errors, causing the validation MAE to explode to $375\text{ counts/s}$.

### 3.2 Physical Lesson for Future Model Designs
Residual modeling relative to $r_T$ is fundamentally sound in physical intuition, but **it cannot be formulated as an addition in logarithmic space**. Any residual forecasting architecture must operate either:
1. Directly in linear count space using an adaptive Huber loss function that tolerates Poisson noise without log-transforming the base signal.
2. In log-space with a heavily constrained, bounded tanh output gating $|\hat{\Delta}| \le 0.05$ to prevent explosive multiplicative scaling.

---

## 4. Evaluation of the Retained Reference Model (`cnn_lstm_direct`)

Because neither candidate model surpassed the reference architecture, `cnn_lstm_direct` was retained as the optimal model. Analyzing its performance on the validation cohort:

1. **Overall Performance:**
   - MAE: **$20.275\text{ counts/s}$** (virtually identical to persistence at $20.284\text{ counts/s}$, $+0.05\%$ skill).
   - Relative MAE: **$4.55\%$** on mean flux of $445.5\text{ counts/s}$.
   - Skill vs Rolling Mean: **$+21.91\%$** (decisively beats historical window averaging).
2. **Rising Sequences:**
   - On the 9,311 windows where count rates rose by $>10\%$, `cnn_lstm_direct` achieved an MAE of **$12.200\text{ counts/s}$**, beating persistence ($14.928\text{ counts/s}$) by **$+18.27\%$**.
3. **High-Rate Regime Deficit:**
   - At count rates $>1,000\text{ counts/s}$, persistence achieved an MAE of $53.115\text{ counts/s}$, whereas `cnn_lstm_direct` achieved $64.287\text{ counts/s}$.
4. **Historical Test Comparison (Quarantined Feb 5 Benchmark):**
   - On February 5, `cnn_lstm_direct` maintains its verified Phase 4M-S benchmark metrics ($\text{MAE} = 14.032\text{ counts/s}$, $\text{RMSE} = 23.324\text{ counts/s}$, $\text{Skill} = +11.90\%$).

---

## 5. Artifact Delivery

The retrained reference checkpoint and provenance metadata were saved to:
- **Weights:** [`backend/artifacts/real_data_experiments/phase4o/solexs_phase4o_cnn_lstm_direct_20261025T184851Z.pt`](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/artifacts/real_data_experiments/phase4o/solexs_phase4o_cnn_lstm_direct_20261025T184851Z.pt)
- **Metadata:** [`backend/artifacts/real_data_experiments/phase4o/solexs_phase4o_cnn_lstm_direct_20261025T184851Z_metadata.json`](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/artifacts/real_data_experiments/phase4o/solexs_phase4o_cnn_lstm_direct_20261025T184851Z_metadata.json)
- **Visual Plots:** [`backend/artifacts/real_data_experiments/phase4o/plots/`](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/artifacts/real_data_experiments/phase4o/plots/)
