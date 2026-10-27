# Phase 4S: Controlled Chronological Evaluation Protocol

**Project:** Solar AI Flare  
**Backend:** `backend/`  
**Phase:** Phase 4S — May 2024 Real-Data Ingestion, Scientific Validation, and Controlled Forecast Training  
**Protocol Version:** `phase4s_protocol_v1.0`  
**Status:** `ACTIVE_AND_ENFORCED`

---

## 1. Executive Summary

This document establishes the experimental protocol for self-supervised sequence-to-sequence forecasting on genuine SoLEXS SDD2 Level-1 count rates. The protocol guarantees absolute temporal independence, complete quarantine of final test data during training and hyperparameter tuning, strict scaler isolation, and rigorous baseline benchmarking across diverse solar eruptive activity regimes.

---

## 2. Walk-Forward Chronological Partitioning

To avoid look-ahead bias and data leakage, data partitioning strictly obeys chronological sequence boundaries. No random shuffles, k-fold splits, or bidirectional convolutions across temporal boundaries are permitted:

```
[--- Training: May 8-12 (5 days) ---] [Val: May 13 (1 day)] [--- Held-Out Test: May 14-15 (2 days) ---]
       431,069 sliding windows             86,185 windows                  172,439 sliding windows
```

### Partition Allocations

| Split Partition | Dates Covered | Duration | Finite Records | Sliding Windows ($L_{in}=60, H=10$) | Primary Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Training** | May 8–12, 2024 | 5 days | 431,977 | **431,069** | Model parameter optimization via backpropagation. |
| **Validation** | May 13, 2024 | 1 day | 86,396 | **86,185** | Epoch checkpoint selection; model restoration on best val loss. |
| **Quarantined Test** | May 14–15, 2024 | 2 days | 172,793 | **172,439** | Held-out benchmark; evaluated only after training is completed. |
| **Total** | May 8–15, 2024 | 8 days | 691,166 | **689,693** | Unbroken chronological sequence across 21 valid segments. |

---

## 3. Strict Scaler Isolation Protocol

Feature normalization must never introduce statistical parameters computed over unseen future distributions.

1. **Scaler Formulation:** $\log_{10}(x + 1)$ logarithmic transform followed by z-score standardization:
   $$z = \frac{\log_{10}(x + 1) - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
2. **Fitting Protocol:** $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ are calculated **strictly from the finite records of the May 8–12 training partition**:
   $$\mu_{\text{train}} = 2.7411, \quad \sigma_{\text{train}} = 0.4569$$
3. **Quarantine Enforcement:** Zero observations from May 13 (Validation) or May 14–15 (Test) are passed to `scaler.fit()`.
4. **Inverse Transformation:** Predictions generated in normalized space are inverted strictly via:
   $$\hat{x} = 10^{(z \cdot \sigma_{\text{train}} + \mu_{\text{train}})} - 1$$
   with hard clipping at $0.0\text{ counts/s}$ to preserve physical non-negativity.

---

## 4. Benchmark Models & Transfer Evaluation

All models are evaluated on identical, sample-matched test window cohorts:

1. **Phase 4S Direct CNN-LSTM:**
   - 1D Convolution (32 channels, kernel size 3) + ReLU + MaxPool(2) + 2-layer LSTM (hidden size 64, dropout 0.2) + Direct Linear Output ($H=10$).
   - Trained on May 8–12 solar maximum data; restored to best validation checkpoint (epoch 2, val loss = 0.006586).
2. **Persistence Baseline (Physical Zero-Skill Reference):**
   - $\hat{y}_{t+h} = x_t$ for all horizons $h \in \{1, \dots, 10\}$.
3. **Rolling Mean Baseline:**
   - $\hat{y}_{t+h} = \frac{1}{60} \sum_{i=0}^{59} x_{t-i}$ across the 60s input history.
4. **Phase 4O Reference Model Transfer:**
   - Frozen checkpoint `solexs_phase4o_cnn_lstm_direct_20261025T184851Z.pt` trained on February 2026 data.
   - Evaluated with its native frozen scaler parameters ($\mu = 2.2473, \sigma = 0.6966$) directly on the May 14–15 test partition without retraining, testing domain transfer across 27 months.

---

## 5. Cohort Stratification & Metric Definitions

To prevent aggregate metrics from masking performance differences between baseline noise and extreme solar flares, evaluations are stratified into 7 distinct cohorts:

1. `Combined_Test_May14_15`: Full quarantined test partition (172,439 samples).
2. `Day7_May14_X8_7_Flare`: Peak eruptive day featuring the historic X8.7 solar flare (86,220 samples).
3. `Day8_May15_Active_Decay`: Active post-eruptive decay day (86,219 samples).
4. `Regime_REG_QUIET_lt200`: Quiescent coronal background where target count rate $< 200\text{ c/s}$.
5. `Regime_REG_MOD_200_1000`: Moderate solar activity where target count rate $\in [200, 1000]\text{ c/s}$.
6. `Regime_REG_HIGH_gt1000`: Extreme coronal flare emission where target count rate $> 1000\text{ c/s}$.
7. `Rising_Surge_DeltaC_gt50`: Rapid impulsive rising intervals where $x_t - x_{t-10} > 50\text{ c/s}$.

### Mathematical Metrics

For each cohort and forecast horizon $h \in \{1, \dots, 10\}$:

- **Mean Absolute Error (MAE):** $\text{MAE}_h = \frac{1}{N} \sum_{i=1}^N |y_{i, h} - \hat{y}_{i, h}|$
- **Root Mean Squared Error (RMSE):** $\text{RMSE}_h = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_{i, h} - \hat{y}_{i, h})^2}$
- **Relative MAE (%):** $\text{RelMAE}_h = \frac{\text{MAE}_h}{\bar{y}} \times 100\%$
- **Skill Score vs Persistence (%):**
  $$\text{Skill}_{\text{persist}, h} = \left( 1 - \frac{\text{MAE}_{\text{model}, h}}{\text{MAE}_{\text{persist}, h}} \right) \times 100\%$$
- **Skill Score vs Rolling Mean (%):**
  $$\text{Skill}_{\text{rolling}, h} = \left( 1 - \frac{\text{MAE}_{\text{model}, h}}{\text{MAE}_{\text{rolling}, h}} \right) \times 100\%$$
