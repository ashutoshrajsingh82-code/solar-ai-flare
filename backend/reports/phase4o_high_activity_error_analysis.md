# Phase 4O: High-Activity Forecast Error Diagnostics & Regime Analysis

**Document Version:** 1.0.0  
**Audit Date:** 2026-10-26  
**Canonical Designation:** `PHASE4O_HIGH_ACTIVITY_ERROR_ANALYSIS`  
**Parent Phase:** Phase 4O (High-Activity Forecasting Improvement & Leakage-Safe Model Comparison)  
**Evaluated Cohorts:** Chronological Training Partition (`Jan 31 – Feb 2`) and Validation Partition (`Feb 3 – Feb 4`)  

---

## 1. Executive Summary

This diagnostic investigation evaluates the failure modes of the reference SoLEXS SDD2 autoregressive forecasting model during high solar activity, steep rate-of-change sequences, and high-flux active region emissions. 

In strict adherence to Requirement 3, **all error diagnostics, thresholds, and regime definitions were established strictly using the training and validation cohorts**. The previously inspected February 5 test set was completely quarantined from model diagnostic decisions.

### Primary Diagnostic Discoveries
1. **The Stable-Regime Offset Trap:** On the validation cohort, the direct CNN-LSTM forecaster underperformed persistence during "Stable" regimes (Skill: **$-7.56\%$**, MAE: $25.87$ vs $24.05\text{ counts/s}$). In stable conditions, predicting raw absolute count rates through dense neural layers introduces small fractional errors that translate into large absolute count errors at high baselines, whereas persistence has zero drift.
2. **Severe Inversion Above 500 counts/s:** On the validation cohort, model skill is positive for all count rates below 500 counts/s (+12% to +25%), but inverts to **$-4.62\%$** in the 500–1,000 counts/s regime and collapses to **$-21.03\%$** at count rates $>1,000\text{ counts/s}$ (MAE: $64.29$ vs persistence $53.12\text{ counts/s}$).
3. **Prediction Lag on Impulsive Rises:** During steep rate-of-change events, the model's MSE loss function on log-compressed rates penalizes high-rate slopes insufficiently, causing the output curve to systematically lag behind impulsive solar ascents.

---

## 2. Training-Derived Regime Definitions

To avoid lookahead bias, all operational rate-of-change thresholds were calculated exclusively from the 174,334 training windows (`2026-01-31` to `2026-02-02`):

Let $\Delta_{10}(T) = r_T - r_{T-10}$ represent the 10-second change in native count rates prior to the forecast origin, and let $\text{rel\_change} = \Delta_{10} / \max(1.0, r_{T-10})$.
- **Training Percentiles:**
  - $10^{\text{th}}$ Percentile: $-16.13\%$
  - $25^{\text{th}}$ Percentile: $-6.67\%$
  - $50^{\text{th}}$ Percentile (Median): $0.00\%$
  - $75^{\text{th}}$ Percentile: $+7.28\%$
  - $90^{\text{th}}$ Percentile: $+19.44\%$

### Formal Operational Definitions
- **Rising Sequence:** $\text{rel\_change} > +10.0\%$ (Count rate increased by $>10\%$ over the preceding 10 seconds).
- **Stable Sequence:** $-10.0\% \le \text{rel\_change} \le +10.0\%$ (Rate within nominal Poisson fluctuation range).
- **Falling Sequence:** $\text{rel\_change} < -10.0\%$ (Count rate decayed by $>10\%$ over the preceding 10 seconds).

---

## 3. Error Breakdown by Operational Regimes (Validation Cohort: Feb 3 – Feb 4)

### 3.1 Error by Native Count-Rate Quantile

Evaluating the 48,523 validation windows across count-rate brackets:

| Native Count-Rate Bracket (`counts/s`) | Window Count ($N$) | Mean Rate (`counts/s`) | Direct CNN-LSTM MAE | Persistence MAE | Direct Model Skill vs Persistence | Physical Interpretation |
|---|---|---|---|---|---|---|
| **$0 \le r < 50$** | 11,272 (23.2%) | 33.4 | **4.82** | 6.45 | **+25.27%** | Quiet background |
| **$50 \le r < 100$** | 7,878 (16.2%) | 71.8 | **7.26** | 9.55 | **+23.98%** | Moderate background |
| **$100 \le r < 250$** | 10,113 (20.8%) | 171.9 | **12.32** | 14.97 | **+17.72%** | Active region emission |
| **$250 \le r < 500$** | 5,730 (11.8%) | 354.8 | **19.33** | 22.09 | **+12.49%** | Pre-flare enhancement |
| **$500 \le r < 1000$** | 7,470 (15.4%) | 723.7 | 33.11 | **31.65** | **-4.62%** | **Inversion Threshold** |
| **$r \ge 1000$** | 6,060 (12.5%) | 1,897.5 | 64.29 | **53.12** | **-21.03%** | **Severe Failure Regime** |

### 3.2 Error by Dynamical Trend Regime

| Dynamical Trend Regime | Window Count ($N$) | Mean Rate (`counts/s`) | Direct CNN-LSTM MAE | Persistence MAE | Skill Score vs Persistence |
|---|---|---|---|---|---|
| **Rising ($>+10\%$)** | 9,311 (19.2%) | 125.4 | **12.20** | 14.93 | **+18.27%** |
| **Stable ($-10\%\dots +10\%$)** | 30,811 (63.5%) | 602.8 | 25.87 | **24.05** | **-7.56%** |
| **Falling ($<-10\%$)** | 8,401 (17.3%) | 224.6 | **8.70** | 12.40 | **+29.84%** |

**Diagnostic Analysis of the Stable Regime Deficit:**
Notice that the mean count rate during the "Stable" regime on the validation set is **$602.8\text{ counts/s}$**, compared to $125.4\text{ counts/s}$ during Rising sequences. When a solar active region emits steady high flux ($\sim 1,000$ to $2,000\text{ counts/s}$), persistence anchors the forecast precisely to $r_T$. The direct neural forecaster attempts to generate absolute numbers through unconstrained dense weights, introducing level-prediction jitter that degrades its skill by $-7.56\%$.

---

## 4. Multi-Step Forecast Error Progression (Validation Set)

| Step | Horizon ($\Delta t$) | Direct CNN-LSTM MAE (`counts/s`) | Persistence MAE (`counts/s`) | Rolling Mean MAE (`counts/s`) | Skill vs Persistence |
|---|---|---|---|---|---|
| **$t+1$** | 1 second | 19.98 | 19.62 | 22.84 | -1.83% |
| **$t+2$** | 2 seconds | 20.08 | 19.57 | 23.47 | -2.61% |
| **$t+3$** | 3 seconds | 19.82 | 19.85 | 24.11 | **+0.15%** |
| **$t+4$** | 4 seconds | 19.85 | 19.97 | 24.78 | **+0.60%** |
| **$t+5$** | 5 seconds | 20.00 | 20.25 | 25.47 | **+1.23%** |
| **$t+6$** | 6 seconds | 20.24 | 20.48 | 26.20 | **+1.17%** |
| **$t+7$** | 7 seconds | 20.35 | 20.67 | 26.96 | **+1.55%** |
| **$t+8$** | 8 seconds | 20.55 | 20.84 | 27.76 | **+1.39%** |
| **$t+9$** | 9 seconds | 20.81 | 21.03 | 28.58 | **+1.05%** |
| **$t+10$** | 10 seconds | 21.07 | 21.26 | 29.45 | **+0.89%** |

*Key Takeaway:* In the very short horizons ($t+1$ and $t+2$), persistence is marginally superior on the validation set. Beyond $t+3$, the direct neural model gains slight positive skill over persistence, while rolling mean degrades rapidly across all steps.

---

## 5. Visual Diagnostics: Time-Series Case Studies

Representative forecast windows from the validation set were plotted and archived under `backend/artifacts/real_data_experiments/phase4o/plots/`:

1. **Quiet-Sun Validation Window:**  
   ![Quiet Sun Validation](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/artifacts/real_data_experiments/phase4o/plots/phase4o_quiet_sun_validation.png)  
   *Metadata:* Observation Date: 2026-02-04 | Origin: 02:45:00 UTC | Instrument: SoLEXS SDD2 (2–22 keV) | Split: Validation.  
   *Observation:* Under low flux (<40 counts/s), neural forecasts track the gentle decay accurately, smoothing out Poisson noise.

2. **Impulsive Rate Rise Window:**  
   ![Impulsive Rise Validation](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/artifacts/real_data_experiments/phase4o/plots/phase4o_impulsive_rise_validation.png)  
   *Metadata:* Observation Date: 2026-02-04 | Origin: 14:15:30 UTC | Instrument: SoLEXS SDD2 (2–22 keV) | Split: Validation.  
   *Observation:* When rate of change exceeds +25%, the direct neural forecaster lags behind the upward inflection, under-predicting the first 5 seconds of the burst.

3. **High-Flux Steady Window:**  
   ![High Flux Validation](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/artifacts/real_data_experiments/phase4o/plots/phase4o_high_flux_regime_validation.png)  
   *Metadata:* Observation Date: 2026-02-03 | Origin: 23:15:00 UTC | Instrument: SoLEXS SDD2 (2–22 keV) | Split: Validation.  
   *Observation:* At high baseline emission (>1,200 counts/s), small fractional offsets in the dense layer output generate steady ~50 counts/s absolute errors, while persistence matches the true plateau.
