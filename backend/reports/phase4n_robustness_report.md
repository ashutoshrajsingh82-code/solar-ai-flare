# Phase 4N: Robustness, Generalization & Failure Mode Analysis Report

**Document Version:** 1.0.0  
**Audit Date:** 2026-10-25  
**Canonical Designation:** `PHASE4N_ROBUSTNESS_AND_FAILURE_MODE_REPORT`  
**Parent Phase:** Phase 4N (Independent Reproducibility, Robustness & Generalization Validation)  
**Evaluated Cohort:** 6 Daily Aditya-L1 SoLEXS SDD2 Observational Products (`2026-01-31` to `2026-02-05`)  

---

## 1. Executive Summary

While the Phase 4M-S self-supervised CNN-LSTM model successfully reproduced its reported held-out test metrics on 2026-02-05, an empirical evaluation of model robustness across different solar regimes and chronological periods reveals significant operational boundaries:

1. **Generalization Scope Limitation:** No untouched observational periods beyond 2026-02-05 currently exist in `E:\SolarFlareData` or downloaded local archives. All available candidate data from ISRO PRADAN spans the single 6-day epoch (2026-01-31 to 2026-02-05).
2. **Count-Rate Regime Dependency:** The model exhibits strong positive skill over persistence (+14% to +25%) during quiet-to-moderate solar conditions (<500 counts/s). However, in the extreme count-rate flare regime (>1,000 counts/s), **model skill becomes negative (-2.76%)**, with persistence providing superior short-term forecasts.
3. **Eruptive Transition Inertia:** Across all evaluated days, major flare events (e.g., the M-class event on 2026-02-01 and high-flux post-flare decay on 2026-02-02) cause severe neural lag, resulting in persistence outperforming the model by over 100% on those eruptive dates.

---

## 2. Daily Coverage and Chronological Robustness

Every daily Level-1 FITS file in the SoLEXS archive was evaluated with the frozen Phase 4M-S checkpoint.

### 2.1 Daily Observational Statistics and Coverage Table

| Observation Date | Partition in Phase 4M-S | Duty Cycle | Contiguous Segments | Evaluated Windows (60s -> 10s) | Mean Target Rate (`counts/s`) | Model MAE (`counts/s`) | Persist MAE (`counts/s`) | Rolling Mean MAE (`counts/s`) | Skill Score vs Persistence | Skill Score vs Rolling Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| **2026-01-31** | Train (Quiet Sun) | 99.99% | 3 | 86,179 | 77.89 | **7.22** | 9.44 | 7.58 | **+23.43%** | **+4.68%** |
| **2026-02-01** | Train (M-Class Flare) | 52.24% | 4 | 44,087 | 570.73 | 68.46 | **23.94** | 48.12 | **-185.97%** | -42.25% |
| **2026-02-02** | Train (Post-Flare High Flux) | 52.04% | 3 | 44,068 | 2478.61 | 120.07 | **50.58** | 89.69 | **-137.39%** | -33.87% |
| **2026-02-03** | Validation (High Activity) | 5.91% | 1 | 4,308 | 1226.02 | 39.60 | **38.41** | 40.32 | **-3.09%** | **+1.80%** |
| **2026-02-04** | Validation (Recovery) | 52.45% | 3 | 44,215 | 369.50 | **18.39** | 18.52 | 24.56 | **+0.68%** | **+25.12%** |
| **2026-02-05** | Held-Out Test (C-Class Activity) | 52.44% | 3 | 44,221 | 281.60 | **14.03** | 15.93 | 14.77 | **+11.90%** | **+5.03%** |

*Companion CSV files:* [`phase4n_daily_coverage.csv`](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/reports/phase4n_daily_coverage.csv) and [`phase4n_generalization_metrics.csv`](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/reports/phase4n_generalization_metrics.csv).

### 2.2 Physical Explanation of Cross-Day Behavior
1. **Quiet Conditions (2026-01-31 & 2026-02-05):** When solar activity is steady or dominated by micro-variability (mean rates 78–282 counts/s), the CNN temporal convolutions and LSTM recurrent state extract smooth autocorrelation patterns, outperforming persistence by +12% to +23%.
2. **Major Solar Eruption (2026-02-01 & 2026-02-02):** When an M-class flare erupted on Feb 1, count rates spiked by three orders of magnitude from ~100 to 27,465 counts/s. During rapid impulsive rises and high-amplitude post-flare turbulence:
   - The model's $\log_{10}(x+1)$ transform compresses high rates, while its smooth MSE loss penalizes oscillations.
   - The model introduces an inertial prediction lag (under-forecasting the rise and over-forecasting the decay).
   - Persistence ($r_{T+k} = r_T$) provides an instantaneous benchmark that reacts instantaneously, producing less than half the MAE of the neural forecaster on those days.

---

## 3. Generalization Breakdown on Held-Out Test Set (2026-02-05)

### 3.1 Performance by Count-Rate Quantile

Evaluating the 44,221 held-out test windows across count-rate ranges highlights the regime boundary where neural forecasting ceases to add value over simple persistence:

| Count-Rate Range (`counts/s`) | Window Count ($N$) | Mean Rate (`counts/s`) | Model MAE (`counts/s`) | Persistence MAE (`counts/s`) | Relative MAE (%) | Skill vs Persistence | Regimes / Behavior |
|---|---|---|---|---|---|---|---|
| **$0 \le r < 50$** | 10,721 (24.2%) | 35.4 | **5.00** | 6.71 | 14.13% | **+25.52%** | Quiet background |
| **$50 \le r < 100$** | 12,395 (28.0%) | 70.5 | **7.20** | 9.56 | 10.21% | **+24.69%** | Moderate background |
| **$100 \le r < 250$** | 7,706 (17.4%) | 149.3 | **11.33** | 13.95 | 7.59% | **+18.83%** | Active region emission |
| **$250 \le r < 500$** | 6,585 (14.9%) | 372.7 | **19.12** | 22.25 | 5.13% | **+14.04%** | Pre-flare enhancement |
| **$500 \le r < 1000$** | 2,768 (6.3%) | 646.3 | **29.22** | 29.33 | 4.52% | **+0.40%** | Flare transition threshold |
| **$r \ge 1000$** | 4,046 (9.2%) | 1,435.1 | 45.38 | **44.16** | 3.16% | **-2.76%** | **Flare peak regime (Failure Mode)** |

> [!WARNING]
> **Key Empirical Discovery:** Model skill is strictly positive for all count rates below 1,000 counts/s. At $r \ge 1,000\text{ counts/s}$, persistence beats the CNN-LSTM. The model should not be deployed for unassisted forecasting during flare peak conditions without a hybrid persistence-handover rule.

### 3.2 Performance in Stable vs. Rapidly Changing Regimes

Using an operational definition of volatility based on the relative standard deviation of the 60-second input window ($\sigma_{\text{input}} / \mu_{\text{input}}$):
- **Stable Regime:** Volatility $\le 50\text{th}$ percentile ($\le 0.1083$)
- **Moderate Regime:** $50\text{th} < \text{Volatility} \le 90\text{th}$ percentile ($0.1083 < \text{vol} \le 0.1794$)
- **Rapidly Changing Regime:** Volatility $> 90\text{th}$ percentile ($> 0.1794$)

| Dynamical Regime | Window Count ($N$) | Mean Rate (`counts/s`) | Model MAE (`counts/s`) | Persistence MAE (`counts/s`) | Relative MAE (%) | Skill vs Persistence |
|---|---|---|---|---|---|---|
| **Stable ($\le p50$)** | 22,111 | 501.9 | **21.38** | 23.18 | 4.26% | **+7.79%** |
| **Moderate ($p50-p90$)** | 17,688 | 65.5 | **6.81** | 8.98 | 10.40% | **+24.22%** |
| **Rapidly Changing ($> p90$)** | 4,422 | 47.1 | **6.20** | 7.43 | 13.17% | **+16.51%** |

*Interpretation:* The model successfully captures volatility in low-to-moderate count-rate regimes (e.g. Poisson fluctuation and small active region oscillations), maintaining +16.5% skill. The degradation seen in §3.1 occurs specifically when rapid change is paired with massive physical count-rate surges ($>1,000\text{ counts/s}$).

---

## 4. Multi-Horizon Step Error Progression

The model predicts 10 forward time steps ($t+1\text{s}$ through $t+10\text{s}$). Error metrics and skill scores were audited across each step:

| Forecast Step | Horizon ($\Delta t$) | Model MAE (`counts/s`) | Persistence MAE (`counts/s`) | Rolling Mean MAE (`counts/s`) | Skill vs Persistence | Skill vs Rolling Mean |
|---|---|---|---|---|---|---|
| **$t+1$** | 1 second | 14.02 | 15.81 | 14.11 | **+11.31%** | +0.63% |
| **$t+2$** | 2 seconds | 14.12 | 15.83 | 14.25 | **+10.79%** | +0.89% |
| **$t+3$** | 3 seconds | 13.62 | 15.88 | 14.40 | **+14.20%** | +5.38% |
| **$t+4$** | 4 seconds | 13.56 | 15.75 | 14.54 | **+13.88%** | +6.73% |
| **$t+5$** | 5 seconds | 13.71 | 15.91 | 14.69 | **+13.82%** | +6.68% |
| **$t+6$** | 6 seconds | 13.95 | 15.93 | 14.84 | **+12.46%** | +6.03% |
| **$t+7$** | 7 seconds | 14.08 | 15.97 | 15.00 | **+11.85%** | +6.11% |
| **$t+8$** | 8 seconds | 14.29 | 15.97 | 15.15 | **+10.48%** | +5.65% |
| **$t+9$** | 9 seconds | 14.37 | 16.03 | 15.31 | **+10.36%** | +6.13% |
| **$t+10$** | 10 seconds | 14.59 | 16.21 | 15.47 | **+9.95%** | +5.65% |

**Key Observations:**
1. **Persistent Advantage:** The model outperforms both persistence and rolling mean at every single forecast step.
2. **Horizon Error Growth:** Model MAE rises gently from 14.02 at $t+1$ to 14.59 at $t+10$ (a 4.1% increase over 10 seconds).
3. **Rolling Mean Degradation:** Rolling mean baseline deteriorates steadily from 14.11 at $t+1$ to 15.47 at $t+10$ (a 9.6% degradation), confirming that averaging past history becomes increasingly suboptimal for forward projection.

---

## 5. Residual Diagnostics and Error Distribution

Evaluating the full residual vector $\mathbf{e} = \hat{\mathbf{y}} - \mathbf{y}$ across all $442,210$ forecast points on 2026-02-05:
- **Mean Residual Bias:** $-0.4221\text{ counts/s}$ (minimal negative bias, indicating slight under-prediction).
- **Residual Standard Deviation:** $23.32\text{ counts/s}$
- **Median Residual:** $-0.6171\text{ counts/s}$

### Residual Quantiles
- **$1^{\text{st}}$ Percentile (Under-prediction tail):** $-83.26\text{ counts/s}$
- **$5^{\text{th}}$ Percentile:** $-31.89\text{ counts/s}$
- **$25^{\text{th}}$ Percentile:** $-8.26\text{ counts/s}$
- **$50^{\text{th}}$ Percentile:** $-0.62\text{ counts/s}$
- **$75^{\text{th}}$ Percentile:** $+7.31\text{ counts/s}$
- **$95^{\text{th}}$ Percentile:** $+35.46\text{ counts/s}$
- **$99^{\text{th}}$ Percentile (Over-prediction tail):** $+69.31\text{ counts/s}$

The central 50% of forecast errors fall within $[-8.26, +7.31]\text{ counts/s}$, demonstrating tight calibration in typical conditions. The heavy negative tail ($-83.26\text{ counts/s}$) reflects the under-prediction of steep solar flare ascents.

---

## 6. Case Study: Top Large-Error Windows

Inspecting the 10 windows with the largest mean absolute error:

| Timestamp (UTC) | Model MAE (`counts/s`) | Persist MAE (`counts/s`) | Forecast Origin Rate | Target Mean Rate | Predicted Mean Rate | Physical Solar Event |
|---|---|---|---|---|---|---|
| **2026-02-05 04:32:55** | 185.3 | 144.7 | 1,653.0 | 1,797.7 | 1,612.4 | Rapid C-class flare ramp-up |
| **2026-02-05 04:33:20** | 181.9 | 175.0 | 1,627.0 | 1,802.0 | 1,620.1 | Rapid flare ramp-up |
| **2026-02-05 04:33:55** | 153.7 | 127.0 | 1,693.0 | 1,820.0 | 1,666.3 | Near flare peak |
| **2026-02-05 04:34:02** | 144.6 | 110.5 | 1,780.0 | 1,890.5 | 1,745.9 | Near flare peak |
| **2026-02-05 04:27:57** | 141.6 | 117.7 | 1,569.0 | 1,686.7 | 1,545.1 | Early impulsive rise |
| **2026-02-05 04:29:43** | 140.4 | 114.2 | 1,594.0 | 1,708.2 | 1,567.8 | Early impulsive rise |
| **2026-02-05 04:21:47** | 137.5 | 121.3 | 1,563.0 | 1,684.3 | 1,546.8 | Onset of flare event |
| **2026-02-05 04:35:27** | 137.3 | 114.2 | 1,795.0 | 1,909.2 | 1,771.9 | Flare maximum |
| **2026-02-05 04:43:09** | 134.8 | 117.0 | 1,486.0 | 1,603.0 | 1,468.2 | Early flare decay |
| **2026-02-05 04:31:23** | 134.7 | 127.1 | 1,618.0 | 1,745.1 | 1,610.4 | Rapid impulsive rise |

**Key Diagnostic Insight:** All 10 largest error cases occurred within a single 22-minute window on `2026-02-05` between `04:21 UTC` and `04:43 UTC`. During this event, SoLEXS detected a C-class solar flare where flux rose from ~1,500 to ~1,910 counts/s. Because the model predicts a forward mean of ~1,612 counts/s while the actual count rate surged to ~1,798 counts/s, it lagged by approximately 180 counts/s. Persistence also had large errors (~145 counts/s), but by retaining the most recent rate, it lagged less than the neural model.

---

## 7. Time-Aware Uncertainty Quantification

Because high-cadence solar time series exhibit strong serial autocorrelation, standard i.i.d. Gaussian confidence intervals underestimate interval coverage during flare transitions. 

Using empirical block-quantile prediction intervals calibrated on the held-out test series:
- **90% Empirical Error Interval:** $\hat{y}_{T+h} \in [\hat{y} - 31.89, \hat{y} + 35.46]\text{ counts/s}$
- **95% Empirical Error Interval:** $\hat{y}_{T+h} \in [\hat{y} - 52.40, \hat{y} + 54.10]\text{ counts/s}$

*Methodological Limitation:* Empirical conformal intervals calibrated on 2026-02-05 provide valid coverage during quiet-to-moderate solar conditions, but underestimate uncertainty during M/X-class eruptive events (such as 2026-02-01). Operational deployment requires regime-dependent interval scaling based on background count-rate variance.
