# Phase 4T: Scientific Claims, Cohort Conditioning, and Statistical Robustness Audit

**Project:** Solar AI Flare  
**Backend:** `backend/`  
**Phase:** Phase 4T — Independent Reproducibility Audit and Forecasting-Skill Verification  
**Audit Date:** 2026-10-26  
**Auditor:** Independent Automated Audit Subsystem  
**Audit Finding:** `TARGET_CONDITIONING_FLAGGED_CAUSAL_SKILL_ROBUST_BUT_PERSISTENCE_DOMINATES_OVERALL`

---

## 1. Executive Summary

This audit critically evaluates the scientific definitions, statistical methodologies, and substantive conclusions presented in Phase 4S. 

### Core Audit Discoveries:
1. **Target-Conditioned Leakage in Phase 4S Regime Definition:** The Phase 4S regime cohorts (`REG_QUIET`, `REG_MOD`, `REG_HIGH`) were constructed by inspecting future target values (`np.max(y_test, axis=1)`), not the causal state at forecast origin ($x_t$). While acceptable as an ex-post diagnostic of future states, this conditioning constitutes look-ahead bias if treated as an operational decision gate.
2. **Causal Re-Conditioning Verification:** When regimes are independently re-defined using strictly causal forecast-origin observations ($x_t \in [200, 1000]\text{ c/s}$), the **+22% skill score over persistence is robustly preserved** (+22.70% on May 14, +21.36% on May 15).
3. **Time-Aware Block Bootstrap (95% CI):** Standard i.i.d. bootstrapping on 1-second overlapping sliding windows is mathematically invalid due to 69-second serial autocorrelation. A 300-second block bootstrap confirms that the moderate activity advantage has a 95% confidence interval of **[+21.40%, +22.89%]**.
4. **Overall Physical Reality — Persistence Decisively Dominates:** Despite the moderate-regime advantage, **Persistence overwhelmingly dominates the full held-out test partition** ($h=10\text{s}$: Persistence MAE 37.36 c/s vs CNN-LSTM MAE 78.86 c/s; skill = -111.07%, 95% CI [-188.67%, -29.79%]).
5. **Rejection of Premature Generalization Claims:** Claims that May 2024 training proves multi-year solar cycle generalization are firmly rejected. An 8-day single-epoch experiment cannot establish solar cycle invariance.

---

## 2. Cohort Conditioning Audit: Target-Based vs Causal Origin-Based

In [`backend/scripts/run_phase4s_experiments.py`](file:///c:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/scripts/run_phase4s_experiments.py#L362-L369), the code defined regime membership as:

```python
# Phase 4S Original Implementation (Target-Conditioned)
peak_target = np.max(y_test, axis=1)  # LOOK-AHEAD: uses future 10s target
cohorts_eval["Regime_REG_QUIET_lt200"] = (peak_target < 200.0)
cohorts_eval["Regime_REG_MOD_200_1000"] = (peak_target >= 200.0) & (peak_target <= 1000.0)
cohorts_eval["Regime_REG_HIGH_gt1000"] = (peak_target > 1000.0)
```

### Critical Audit Assessment:
- **Conditioning on the Future:** Grouping windows based on the maximum value attained during the future forecast window ($t+1, \dots, t+10$) requires knowing the future before making the forecast. In an operational context, an alert or forecasting pipeline cannot inspect $y_{t+1:t+10}$ to select which model or regime to use.
- **Independent Causal Formulation:** The audit independently implemented a strictly causal, origin-conditioned regime stratification:
  $$x_t = X_{\text{test}}[:, -1, 0]$$
  - `Causal_REG_QUIET_lt200`: $x_t < 200.0\text{ counts/s}$
  - `Causal_REG_MOD_200_1000`: $200.0 \le x_t \le 1000.0\text{ counts/s}$
  - `Causal_REG_HIGH_gt1000`: $x_t > 1000.0\text{ counts/s}$

### Comparative Empirical Results ($t+10\text{s}$)

| Stratification Conditioning | Cohort Definition | Sample Size ($N$) | Phase 4S CNN-LSTM MAE | Persistence MAE | Skill Score vs Persistence |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Phase 4S Target-Conditioned** | `Regime_REG_MOD_200_1000` ($y_{\max} \in [200, 1000]$) | 89,452 | **17.44 c/s** | 22.45 c/s | **+22.30%** |
| **Audit Causal Origin-Conditioned** | `Causal_REG_MOD_200_1000` ($x_t \in [200, 1000]$) | 77,643 | **18.57 c/s** | 23.86 c/s | **+22.17%** |

> [!NOTE]
> Re-conditioning on the observable state at forecast origin reduces the sample size from 89,452 to 77,643 (because windows starting below 200 c/s that surge into moderate activity, or starting in moderate activity and surging into extreme activity, are appropriately assigned to their causal origin). However, **the skill score remains essentially identical (+22.17% causal vs +22.30% target-conditioned)**.

---

## 3. Subgroup Stability Across Individual Days & Segments

To test whether the moderate-regime advantage was driven by a single isolated event or represents consistent behavior, the causal moderate cohort was evaluated across separate solar days:

| Test Day | Flare Context | Causal Moderate Samples ($N$) | CNN-LSTM MAE (`c/s`) | Persistence MAE (`c/s`) | Skill Score vs Persistence |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **May 14, 2024** | Peak Solar Max Day (Historic X8.7) | 51,304 | 17.98 | 23.26 | **+22.70%** |
| **May 15, 2024** | Active Post-Eruptive Coronal Decay | 26,339 | 19.67 | 25.01 | **+21.36%** |

**Conclusion:** The moderate activity skill is **reproducible across both days independently**. It is not an artifact of a single segment or a localized burst.

---

## 4. Time-Aware Block Bootstrap (Autocorrelation-Corrected)

### Why Standard I.I.D. Bootstrapping Fails:
Sliding windows sampled at a 1-second cadence from a 70-second window ($L_{\text{in}} + H = 60 + 10$) share 69 seconds of identical physical measurements. Treating consecutive windows as independent identically distributed (i.i.d.) observations produces artificially deflated standard errors and falsely tight confidence bounds.

### Time-Aware Block Bootstrap Protocol:
- **Resampling Unit:** Non-overlapping blocks of $L_{\text{block}} = 300\text{ seconds}$ (5 minutes), spanning the full 172,439-sample test partition across 574 blocks.
- **Iterations:** $B = 500$ bootstrap replicates with replacement.
- **Statistic:** Empirical Skill Score vs Persistence:
  $$\text{Skill}_b = \left( 1 - \frac{\text{MAE}_{\text{CNN}, b}}{\text{MAE}_{\text{persist}, b}} \right) \times 100\%$$

### Empirical Bootstrap Confidence Intervals ($t+10\text{s}$)

| Evaluation Cohort | Resampling Method | Median Skill | 95% Bootstrap Confidence Interval | Statistical Significance |
| :--- | :--- | :---: | :---: | :---: |
| **Overall Combined Test (May 14–15)** | 300s Block Bootstrap | **-106.13%** | **[-188.67%, -29.79%]** | **Persistence significantly superior ($p < 0.001$)** |
| **Target-Conditioned Moderate** | 300s Block Bootstrap | **+22.32%** | **[+21.65%, +22.86%]** | **CNN-LSTM significantly superior ($p < 0.001$)** |
| **Causal Origin-Conditioned Moderate** | 300s Block Bootstrap | **+22.23%** | **[+21.40%, +22.89%]** | **CNN-LSTM significantly superior ($p < 0.001$)** |

The 95% confidence interval for causal moderate skill strictly excludes zero ($[+21.40\%, +22.89\%]$), confirming that the moderate-activity skill advantage is statistically robust.

---

## 5. Audit of Substantive Scientific Claims

| Phase 4S Claim | Audit Verification | Scientific Verdict | Audit Rationale |
| :--- | :---: | :---: | :--- |
| **"CNN-LSTM achieves +22.3% skill over persistence in moderate flare regimes"** | Verified | **SUBSTANTIATED WITH QUALIFICATION** | Confirmed empirically across both target-based and causal origin-based conditioning. Must be qualified as a subgroup result. |
| **"May 2024 model adapts better to coronal decay than February 2026 model"** | Verified | **SUBSTANTIATED** | On May 15 decay, Phase 4S MAE is 55.40 c/s vs Phase 4O MAE 76.89 c/s (27.9% error reduction). |
| **"CNN-LSTM is superior to persistence for real-time solar flare forecasting"** | Unverified | **REFUTED** | Persistence is vastly superior overall (MAE 37.36 vs 78.86 c/s) and on extreme flare spikes (107.34 vs 321.25 c/s). |
| **"Lower error than Phase 4O model proves superiority"** | Invalid | **METHODOLOGICALLY FLAWED** | Lower error relative to an inferior neural model does not confer skill over physical persistence. |
| **"May 2024 training proves solar cycle generalization"** | Unverified | **REJECTED AS UNPROVEN** | 8 days from May 2024 is a single solar eruptive sequence; solar cycle generalization requires multi-year coverage. |
| **"Missing flare events are negative labels"** | Enforced | **PROHIBITED (`B-LBL-NEG-007`)** | Zero binary flare labels were used. Model is strictly continuous count-rate regression. |
| **"Count rates represent physical solar irradiance in $W/m^2$"** | Enforced | **QUARANTINED (`B-SX-FLUX-003`)** | Native units `counts/s` (2–22 keV) strictly maintained; zero flux conversion attempted. |

---

## 6. Audit Conclusion

The scientific claim of subgroup forecasting skill in moderate solar activity is substantiated and survives causal re-conditioning and block bootstrap verification. However, the overarching conclusion remains clear: **for overall solar flare forecasting, physical Persistence remains the dominant and superior baseline**.
