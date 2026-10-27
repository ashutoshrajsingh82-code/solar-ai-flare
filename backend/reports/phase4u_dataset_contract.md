# Phase 4U: Multi-Epoch Real-Data Specification & Dataset Contract

**Project:** Solar AI Flare  
**Backend:** `backend/`  
**Phase:** Phase 4U — Multi-Epoch Real-Data Expansion and Untouched Holdout Design  
**Contract Version:** `phase4u_contract_v1.0`  
**Status:** `RATIFIED_AND_ENFORCED`

---

## 1. Specification Overview

This contract establishes the formal specification for self-supervised sequence-to-sequence autoregressive forecasting using genuine Aditya-L1 SoLEXS Level-1 observational time series across multiple solar epochs.

All data loaders, windowing generators, preprocessing scalers, benchmark models, and evaluation pipelines must strictly satisfy this contract. Any violation triggers an immediate, fail-closed pipeline halt.

---

## 2. Permitted Source Products & Provenance Contract

1. **Permitted Instrument & Detector:**
   - Instrument: **Solar Low Energy X-ray Spectrometer (SoLEXS)**.
   - Active Detector: **Silicon Drift Detector 2 (SDD2)**.
   - Primary Extension: `RATE` binary table containing continuous 1.0-second records.
2. **Authoritative Provenance:**
   - Must originate from authentic **ISRO ISSDC PRADAN Level-1 daily packages** (`AL1_SLX_L1_<YYYYMMDD>_v<VER>.zip`).
   - Every archive must possess a cryptographically verified SHA-256 digest recorded in the dataset inventory.
   - Synthetic, simulated, or linearly interpolated observations are strictly prohibited.
3. **Excluded Subsystems & Products:**
   - SDD1 products: Excluded under `EXCL-AUXILIARY-GTI` (zero valid GTI intervals).
   - Pulse Invariant (PI) spectral tables: Excluded under `EXCL-NON-TIMESERIES-HISTOGRAM`.
   - HEL1OS Level-1 data: Quarantined under active blockers `B-HL-CALIB-001`, `B-HL-VERSION-006`, and `B-HL-QUALITY-005`.

---

## 3. Physical Semantics, Columns & Timing

1. **Time Coordinate (`TIME` column):**
   - Header metadata: `HDUCLAS1 = 'LIGHTCURVE'`, `TIMEUNIT = 's'`, `TIMEDEL = 1`, `MJDREFI = 40587`.
   - Scale: Continuous Unix seconds elapsed since 1970-01-01T00:00:00Z.
   - Timestamps must be strictly monotonic ($t_{i+1} > t_i$) with uniform cadence ($\Delta t = 1.0\text{s}$) across active segments. Zero duplicate or out-of-order timestamps are permitted.
2. **Measurement Coordinate (`COUNTS` column):**
   - Header metadata: `HDUCLAS2 = 'TOTAL'`, `HDUCLAS3 = 'COUNTS'`.
   - Units: Native detector count rate (**`counts/s`**) in the **2–22 keV** physical energy band.
   - Non-negativity: $x_i \ge 0.0$ for all finite valid measurements.
3. **No Uncalibrated Flux Conversion (`B-SX-FLUX-003`):**
   - Count rates must NEVER be converted to physical solar irradiance ($W/m^2$) or GOES flare classes (C, M, X) using heuristic, uncalibrated scaling factors.

---

## 4. Quality Filtering, Missing-Value & Segmentation Policy

1. **Missing Data Policy:**
   - Missing observations (e.g. orbital telemetry gaps, nightside passes, or calibration dropouts) must be represented strictly as `NaN`.
   - Imputation, interpolation, or forward-filling of missing values is strictly prohibited.
2. **Contiguous Segmentation Rule:**
   - Observation arrays are segmented into independent, contiguous blocks whenever:
     $$\Delta t > 1.0\text{ second} \quad \text{OR} \quad x_i \text{ is NaN} \quad \text{OR} \quad x_i < 0.0$$
3. **Minimum Segment Length:**
   - A segment is eligible for windowing if and only if:
     $$L_{\text{segment}} \ge L_{\text{in}} + H = 60 + 10 = 70\text{ seconds}$$
   - Segments shorter than 70 seconds are discarded without bridging.

---

## 5. Sliding Window Generation & Partition Boundaries

1. **Window Dimensions:**
   - Input sequence history: $L_{\text{in}} = 60\text{ seconds}$ ($t-59, \dots, t$).
   - Forecast horizon: $H = 10\text{ seconds}$ ($t+1, \dots, t+10$).
   - Step size: 1.0 second stride within each contiguous segment.
2. **Boundary Isolation:**
   - Sliding windows must remain **strictly within a single contiguous segment**.
   - No window may cross midnight UTC daily boundaries or bridge across missing-value gaps.
   - No window may cross split boundaries (e.g., from Training into Validation, or from Validation into Test).

---

## 6. Preprocessing & Scaler Isolation Contract

1. **Mathematical Transformation:**
   $$z = \frac{\log_{10}(x + 1) - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
2. **Strict Fitting Isolation:**
   - Scaler parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) must be computed **strictly and exclusively from finite records belonging to the designated Training partition**.
   - Zero observations from Validation partitions or Holdout partitions may enter `scaler.fit()`.
3. **Inversion & Non-Negativity:**
   - Predictions are inverted back to native count rates via:
     $$\hat{x} = \max\left(0.0, \, 10^{(z \cdot \sigma_{\text{train}} + \mu_{\text{train}})} - 1.0\right)$$

---

## 7. Baseline Definitions & Target Alignment

1. **Target Alignment:** All models and baselines must be evaluated on **identical test window timestamps and identical ground-truth target vectors** $y_{t+1:t+10}$.
2. **Persistence Baseline (Zero-Skill Reference):**
   $$\hat{y}_{t+h}^{\text{persist}} = x_t \quad \forall h \in \{1, \dots, 10\}$$
   where $x_t$ is strictly the final observed count rate at forecast origin.
3. **Rolling Mean Baseline:**
   $$\hat{y}_{t+h}^{\text{rolling}} = \frac{1}{60} \sum_{i=0}^{59} x_{t-i} \quad \forall h \in \{1, \dots, 10\}$$

---

## 8. Causal Operational Activity Cohorts

To prevent informational look-ahead bias (identified and audited in Phase 4T), all physical activity cohorts must be conditioned **strictly on the observable count rate at forecast origin ($x_t = X[:, -1, 0]$)**:

| Activity Cohort | Conditioning Rule | Physical Interpretation |
| :--- | :--- | :--- |
| **Causal Quiet Coronal** (`Causal_REG_QUIET`) | $x_t < 200.0\text{ counts/s}$ | Quiescent background emission |
| **Causal Moderate Flaring** (`Causal_REG_MOD`) | $200.0 \le x_t \le 1000.0\text{ counts/s}$ | Moderate flares and active coronal loops |
| **Causal Extreme Flaring** (`Causal_REG_HIGH`) | $x_t > 1000.0\text{ counts/s}$ | Major M/X-class solar eruptions |
| **Causal Impulsive Surge** (`Causal_SURGE_50`) | $x_t - x_{t-10} > 50.0\text{ counts/s}$ | Rapid count rate rise in recent 10s history |

Conditioning on future target peaks ($\max y_{t+1:t+10}$) is prohibited for operational decision analysis.

---

## 9. Evaluation Metrics

For each cohort and horizon $h \in \{1, \dots, 10\}$:
- **MAE:** $\frac{1}{N} \sum_{i=1}^N |y_{i, h} - \hat{y}_{i, h}|$
- **RMSE:** $\sqrt{\frac{1}{N} \sum_{i=1}^N (y_{i, h} - \hat{y}_{i, h})^2}$
- **Skill Score vs Persistence (%):**
  $$\text{Skill}_{\text{persist}, h} = \left( 1 - \frac{\text{MAE}_{\text{model}, h}}{\text{MAE}_{\text{persist}, h}} \right) \times 100\%$$
- **Autocorrelation-Corrected Uncertainty:** Confidence intervals must be computed using time-aware block bootstrap ($L_{\text{block}} \ge 300\text{s}$) to account for 69s sliding window overlap.
