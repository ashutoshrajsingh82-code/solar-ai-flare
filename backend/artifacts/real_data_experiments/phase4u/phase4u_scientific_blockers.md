# Phase 4U: Comprehensive Scientific Blocker Ledger

**Project:** Solar AI Flare  
**Backend:** `backend/`  
**Phase:** Phase 4U — Multi-Epoch Real-Data Expansion and Untouched Holdout Design  
**Ledger Version:** `phase4u_blockers_v1.0`  
**Status:** `ALL_BLOCKERS_ENFORCED_FAIL_CLOSED`

---

## 1. Executive Summary

This document records the official, fail-closed blocker register governing Phase 4U and all subsequent phases. Every blocker represents an essential physical, instrumental, or methodological safeguard. 

No blocker may be cleared by heuristic assumptions, synthetic substitutions, or unauthenticated shortcuts.

---

## 2. Active Blocker Register

| Blocker ID | Affected Subsystem / Domain | Blocker Title | Status | Fail-Closed Enforcement Rule |
| :--- | :--- | :--- | :---: | :--- |
| **B-SX-FLUX-003** | SoLEXS Light Curves | Physical Irradiance ($W/m^2$) Conversion Blocked | **ACTIVE** | Prohibits converting detector count rates (`counts/s`) to physical flux ($W/m^2$) or GOES flare classes (C, M, X) using uncalibrated scaling. Native units (2–22 keV `counts/s`) must be preserved. |
| **B-LBL-NEG-007** | Catalogues & Labels | Negative Flare Label Fabrication Prohibited | **ACTIVE** | Prohibits interpreting the absence of flare catalogue entries as negative classification labels. Models must operate strictly on self-supervised physical count-rate forecasting. |
| **B-HL-CALIB-001** | HEL1OS Level-1 Data | Official ISRO Response Matrix Calibration Missing | **ACTIVE** | HEL1OS hard X-ray data remain quarantined pending official release of validated Level-1 detector response matrices. |
| **B-HL-VERSION-006** | HEL1OS Level-1 Data | V111 vs V211 Product Version Precedence Unresolved | **ACTIVE** | Colliding processing versions in HEL1OS daily telemetry dumps prevent deterministic product selection; quarantined. |
| **B-HL-QUALITY-005** | HEL1OS Telemetry | Housekeeping Saturation & Pile-Up Thresholds Undefined | **ACTIVE** | HEL1OS housekeeping counters lack official rejection thresholds; quarantined from automated pipelines. |
| **B-EPOCH3-ACQ-008** | Multi-Epoch Holdout | Untouched Epoch 3 (Oct 2024) Unacquired Locally | **ACTIVE** | Candidate holdout epoch (`P-2024Q4-OCT`, October 1–7, 2024) is not available locally. Multi-epoch model training and holdout evaluation remain strictly blocked until authorized PRADAN acquisition. |

---

## 3. Detailed Blocker Specifications

### Blocker B-SX-FLUX-003: Physical Solar Irradiance Conversion Blocked
- **Physical Context:** SoLEXS SDD2 records native pulse counts in the 2–22 keV band. Converting count rates ($C$) to physical energy flux ($F$ in $W/m^2$) requires convolution with detector effective area $A(E)$, energy resolution $R(E, E')$, pulse pile-up correction, and an assumed coronal plasma spectral model (e.g. isothermal bremsstrahlung + emission lines via CHIANTI).
- **Enforcement:** In the absence of an officially validated, automated ISRO spectrum-to-flux pipeline in the repository, all inputs, targets, and outputs must remain in native detector count rates (**`counts/s`**). Describing model outputs as physical solar irradiance or GOES flux is prohibited.

### Blocker B-LBL-NEG-007: Negative Flare Label Fabrication Prohibited
- **Methodological Context:** Flare catalogues (e.g., GOES flare event lists or preliminary ISRO solar flare lists) have observational detection thresholds, telemetry dropouts, and reporting latencies. Treating unlisted time bins as "no flare" (negative label) introduces massive label contamination and false negatives.
- **Enforcement:** Zero classification targets are synthesized. The task is strictly defined as continuous, self-supervised regression on physical count-rate time series.

### Blocker B-EPOCH3-ACQ-008: Untouched Third Epoch Unacquired Locally
- **Methodological Context:** Methodological rigor requires that a model be evaluated against a genuinely untouched holdout epoch that was never inspected during model iteration. All 14 calendar days of locally available SoLEXS data (May 8–15, 2024 and Jan 31–Feb 5, 2026) were examined in Phases 4M-S through 4T.
- **Enforcement:** Model training in Phase 4V cannot commence until the operator acquires the reserved October 1–7, 2024 archive via authorized PRADAN download.
