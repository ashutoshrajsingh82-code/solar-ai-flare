# Phase 4M-S: Dataset Eligibility and Scientific Exclusion Report

**Document Version:** 1.0.0  
**Date:** 2026-10-25  
**Designation:** `CANONICAL_ELIGIBILITY_AND_EXCLUSION_REPORT`  
**Parent Phase:** Phase 4M-S (Self-Supervised Real Observational Data Training)  

---

## 1. Scientific Motivation and Task Formulation

The overarching goal of Phase 4M-S is to determine whether a scientifically scoped, experimental model can be trained on actual Aditya-L1 solar observations (SoLEXS and/or HEL1OS) **without falsifying ground-truth labels**.

### 1.1 The Supervised Learning Trap
In supervised solar flare prediction, models attempt to classify whether an observational window will produce a $\ge$ C-class solar flare in the next 12–24 hours. However, as demonstrated in Phase 4M-R:
1. **Catalog Completeness Asymmetry:** Catalogs like NOAA SWPC or ISRO Aditya-L1 Solar Flare Event Lists record high-confidence flare events, but they **never** catalog quiet-Sun intervals.
2. **Negative Label Fabrication Hazard:** In the 6 days of available real SoLEXS observations (2026-01-31 to 2026-02-05), only 2 flares are catalogued. If an unprincipled pipeline assumes that every non-flare second is a "negative" (quiet Sun), it generates over 272,000 fabricated negative labels. 
3. **Physical Reality:** Real solar X-ray observations contain microflares, nanoflares, and active-region evolution. Labeling real physical flux variations as "quiet Sun" forces the neural network to suppress genuine pre-flare signatures, rendering the model scientifically invalid and operationally dangerous.

### 1.2 The Self-Supervised Resolution
Self-supervised sequence forecasting resolves this impasse completely:
- **Task:** Predict future native count rates ($t \in [T+1, T+10]$ s) given past native count rates ($t \in [T-59, T]$ s).
- **Zero Label Dependency:** The supervision signal comes entirely from the physical observation sequence itself.
- **Physical Invariance:** Native count rates (`counts/s`) are well-defined physical Poisson count rates directly output by the silicon drift detector (SDD).
- **Decoupling from Uncalibrated Flux:** The task operates in detector count space (2.0–22.0 keV) and avoids routing uncalibrated data through the physical flux converter (`W/m^2`), which is blocked by `B-SX-FLUX-003`.

---

## 2. Eligibility Adjudication Matrix

| Instrument / Cohort | Product Identifier | Physical / Measurement Semantics | Temporal Ordering & Integrity | Quality / GTI Semantics | Decision | Adjudication Rationale |
|---|---|---|---|---|---|---|
| **SoLEXS SDD2** | `solexs_l1_20260131` through `20260205_v01.fits` | Verified `counts/s` in 2.0–22.0 keV band (`HDUCLAS3 = 'COUNTS'`). | Verified UTC Unix seconds (`TIMEUNIT = 's'`, strictly monotonic, $\Delta t = 1.0\text{ s}$). | Verified `GTI_SDD2` extension, IEEE NaN on unobserved night intervals. | **ELIGIBLE** | Selected for self-supervised autoregressive forecasting. Standalone, continuous, internally consistent. |
| **SoLEXS SDD1** | Extension `RATE_SDD1` in daily L1 FITS files | Inactive / NaN throughout 6-day cohort. | Identical time axis to SDD2. | `GTI_SDD1` table contains 0 rows (detector uncalibrated or powered down). | **EXCLUDED** | Inactive detector would inject degenerate zero-variance data or all-NaN sequences. |
| **SoLEXS Spectral Histograms** | `E:\SolarFlareData\solexs\level1\*\pha\*` and PI products | Spectral channel count histograms (PI/PHA). | Integrated over orbit chunks; non-continuous time series. | Energy calibration matrices (RMF/ARF) unadjudicated. | **EXCLUDED** | Spectral histograms require forward-folding detector response models; cannot be ingested as 1D time sequences. |
| **HEL1OS Level-1 Lightcurves** | `E:\SolarFlareData\hel1os\level1\202402*\*` | Pulse counts in 16 energy bands (10–150 keV). | Unadjudicated discrepancy between `MJD` and `ISOT` columns (`B-HL-TIME-004`). | Unresolved V111 vs V211 counterpart differences (`B-HL-VERSION-006`); unthresholded HK saturation (`B-HL-QUALITY-005`). | **EXCLUDED** | Blocked by Phase 4K blockers `B-HL-VERSION-006`, `B-HL-TIME-004`, and `B-HL-QUALITY-005`. |
| **Synthetic / Smoke Fixtures** | `backend/artifacts/synthetic_smoke/*`, `E:\SolarFlareData\pipeline\*` | Synthetic mock counts / labels. | Synthetic linear sequences. | Mock quality flags. | **EXCLUDED** | Prohibited by Requirement 9 ("Never substitute synthetic data silently"). |
| **NOAA / Catalog Flare Labels** | NOAA SWPC Solar Event Reports (2026-01-31 to 2026-02-05) | 2 C/M-class catalogued events; 0 verified quiet-Sun events. | Sparse interval points. | Missing quiet-Sun ground truth. | **EXCLUDED** | Prohibited by Requirement 5 ("Do not generate binary flare labels, treat missing NOAA catalogue events as negative labels"). |

---

## 3. Detailed Scientific Rationale for Exclusions

### 3.1 Blocker Analysis for HEL1OS Products (`B-HL-VERSION-006`, `B-HL-TIME-004`, `B-HL-QUALITY-005`)
The HEL1OS dataset contains 44 Level-1 FITS files covering February 2024. Despite having multiple energy bands, it fails the eligibility gates on three counts:
1. **Version Counterpart Discrepancy (`B-HL-VERSION-006`):** For identical observation dumps, the repository contains both `V111` and `V211` files. Differences in pulse-height calibration and background rejection between these versions remain undocumented in ISRO public documentation.
2. **Timestamp Ambiguity (`B-HL-TIME-004`):** The `MJD` column and the `ISOT` string column differ by variable millisecond offsets. Training sequence models on unanchored time series would introduce non-physical phase jitter.
3. **Saturation & Quality Ambiguity (`B-HL-QUALITY-005`):** Housekeeping parameters `SAT_CNT` (saturation count) and `DEAD_TIME` lack documented physical threshold bounds. Masking saturated intervals without authoritative guidelines risks invalidating data during the most intense solar activity.

### 3.2 SoLEXS SDD1 Inactivity
SoLEXS houses two identical Silicon Drift Detectors: SDD1 and SDD2. During the 2026-01-31 to 2026-02-05 observation window:
- HDU `RATE_SDD1` contains 518,400 rows, all populated with IEEE NaN.
- HDU `GTI_SDD1` contains 0 rows (an empty table).
- In contrast, HDU `RATE_SDD2` contains 272,223 finite valid records and 50 Good Time Intervals.
- **Action Taken:** SDD1 was cleanly excluded at the FITS ingestion boundary. No synthetic imputation or zero-filling was applied.

### 3.3 Protection Against Negative Label Falsification
The scientific community recognizes label noise as the primary cause of model failure in space weather operations. In this 6-day observation window:
- NOAA reported an M-class flare on 2026-02-01 and a C-class flare on 2026-02-05.
- Zero quiet-Sun intervals were officially certified by any solar observatory.
- Under Requirement 5, **no binary flare labels were produced**. Unobserved intervals during spacecraft night are treated as unknown/unobserved data gaps, not quiet-Sun negatives.

---

## 4. Preservation of Scientific Integrity and Pipeline Separation

1. **Experimental Namespace Isolation:** All code, models, and outputs reside in `backend/ml/experiments/` and `backend/artifacts/real_data_experiments/`.
2. **Registry Protection:** The production model registry (`backend/artifacts/model_registry/registry_index.json`) is untouched. The experimental checkpoint has `production_registry_eligible: False`.
3. **No Blocker Downgrades:** Scientific blockers `B-HL-VERSION-006`, `B-HL-TIME-004`, `B-HL-QUALITY-005`, and `B-SX-FLUX-003` remain active and unresolved until authoritative ISRO calibration documentation is delivered.
