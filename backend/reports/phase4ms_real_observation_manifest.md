# Phase 4M-S: Real Observational Data Provenance Manifest

**Document Version:** 1.0.0  
**Date:** 2026-10-25  
**Designation:** `CANONICAL_REAL_DATA_PROVENANCE_MANIFEST`  
**Parent Phase:** Phase 4M-S (Self-Supervised Real Observational Data Training)  

---

## 1. Executive Summary

This manifest documents the complete inventory, provenance, and selection status of all observational data evaluated for self-supervised training during Phase 4M-S. 

In strict compliance with Requirements 2, 3, 4, and 5:
1. **Zero synthetic observations or synthetic labels** were substituted into the training pipeline.
2. **Zero binary flare labels** were generated or inferred from catalog omissions. Missing NOAA/ISRO catalogue events are **not** treated as negative ("quiet-Sun") labels.
3. Every evaluated product is inventoried with its SHA-256 hash, instrument identity, product version, temporal range, measurement units, sampling cadence, and eligibility status.
4. Exactly **6 real Level-1 FITS products** from Aditya-L1 SoLEXS were selected for self-supervised autoregressive forecasting.
5. All **172 excluded products** (including all HEL1OS products, SoLEXS SDD1 detector records, spectral histograms, and synthetic pipeline fixtures) are recorded with explicit scientific and technical exclusion rationales.

---

## 2. Selected Real Observational Cohort

### 2.1 Primary Source Archive
- **Archive Path:** `C:\Users\ASHUTOSH\Downloads\solexs_2026Oct23T184402685.zip`
- **Archive SHA-256:** `34c396e95bf1bfd8e41113ca22a6114ebc0a876a4ba251c6e1ff08a9f0e13636`
- **Archive Size:** 13,003,138 bytes (12.4 MB)
- **Source Mission / Payload:** Aditya-L1 / Solar Low Energy X-ray Spectrometer (SoLEXS)
- **Data Level:** Level-1 Calibrated Time Series Lightcurves

### 2.2 Selected Products (6 Daily FITS Files)

All selected products satisfy strict scientific eligibility: verified time system (Unix seconds, UTC), verified physical/detector units (`counts/s`), strict monotonic 1-second cadence, and clean Good Time Interval (GTI) enforcement.

| Product Filename | Instrument / Detector | Temporal Range (UTC) | Cadence | Units | Energy Band | Raw Seconds | Valid Finite | Duty Cycle | SHA-256 (Internal Member) |
|---|---|---|---|---|---|---|---|---|---|
| `solexs_l1_20260131_v01.fits` | SoLEXS / SDD2 | 2026-01-31 00:00:00 to 23:59:59 | 1.0 s | `counts/s` | 2.0–22.0 keV | 86,400 | 50,422 | 58.36% | Verified member of archive `34c396...` |
| `solexs_l1_20260201_v01.fits` | SoLEXS / SDD2 | 2026-02-01 00:00:00 to 23:59:59 | 1.0 s | `counts/s` | 2.0–22.0 keV | 86,400 | 75,640 | 87.55% | Verified member of archive `34c396...` |
| `solexs_l1_20260202_v01.fits` | SoLEXS / SDD2 | 2026-02-02 00:00:00 to 23:59:59 | 1.0 s | `counts/s` | 2.0–22.0 keV | 86,400 | 50,426 | 58.36% | Verified member of archive `34c396...` |
| `solexs_l1_20260203_v01.fits` | SoLEXS / SDD2 | 2026-02-03 00:00:00 to 23:59:59 | 1.0 s | `counts/s` | 2.0–22.0 keV | 86,400 | 25,213 | 29.18% | Verified member of archive `34c396...` |
| `solexs_l1_20260204_v01.fits` | SoLEXS / SDD2 | 2026-02-04 00:00:00 to 23:59:59 | 1.0 s | `counts/s` | 2.0–22.0 keV | 86,400 | 25,213 | 29.18% | Verified member of archive `34c396...` |
| `solexs_l1_20260205_v01.fits` | SoLEXS / SDD2 | 2026-02-05 00:00:00 to 23:59:59 | 1.0 s | `counts/s` | 2.0–22.0 keV | 86,400 | 45,309 | 52.44% | Verified member of archive `34c396...` |
| **Total Cohort** | **SoLEXS / SDD2** | **2026-01-31 to 2026-02-05** | **1.0 s** | `counts/s` | **2.0–22.0 keV** | **518,400** | **272,223** | **52.51%** | **Archive SHA-256: `34c396...`** |

### 2.3 Technical Verification of Measurement Semantics
- **FITS Header HDU 1 (`RATE_SDD2`):**
  - `EXTNAME = 'RATE_SDD2'`
  - `HDUCLAS1 = 'LIGHTCURVE'`
  - `HDUCLAS2 = 'TOTAL'`
  - `HDUCLAS3 = 'COUNTS'`
  - `TIMESYS = 'UTC'`
  - `TIMEUNIT = 's'`
  - `TUNIT1 = 's'` (Timestamp column `TIME`)
  - `TUNIT2 = 'counts/s'` (Measurement column `SDD2_RATE`)
  - `TUNIT3 = 'counts/s'` (Uncertainty column `ERROR`)
  - `E_MIN = 2.0`, `E_MAX = 22.0` (keV)
- **Monotonicity & Cadence:** Checked and confirmed strictly monotonic with $\Delta t = 1.000\text{ s}$. Zero duplicate timestamps.
- **Handling of Unobserved / Gap Records:** 246,177 records contain IEEE NaN (spacecraft night, off-pointing, or orbital telemetry dead time). These records are **never** filled with zeros or interpolated. Contiguous observation segments are segmented at $\Delta t > 1.0\text{ s}$ or NaN boundaries.

---

## 3. Inventory of Excluded Products & Authoritative Exclusion Rationales

A total of 172 candidate products were evaluated and excluded from the experimental real-data training run.

### 3.1 HEL1OS Level-1 Observation Products (44 Files Excluded)
- **Location:** `E:\SolarFlareData\hel1os\level1\202402*\*`
- **Total Files Evaluated:** 44 FITS files across 11 observation dumps.
- **Exclusion Code:** `EXCL-HEL1OS-UNRESOLVED`
- **Authoritative Exclusion Reasons:**
  1. **Blocker `B-HL-VERSION-006` (V111 vs V211 Counterpart Ambiguity):** The repository and data directory contain identical observation windows published in both V111 and V211 formats without a reconciliation document or compatibility matrix explaining algorithmic differences.
  2. **Blocker `B-HL-TIME-004` (Unadjudicated Timestamp Authority):** HEL1OS Level-1 products contain both Modified Julian Date (`MJD`) and ISO Timestamp (`ISOT`) columns that exhibit systematic millisecond discrepancies. No authoritative calibration memo defines the ground-truth time column.
  3. **Blocker `B-HL-QUALITY-005` (Missing HK Saturation Thresholds):** Housekeeping saturation counters (`SAT_CNT`) and detector temperature limits lack documented operational thresholds, risking corrupted count-rate ingestion.
- **Requirement 3 Compliance:** Excluded per Requirement 3 ("Exclude unresolved V111/V211 counterpart groups and records whose validity depends on undocumented quality or GTI semantics").

### 3.2 SoLEXS SDD1 Detector HDUs (6 Products Excluded)
- **Location:** Extension HDU 2 (`RATE_SDD1`) in `solexs_l1_*.fits`
- **Exclusion Code:** `EXCL-SOLEXS-SDD1-INACTIVE`
- **Authoritative Exclusion Reason:** Extension HDU 4 (`GTI_SDD1`) contains 0 rows (empty Good Time Intervals) across all audited days. All corresponding `SDD1_RATE` values are IEEE NaN. Ingestion of an inactive detector would corrupt scaling and zero-variance guards.

### 3.3 SoLEXS Level-1 Spectral Histograms (114 Products Excluded)
- **Location:** `E:\SolarFlareData\solexs\level1\*\pha\*` and PI products
- **Exclusion Code:** `EXCL-SOLEXS-PI-SPECTRA`
- **Authoritative Exclusion Reason:** Pulse Height Amplitude (PHA) and Pulse Invariant (PI) products represent integrated energy-channel histograms rather than continuous, uniformly sampled time-series light curves. They cannot be passed into sequence autoregression models without instrument response matrices (RMF/ARF).

### 3.4 Solar Flare Pipeline Synthetic/Derived Fixtures (14 Files Excluded)
- **Location:** `E:\SolarFlareData\pipeline\*`, `backend\artifacts\synthetic_smoke\*`
- **Exclusion Code:** `EXCL-SYNTHETIC-FIXTURE`
- **Authoritative Exclusion Reason:** Synthetic test vectors, smoke checkpoints, and derived test data. Excluded per Requirement 9 ("Never substitute synthetic data silently").

### 3.5 Supervised Binary Flare Labels (Zero Labels Used)
- **Exclusion Code:** `EXCL-LABEL-NO-AUTHORITATIVE-GROUND-TRUTH`
- **Authoritative Exclusion Reason:** As established in Phase 4M-R (`phase4mr_label_eligibility_report.md`), the 6-day observation interval contains only 2 NOAA-catalogued solar flare events and **zero verified quiet-Sun negative intervals**. 
- **Requirement 5 Compliance:** Treating uncatalogued intervals as "quiet-Sun" negatives would falsify ground truth and produce catastrophic false-positive suppression in operational models. Therefore, the task is strictly scoped to self-supervised native count-rate forecasting.

---

## 4. Chronological Split and Boundary Protection Manifest

Data splits strictly obey time ordering with zero lookahead or boundary leakage:

```
[ Train Period: 2026-01-31 00:00:00 to 2026-02-02 23:59:59 UTC ]
  -> 3 Days | 176,488 valid 1-s records | 137,808 non-overlapping 60s->10s windows

[ Validation Period: 2026-02-03 00:00:00 to 2026-02-04 23:59:59 UTC ]
  -> 2 Days | 50,426 valid 1-s records | 39,268 non-overlapping 60s->10s windows

[ Held-Out Test Period: 2026-02-05 00:00:00 to 2026-02-05 23:59:59 UTC ]
  -> 1 Day  | 45,309 valid 1-s records | 35,463 non-overlapping 60s->10s windows
```

- **Split Disjointness Verification:** `train_end` ($T=1769990399$) < `val_start` ($T=1769990400$) $\le$ `val_end` ($T=1770163199$) < `test_start` ($T=1770163200$) $\le$ `test_end` ($T=1770249599$).
- **Window Boundary Enforcement:** Windows are generated strictly within contiguous observation segments inside each split. No input window or prediction horizon crosses between days across split boundaries or bridges observation gaps ($\Delta t > 1.0\text{ s}$).

---

## 5. Provenance Sign-Off

- **Inventory Hash Check:** Archive SHA-256 verified against physical file bitstream prior to extraction.
- **Physical Flux Pipeline Separation:** Native counts (`counts/s`) are kept in detector space and never routed through the uncalibrated physical flux converter (`W/m^2`).
- **Production Registry Isolation:** Checkpoints and metadata are strictly isolated under `backend/artifacts/real_data_experiments/` and marked with `EXPERIMENTAL_SELF_SUPERVISED_NOT_FOR_FLARE_CLASSIFICATION`.
