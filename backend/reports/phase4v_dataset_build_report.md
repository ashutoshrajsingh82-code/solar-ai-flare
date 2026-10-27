# Phase 4V: Dataset Build Report

**Project:** Solar AI Flare  
**Phase:** Phase 4V — Multi-Epoch Real-Data Forecasting Pipeline and Gated Training  
**Report Version:** `phase4v_dataset_build_report_v1.0`  
**Generated:** 2026-10-26T16:20:00Z  
**Status:** `PHASE4V_DEVELOPMENT_TRAINING_COMPLETED_HOLDOUT_PENDING`

---

## 1. Executive Summary

Phase 4V constructed a multi-epoch SoLEXS SDD2 Level-1 count-rate dataset from two
independently acquired, hash-verified daily archives. The pipeline ingested 14 distinct
daily light curves spanning two solar activity regimes and applied the Phase 4U dataset
contract strictly throughout.

| Item | Value |
|------|-------|
| Development epochs | 2 (May 2024 Solar Max; Jan-Feb 2026 Quiet Sun) |
| Total unique daily products | 14 |
| Total daily packages confirmed | 14/14 |
| Archive integrity | SHA-256 verified (bit-level) |
| Synthetic data used | **None** |
| Binary flare labels used | **None** |
| HEL1OS data used | **None** |
| SDD1 data used | **None** |
| Irradiance conversion performed | **None** |

---

## 2. Source Archives and Provenance

### Archive 1 — Epoch 1: May 2024 Solar Maximum

| Field | Value |
|-------|-------|
| Archive ID | `ARCHIVE_MAY2024` |
| Filename | `solexs_2026Oct25T200126271.zip` |
| Local Path | `C:\Users\ASHUTOSH\Downloads\solexs_2026Oct25T200126271.zip` |
| SHA-256 | `8df85ed75b6f00052d7f45a6764b187c700977f931504fb79fdfe56f10df9db2` |
| File Size | 92,592,109 bytes |
| ZIP Integrity | Verified (testzip clean) |
| Source | ISRO ISSDC PRADAN Level-1 daily packages |
| Epoch Description | Solar Cycle 25 Maximum / AR 13664 X-class eruptive period |
| Calendar Range | May 8–15, 2024 (8 daily packages) |
| Prior Project Usage | Phase 4S training/validation/test; Phase 4T audit |

### Archive 2 — Epoch 2: January-February 2026 Quiet Sun

| Field | Value |
|-------|-------|
| Archive ID | `ARCHIVE_FEB2026` |
| Filename | `solexs_2026Oct23T184402685.zip` |
| Local Path | `C:\Users\ASHUTOSH\Downloads\solexs_2026Oct23T184402685.zip` |
| SHA-256 | `34c39624194ee2da95bbf8e3dfed5ac002c55a517a31bc4091bfdd0d2b4671c2` |
| File Size | 29,334,261 bytes |
| ZIP Integrity | Verified (testzip clean) |
| Source | ISRO ISSDC PRADAN Level-1 daily packages |
| Epoch Description | Aditya-L1 mission phase, quiet-sun to moderate flaring |
| Calendar Range | January 31 – February 5, 2026 (6 daily packages) |
| Prior Project Usage | Phase 4M-S training/validation/test; Phase 4N, 4O, 4R audit |

---

## 3. Per-Product Observation Inventory (14 Daily Products)

| Product Key | Date | Total Bins | Finite Records | NaN Records | Duty Cycle | Contiguous Segs | Peak c/s |
|-------------|------|-----------|---------------|-------------|------------|----------------|---------|
| SoLEXS_SDD2_20240508 | 2024-05-08 | 86,400 | 86,395 | 5 | 99.99% | 3 | 8,086 |
| SoLEXS_SDD2_20240509 | 2024-05-09 | 86,400 | 86,398 | 2 | 99.99% | 2 | 14,555 |
| SoLEXS_SDD2_20240510 | 2024-05-10 | 86,400 | 86,396 | 4 | 99.99% | 4 | 19,937 |
| SoLEXS_SDD2_20240511 | 2024-05-11 | 86,400 | 86,398 | 2 | 99.99% | 2 | 22,152 |
| SoLEXS_SDD2_20240512 | 2024-05-12 | 86,400 | 86,396 | 4 | 99.99% | 2 | 8,064 |
| SoLEXS_SDD2_20240513 | 2024-05-13 | 86,400 | 86,396 | 4 | 99.99% | 3 | 5,238 |
| SoLEXS_SDD2_20240514 | 2024-05-14 | 86,400 | 86,395 | 5 | 99.99% | 3 | 29,568 |
| SoLEXS_SDD2_20240515 | 2024-05-15 | 86,400 | 86,398 | 2 | 99.99% | 2 | 20,022 |
| SoLEXS_SDD2_20260131 | 2026-01-31 | 86,400 | 86,391 | 9 | 99.99% | 3 | 451 |
| SoLEXS_SDD2_20260201 | 2026-02-01 | 86,400 | 45,132 | 41,268 | 52.24% | 4 | 27,465 |
| SoLEXS_SDD2_20260202 | 2026-02-02 | 86,400 | 44,965 | 41,435 | 52.04% | 3 | 23,632 |
| SoLEXS_SDD2_20260203 | 2026-02-03 | 86,400 | 5,109 | 81,291 | 5.91% | 1 | 2,579 |
| SoLEXS_SDD2_20260204 | 2026-02-04 | 86,400 | 45,317 | 41,083 | 52.45% | 3 | 4,019 |
| SoLEXS_SDD2_20260205 | 2026-02-05 | 86,400 | 45,309 | 41,091 | 52.44% | 3 | 2,030 |

---

## 4. FITS Validation Checks Applied

For every daily product, the following checks were applied by `validate_solexs_fits_table()`:

- ✅ RATE binary table HDU present and populated
- ✅ `TIME` and `COUNTS` columns present
- ✅ `TIMEUNIT='s'` or equivalent (Unix seconds)
- ✅ `HDUCLAS3='COUNTS'` confirming native count-rate semantics
- ✅ TIME column strictly monotonically increasing
- ✅ Zero duplicate timestamps
- ✅ All TIME values finite (no NaN in time column)
- ✅ COUNTS column treated as raw count rates (counts/s), not converted to W/m²

---

## 5. Exclusions and Reasons

| Excluded Category | Blocker ID | Reason |
|-------------------|------------|--------|
| SoLEXS SDD1 | `EXCL-AUXILIARY-GTI` | Zero valid GTI intervals; inactive detector |
| PI spectral histograms | `EXCL-NON-TIMESERIES-HISTOGRAM` | Not continuous time series |
| HEL1OS Level-1 data | `B-HL-CALIB-001`, `B-HL-VERSION-006`, `B-HL-QUALITY-005` | Unresolved calibration, version ambiguity, and quality blockers |
| Binary flare labels | `B-LBL-NEG-007` | No ground truth labels; self-supervised approach only |
| Physical irradiance (W/m²) | `B-SX-FLUX-003` | No validated SoLEXS→GOES conversion exists |
| October 2024 holdout data | `B-EPOCH3-ACQ-008` | Reserved untouched holdout; not yet acquired locally |

---

## 6. Deduplication

- 14 products scanned across 2 archives.  
- 0 duplicate product keys detected.  
- Product key format: `SoLEXS_SDD2_YYYYMMDD` (instrument + detector + observation date).  
- Cross-archive collision check passed: no observation date appears in both archives.

---

## 7. Contiguous Segmentation Summary

All products were segmented at boundaries where:
- `Δt > 1.0 s` (telemetry gap)
- `counts[i]` is NaN (missing observation)
- `counts[i] < 0.0` (physically invalid)

Segments shorter than 70 seconds (= SEQ_LEN + HORIZON = 60 + 10) were discarded.

Total contiguous segments eligible for windowing:
- **Epoch 1 (May 2024):** 21 segments (sum across 8 days; all products ≥99.99% duty cycle)
- **Epoch 2 (Jan-Feb 2026):** 17 segments (sum across 6 days; Feb 2026 products exhibit orbital eclipse NaN gaps ~47% duty cycle on some days)

---

## 8. Measurement Semantics

- **Instrument:** Solar Low Energy X-ray Spectrometer (SoLEXS)
- **Detector:** Silicon Drift Detector 2 (SDD2)
- **Primary HDU:** `RATE` binary table
- **Time column:** `TIME` in Unix seconds elapsed since 1970-01-01T00:00:00Z (`MJDREFI = 40587`)
- **Measurement column:** `COUNTS` in native count rate (counts/s), 2–22 keV energy band
- **Cadence:** 1.0 second uniform
- **No irradiance conversion:** Count rates treated as native physical quantities only
