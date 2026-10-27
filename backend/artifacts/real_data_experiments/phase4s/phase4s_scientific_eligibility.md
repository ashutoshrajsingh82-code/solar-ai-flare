# Phase 4S: Scientific Eligibility Gate and Blocker Ledger

**Project:** Solar AI Flare  
**Backend:** `backend/`  
**Phase:** Phase 4S — May 2024 Real-Data Ingestion, Scientific Validation, and Controlled Forecast Training  
**Evaluation Date:** 2026-10-26  
**Status:** `ELIGIBLE_FOR_CONTROLLED_FORECASTING`

---

## 1. Executive Summary

This report establishes the scientific eligibility of the genuine observational data ingested from `solexs_2026Oct25T200126271.zip` for sequence-to-sequence autoregressive forecasting. 

In strict adherence to the project's non-negotiable scientific boundaries, only genuine Level-1 SoLEXS SDD2 count-rate time series are admitted to the forecasting pipeline. All secondary products, uncalibrated instruments, and physically unvalidated conversion routines remain strictly quarantined under active blockers.

---

## 2. Product Eligibility Matrix

A comprehensive programmatic audit of all 78 products contained in the Phase 4S archive and candidate pool yields the following eligibility determinations:

| Instrument | Detector / Subsystem | Product Type | Count | Eligibility Decision | Scientific Rationale & Blocker Code |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **SoLEXS** | **SDD2** | Light Curve (`.lc.gz`) | **8** | **ELIGIBLE** | **Authorized for Self-Supervised Forecasting.** 1.0s cadence, continuous exposure, native `counts/s` (2–22 keV). |
| **SoLEXS** | **SDD1** | Light Curve (`.lc.gz`) | 8 | **EXCLUDED** | `EXCL-AUXILIARY-GTI`: SDD1 auxiliary exposure tables indicate inactive or zero valid GTI intervals in this epoch. |
| **SoLEXS** | SDD1 / SDD2 | GTI Table (`.gti.gz`) | 16 | **EXCLUDED** | `EXCL-AUXILIARY-GTI`: Auxiliary exposure metadata; lightcurve segmentation is executed directly on finite records. |
| **SoLEXS** | SDD1 / SDD2 | PI Histogram (`.pi.gz`) | 16 | **EXCLUDED** | `EXCL-NON-TIMESERIES-HISTOGRAM`: Pulse Invariant spectral histograms are non-time-series products unsuitable for autoregression. |
| **HEL1OS** | CZT / NaI | Level-1 Light Curves | 24 | **QUARANTINED** | `QUARANTINED_BLOCKED_CALIBRATION`: Active blockers `B-HL-CALIB-001`, `B-HL-VERSION-006`, `B-HL-QUALITY-005`. |
| **Catalogue** | GOES / ISRO | Event Lists | N/A | **PROHIBITED** | `B-LBL-NEG-007`: Prohibits generating binary flare labels or interpreting absence of catalogue events as negative labels. |

---

## 3. Active Blocker Ledger

The scientific blockers governing the Solar AI Flare repository are reaffirmed as active, uncompromised, and strictly enforced:

### Blocker B-SX-FLUX-003: Physical Solar Irradiance Conversion Blocked
- **Policy:** Count rates (`counts/s`) must NEVER be converted to physical flux units ($W/m^2$) or GOES flare classes (C, M, X) using uncalibrated linear or heuristic scaling factors.
- **Enforcement:** The model forecasts native detector count rates ($counts/s$) in the 2–22 keV band. No physical flux conversion is attempted.

### Blocker B-LBL-NEG-007: Prohibited Negative Flare Labeling
- **Policy:** Missing flare catalogue entries must NEVER be treated as negative training labels.
- **Enforcement:** Zero classification labels are generated. The objective is 100% self-supervised regression on continuous physical count rates.

### Blockers B-HL-CALIB-001, B-HL-VERSION-006, B-HL-QUALITY-005: HEL1OS Ingestion Quarantine
- **Policy:** HEL1OS products remain quarantined pending official ISRO response matrix calibration (`B-HL-CALIB-001`), resolution of V111/V211 product version discrepancies (`B-HL-VERSION-006`), and housekeeping saturation thresholds (`B-HL-QUALITY-005`).
- **Enforcement:** HEL1OS data are excluded from all model training and evaluation partitions.

---

## 4. Cadence, Timing, and Monotonicity Validation

Astropy inspection of all 8 daily SoLEXS SDD2 FITS files confirmed:
- **Cadence:** Uniform 1.0-second time resolution (`TIMEDEL = 1`, `TIMEUNIT = 's'`).
- **Time Representation:** Unix seconds since 1970-01-01T00:00:00Z (`MJDREFI = 40587`).
- **Monotonicity:** $t_{i+1} - t_i = 1.0$ across contiguous observation blocks. Zero duplicate or negative time steps.
- **Boundary Preservation:** Exactly 28 missing 1s bins are detected and segmented cleanly, ensuring that zero interpolated or synthetic values enter the training or evaluation windows.

---

## 5. Eligibility Conclusion

The 8 genuine SoLEXS SDD2 daily Level-1 light curves for May 8–15, 2024 satisfy all criteria for scientific eligibility. The dataset provides 691,172 valid physical observations capturing extreme solar eruptive dynamics under unbroken provenance.
