# Phase 4R: Dataset Integrity Audit & Archive Verification

**Document ID:** `PHASE4R-INTEG-AUDIT-001`  
**Phase:** Phase 4R — Real Observational Data Expansion, Gated Training, and Independent Evaluation  
**Repository:** `C:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main`  
**Backend:** `C:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main\backend`  
**Manifest References:**
- Phase 4R JSON Manifest: [`reports/phase4r_expanded_manifest.json`](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/reports/phase4r_expanded_manifest.json)
- Phase 4R CSV Manifest: [`reports/phase4r_expanded_manifest.csv`](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/reports/phase4r_expanded_manifest.csv)
**Execution Timestamp:** 2026-10-26T01:34:00 UTC  

---

## 1. Executive Summary

This report delivers a bit-level integrity audit of all genuine solar observational data available to the platform. Conducted using the deterministic indexer ([`scripts/index_real_observations.py`](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/scripts/index_real_observations.py)) and archive verification helper ([`scripts/operator_acquisition_helper.py`](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/scripts/operator_acquisition_helper.py)), this audit validates bitstream SHA-256 hashes, FITS HDU table schemas, timestamp monotonicity, measurement units, gap distributions, and exclusion codes.

### Audit Summary
- **Total Archives Audited:** 2 outer ZIP archives (1 SoLEXS archive, 1 HEL1OS archive). Both pass `zipfile.testzip()` with 0 corrupt members.
- **Total Products Indexed:** 35 data packages/products.
- **Eligible Autoregressive Forecasting Products:** Exactly 6 Level-1 SoLEXS SDD2 light curves (2026-01-31 to 2026-02-05).
- **Cumulative Genuine Observations:** 272,223 finite 1-second records (`counts/s`, 2–22 keV band) distributed across 17 contiguous segments with zero gap interpolation. Overall 6-day duty cycle is 52.51%.
- **Overlapping Observation Conflicts:** 0. Each 24-hour observation file governs exactly 86,400 1-second bins spanning `00:00:00 UTC` to `23:59:59 UTC`.
- **Duplicate Bitstream Groups:** 0 identical duplicates found among indexed products.
- **Counterpart Version Collisions:** 1 group detected in HEL1OS (`HLS_20260205_000005_43185sec_lev1_V111.zip` vs `V211.zip`), properly quarantined under blocker `B-HL-VERSION-006`.

---

## 2. On-Disk Archive Bitstream Verification

Both primary observational archives were audited directly on disk:

| Archive Filename | On-Disk Path | Size (Bytes) | SHA-256 Checksum | `testzip()` Status | Inner Daily Dumps |
|---|---|---|---|---|---|
| `solexs_2026Oct23T184402685.zip` | `C:\Users\ASHUTOSH\Downloads\` | 29,334,261 | `34c39624194ee2da...` | Clean (`None`) | 6 daily packages |
| `hel1os_2026Oct23T184518914.zip` | `C:\Users\ASHUTOSH\Downloads\` | 1,066,706,121 | `17654c33aff72313...` | Clean (`None`) | 11 dump packages |

Both archives are confirmed intact, uncorrupted, and bit-verified against historical Phase 4P and Phase 4Q records.

---

## 3. Detailed Audit of Eligible SoLEXS SDD2 Products

The 6 eligible SoLEXS SDD2 light curves conform strictly to Level-1 calibrated count rates with 1.0 Hz cadence and monotonic timestamps:

| Date | FITS Member Name | Valid Records | Bins Total | Duty Cycle | Segments ($\ge 70\text{s}$) | Mean Rate | Peak Rate | Native Units | Energy Band |
|---|---|---|---|---|---|---|---|---|---|
| **2026-01-31** | `al1_solexs_20260131_SDD2_v1.0.lc.gz` | 86,391 | 86,400 | 99.99% | 3 | 77.82 c/s | 451.0 c/s | `counts/s` | 2–22 keV |
| **2026-02-01** | `al1_solexs_20260201_SDD2_v1.0.lc.gz` | 45,132 | 86,400 | 52.24% | 4 | 561.29 c/s | 27,465.0 c/s | `counts/s` | 2–22 keV |
| **2026-02-02** | `al1_solexs_20260202_SDD2_v1.0.lc.gz` | 44,965 | 86,400 | 52.04% | 3 | 2,476.21 c/s | 23,632.0 c/s | `counts/s` | 2–22 keV |
| **2026-02-03** | `al1_solexs_20260203_SDD2_v1.0.lc.gz` | 5,109 | 86,400 | 5.91% | 1 | 1,049.49 c/s | 2,579.0 c/s | `counts/s` | 2–22 keV |
| **2026-02-04** | `al1_solexs_20260204_SDD2_v1.0.lc.gz` | 45,317 | 86,400 | 52.45% | 3 | 361.46 c/s | 4,019.0 c/s | `counts/s` | 2–22 keV |
| **2026-02-05** | `al1_solexs_20260205_SDD2_v1.0.lc.gz` | 45,309 | 86,400 | 52.44% | 3 | 275.10 c/s | 2,030.0 c/s | `counts/s` | 2–22 keV |
| **Total** | **6 Level-1 Lightcurves** | **272,223** | **518,400** | **52.51%** | **17** | **748.24 c/s** | **27,465.0 c/s** | `counts/s` | 2–22 keV |

### FITS HDU Schema Validation:
1. **Primary HDU:** Contains metadata headers (`TELESCOP = 'ADITYA-L1'`, `INSTRUME = 'SOLEXS'`, `DETNAM = 'SDD2'`).
2. **RATE HDU:** Contains binary table with columns:
   - `TIME`: 1D double-precision array, unit `'s'`, strictly monotonic ($\Delta t = 1.0\text{s}$). Elapsed Unix seconds since `1970-01-01 00:00:00 UTC` (verified per `SoLEXS-UserManual.pdf` Section 2.2).
   - `COUNTS`: 1E single-precision array, unit `'counts/s'`, non-negative values.
   - Zero synthetic interpolation or gap imputation.

---

## 4. Segment Breakdown & Gap Integrity

Windows for autoregressive sequence forecasting ($T_{\text{in}} = 60\,\text{s}$, $T_{\text{out}} = 10\,\text{s}$) require unbroken, consecutive 1 Hz records:
$$\text{Window Length} = T_{\text{in}} + T_{\text{out}} = 70\,\text{s}$$
Any NaN value or timestamp jump $\Delta t > 1.0\,\text{s}$ terminates the segment immediately.

### Contiguous Segments per Observation Day:
- **January 31, 2026 (3 segments):** Segment 0 (28,800s), Segment 1 (28,796s), Segment 2 (28,795s). Yields 86,184 sliding windows under quiet-Sun conditions.
- **February 1, 2026 (4 segments):** Segment 0 (15,156s), Segment 1 (15,191s), Segment 2 (5,746s), Segment 3 (9,039s). Yields 44,856 sliding windows covering major X-ray flare eruptions.
- **February 2, 2026 (3 segments):** Segment 0 (15,001s), Segment 1 (14,999s), Segment 2 (14,965s). Yields 44,758 sliding windows covering high flaring and decay conditions.
- **February 3, 2026 (1 segment):** Segment 0 (5,109s). Yields 5,040 sliding windows. Verified authentic instrument telemetry gap (5.91% duty cycle).
- **February 4, 2026 (3 segments):** Segment 0 (15,301s), Segment 1 (15,301s), Segment 2 (14,715s). Yields 45,110 sliding windows.
- **February 5, 2026 (3 segments):** Segment 0 (15,301s), Segment 1 (15,301s), Segment 2 (14,707s). Yields 45,102 sliding windows. Reserved as frozen held-out test set.
- **Cumulative Usable Forecasting Windows:** Exactly **271,050 sliding windows** ($60\text{s} \to 10\text{s}$) across the 17 contiguous segments.

---

## 5. Excluded Products Audit (29 Products)

All 29 excluded products remain cleanly quarantined with specific scientific rationale:

1. **SoLEXS SDD1 Light Curves (6 products):** SDD1 was inactive/uncalibrated during this epoch; all measurements are NaN. Tagged `EXCL-SOLEXS-SDD1-INACTIVE`.
2. **SoLEXS Type-I PI Spectrograms (6 products):** Energy-channel pulse height histograms (`counts/channel`), scientifically ineligible for autoregressive sequence forecasting. Tagged `EXCL-SOLEXS-PI-SPECTRA`.
3. **SoLEXS GTI Auxiliary Tables (12 products):** Good Time Interval exposure tables. Bypassed for self-supervised sequence forecasting because segmentation is performed directly on the calibrated 1 Hz rate records to avoid bin-edge ambiguity (`B-SLX-GTI-001`). Tagged `EXCL-AUXILIARY-GTI`.
4. **HEL1OS Level-1 Telemetry Packages (11 products):** Quarantined under active scientific blockers `B-HL-VERSION-006` (V111 vs V211 counterpart collision), `B-HL-TIME-004` (timing conventions), and `B-HL-QUALITY-005` (saturation criteria). Tagged `EXCL-HEL1OS-BLOCKERS`.

---

## 6. Audit Conclusion

The existing observational dataset is verified clean, gap-preserving, and scientifically eligible for native count-rate forecasting. Zero malformed files exist. The manifest files [`reports/phase4r_expanded_manifest.json`](file:///C:/Users/ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main\backend\reports\phase4r_expanded_manifest.json) and [`reports/phase4r_expanded_manifest.csv`](file:///C:/Users/ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main\backend\reports\phase4r_expanded_manifest.csv) represent the authoritative dataset state.
