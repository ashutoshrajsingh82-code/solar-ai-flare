# Phase 4S: Data Integrity and Observational Provenance Audit

**Project:** Solar AI Flare  
**Backend:** `backend/`  
**Phase:** Phase 4S — May 2024 Real-Data Ingestion, Scientific Validation, and Controlled Forecast Training  
**Audit Date:** 2026-10-26  
**Status:** `AUDIT_PASSED_PROVENANCE_VERIFIED`

---

## 1. Executive Summary

This audit establishes the observational provenance, bit-level integrity, time-series continuity, and physical validity of the newly supplied SoLEXS observational dataset covering May 8–15, 2024. The dataset captures Solar Cycle 25 maximum conditions, including the historic Solar Active Region 13664 / 13663 eruptive sequence and the May 14, 2024 X8.7 solar flare.

All non-negotiable scientific safeguards have been strictly maintained:
1. **Genuine Observational Data Only:** Zero synthetic data or simulated curves were ingested.
2. **Native Physical Semantics:** Data units are strictly preserved as native detector count rates (`counts/s`, 2–22 keV) from SoLEXS SDD2 Level-1 light curves.
3. **No Flare Classification / Negative Label Generation:** Observational time series are evaluated strictly via self-supervised autoregressive forecasting. Missing events are never interpreted as negative flare labels (`B-LBL-NEG-007`).
4. **No Uncalibrated Irradiance Conversion:** Physical solar irradiance ($W/m^2$) conversion remains quarantined under blocker `B-SX-FLUX-003`.

---

## 2. Archive Verification and Bit-Level Integrity

The candidate archive was ingested and verified on disk at the authorized location:

| Parameter | Specification | Measured Value | Verification Result |
| :--- | :--- | :--- | :--- |
| **Archive File** | `solexs_2026Oct25T200126271.zip` | `solexs_2026Oct25T200126271.zip` | **MATCH** |
| **Storage Path** | `C:\Users\ASHUTOSH\Downloads\...` | `C:\Users\ASHUTOSH\Downloads\...` | **CONFIRMED** |
| **Byte Size** | 92,592,109 bytes | 92,592,109 bytes | **EXACT MATCH** |
| **SHA-256 Digest** | `8df85ed75b6f00052d7f45a6764b187c700977f931504fb79fdfe56f10df9db2` | `8df85ed75b6f00052d7f45a6764b187c700977f931504fb79fdfe56f10df9db2` | **EXACT BIT MATCH** |
| **Archive CRC32/ZIP Test** | Zero corrupted entries (`testzip() == None`) | Zero corrupted entries (`None`) | **PASSED** |
| **ISRO PRADAN Provenance** | Authorized ISSDC level-1 batch | Download status `DOWNLOADED_HASH_VERIFIED` | **VERIFIED** |

---

## 3. Product Inventory and Daily Breakdown

The archive unpacks into exactly 8 daily observation packages (`AL1_SLX_L1_20240508_v1.0.zip` through `AL1_SLX_L1_20240515_v1.0.zip`), containing Level-1 SDD2 light curves, Good Time Interval (GTI) tables, and Pulse Invariant (PI) spectral histograms.

Each daily SDD2 light curve contains precisely 86,400 1-second records (one full UTC solar day). Across the 8-day period, the dataset exhibits an extraordinary **99.9959% observational duty cycle**, with only 28 total unobserved/NaN records out of 691,200 time bins:

| Observation Date | Total Bins | Finite Records | NaN Bins | Contiguous Segments | Min Rate (`c/s`) | Median Rate (`c/s`) | Peak Rate (`c/s`) | Solar Activity Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **2024-05-08** | 86,400 | 86,395 | 5 | 2 | 26.00 | 66.00 | 5,690.00 | Moderate solar activity, multiple M-class events |
| **2024-05-09** | 86,400 | 86,398 | 2 | 3 | 29.00 | 79.00 | 7,858.00 | Elevated baseline, major active region emergence |
| **2024-05-10** | 86,400 | 86,396 | 4 | 3 | 27.00 | 102.00 | 14,888.00 | Extreme activity; superstorm conditions begin |
| **2024-05-11** | 86,400 | 86,398 | 2 | 2 | 34.00 | 104.00 | 22,152.00 | Major X-class flare activity, extreme coronal emission |
| **2024-05-12** | 86,400 | 86,396 | 4 | 3 | 31.00 | 97.00 | 7,677.00 | Persistent high activity, high surge frequency |
| **2024-05-13** | 86,400 | 86,396 | 4 | 3 | 34.00 | 87.00 | 9,977.00 | Validation partition; multiple X/M surges |
| **2024-05-14** | 86,400 | 86,395 | 5 | 3 | 30.00 | 81.00 | **29,568.00** | **Quarantined Test; Historic X8.7 solar flare** |
| **2024-05-15** | 86,400 | 86,398 | 2 | 2 | 26.00 | 78.00 | **20,022.00** | **Quarantined Test; Major limb X-flare and decay** |
| **TOTAL** | **691,200** | **691,172** | **28** | **21** | **26.00** | **86.00** | **29,568.00** | **Solar Cycle 25 Maximum Surge Sequence** |

---

## 4. FITS Header & Scientific Integrity Audit

Every Level-1 FITS binary table was extracted and programmatically verified using Astropy:
1. **Primary HDU & Extension:** `RATE` binary table extension confirmed.
2. **Timing System:**
   - `HDUCLAS1 = 'LIGHTCURVE'`
   - `HDUCLAS2 = 'TOTAL'`
   - `HDUCLAS3 = 'COUNTS'`
   - `TIMEUNIT = 's'`
   - `TIMEDEL = 1`
   - `MJDREFI = 40587` (Unix Epoch 1970-01-01T00:00:00Z)
   - Timestamps are strictly monotonic with zero backward steps or duplicate timestamps.
3. **Count-Rate Semantics:**
   - Units: `counts/s`
   - Physical Band: 2–22 keV (Silicon Drift Detector 2)
   - Detector: SDD2 active, nominal thermal stabilization.
4. **GTI and Exposure Alignment:**
   - SDD2 GTI tables confirm continuous active exposure throughout each day.
   - Contiguous lightcurve segmentation strictly respects the 28 missing 1s bins, producing 21 clean continuous segments with zero interpolated values.

---

## 5. Frozen Artifact and Historic Manifest Integrity

All historical benchmark artifacts and manifests were verified bit-for-bit unchanged:

| Artifact | Historical Phase | Verified SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- |
| `phase4p_expanded_dataset_manifest.json` | Phase 4P | `b5a3918df6753de873847e3346649a81370189ce24040482b5780a34e145d7a6` | **UNMUTATED** |
| `phase4q_expanded_dataset_manifest.json` | Phase 4Q | `c964a40be3a0d948abd070473ffd688b3761b85f74ab7c978165c854aeaecc75` | **UNMUTATED** |
| `phase4r_expanded_manifest.json` | Phase 4R | Generated and verified | **UNMUTATED** |
| `solexs_self_supervised_cnnlstm_20261025T115207Z.pt` | Phase 4M-S | `815ded9a605fa4bb410791b3a64d07624d6f3d03ffaf02953b67c7f1f8e53d45` | **UNMUTATED** |
| `solexs_phase4o_cnn_lstm_direct_20261025T184851Z.pt` | Phase 4O | `d6076ab20c99fdc852d338c47f9b7248250a333c4708916a8ced8de3a6336514` | **UNMUTATED** |

---

## 6. Audit Conclusion

The May 8–15, 2024 observational dataset provides the exact high-flux, surge-rich dynamics necessary to evaluate count-rate forecasting under extreme solar flare conditions. All data integrity, provenance, and header checks pass unconditionally.
