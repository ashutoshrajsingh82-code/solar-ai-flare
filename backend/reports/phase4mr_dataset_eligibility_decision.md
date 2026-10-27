# Phase 4M-R: Dataset Eligibility Gate Decision

**Decision Date**: 2026-10-25  
**Canonical Status**: `ALL_CANDIDATE_SUBSETS_INELIGIBLE`  
**Training Action**: `TRAINING_PROHIBITED_FAIL_CLOSED`  
**Model Fitting Occurred**: `FALSE`  
**Checkpoints Created**: `NONE`  

---

## 1. Scope and Objective

Phase 4M-R evaluates whether any scientifically defensible subset of the actual Aditya-L1 (SoLEXS and HEL1OS) observational archives can support supervised machine learning model training at this time.

In accordance with strict scientific guidelines and prohibited shortcuts:
1. Synthetic observations and synthetic labels are strictly barred from real-data training.
2. Missing catalogue entries and unobserved intervals must never be converted into negative labels.
3. Partially resolved global gates must not be used to bypass unresolved instrument contracts unless those rules are demonstrably irrelevant to the specific candidate subset.
4. If no candidate subset passes all 10 gates, training must not be forced.

---

## 2. Evaluation of Candidate Subsets Across the 10 Eligibility Gates

Three candidate observational subsets were defined and systematically audited against all 10 eligibility criteria:
- **Candidate Subset A**: Dual-Instrument Fusion (`SoLEXS SDD2 LC + HEL1OS L1 LC`)
- **Candidate Subset B**: SoLEXS-Only (`SoLEXS SDD2 Lightcurves`)
- **Candidate Subset C**: HEL1OS-Only (`HEL1OS Level-1 Lightcurves`)

### 2.1 Summary Matrix

| Gate # | Eligibility Gate Description | Subset A (Fusion) | Subset B (SoLEXS-Only) | Subset C (HEL1OS-Only) |
|---|---|:---:|:---:|:---:|
| **Gate 1** | Documented timestamp semantics | `BLOCKED` (R07/R08) | `PASS` (R01) | `BLOCKED` (R07/R08) |
| **Gate 2** | Valid observation & exposure (GTI) | `BLOCKED` (R03/R12) | `BLOCKED` (R03) | `BLOCKED` (R12) |
| **Gate 3** | Applicable quality & saturation treatment | `BLOCKED` (R14) | `BLOCKED` (R02) | `BLOCKED` (R14) |
| **Gate 4** | Product / version compatibility | `BLOCKED` (R05/R13) | `BLOCKED` (R05) | `BLOCKED` (R13) |
| **Gate 5** | Calibration compatibility (physical flux) | `BLOCKED` (R06) | `BLOCKED` (R06) | `BLOCKED` (R09) |
| **Gate 6** | Verified label provenance | `PASS` | `PASS` | `PASS` |
| **Gate 7** | Positive & negative label justification | `BLOCKED` (B-LBL-007) | `BLOCKED` (B-LBL-007) | `BLOCKED` (B-LBL-007) |
| **Gate 8** | Chronological train/val/test feasibility | `BLOCKED` | `BLOCKED` | `BLOCKED` |
| **Gate 9** | Adequate class counts in each partition | `BLOCKED` | `BLOCKED` | `BLOCKED` |
| **Gate 10** | Leakage-free windowing & transforms | `BLOCKED` | `BLOCKED` | `BLOCKED` |
| **OVERALL** | **Subset Training Eligibility** | **`INELIGIBLE`** | **`INELIGIBLE`** | **`INELIGIBLE`** |

---

## 3. Detailed Gate-by-Gate Analysis

### Gate 1 — Documented Timestamp Semantics
- **SoLEXS**: RATE.TIME is verified as Unix seconds since `1970-01-01 00:00:00 UTC` (*SoLEXS User Manual v1.0* p.6 §2.2). Gate 1 passes for SoLEXS-only.
- **HEL1OS**: The L1 light curve contains both `MJD` and `ISOT` columns. While the manual documents their general format (*HEL1OS User Manual v1.2* Appendix B.3), no issuer document establishes which stored representation is authoritative in case of disagreement, what tolerance is permitted, or how discrepancies must be handled (Blocker `B-HL-TIME-004`).
- **Result**: Subsets A and C are blocked.

### Gate 2 — Valid Observation and Exposure Interpretation
- **SoLEXS**: The archives provide 12 GTI tables. However, no authoritative ISRO document specifies whether interval endpoints are open or closed ($[a, b)$ vs $[a, b]$), how empty or overlapping intervals are resolved, or whether a library convention applies (Blocker `B-SLX-GTI-001`). Without an explicit GTI contract, valid detector exposure cannot be distinguished from invalid periods.
- **HEL1OS**: Level-1 GTIs cover the entire data dump and are documented as ignorable for Level-1 analysis (*HEL1OS User Manual v1.2* p.13 §4). No intra-dump GTI or pointing validity table exists.
- **Result**: All subsets are blocked.

### Gate 3 — Applicable Quality and Saturation Treatment
- **SoLEXS**: No quality column exists in the L1 RATE binary table. The Type-I PI spectral quality flag documented in the manual applies only to download filters, not lightcurve sample masking.
- **HEL1OS**: Telemetry housekeeping tables contain candidate quality and saturation fields (`fehkstat`, `czt1satctr1`, `cdte1pilectr`). However, no versioned L1 HK/quality specification defines bit meanings, numerical saturation thresholds, or authorized row-rejection actions (Blocker `B-HL-QUALITY-005`).
- **Result**: All subsets are blocked.

### Gate 4 — Product and Version Compatibility
- **SoLEXS**: Daily archive files span 6 days. Compatibility across pipeline updates, software versions, and metadata fields (`CREATOR`, `OBS_ID`) lacks an authoritative compatibility matrix (Blocker `B-SLX-PROV-002`).
- **HEL1OS**: Dumps include 10 V111 packages and 1 V211 counterpart package (`HLS_20260205_000005_43185sec`). Official release notes do not identify V111 versus V211 or provide rules for counterpart selection or calibration compatibility (Blocker `B-HL-VERSION-006`).
- **Result**: All subsets are blocked.

### Gate 5 — Calibration Compatibility (Physical Flux Requirements)
- The existing machine learning feature-engineering and nowcast/forecast pipeline (`ml/features/engineering.py`, `app/services/preprocess_service.py`) strictly requires physical flux in $W/m^2$ (`solexs_flux`, `helios_flux`).
- SoLEXS RATE provides native counts per second (2–22 keV). While `solexs_tools-1.1` supplies ARF/RMF files for forward-folding spectral analysis, no audited L1 count-rate-to-physical-flux conversion or uncertainty prescription is documented (Blocker `B-SLX-CAL-003`). Feeding raw counts into the pipeline raises a `ValueError` by design.
- HEL1OS CTR provides native counts per second; no flux conversion exists for L1.
- **Result**: All subsets are blocked.

### Gate 6 — Verified Label Provenance
- The NOAA/NCEI GOES-R XRS L2 Flare Report (`sci_xrsf-l2-flrpt_geo_y2026_v1-0-1.csv`) is verified, immutable (`b782aa60847e7cf0862bc8db281bb833680c50f8adf71bd24b21414d7e490de8`), and accompanied by official metadata and documentation.
- **Result**: Gate 6 passes across all subsets.

### Gate 7 — Positive and Negative Label Justification
- While 123 discrete flare events (73 M/X flares) provide positive candidate targets, **trustworthy negative labels cannot be established**.
- The GOES flare report is an automated detection catalogue, not a certification of quiet Sun. Missing catalogue entries do not prove absence of flares.
- Project rules strictly forbid treating unobserved, ambiguous, or uncatalogued intervals as negative.
- When unverified intervals remain strictly unlabeled (`NaN`), the defensible negative sample count is **zero**.
- Supervised binary classification cannot be trained on 100% positive data or fabricated negatives (Blocker `B-LBL-NEG-007`).
- **Result**: All subsets are blocked.

### Gate 8 & 9 — Chronological Feasibility and Class Representation
- Creating a chronological train / validation / test partition with zero negative examples results in partitions with 0 negatives in train, 0 negatives in validation, and 0 negatives in test.
- This violates the explicit minimum class-count policy (`minimums['negative'] > 0`).
- **Result**: All subsets are blocked.

### Gate 10 — Leakage Prevention
- The codebase enforces temporal ordering and past-only rolling feature extraction on test fixtures. However, because no valid chronological split can be formed from the real dataset, leakage-free training on real data cannot be validated.
- **Result**: Blocked.

---

## 4. Final Scientific Decision

**NO CANDIDATE SUBSET PASSES THE ELIGIBILITY GATES.**

- In accordance with Step 5 instructions: **Training is NOT forced.**
- No model fitting, parameter estimation, hyperparameter tuning, or test set evaluation was performed.
- No model was registered in the production model registry.
- All 178 FITS products remain excluded.
- The pipeline remains safely in a fail-closed, non-training state.

**Machine-Readable Companion**: `backend/reports/phase4mr_dataset_eligibility_decision.json`
