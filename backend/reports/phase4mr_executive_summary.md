# Phase 4M-R: Real Observational Data Training Gate — Executive Summary

**Phase**: 4M-R (Real Observational Data Training Gate)  
**Date**: 2026-10-25  
**Canonical Gate Result**: **`ALL_CANDIDATE_SUBSETS_INELIGIBLE`**  
**Training Execution**: **`TRAINING_NOT_ATTEMPTED_FAIL_CLOSED`**  
**Model Fitting Occurred**: **`FALSE`**  
**Checkpoint Created**: **`NONE`**  
**Production Registry Modified**: **`FALSE`**  

---

## 1. Primary Findings

1. **Complete Real Observational Inventory**:
   - The configured data root `E:\SolarFlareData` and downloaded archive paths were audited read-only in memory.
   - **178 candidate FITS products** were structurally inventoried: 24 SoLEXS products across 6 daily archive packages (`2026-01-31` to `2026-02-05`) and 154 HEL1OS products across 11 dump packages (`2026-02-01` to `2026-02-05`).
   - All 178 products are structurally readable, but **0 of 178 products are scientifically valid for model training at this time**. Structural readability is not scientific validity. Every product is documented with SHA-256 digests and explicit exclusion reasons in `backend/reports/phase4mr_real_observation_inventory.json` and `backend/reports/phase4mr_real_observation_manifest.md`.

2. **Label Eligibility and the Negative-Label Contract**:
   - The official NOAA/NCEI GOES-R XRS L2 Flare Report (`sci_xrsf-l2-flrpt_geo_y2026_v1-0-1.csv`, SHA-256 `b782aa60847e7cf0862bc8db281bb833680c50f8adf71bd24b21414d7e490de8`) contains 123 verified flare events (73 M/X-class flares) overlapping the Aditya-L1 observation period.
   - **Absence of a catalogue entry does not establish quiet Sun / negative flare occurrence.**
   - In accordance with scientific integrity guidelines, missing catalogue entries, unobserved intervals, and telemetry dropouts must not be converted into negative labels.
   - When unverified intervals remain strictly unlabeled, the defensible negative class count is **zero**. Supervised binary classification cannot be trained on fabricated negatives.

3. **Systematic 10-Gate Subset Evaluation**:
   - All three candidate subsets—Dual-Instrument Fusion, SoLEXS-Only, and HEL1OS-Only—were evaluated against the 10 training eligibility gates.
   - Every candidate subset fails multiple gates:
     - **Gate 2 (Exposure/GTI)**: Undocumented SoLEXS GTI endpoint closure and overlap precedence (Blocker `B-SLX-GTI-001`).
     - **Gate 3 (Quality/Saturation)**: Uninterpreted HEL1OS HK saturation counters without bit definitions or row-rejection thresholds (Blocker `B-HL-QUALITY-005`).
     - **Gate 4 (Versions)**: Unadjudicated HEL1OS V111 vs V211 counterpart relationship (Blocker `B-HL-VERSION-006`).
     - **Gate 5 (Calibration)**: Native count rates (counts/s, cts/sec) without an audited L1 RATE-to-physical-flux conversion, whereas the ML feature pipeline requires physical flux ($W/m^2$) (Blocker `B-SLX-CAL-003`).
     - **Gate 7 (Labels)**: Zero defensible negative examples (Blocker `B-LBL-NEG-007`).
     - **Gates 8 & 9 (Splits & Classes)**: Infeasible chronological binary partition with 0 negatives.
   - Canonical dataset eligibility decision is **`INELIGIBLE`** (`training_allowed = false`).

4. **Fail-Closed Safety Guarantee**:
   - Because no candidate subset passed the eligibility gates, **training was not forced**.
   - No model fitting, parameter optimization, threshold tuning, or evaluation on test data occurred.
   - No synthetic observations or synthetic labels were used.
   - The production model registry remains untouched.

5. **Test and Environment Verification**:
   - Using the active system environment `C:\Python313\python.exe` (Python 3.13.4):
     - **Focused Tests**: 31 passed, 0 failed (1.78s).
     - **Non-ML Regression Tests**: 80 passed, 0 failed (9.33s).
     - **Full Backend Test Suite**: 235 passed, 0 failed (37.44s).
     - **Training Readiness Dry Run**: Exit 1 (fail-closed refusal as designed).

---

## 2. Deliverables Index

| Deliverable | Path | Description |
|---|---|---|
| **Real Observation Inventory (JSON)** | `backend/reports/phase4mr_real_observation_inventory.json` | 178 FITS products with archive paths, hashes, semantics, and exclusion reasons |
| **Real Observation Manifest (Markdown)** | `backend/reports/phase4mr_real_observation_manifest.md` | Human-readable provenance manifest and instrument breakdown |
| **Label Eligibility Report** | `backend/reports/phase4mr_label_eligibility_report.md` | Audit of GOES flare events, negative-label absence, and supervised classification limits |
| **Dataset Eligibility Decision (JSON)** | `backend/reports/phase4mr_dataset_eligibility_decision.json` | Machine-readable evaluation of all 3 candidate subsets across 10 gates |
| **Dataset Eligibility Decision (Markdown)** | `backend/reports/phase4mr_dataset_eligibility_decision.md` | Detailed gate-by-gate analysis and rationale |
| **Blockers & Remediation Plan** | `backend/reports/phase4mr_blockers_and_remediation_plan.md` | 7 active blockers and actionable 4-step remediation plan |
| **Test & Environment Report** | `backend/reports/phase4mr_test_and_environment_report.md` | Environment manifest, disk metrics, and test execution results (235 passed) |
