# Phase 4M-R: Exact Blockers and Minimal Remediation Plan

**Audit Date**: 2026-10-25  
**Current Gate Status**: `BLOCKED_SCIENTIFIC_SEMANTICS_AND_LABELS_UNRESOLVED`  

---

## 1. Inventory of Active Blockers

Real observational model training cannot proceed until each of the following blockers is resolved by authoritative evidence. No assumptions, inferences, or library defaults may be substituted.

| Blocker ID | Affected Rules | Instrument / Target | Blocker Description | Required Authoritative Artifact |
|---|---|---|---|---|
| **`B-SLX-GTI-001`** | Rule R03 | SoLEXS SDD2 | Undocumented GTI `START`/`STOP` interval endpoint closure ($[a, b)$ vs $[a, b]$), overlap precedence, and empty/reversed interval handling. | Official, versioned SoLEXS Level-1 FITS SIS / Data Dictionary with explicit interval semantics. |
| **`B-SLX-PROV-002`** | Rule R05 | SoLEXS L1 | Undocumented schema compatibility across daily archives, pipeline versions (`solexs_pipeline-1.4`), and metadata fields (`OBS_ID`, `DATE`). | Official SoLEXS L1 provenance and product-version compatibility matrix. |
| **`B-SLX-CAL-003`** | Rule R06 | SoLEXS SDD2 | Lightcurve `RATE` is native counts/s; no audited conversion to physical flux ($W/m^2$) or uncertainty prescription exists for Level-1 light curves. | Official ISRO calibration document or CALDB specification authorizing an L1 lightcurve RATE-to-physical-flux conversion equation, response matrix, and uncertainty propagation. |
| **`B-HL-TIME-004`** | Rules R07, R08 | HEL1OS L1 | Both `MJD` and `ISOT` are stored; manual defines them but does not state which representation is authoritative when they disagree, nor allowed tolerance. | Versioned HEL1OS L1 FITS SIS / Time-Correlation specification defining primary time authority and discrepancy handling. |
| **`B-HL-QUALITY-005`** | Rule R14 | HEL1OS L1 | Housekeeping telemetry contains candidate saturation and pile-up counters (`fehkstat`, `czt1satctr1`, `cdte1pilectr`), but lacks bit definitions, numerical thresholds, or authorized row-rejection actions. | Official HEL1OS Level-1 HK/Quality specification defining bitmasks, saturation thresholds, and authorized rejection actions. |
| **`B-HL-VERSION-006`** | Rule R13 | HEL1OS L1 | Counterpart products exist for `2026-02-05` dump labeled `_lev1_V111` and `_lev1_V211`. Official release notes do not identify or govern these versions. | Official HEL1OS release note or supersession notice explicitly mapping V111 vs V211 and prescribing counterpart handling. |
| **`B-LBL-NEG-007`** | Gate 7 | Training Labels | The GOES Flare Report is an automated detection catalogue. Absence of an event does not prove solar quiescence. Defensible negative sample count is 0. | Defensible negative-label contract: an authoritative quiet-Sun event-absence certification (e.g. continuous GOES XRS baseline irradiance thresholding during verified instrument coverage) establishing verified non-flare intervals. |

---

## 2. Minimal Remediation Plan

To enable real-data training as soon as defensible without compromising scientific integrity, the following minimal sequential steps are required:

### Step 1: Submit Inquiry Requests to Instrument Teams
The formal inquiry drafts prepared in Phase 4K and updated in Phase 4L must be submitted through verified official ISRO/PRADAN channels:
- **`REQ-SLX-001`**: Submitted to the SoLEXS Principal Investigator / ISSDC data contact (covers `B-SLX-GTI-001`, `B-SLX-PROV-002`, `B-SLX-CAL-003`).
- **`REQ-HL-002`**: Submitted to the HEL1OS Principal Investigator / ISSDC data contact (covers `B-HL-TIME-004`, `B-HL-QUALITY-005`, `B-HL-VERSION-006`).

Draft artifacts:
- `backend/reports/phase4l_solexs_inquiry_draft_v2.md` (SHA-256: `d156f0f9290037eb7c040e17a8cb876d4f523b9c8473794a40b4583c30af6284`)
- `backend/reports/phase4l_hel1os_inquiry_draft_v2.md` (SHA-256: `093203526ed63efb0ccd54ac43325738c8859eaeeec5381fc4dce080d53cbde7`)

### Step 2: Acquire and Ingest Authoritative Artifacts
Upon receipt of official responses or updated documentation from ISSDC/PRADAN:
1. Deposit original artifacts into `E:\SolarFlareData\source_docs\isro\`.
2. Compute and log immutable SHA-256 digests.
3. Register the documents in `backend/reports/phase4e_document_register.json` following the response-ingestion procedure (`backend/reports/phase4l_response_ingestion_and_adjudication_v2.md`).
4. Re-run the canonical Phase 3C semantic audit (`python scripts/fits_semantics_audit.py`).

### Step 3: Establish a Defensible Quiet-Sun Contract
To resolve Blocker `B-LBL-NEG-007`:
1. Use continuous GOES 1-minute measurements (`goes_xrs_1min.csv` or `sci_xrsf-l2-flrpt_geo_y2026_v1-0-1.csv`) during verified instrument observation periods.
2. Formulate an explicit, documented physical irradiance threshold for solar quiescence (e.g. NOAA SWPC definition of B-level or below, $I_{XRS-B} < 10^{-7}\text{ W/m}^2$, verified across full forecast horizons).
3. Confirm that no satellite eclipse, flare tail, or solar proton event contaminated the quiet window.
4. Document the negative-label contract with mathematical bounds and peer-reviewed citations before assigning $Y = 0$.

### Step 4: Validate Dataset Eligibility Preflight
Once semantics and labels are verified:
1. Run `python scripts/training_readiness_dry_run.py` to confirm exit 0.
2. Confirm positive and negative class counts $\ge N_{min}$ across all splits.
3. Fit preprocessors and feature engineering strictly on the training partition.
4. Train a baseline classical model (e.g. Climatology, Logistic Regression, Random Forest) on real observational data.
5. Evaluate on validation data for threshold selection.
6. Evaluate once on the held-out chronological test set.
7. Register the resulting model as an experimental real-data model.
