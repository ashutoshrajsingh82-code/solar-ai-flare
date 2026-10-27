# Phase 4Q: Scientific Eligibility Audit & Blocker Register

**Document ID:** `PHASE4Q-ELIG-AUDIT-001`  
**Phase:** Phase 4Q — Authorized Observational Data Acquisition, Scientific Eligibility Gate, and Multi-Period Readiness  
**Repository:** `C:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main`  
**Backend:** `C:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main\backend`  
**Authoritative Documents Audited:**
1. `SoLEXS-UserManual.pdf` (Aditya-L1 SoLEXS Data Analysis Guide Version 1.0, ISRO URSC)
2. `SoLEXS_Release_Updates.txt` (Updated 2026-06-09, ISRO SoLEXS Instrument Team)
3. `HEL1OS_ReleaseNotes.pdf` (Version 1.0, Aug 20, 2026, ISRO SAC/ISSDC)
4. `HEL1OS_UserManual.pdf` (Version 1.2, ISRO)
**Execution Timestamp:** 2026-10-26T01:23:00 UTC  

---

## 1. Executive Summary

This document performs a comprehensive scientific audit of all data ingestion and labeling gates in the Solar AI Flare platform. In accordance with Phase 4Q Requirement 5 ("Use the existing Phase 4P blocker register; do not clear any blocker based solely on code convenience or a successful FITS parse"), this audit evaluates new primary documentation discovered in the local environment and formally records the adjudication status of all six active blockers.

### Master Determination
**All six scientific blockers remain ACTIVE and UNRESOLVED.**  
While the discovered primary documentation provides invaluable clarification on instrument operations and pipeline history, none of the formal criteria required to clear these blockers have been satisfied. Native SoLEXS SDD2 count-rate forecasting remains the only scientifically eligible task.

---

## 2. Master Blocker Status & Evidence Register

| Blocker ID | Instrument | Scientific Domain | Reviewer Decision | Decision Date | Authoritative Document Reference | Key Passage / Technical Rationale |
|---|---|---|---|---|---|---|
| **`B-HL-VERSION-006`** | HEL1OS | Pipeline Version Collision | **ACTIVE (UNRESOLVED)** | 2026-10-26 | `HEL1OS_ReleaseNotes.pdf` (Page 4, Data Release 2) | Pipeline re-generated products to patch clock roll-overs and re-structure folders; however, both V111 and V211 remain simultaneously in PRADAN Level-1 dumps without automated deduplication. |
| **`B-HL-TIME-004`** | HEL1OS | Timing Convention Authority | **ACTIVE (UNRESOLVED)** | 2026-10-26 | `HEL1OS_UserManual.pdf` (Section 3.1) | Sub-second timing discrepancies persist between MJD headers and ISOT time columns; light-travel time corrections to Earth geocenter remain unstandardized. |
| **`B-HL-QUALITY-005`** | HEL1OS | Sensor Saturation Criteria | **ACTIVE (UNRESOLVED)** | 2026-10-26 | `HEL1OS_ReleaseNotes.pdf` (Page 5, Data Release 3) | Temperature-read module on CZT caused intermittent event gaps until June 2024; hardware dead-time saturation thresholds remain uncalibrated for high-fluence bursts. |
| **`B-SX-FLUX-003`** | SoLEXS | Count-to-Flux Conversion | **ACTIVE (UNRESOLVED)** | 2026-10-26 | `SoLEXS-UserManual.pdf` (Section 2.4 & Section 4) | Level-1 light curves provide native count rates (`counts/s`, 2–22 keV). Conversion to physical flux ($\text{W/m}^2$) requires forward-folding spectral models with CALDB ARF/RMF response matrices in XSPEC/Sherpa. |
| **`B-SLX-GTI-001`** | SoLEXS | GTI Endpoint Semantics | **ACTIVE (BYPASSED FOR SELF-SUP)** | 2026-10-26 | `SoLEXS-UserManual.pdf` (Section 2.1.1) | Auxiliary GTI exposure tables define observation windows, but bin edge truncation ambiguity remains. Bypassed by segmenting strictly on continuous valid records ($\Delta t = 1.0\text{s}$, $c_i \ge 0$). |
| **`B-LBL-NEG-007`** | Platform | Ground Truth Labeling | **ACTIVE (UNRESOLVED)** | 2026-10-26 | GOES Flare Catalog ReadMe & Event Inventory | Uncatalogued intervals cannot be coded as negative ("quiet-Sun") ground truth. Supervised classification remains blocked; self-supervised sequence forecasting is enforced. |

---

## 3. In-Depth Analysis of Individual Blockers

### 3.1 Blocker `B-HL-VERSION-006`: HEL1OS V111 vs V211 Pipeline Precedence
- **Audit Findings:** `HEL1OS_ReleaseNotes.pdf` (Section "Data Release 2: Feb 01, 2025", page 4) explicitly clarifies the origin of the version counterpart collision:
  > *"In the second release of the HEL1OS payload data... the data pertaining to the period Jul 01, 2024 till Sep 30, 2024 which was released in the first data release have been re-generated... Owing to the above changes, users who had downloaded the data after the first data release... are requested to use the regenerated products which are part of the second release."*
- **Reason Blocker Remains Active:** In the local archive `hel1os_2026Oct23T184518914.zip`, dump `HLS_20260205_000005_43185sec` contains both `_lev1_V111.zip` and `_lev1_V211.zip`. PRADAN distributes both versions concurrently. Without an automated, authoritative reconciliation rule in the ingestion pipeline that deterministically supersedes V111 without data loss or schema mismatch, HEL1OS Level-1 packages remain quarantined under `EXCL-HEL1OS-BLOCKERS`.

---

### 3.2 Blocker `B-HL-TIME-004`: HEL1OS Timing Semantics & Geocenter Travel Time
- **Audit Findings:** Neither the `HEL1OS_UserManual.pdf` nor `HEL1OS_ReleaseNotes.pdf` resolves the fundamental timing discrepancy between Modified Julian Date (`MJD_START`) headers and ISOT time columns. Furthermore, because Aditya-L1 orbits the Sun-Earth L1 Lagrange point approximately $1.5 \times 10^6\,\text{km}$ sunward of Earth, X-rays arrive at Aditya-L1 approximately 5.0 seconds before arriving at Earth geostationary satellites (GOES):
  $$\Delta t_{\text{L1-Earth}} = \frac{d_{\text{L1}}}{c} \approx \frac{1.5 \times 10^9\,\text{m}}{3.0 \times 10^8\,\text{m/s}} \approx 5.0\,\text{s}$$
- **Reason Blocker Remains Active:** No standardized geocenter light-travel time correction has been applied to HEL1OS Level-1 timestamps. Mixing HEL1OS with SoLEXS or GOES without sub-second spatial-temporal synchronization would inject phase misalignment into autoregressive sequence models.

---

### 3.3 Blocker `B-HL-QUALITY-005`: HEL1OS Quality & Dead-Time Saturation Thresholds
- **Audit Findings:** Page 5 of `HEL1OS_ReleaseNotes.pdf` notes:
  > *"The CZT's read-out application specific integrated circuit (ASIC) provides a Temperature-read module which reads the device temperature intermittently... resulting in gaps in the CZT eventlist... For any meaningful scientific interpretation of the data, users are requested to get in touch with the HEL1OS instrument team."*
- **Reason Blocker Remains Active:** Instrument-level data gaps and dead-time pileup during strong solar flares cannot be filtered automatically without official saturation threshold curves.

---

### 3.4 Blocker `B-SX-FLUX-003`: Validated SoLEXS Count-Rate to Physical Flux Conversion
- **Audit Findings:** `SoLEXS-UserManual.pdf` (Section 2.4, page 7 and Section 4, page 10) provides definitive authoritative guidance:
  > *"The Calibration Database (CALDB) for SoLEXS provides essential files that describe the instrument's response characteristics... Auxiliary Response File (ARF) and Redistribution Matrix File (RMF)... The most important radiation mechanism producing X-ray emission in SoLEXS's energy range are thermal bremsstrahlung continuum and line emission... This process requires spectral fitting of the data from Type-I Pulse Invariant (PI) spectral files and the calibration files (ARF and RMF). Tools such as XSPEC, Sherpa, or OSPEX can be used for spectral fitting."*
- **Scientific Impact:** Converting SoLEXS count rates ($2\text{--}22\,\text{keV}$) to physical radiant flux ($\text{W/m}^2$, $1.0\text{--}8.0\,\text{Å}$) cannot be achieved by scalar multiplication. It is a nonlinear inverse problem requiring thermal plasma model forward-folding.
- **Enforced Safeguard:** The platform strictly prohibits describing count-rate forecasts as physical flux ($\text{W/m}^2$) or claiming GOES flare classification capability. All operations remain in native units ($\text{counts/s}$).

---

### 3.5 Blocker `B-SLX-GTI-001`: SoLEXS GTI Endpoint Semantics
- **Audit Findings:** Section 2.1.1 of `SoLEXS-UserManual.pdf` describes the GTI file as listing time intervals where the detector was collecting valid data based on FOV, detector health, and operational state.
- **Bypass Validation:** To eliminate GTI bin-edge truncation ambiguity, the self-supervised pipeline bypasses GTI files and segments directly on the calibrated 1 Hz rate table:
  $$\Delta t_i = t_i - t_{i-1} = 1.0\,\text{s} \quad \text{and} \quad c_i \ge 0, \; c_i \in \mathbb{R}$$
  This operation is fully validated and verified with automated unit tests. The blocker remains open for higher-level Level-2 physical flux pipelines.

---

### 3.6 Blocker `B-LBL-NEG-007`: Defensible Negative Ground Truth Labels
- **Audit Findings:** No catalogue or satellite instrument provides an authoritative "absence of flaring activity" label. Solar coronal heating consists of continuous micro-flaring below GOES A-class background levels.
- **Enforced Safeguard:** Treating uncatalogued intervals as negative samples is prohibited. The platform operates exclusively in a self-supervised autoregressive forecasting paradigm, learning continuous temporal dynamics without binary labels.

---

## 4. Scientific Task Eligibility Summary

| Task Description | Eligibility Status | Permitted Scope & Safeguards |
|---|---|---|
| **Self-Supervised SDD2 Count-Rate Sequence Forecasting** | **ELIGIBLE** | Native `counts/s` (2–22 keV), 60s input history, 10s forecast horizon, gap-free contiguous segments. |
| **Physical Irradiance Forecasting ($\text{W/m}^2$)** | **BLOCKED** | Blocked by `B-SX-FLUX-003`; requires CALDB forward-folding in XSPEC. |
| **Binary / Multi-Class Flare Classification** | **BLOCKED** | Blocked by `B-LBL-NEG-007`; negative flare intervals cannot be defined defensively. |
| **Multi-Instrument HEL1OS + SoLEXS Fusion** | **BLOCKED** | Blocked by `B-HL-VERSION-006`, `B-HL-TIME-004`, and `B-HL-QUALITY-005`. |
| **Automated Data Expansion from PRADAN** | **READY (PENDING LOGIN)** | Operator SOP established; awaits authenticated user download execution. |
