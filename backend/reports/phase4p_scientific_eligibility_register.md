# Phase 4P: Scientific Eligibility & Blocker Register

**Document ID:** `PHASE4P-BLOCKER-REG-001`  
**Phase:** Phase 4P — Real Observational Data Expansion & Multi-Period Generalization  
**Repository:** `C:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main`  
**Backend:** `C:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main\backend`  
**Audit Scope:** Aditya-L1 SoLEXS & HEL1OS Observational Data Ingestion Pipelines  
**Execution Timestamp:** 2026-10-26T01:02:00 UTC  

---

## 1. Executive Summary

This register provides a formal, exhaustive evaluation of all active scientific blockers governing real-data ingestion, preprocessing, and model training in the Solar AI Flare platform. In accordance with Non-negotiable Requirement 4 ("Revisit active scientific blockers with existing evidence: B-HL-VERSION-006, B-HL-TIME-004, B-HL-QUALITY-005, B-SX-FLUX-003, B-SLX-GTI-001, B-LBL-NEG-007. Explain why none are currently resolved by existing evidence"), this audit confirms that **all six blockers remain ACTIVE and UNRESOLVED**.

No blocker has been cleared without official, authoritative ISSDC documentation or empirical calibration tables. Bypasses implemented for self-supervised sequence forecasting (such as contiguous valid-record segmentation) are explicitly documented as operational workarounds, not scientific resolutions.

---

## 2. Master Blocker Status Matrix

| Blocker ID | Instrument | Domain | Status | Impact on Model Training | Resolution Prerequisite |
|---|---|---|---|---|---|
| **`B-HL-VERSION-006`** | HEL1OS | Data Integrity / Pipeline | **ACTIVE (UNRESOLVED)** | Complete exclusion of HEL1OS Level-1 packages | Authoritative ISSDC release notes adjudicating V111 vs V211 pipeline differences |
| **`B-HL-TIME-004`** | HEL1OS | Time Semantics / Geocenter | **ACTIVE (UNRESOLVED)** | Prohibits multi-instrument cross-alignment | Specification of authoritative timestamp column (MJD vs ISOT) and light-travel correction |
| **`B-HL-QUALITY-005`** | HEL1OS | Instrument Quality / HK | **ACTIVE (UNRESOLVED)** | Prohibits filtering of invalid detector events | Published Housekeeping (HK) high-voltage saturation and temperature thresholds |
| **`B-SX-FLUX-003`** | SoLEXS | Physical Units / Radiometry | **ACTIVE (UNRESOLVED)** | Forecasts restricted to native `counts/s`; flux conversion prohibited | Calibrated Effective Area ($A_{\text{eff}}(E)$) curves and response matrix files (RMF/ARF) |
| **`B-SLX-GTI-001`** | SoLEXS | Telemetry Exposure / GTIs | **ACTIVE (BYPASSED FOR SELF-SUP)** | Auxiliary GTI tables excluded; segmentation done on contiguous records | Formal documentation of GTI start/stop edge convention relative to rate sampling bins |
| **`B-LBL-NEG-007`** | Platform | Supervised Ground Truth | **ACTIVE (UNRESOLVED)** | Prohibits binary flare classification; self-supervised forecasting only | Definitive quiet-Sun catalogue or solar background irradiance threshold agreement |

---

## 3. In-Depth Blocker Analysis

### 3.1 Blocker `B-HL-VERSION-006`: HEL1OS Pipeline Version Counterpart Collision
- **Affected Data:** Aditya-L1 HEL1OS Level-1 dump archives, specifically duplicate dump instances such as:  
  `HLS_20260205_000005_43185sec_lev1_V111.zip` vs `HLS_20260205_000005_43185sec_lev1_V211.zip`.
- **Scientific Issue:** The secondary archive contains simultaneous Level-1 packages covering identical observation durations with conflicting pipeline versions (`V111` and `V211`). A bit-level comparison reveals significant differences in event row counts, calibration factors, and flagging.
- **Repository Evidence:** The automated indexer ([`index_real_observations.py`](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/scripts/index_real_observations.py)) flagged this counterpart collision during inventory scanning. No document in the repository or ISSDC download manifest specifies whether V211 supersedes V111 or represents an experimental re-reduction.
- **Resolution Condition:** Official documentation from the HEL1OS Instrument Team (SAC/ISRO) detailing pipeline version change logs, specifying whether V211 deprecates V111, or providing a rule for multi-version reconciliation.

---

### 3.2 Blocker `B-HL-TIME-004`: HEL1OS Timestamp Authority & Discrepancies
- **Affected Data:** All HEL1OS CdTe and CZT event lists and light curve tables.
- **Scientific Issue:** HEL1OS FITS tables provide multiple timing references: Modified Julian Date (`MJD_START`, `MJD_END`) in primary headers, spacecraft clock seconds (`TIME`), and UTC ISO strings (`ISOT`). When converted to UTC seconds, discrepancies of up to several hundred milliseconds appear between MJD-derived and ISOT-derived timestamps. Furthermore, it is unstated whether light-travel time corrections from the Sun-Earth L1 Lagrange point to Earth geocenter have been applied.
- **Scientific Impact:** Cross-instrument correlation between HEL1OS hard X-rays (10–150 keV) and SoLEXS soft X-rays (2–22 keV) or GOES irradiance cannot be performed without sub-second timing fidelity.
- **Resolution Condition:** Authoritative specification from ISRO establishing the definitive primary timing column, clock drift correction status, and coordinate reference frame (Barycentric vs Geocentric vs Spacecraft Frame).

---

### 3.3 Blocker `B-HL-QUALITY-005`: Missing HEL1OS Quality & Housekeeping Saturation Thresholds
- **Affected Data:** HEL1OS High-Energy event packets.
- **Scientific Issue:** Fast hard X-ray bursts induce pulse pileup and detector dead-time in CZT/CdTe sensors. Level-1 HEL1OS data packages include Housekeeping (HK) tables containing sensor temperatures, bias voltages, and dead-time counters. However, no threshold criteria are defined to determine when an event is physically corrupted by HV instability or dead-time saturation.
- **Scientific Impact:** Ingesting unflagged saturated records introduces artificial truncation artifacts into model training.
- **Resolution Condition:** Publication of instrument quality threshold criteria (e.g. maximum acceptable dead-time fraction, HV tolerance ranges).

---

### 3.4 Blocker `B-SX-FLUX-003`: Physical Irradiance Flux Conversion
- **Affected Data:** All SoLEXS SDD1 and SDD2 light curve tables.
- **Scientific Issue:** SoLEXS Level-1 light curves record count rates in native $\text{counts/s}$ integrated across the 2–22 keV detector bandwidth. Converting $\text{counts/s}$ to physical radiant flux ($\text{W/m}^2$ or $\text{erg cm}^{-2} \text{s}^{-1}$) requires folding a solar spectral model $I(E)$ through the energy-dependent effective area $A_{\text{eff}}(E)$ and detector Redistribution Matrix Function (RMF):
  $$\text{Count Rate} = \int_{2\,\text{keV}}^{22\,\text{keV}} I(E) \cdot A_{\text{eff}}(E) \cdot R(E, E') \, dE$$
- **Current Status:** No calibrated Ancillary Response Files (ARF) or RMF response matrices are included in the ISSDC Level-1 distribution.
- **Scientific Safeguard:** All forecasting models, scalers, losses, and evaluation metrics operate strictly in native count-rate units ($\text{counts/s}$). Models are explicitly prohibited from predicting physical $\text{W/m}^2$ flux or GOES flare classes (C, M, X).

---

### 3.5 Blocker `B-SLX-GTI-001`: GTI Table Closure & Bin Edge Ambiguity
- **Affected Data:** SoLEXS SDD2 auxiliary Good Time Interval FITS files (`*.gti.gz`).
- **Scientific Issue:** The GTI auxiliary tables provide interval boundaries `[START, STOP]`. However, due to clock truncation and leap second conventions, the exact correspondence between GTI edges and the discrete 1.0s light curve bin centers is ambiguous at interval endpoints.
- **Self-Supervised Bypass:** In Phase 4M-S and maintained in Phase 4P, we bypass the GTI auxiliary tables for model windowing by segmenting strictly on valid records in the calibrated light curve:
  $$\Delta t_i = t_{i} - t_{i-1} = 1.0\,\text{s} \quad \text{and} \quad c_i \ge 0, \; c_i \in \mathbb{R}$$
  Any interval where $\Delta t_i \ne 1.0\,\text{s}$ or $c_i = \text{NaN}$ triggers an immediate segment boundary. This guarantees zero gap interpolation while leaving the formal scientific blocker `B-SLX-GTI-001` open for higher-level products.

---

### 3.6 Blocker `B-LBL-NEG-007`: Prohibiting Negative Flare Labels in Uncatalogued Intervals
- **Affected Data:** Platform Ground Truth Labeling & Catalog Matching.
- **Scientific Issue:** Standard flare catalogues (GOES, RHESSI, Hinode) compile events based on specific instrumental detection thresholds (e.g. 4-sigma above 4-hour background). Uncatalogued time periods frequently contain microflares, thermal loop brightenings, and active region evolution.
- **Methodological Rule:** Absence of a catalogued flare event in an interval must **never** be coded as a negative ("quiet-Sun") ground truth label for binary classification. Treating uncatalogued intervals as negative samples introduces massive label noise into supervised classifiers.
- **Impact:** The platform operates exclusively in a self-supervised autoregressive forecasting paradigm, learning the continuous temporal dynamics of solar X-ray count rates without synthetic binary labels.

---

## 4. Conclusion & Audit Summary

All six scientific blockers are actively enforced. The self-supervised forecasting model respects all physical boundaries:
1. HEL1OS remains quarantined under blockers `B-HL-VERSION-006`, `B-HL-TIME-004`, and `B-HL-QUALITY-005`.
2. SoLEXS forecasts remain bounded to native `counts/s` under `B-SX-FLUX-003`.
3. Contiguous segment extraction safely bypasses `B-SLX-GTI-001` without data distortion.
4. Binary classification remains blocked under `B-LBL-NEG-007`.
