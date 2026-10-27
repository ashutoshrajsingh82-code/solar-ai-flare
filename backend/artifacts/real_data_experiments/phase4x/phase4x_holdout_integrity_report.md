# Phase 4X: Independent Holdout Acquisition Verification and Integrity Report

**Phase:** Phase 4X — Independent Holdout Acquisition Verification and Frozen-Model Evaluation  
**Execution Timestamp:** 2026-10-26T17:18:00Z  
**Repository:** `solar-ai-flare`  
**Target Holdout Epoch:** October 1–7, 2024 (Period ID: `P-2024Q4-OCT`)  
**Final Status:** `BLOCKED_INDEPENDENT_HOLDOUT_NOT_ACQUIRED`  
**Holdout Evaluation Executed:** **NO** (Evaluation strictly prohibited and blocked pending legitimate acquisition)

---

## 1. Executive Summary and Primary Determination

Phase 4X governs the independent holdout evaluation of the frozen Phase 4V multi-epoch CNN-LSTM against the reserved **October 1–7, 2024** observation epoch, in accordance with the pre-registered Phase 4W protocol.

Prior to opening any evaluation pipeline, an exhaustive audit was conducted across local storage and repository directories to verify whether the reserved October 2024 Level-1 SoLEXS data has been acquired.

### Primary Determination
$$\mathbf{Status:\ BLOCKED\_INDEPENDENT\_HOLDOUT\_NOT\_ACQUIRED}$$

1. **Absence of Holdout Data:** The reserved October 1–7, 2024 observation epoch is **not present locally** in `C:\Users\ASHUTOSH\Downloads\`, within the repository, or in any configured data storage path.
2. **Evaluation Non-Execution:** In strict compliance with Governance Safeguards 2, 3, 4, and 5, **no holdout evaluation occurred**, no synthetic data was generated, and no development data was substituted.
3. **Frozen State Maintained:** The Phase 4V model weights ([`solexs_phase4v_cnn_lstm_20261026T163620Z.pt`](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/artifacts/real_data_experiments/phase4v/solexs_phase4v_cnn_lstm_20261026T163620Z.pt)), normalizer parameters, preprocessing routines, and inference scripts remain strictly frozen and unaltered.

---

## 2. Exhaustive Local Storage and Inventory Audit

A comprehensive scan of local storage was executed to establish data readiness:

| Search Target / Directory | Search Pattern | Files Found | Status / Finding |
|---|---|---|---|
| `C:\Users\ASHUTOSH\Downloads\` | `*solexs*` | 8 items | Only May 2024 and Feb 2026 archives exist; zero October 2024 files. |
| `C:\Users\ASHUTOSH\Downloads\` | `*202410*` | 0 items | Zero October 2024 packages found. |
| `solar-ai-flare-main/backend/` | `*202410*` | 0 items | Zero October 2024 observation files found in codebase. |
| `solar-ai-flare-main/backend/` | `*13842*` | 0 items | Zero AR 13842 observation files found. |

### Status of Existing Local Observation Archives
The local filesystem contains only the two previously examined development archives:
- **Epoch 1 (May 2024):** `solexs_2026Oct25T200126271.zip` (92.59 MB, SHA256: `8df85ed75b6f00052d7f45a6764b187c700977f931504fb79fdfe56f10df9db2`). Used in Phase 4S, 4T, 4V, 4W.
- **Epoch 2 (Feb 2026):** `solexs_2026Oct23T184402685.zip` (29.33 MB, SHA256: `34c39624194ee2da95bbf8e3dfed5ac002c55a517a31bc4091bfdd0d2b4671c2`). Used in Phase 4M-S, 4N, 4O, 4R, 4V, 4W.

Neither archive contains observations from October 2024.

---

## 3. Scientific Context and Non-Substitution Mandate

The reserved holdout period spans **October 1–7, 2024**:
- **Coronal Active Region:** Captures Active Region 13842 during its transit across the solar disk, including the major eruptive sequence featuring the historic **X9.0 flare on October 3, 2024**.
- **Instrument Software Version:** Represents the first operational epoch processed under SoLEXS Level-1 Pipeline **Version 1.1 / Data Version 1.0** (released officially by ISRO ISSDC on 2025-02-14), providing a genuine calibration baseline distinct from the May 2024 PV phase.
- **Strict Prohibition Against Data Substitution:**
  - Under no circumstances may development data (May 2024 or February 2026) be re-evaluated as an "independent holdout".
  - Synthetic data generation (e.g. Poisson or pseudo-flare simulations) is strictly prohibited as a holdout substitute.
  - Splicing or interpolating unobserved periods is strictly forbidden.

---

## 4. Authorized ISRO ISSDC PRADAN Acquisition Protocol

In Phase 4P and 4Q, live network probing of the PRADAN download endpoint (`https://pradan1.issdc.gov.in/al1/protected/downloadData/`) confirmed that automated unauthenticated downloading is blocked by ISRO's Keycloak Single Sign-On (SSO) gateway (HTTP 302 redirect to `https://idp.issdc.gov.in/auth/`).

Bypassing authentication or scraping the portal is prohibited by ISRO policy and project governance. Acquisition requires human operator authentication:

### Operator Standard Operating Procedure (SOP)
1. **Navigate to Portal:** Open an authorized web browser and navigate to:
   $$\text{URL: } \texttt{https://pradan1.issdc.gov.in/al1/}$$
2. **Authenticate:** Log in using verified ISRO ISSDC PRADAN user credentials.
3. **Configure Search Filters:**
   - **Payload / Instrument:** `SoLEXS` (Solar Low Energy X-ray Spectrometer)
   - **Data Processing Level:** `Level-1`
   - **Observation Time Range:** `2024-10-01 00:00:00 UTC` to `2024-10-07 23:59:59 UTC`
4. **Product Selection:** Select all daily SDD2 1-second cadence lightcurve FITS products (`al1_solexs_l1_*.fits`).
5. **Cart Checkout & Download:** Add the 7 daily packages to the cart and download the official zip archive to `C:\Users\ASHUTOSH\Downloads`.
6. **Provenance Logging:** Preserve the download metadata, including the official session token hash (redacted), download timestamp, and package manifest.

---

## 5. Pre-Registered Holdout Integrity Verification Protocol (Post-Acquisition Gate)

Once the October 2024 archive is placed in `C:\Users\ASHUTOSH\Downloads\`, the following verification gates must execute and pass before any model evaluation is permitted:

1. **Bitstream Checksum Verification:** Compute and log SHA-256 hashes for the parent zip archive and each nested FITS file.
2. **FITS Header Schema Compliance:** Verify PRIMARY and LIGHTCURVE HDU headers:
   - `TELESCOP = 'ADITYA-L1'`
   - `INSTRUME = 'SOLEXS'`
   - `DETECTOR = 'SDD2'`
   - `DATATYPE = 'LIGHTCURVE'`
   - `CREATOR = 'SoLEXS_L1_Pipeline_v1.1'`
3. **Temporal Uniformity & Cadence:**
   - Validate 1-second time bins ($\Delta t = 1.0\text{ s} \pm 0.05\text{ s}$).
   - Inspect Good Time Intervals (GTI) table for orbit night, passage flags, and South Atlantic Anomaly (SAA) exclusions.
   - Enforce segment continuity ($\ge 70\text{ s}$ per contiguous segment, zero interpolation across missing bins).
4. **Quality Indicator & Non-Finite Masking:** All NaN or negative count rates must be masked out and excluded from evaluation windows.

---

## 6. Formal Statement of Evaluation Non-Execution

- **Holdout Evaluation Executed:** **NO.**
- **October 2024 Inferences Generated:** **0.**
- **Checkpoints Modified:** **0.**
- **Governance Compliance:** **100%.**
- **Action Required:** Operator must perform authenticated PRADAN acquisition of the October 1–7, 2024 dataset to unblock Phase 4X evaluation.
