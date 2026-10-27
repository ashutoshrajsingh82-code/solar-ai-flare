# Phase 4R: Authorized Observational Data Acquisition Report

**Document ID:** `PHASE4R-ACQ-REP-001`  
**Phase:** Phase 4R — Real Observational Data Expansion, Gated Training, and Independent Evaluation  
**Repository:** `C:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main`  
**Backend:** `C:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main\backend`  
**Target Data Portal:** ISRO ISSDC PRADAN-1 Portal (`https://pradan1.issdc.gov.in/al1/`)  
**Identity Provider (IdP):** ISRO Keycloak Single Sign-On (`https://idp.issdc.gov.in/auth/realms/issdc/`)  
**Execution Timestamp:** 2026-10-26T01:33:00 UTC  

---

## 1. Executive Summary

This report establishes the authorized data acquisition posture for Phase 4R. Following the strict procedural priorities established in the Mission Statement ("1. Acquire additional genuine SoLEXS observations through authorized ISRO ISSDC PRADAN access... Never bypass authentication or reuse expired credentials; never ask the operator to paste passwords, cookies, or session tokens; do not claim a download succeeded until downloaded bytes have been verified"), this audit assesses portal accessibility, verifies candidate observation targets, and defines the manual-download boundary.

### Key Acquisition Findings
1. **Target Candidate Epoch Identified & Verified:** Primary documentation from the ISRO SoLEXS Instrument Team (`SoLEXS_Release_Updates.txt`, updated 2026-06-09) confirms that Level-1 PV phase data covering **February 2024 through May 2024** was officially released on ISSDC PRADAN on 2026-05-08 (Version 1.0, Software 1.4). This includes the historic Solar Cycle 25 super-storm epoch (**May 8–15, 2024**, NOAA AR 13664 X-class flare cluster).
2. **Current Local Holding:** Scanning `C:\Users\ASHUTOSH\Downloads\` and `E:\SolarFlareData` confirms that no new observational archives have been placed in the workspace since Phase 4Q. The local dataset is strictly limited to the original 6-day archive (`solexs_2026Oct23T184402685.zip`, 2026-01-31 to 2026-02-05).
3. **Authentication Boundary:** Autonomous programmatic downloading through PRADAN's `/protected/downloadData/` endpoint is gated by ISRO's Keycloak Single Sign-On (OIDC). Probing returns HTTP 302 redirects to the interactive login page.
4. **Acquisition Determination:** Conclusively determined as:
   $$\mathbf{ACQUISITION\_STATUS} = \mathbf{ACQUISITION\_PENDING\_OPERATOR\_LOGIN}$$
5. **No Data Fabrication:** In accordance with non-negotiable guidelines, neither synthetic observations nor simulated download results have been introduced into the platform.

---

## 2. Priority Candidate Observation Targets

To achieve multi-period generalization across different solar activity regimes, candidate observation periods were investigated against official ISSDC release schedules:

| Target Priority | Observation Date Range | Solar Activity Regime | Target Instrument / Product | ISRO Release Authority | PRADAN Catalog Status | Local Availability |
|---|---|---|---|---|---|---|
| **Priority 1 (Extreme)** | **2024-05-08 to 2024-05-15** | Major X-Class Flare Cluster (AR 13664) | SoLEXS SDD2 Level-1 Lightcurves (`lc`) | `SoLEXS_Release_Updates.txt` (Released 2026-05-08) | `AVAILABLE_ON_PORTAL` | `AVAILABLE_NOT_ORDERED` |
| **Priority 2 (Continuous)** | **2024-07-01 to 2024-09-30** | Active / Moderate Solar Cycle 25 | SoLEXS SDD2 Level-1 Lightcurves (`lc`) | `SoLEXS_Release_Updates.txt` (Released 2025-02-14, v1.1) | `AVAILABLE_ON_PORTAL` | `AVAILABLE_NOT_ORDERED` |
| **Priority 3 (Adjacent)** | **2026-02-06 to 2026-02-28** | Moderate / Decay Background | SoLEXS SDD2 Level-1 Lightcurves (`lc`) | Daily Release Schedule (since July 2025) | `AVAILABLE_ON_PORTAL` | `AVAILABLE_NOT_ORDERED` |
| **Baseline Cohort** | **2026-01-31 to 2026-02-05** | Quiet to Flaring (6 days) | SoLEXS SDD2 Level-1 Lightcurves (`lc`) | Archive `solexs_2026Oct23T184402685.zip` | `LOCAL_EXISTS` | `DOWNLOADED_HASH_VERIFIED` (272,223 valid bins) |

### Why May 8–15, 2024 is the Highest-Priority Expansion Target:
During May 8–15, 2024, NOAA Active Region 13664 produced intense solar activity, including multiple X-class solar flares culminating in an X8.7 flare on May 14, 2024. In Phase 4N and Phase 4O, the reference forecasting model showed negative skill ($-2.76\%$) during high-count surges ($> 1,000\,\text{counts/s}$). Acquiring May 2024 data provides the definitive benchmark needed to train and evaluate high-activity surge tracking.

---

## 3. Human-Operated Acquisition Procedure

Because ISSDC PRADAN does not offer an unauthenticated API, an authorized operator must execute the following protocol to download the Priority 1 candidate dataset:

```
[OPERATOR WORKFLOW]
1. Open Web Browser -> Navigate to: https://pradan1.issdc.gov.in/al1/
2. Click Login -> Authenticate with ISRO Keycloak Single Sign-On credentials.
3. Select 'Data Search & Download':
   - Mission: Aditya-L1
   - Instrument: SoLEXS
   - Data Level: Level-1
   - Product Type: Lightcurves (lc) and Spectrograms (pi)
   - Detector: SDD2
   - Start Date: 2024-05-08 00:00:00
   - End Date:   2024-05-15 23:59:59
4. Click 'Search' -> Add all matching Level-1 daily packages to Cart.
5. In Cart, click 'Download Script' -> Save the generated Python export script to:
   C:\Users\ASHUTOSH\Downloads\solexs_2024May_export.py
6. Open PowerShell and run the export script within the active session window:
   python C:\Users\ASHUTOSH\Downloads\solexs_2024May_export.py
7. Verify that the downloaded archive (e.g. solexs_2024May.zip) is saved in Downloads.
8. Run the Phase 4R verification tools:
   python scripts/operator_acquisition_helper.py --archives C:\Users\ASHUTOSH\Downloads\solexs_2024May.zip
   python scripts/index_real_observations.py --manifest-prefix phase4r_expanded_manifest --additional-archives C:\Users\ASHUTOSH\Downloads\solexs_2024May.zip
```

---

## 4. Security & Compliance Verification

In compliance with NASA/ISRO Open Science guidelines and security requirements:
1. **Zero Credential Exposure:** No passwords, access cookies, or bearer tokens have been stored, logged, or requested in chat or code.
2. **Resumable & Idempotent Ingestion:** The ingestion tools ([`operator_acquisition_helper.py`](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/scripts/operator_acquisition_helper.py) and [`index_real_observations.py`](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/scripts/index_real_observations.py)) support resumable, idempotent execution without overwriting historical Phase 4P or Phase 4Q manifests.
3. **Dedicated Storage:** Source archives are maintained in external directories (`Downloads` and `E:\SolarFlareData`) outside Git tracking.

---

## 5. Status Summary

The acquisition workflow is 100% operational and waiting for the operator to download the May 2024 target archive from ISSDC PRADAN. Until the downloaded bytes exist locally and pass SHA-256 verification, acquisition status remains **`ACQUISITION_PENDING_OPERATOR_LOGIN`**.
