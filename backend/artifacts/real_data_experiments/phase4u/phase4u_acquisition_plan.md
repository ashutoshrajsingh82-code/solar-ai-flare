# Phase 4U: Independent Observational Epoch Acquisition Plan

**Project:** Solar AI Flare  
**Backend:** `backend/`  
**Phase:** Phase 4U — Multi-Epoch Real-Data Expansion and Untouched Holdout Design  
**Date:** 2026-10-26  
**Status:** `ACQUISITION_PENDING_OPERATOR_PRADAN_LOGIN`

---

## 1. Executive Summary

This document specifies the scientific justification, target selection, product query manifest, and operator procedure for acquiring a **genuinely independent, untouched third observational epoch** of Level-1 SoLEXS data from the ISRO ISSDC PRADAN portal.

In accordance with strict methodological governance, all 14 calendar days of locally available SoLEXS observations (May 8–15, 2024 and January 31–February 5, 2026) are formally classified as **previously examined development data**. To establish true forecasting generalization and prevent test-set optimization, an unexamined third epoch must be reserved as a quarantined holdout before any new model development begins.

---

## 2. Scientific Selection of Reserved Epoch 3

### Primary Target: October 1–7, 2024 (`P-2024Q4-OCT`)

| Parameter | Specification | Scientific Rationale |
| :--- | :--- | :--- |
| **Target Date Range** | **2024-10-01T00:00:00Z to 2024-10-07T23:59:59Z** | 7 contiguous calendar days (168 hours). |
| **Solar Activity Context** | **Extreme Solar Flaring (Active Region 13842)** | AR 13842 generated multiple major flares, including the **historic X9.0 flare on October 3, 2024** (the second-largest solar flare of Solar Cycle 25), accompanied by intense M-class eruptions. |
| **Temporal Separation** | **Distinct Solar Epoch** | Separated by **~5 months** from the May 2024 development epoch and **~16 months** from the Feb 2026 epoch. |
| **ISRO Data Release** | **Data Version 1.0 / Software Version 1.1** | Released officially on 2025-02-14 in batch release Q4-2024. |
| **Detector State** | **SoLEXS SDD2 Nominal** | Active coronal observations under post-PV mission phase thermal equilibrium. |
| **Physical Distribution** | **Diverse Activity Regimes** | Spans quiet pre-flare background ($< 200\text{ c/s}$), moderate flare rises ($200–1,000\text{ c/s}$), and extreme flare peaks ($> 1,000\text{ c/s}$ up to tens of thousands of counts/s). |

### Secondary / Alternative Candidate Epochs

If PRADAN availability or telemetry restrictions impact October 2024, the following ranked alternatives are designated:

1. **Candidate 3B: July 14–20, 2024 (`P-2024Q3-JUL`)**
   - Active Region 13738 / 13740 flaring sequence.
   - ISRO Data Version 1.1 (UTC aligned timestamps, released 2025-02-14).
2. **Candidate 3C: January 10–16, 2025 (`P-2025Q1-JAN`)**
   - Mid-cycle operational phase.
   - ISRO Software Version 1.2 pipeline (released 2025-04-17).

---

## 3. Product Acquisition Manifest

The exact query parameters required for ISRO PRADAN are defined below:

```yaml
acquisition_manifest:
  portal: "ISRO ISSDC PRADAN"
  portal_url: "https://pradan.issdc.gov.in/pradan/"
  mission: "Aditya-L1"
  instrument: "SoLEXS"
  processing_level: "Level-1"
  subsystem: "SDD2"
  data_product_types:
    - "LIGHTCURVE (.lc.gz)"
    - "GTI (.gti.gz)"
  observation_dates:
    start_date_utc: "2024-10-01T00:00:00Z"
    end_date_utc: "2024-10-07T23:59:59Z"
  expected_daily_packages:
    - "AL1_SLX_L1_20241001_v1.0.zip"
    - "AL1_SLX_L1_20241002_v1.0.zip"
    - "AL1_SLX_L1_20241003_v1.0.zip"
    - "AL1_SLX_L1_20241004_v1.0.zip"
    - "AL1_SLX_L1_20241005_v1.0.zip"
    - "AL1_SLX_L1_20241006_v1.0.zip"
    - "AL1_SLX_L1_20241007_v1.0.zip"
  target_local_archive: "C:\\Users\\ASHUTOSH\\Downloads\\solexs_2024Oct_holdout.zip"
```

---

## 4. Step-by-Step Operator Acquisition Checklist

Because ISRO PRADAN enforces authenticated user sessions and CAPTCHA validation, the download **cannot and must not be automated by scraping or credential bypass**. The human operator must complete the following steps:

1. **Portal Sign-in:**
   - Open a browser and navigate to `https://pradan.issdc.gov.in/pradan/`.
   - Log in with authorized ISRO ISSDC user credentials.
2. **Search Criteria Configuration:**
   - Select Mission: **Aditya-L1**.
   - Select Instrument: **SoLEXS**.
   - Select Level: **Level-1**.
   - Set Date Range: From **2024-10-01** To **2024-10-07**.
3. **Cart & Product Selection:**
   - Verify that all 7 daily packages (`20241001` through `20241007`) are returned in search results.
   - Add all 7 packages to the download cart.
4. **Download Execution:**
   - Generate the download script or download the consolidated archive directly.
   - Save the resulting zip file to:
     `C:\Users\ASHUTOSH\Downloads\solexs_2024Oct_holdout.zip`
5. **Checksum & Provenance Verification:**
   - Record the file size and compute the SHA-256 hash.
   - Run the automated intake verifier:
     ```powershell
     python scripts/phase4u_inventory_and_contract.py
     ```
   - Ensure `testzip()` returns zero corrupt members.

---

## 5. Scientific Quarantine Notice

> [!CAUTION]
> **STRICT RETENTION AS UNTOUCHED HOLDOUT:**  
> Once acquired, the October 2024 dataset must be locked immediately under read-only permissions. Zero exploratory model evaluations, loss tuning, or hyperparameter adjustments against October 2024 data are permitted. The data must remain completely unexamined until the final evaluation gate of Phase 4V.
