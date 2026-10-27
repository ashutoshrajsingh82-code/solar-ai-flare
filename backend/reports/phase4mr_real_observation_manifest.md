# Phase 4M-R: Real Observation Inventory and Provenance Manifest

**Audit Date**: 2026-10-25  
**Canonical Status**: `READ_ONLY_AUDIT_COMPLETED`  
**Configured Data Root**: `E:\SolarFlareData`  
**Observational Archives Evaluated**:
1. SoLEXS Level-1: `C:\Users\ASHUTOSH\Downloads\solexs_2026Oct23T184402685.zip` (29,334,261 bytes, SHA-256: `34c39624194ee2da95bbf8e3dfed5ac002c55a517a31bc4091bfdd0d2b4671c2`)
2. HEL1OS Level-1: `C:\Users\ASHUTOSH\Downloads\hel1os_2026Oct23T184518914.zip` (1,066,706,121 bytes, SHA-256: `17654c33aff72313e796ceafb31577224f66f9f6b366438f8d655518ec033e17`)

---

## 1. Executive Summary of Available Products

The configured real-data root `E:\SolarFlareData` contains authoritative documentation and official GOES reference datasets, but does not contain raw or extracted FITS files directly. All candidate Aditya-L1 observational products reside within the verified downloaded nested ZIP archives and were inspected read-only in memory without disk extraction.

A total of **178 candidate FITS products** were structurally audited:
- **SoLEXS**: 24 products across 6 daily archive packages (`2026-01-31` to `2026-02-05`).
- **HEL1OS**: 154 products across 11 data dumps (`2026-02-01` to `2026-02-05`).

All 178 products are structurally readable and conform to basic FITS table structures. However, **structural readability is not scientific validity**. Every candidate product was assessed against instrument manuals, data contracts, and pipeline prerequisites. **178 of 178 products are currently EXCLUDED from real-data model training.**

---

## 2. Archive Provenance and Product Breakdown

### 2.1 SoLEXS Level-1 Products (24 Files)
- **Archive Structure**: Outer zip contains 6 day-wise zip packages:
  - `AL1_SLX_L1_20260131_v1.0.zip`
  - `AL1_SLX_L1_20260201_v1.0.zip`
  - `AL1_SLX_L1_20260202_v1.0.zip`
  - `AL1_SLX_L1_20260203_v1.0.zip`
  - `AL1_SLX_L1_20260204_v1.0.zip`
  - `AL1_SLX_L1_20260205_v1.0.zip`
- **Product Distribution**:
  - **Light Curves (`.lc.gz`)**: 6 products (SDD2). Exactly 86,400 rows per day (518,400 total 1-second records). Fields: `TIME` (Unix seconds since 1970-01-01 UTC) and `COUNTS` (counts/s over 2–22 keV).
  - **Good Time Intervals (`.gti.gz`)**: 12 products (6 SDD1, 6 SDD2). SDD1 GTI tables contain 0 rows (detector inactive/uncalibrated in this phase); SDD2 GTI tables contain 84 total intervals (average 14 intervals/day).
  - **Spectral Products (`.pi.gz`)**: 6 products (SDD2 Type-I PI spectra, 86,400 rows each, channel bins 0–1023).

### 2.2 HEL1OS Level-1 Products (154 Files)
- **Archive Structure**: Outer zip contains 11 dump archives covering 2026-02-01 to 2026-02-05:
  - 10 dumps labeled `_lev1_V111.zip`
  - 1 counterpart dump labeled `_lev1_V211.zip` (`HLS_20260205_000005_43185sec_lev1_V211.zip`, counterpart to identical dump name in V111)
- **Product Distribution**:
  - **Light Curves (`.fits`)**: 44 products (4 detector/band combinations per dump: CdTe1, CdTe2, CZT1, CZT2). Fields: `MJD`, `ISOT`, `CTR` (cts/sec), `STAT_ERR`.
  - **Good Time Intervals (`.fits`)**: 44 products (`gticdte1`, `gticdte2`, `gticzt1`, `gticzt2`).
  - **Spectra (`.fits`)**: 44 products (PHA spectra).
  - **Housekeeping (`.fits`)**: 11 products (`hk_dump.fits`). Contains candidate quality/status counters (`fehkstat`, `czt1satctr1`, `cdte1pilectr`, etc.).
  - **Event Lists (`.fits`)**: 11 products (event-mode photon records).

---

## 3. Documentary Grounding and Measurement Meanings

| Instrument | Product / Extension | Key Fields | Documented Meaning & Units | Documentary Source & Status |
|---|---|---|---|---|
| **SoLEXS** | SDD2 Lightcurve (`RATE`) | `TIME` | Elapsed seconds since 1970-01-01 00:00:00 UTC | *SoLEXS User Manual v1.0* p.6 §2.2 (`VERIFIED`) |
| **SoLEXS** | SDD2 Lightcurve (`RATE`) | `COUNTS` | Native counts per second (2–22 keV) | *SoLEXS User Manual v1.0* p.5 §2.1.2 (`VERIFIED`) |
| **SoLEXS** | SDD2 GTI (`GTI`) | `START`, `STOP` | Interval boundaries in Unix seconds | *SoLEXS User Manual v1.0* p.5 §2.1.1 (`PARTIALLY_RESOLVED`: endpoint closure rule unstated) |
| **HEL1OS** | LC (`LIGHTCURVE`) | `MJD` | Modified Julian Date | *HEL1OS User Manual v1.2* Appendix B.3 (`PARTIALLY_RESOLVED`: authority vs ISOT unstated) |
| **HEL1OS** | LC (`LIGHTCURVE`) | `ISOT` | UTC timestamp string corresponding to MJD | *HEL1OS User Manual v1.2* Appendix B.3 (`PARTIALLY_RESOLVED`: authority vs MJD unstated) |
| **HEL1OS** | LC (`LIGHTCURVE`) | `CTR` | Native count rate (cts/sec) | *HEL1OS User Manual v1.2* Appendix B.3 (`VERIFIED`) |
| **HEL1OS** | LC (`LIGHTCURVE`) | `STAT_ERR` | Poisson uncertainty on count rate (cts/sec) | *HEL1OS User Manual v1.2* Appendix B.3 (`VERIFIED`) |
| **HEL1OS** | GTI (`GTI`) | `START`, `STOP` | Interval covering whole dump | *HEL1OS User Manual v1.2* p.13 §4 (`VERIFIED`: manual explicitly notes L1 GTIs may be ignored) |
| **HEL1OS** | HK (`HOUSEKEEPING`) | `fehkstat`, `czt1satctr1`, etc. | Front-end and saturation counters | *HEL1OS User Manual v1.2* Appendix B.4 (`PARTIALLY_RESOLVED`: no bitmask or row-rejection threshold) |

---

## 4. Per-Product Decision and Exclusion Rationale

| Product Category | Count | Status | Authoritative Reason for Exclusion |
|---|---:|---|---|
| **SoLEXS SDD2 Lightcurves** | 6 | `EXCLUDED` | **(1)** `RATE` is native counts/s; no validated conversion to physical flux ($W/m^2$) is documented for L1 (Blocker `B-SLX-CAL-003`). Existing pipeline requires physical flux. **(2)** GTI `START`/`STOP` endpoint closure and overlap precedence are undocumented (Blocker `B-SLX-GTI-001`). **(3)** No quality mask exists in RATE. **(4)** Trustworthy negative labels cannot be established from the GOES catalogue without fabricating negatives. |
| **SoLEXS SDD1 GTI Tables** | 6 | `EXCLUDED` | Empty (0 intervals). Auxiliary exposure product. |
| **SoLEXS SDD2 GTI Tables** | 6 | `EXCLUDED` | 84 intervals present, but missing authoritative interval endpoint closure rules (`[a, b)` vs `[a, b]`) and overlap precedence. Cannot filter lightcurves without an invented convention. |
| **SoLEXS SDD2 PI Spectra** | 6 | `EXCLUDED` | Energy-binned spectral histogram product, not a continuous time-series observation for lightcurve nowcasting/forecasting. |
| **HEL1OS Lightcurves (V111)** | 40 | `EXCLUDED` | **(1)** Discrepancy authority between stored `MJD` and `ISOT` is undocumented (Blocker `B-HL-TIME-004`). **(2)** HK quality/saturation counters lack documented thresholds or rejection actions (Blocker `B-HL-QUALITY-005`). **(3)** Counterpart relationship between V111 and V211 for 2026-02-05 is unadjudicated (Blocker `B-HL-VERSION-006`). **(4)** Uncalibrated count rates. **(5)** No trustworthy negative labels. |
| **HEL1OS Lightcurves (V211)** | 4 | `EXCLUDED` | Exact counterpart to V111 dump `HLS_20260205_000005_43185sec`. Official release notes do not identify V111/V211 or establish supersession precedence. |
| **HEL1OS GTI Tables** | 44 | `EXCLUDED` | Cover entire dump; documented as ignorable for Level-1 analysis; do not provide intra-dump quality/exposure filtering. |
| **HEL1OS Housekeeping** | 11 | `EXCLUDED` | Auxiliary telemetry; candidate quality flags lack bitwise decision rules. |
| **HEL1OS PHA Spectra & Events** | 55 | `EXCLUDED` | Spectral and photon-list formats, not sequence lightcurves. |

**Machine-Readable Companion**: `backend/reports/phase4mr_real_observation_inventory.json`
