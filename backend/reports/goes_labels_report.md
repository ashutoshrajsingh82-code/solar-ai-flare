# GOES-Based Solar Flare Labeling Report

**Generated UTC:** 2026-10-26T17:41:38.635908+00:00  
**Date Range:** 2026-01-31 00:00:00+00:00 to 2026-02-05 23:59:00+00:00  
**Total Timestamps:** 8,640  
**Cadence:** 1min  

---

## 1. Flare Event Catalog Summary

- **Total Events in Range:** 115
- **M/X Class Events:** 68 (59.1%)

| Flare Class | Event Count | Percentage |
| :---: | :---: | :---: |
| **X** | 6 | 5.2% |
| **M** | 62 | 53.9% |
| **C** | 47 | 40.9% |
| **B** | 0 | 0.0% |
| **A** | 0 | 0.0% |

---

## 2. Nowcast State Class Balance

Four-class state alignment ($t$ is state at time $t$):
- **FLARE:** `start <= t <= peak`
- **DECAY:** `peak < t <= end`
- **PRE_FLARE:** `t within 30m before flare onset` (class $\ge C$)
- **QUIET:** background flux conditions

| State Name | State ID | Row Count | Percentage |
| :--- | :---: | :---: | :---: |
| **QUIET** | 0 | 3,881 | 44.92% |
| **PRE_FLARE** | 1 | 2,568 | 29.72% |
| **DECAY** | 2 | 925 | 10.71% |
| **FLARE** | 3 | 1,266 | 14.65% |

---

## 3. Forecast Target Horizon Analysis

Binary target $y_H(t) = 1$ if flare onset of class $\ge M$ occurs in $(t, t+H]$.

| Horizon | Positive Rows (Valid Only) | Valid Evaluated Rows | Positive Rate (%) | NaN Rows (Masked / Boundary) |
| :---: | :---: | :---: | :---: | :---: |
| **1h** | 2,514 | 6,067 | **41.44%** | 2,573 |
| **3h** | 3,885 | 5,698 | **68.18%** | 2,942 |
| **6h** | 3,977 | 5,185 | **76.70%** | 3,455 |
| **12h** | 3,557 | 4,216 | **84.37%** | 4,424 |
| **24h** | 2,576 | 2,674 | **96.34%** | 5,966 |

---

## 4. Masking and Anti-Leakage Safeguards

- **Total Rows Masked (`forecast_valid = False`):** 2,239 (25.91%)
- **Total Rows Retained (`forecast_valid = True`):** 6,401 (74.09%)
- **Masking Rules Applied:**
  1. *Ongoing Flare/Decay:* Timestamps within active flares of class $\ge C$ are masked to prevent target leakage.
  2. *Coverage Boundary:* Timestamps where $t + H$ exceeds observation coverage are strictly set to `NaN` (never imputed as negative 0).
  3. *Flux Gaps:* Intervals with GOES gaps $\ge 10.0$ minutes invalidate overlapping horizons.

---

## 5. Scientific Integrity Statement

All labels were derived purely from genuine NOAA/HEK GOES flare events and XRS observations. No artificial or synthetic flare events were fabricated to artificially balance classes.