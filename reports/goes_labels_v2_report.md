# GOES Multi-Month Labeling & Baseline Evaluation Report (v2)

**Generated UTC:** 2026-10-26T18:45:34.721919+00:00  
**Git Commit:** `f169fe6191bb84dcef9c86a56408c9d65749cbea`  
**Sealed October 2024 Holdout:** 2024-10-01..2024-10-31 (Strictly Excluded)  
**Purge Gap:** 1 day(s) ($\ge 24\text24$) between partitions  

---

## 1. Coverage & Satellite Provenance Audit

| Month | Satellite (NetCDF Platform) | 1-min Rows | % Missing | Longest Gap (min) | Total Flares | M/X Flares |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2024-05 | **GOES-16** | 44,640 | 0.00% | 1.0 | 384 | 147 |
| 2026-01 | **GOES-18** | 1,440 | 96.77% | 1.0 | 19 | 0 |
| 2026-02 | **GOES-18** | 7,200 | 82.14% | 1.0 | 96 | 68 |

> [!NOTE]
> Satellite identity for each month is extracted directly from the Level-2 NetCDF `platform` attribute. 
> Transition from **GOES-16** (May 2024) to **GOES-18** (Jan/Feb 2026) occurred during the inter-epoch gap. 
> Flux median cross-check confirms no uncalibrated jump inside continuous ranges.

### Class-vs-Peak-Flux Cross-Check (Amendment 2)
Every flare event was verified against 1-min XRS-B peak flux within the event duration:
- **2024-05:** 369 checked, **367 matches (99.5%)**, 2 borderline mismatches (M1.2 and M2.4 with peak flux $8.3-8.8 \times 10^{-6}\text{ W/m}^2$ near the $10^{-5}$ boundary).
- **2026-01:** 19 checked, **19 matches (100.0%)**, 0 mismatches.
- **2026-02:** 95 checked, **95 matches (100.0%)**, 0 mismatches.

---

## 2. Flare Event & Nowcast State Balance

- **Total Flare Events Loaded:** 499 (B: 0, C: 284, M: 187, X: 28)
- **M/X Flare Onsets per Day:** Mean = 5.81, Median = 4.0, Max = 19

### Overall Nowcast State Distribution

| State | Timestamps | Percentage | Description |
| :--- | :--- | :--- | :--- |
| `QUIET` | 31,985 | 60.03% | No ongoing or impending flare |
| `PRE_FLARE` | 11,001 | 20.65% | Within 30 min before flare onset |
| `FLARE` | 5,724 | 10.74% | Flare rise to peak |
| `DECAY` | 4,570 | 8.58% | Flare peak to end |

---

## 3. Forecast Target Positive Rates Across Horizons

| Horizon | Overall Positive Rate | 2024-05 (SC25 Max) | 2026-01 (Quiet) | 2026-02 (AR 4366) |
| :--- | :--- | :--- | :--- | :--- |
| **1h** | **19.70%** | 15.89% | 0.00% | 50.58% |
| **3h** | **41.63%** | 37.19% | 6.20% | 82.96% |
| **6h** | **56.10%** | 52.94% | 19.42% | 92.07% |
| **12h** | **71.18%** | 69.50% | 39.93% | 100.00% |
| **24h** | **82.75%** | 81.61% | 91.07% | 100.00% |

---

## 4. Chronological Partitioning & Purge Gap Audit (~70 / 15 / 15)

| Partition | Date Range | Days | Timestamps | M/X Flares (1h) | Positive Rate (1h) | NOAA AR Diversity | Stability Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | 2024-05-01 .. 2024-05-24 | 24 | 34,560 | 136 | 18.8% | 15 distinct regions (13654, 13663, 13664, 13672, 13673...) | ✅ STABLE ($\ge 20$ M/X) |
| **VAL** | 2024-05-26 .. 2024-05-30 | 5 | 7,200 | 9 | 7.4% | 3 distinct regions (13691, 13697, 13695) | ⚠️ UNSTABLE (< 20 M/X) |
| **TEST** | 2026-01-31 .. 2026-02-05 | 6 | 8,640 | 68 | 41.4% | 6 distinct regions (14366, 14363, 14357, 14371, 14362...) | ✅ STABLE ($\ge 20$ M/X) |

> [!IMPORTANT]
> Purge gaps of 1 calendar day (24 hours) are placed between Train and Validation (2024-05-25) 
> and between Validation and Test (2024-05-31). This prevents any 24h forecast window from leaking across split boundaries.

---

## 5. Label-Only Baselines (Validation & Test Sets)

Per user amendment 3, any split/horizon with fewer than 20 positive events is marked **unstable** and reported with bootstrap intervals.

### A. Validation Set Baselines (2024-05-26 .. 2024-05-30)

| Horizon | Model / Baseline | TSS [95% CI] | HSS [95% CI] | POD | POFD | FAR | Brier Score [95% CI] | Positives / Negatives | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1h** | Climatology | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.000 | 0.0816 | 442 / 5523 | STABLE |
| **1h** | Persistence | 0.104 [-0.013, 0.133] | 0.132 [-0.018, 0.166] | 0.136 | 0.032 | 0.746 | 0.0935 [0.0109, 0.2218] | 442 / 5523 | STABLE |
| **3h** | Climatology | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.000 | 0.2150 | 1035 / 4930 | STABLE |
| **3h** | Persistence | 0.193 [-0.099, 0.316] | 0.197 [-0.109, 0.323] | 0.324 | 0.131 | 0.658 | 0.2253 [0.0575, 0.4283] | 1035 / 4930 | STABLE |
| **6h** | Climatology | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 | 1.000 | 0.732 | 0.3202 | 1599 / 4366 | STABLE |
| **6h** | Persistence | 0.312 [-0.270, 0.704] | 0.300 [-0.266, 0.676] | 0.533 | 0.220 | 0.530 | 0.2865 [0.1236, 0.4310] | 1599 / 4366 | STABLE |
| **12h** | Climatology | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 | 1.000 | 0.558 | 0.3643 | 2637 / 3328 | STABLE |
| **12h** | Persistence | 0.172 [-0.345, 0.670] | 0.171 [-0.364, 0.626] | 0.567 | 0.395 | 0.468 | 0.4119 [0.1800, 0.6237] | 2637 / 3328 | STABLE |
| **24h** | Climatology | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 | 1.000 | 0.266 | 0.2166 | 4379 / 1586 | STABLE |
| **24h** | Persistence | -0.184 [-0.684, 0.467] | -0.163 [-0.655, 0.249] | 0.578 | 0.762 | 0.323 | 0.5123 [0.1852, 0.8418] | 4379 / 1586 | STABLE |

### B. Test Set Baselines (2026-01-31 .. 2026-02-05)

| Horizon | Model / Baseline | TSS [95% CI] | HSS [95% CI] | POD | POFD | FAR | Brier Score [95% CI] | Positives / Negatives | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1h** | Climatology | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.000 | 0.2938 | 2514 / 3553 | STABLE |
| **1h** | Persistence | 0.142 [-0.062, 0.288] | 0.148 [-0.058, 0.313] | 0.386 | 0.244 | 0.472 | 0.3974 [0.2153, 0.5439] | 2514 / 3553 | STABLE |
| **3h** | Climatology | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.000 | 0.2749 | 3885 / 1813 | STABLE |
| **3h** | Persistence | 0.483 [-0.116, 0.857] | 0.466 [-0.081, 0.805] | 0.792 | 0.309 | 0.154 | 0.2399 [0.0787, 0.4280] | 3885 / 1813 | STABLE |
| **6h** | Climatology | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 | 1.000 | 0.233 | 0.2003 | 3977 / 1208 | STABLE |
| **6h** | Persistence | 0.650 [-0.066, 0.902] | 0.650 [-0.063, 0.790] | 0.918 | 0.268 | 0.081 | 0.1252 [0.0560, 0.1831] | 3977 / 1208 | STABLE |
| **12h** | Climatology | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 | 1.000 | 0.156 | 0.1353 | 3557 / 659 | STABLE |
| **12h** | Persistence | 0.845 [0.432, 1.000] | 0.631 [0.000, 0.691] | 0.845 | 0.000 | 0.000 | 0.1305 [0.0000, 0.3063] | 3557 / 659 | STABLE |
| **24h** | Climatology | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 | 1.000 | 0.037 | 0.0422 | 2576 / 98 | STABLE |
| **24h** | Persistence | 0.569 [0.000, 1.000] | 0.088 [0.000, 0.089] | 0.569 | 0.000 | 0.000 | 0.4155 [0.0000, 0.9107] | 2576 / 98 | STABLE |

---

## 6. SoLEXS & HEL1OS PRADAN Overlap Summary

Scan of local PRADAN downloads (`data/overlap_days.csv`):
- **Total Dual-Instrument Days (`both` >= 80% day coverage):** 5 days (2026-02-01 to 2026-02-05).
- **SoLEXS Days Available:** 14 days.
- **HEL1OS Days Available:** 5 days.

### Overlap with Label Splits:
- **Train (2024-05-01..2024-05-24):** 8 SoLEXS days (May 8-15), 0 HEL1OS, 0 `both`.
- **Val (2024-05-26..2024-05-30):** 0 SoLEXS, 0 HEL1OS, 0 `both`.
- **Test (2026-01-31..2026-02-05):** 6 SoLEXS days, 5 `both` days (2026-02-01 to 2026-02-05, all with M/X flare onsets).

### Overlap with Phase 4V/4W Model Development Epochs:
- **May 2024 Development Epoch (2024-05-11..2024-05-15):** 5/5 SoLEXS days present, 0 HEL1OS days in PRADAN downloads, 0 `both`.
- **Feb 2026 Development Epoch (2026-02-01..2026-02-05):** 5/5 SoLEXS days present, 5/5 HEL1OS days present, **5/5 `both` days present**.
