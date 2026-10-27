# Phase 4M-R: Label Eligibility and Exclusion Report

**Audit Date**: 2026-10-25  
**Canonical Finding**: `TRUSTWORTHY_NEGATIVES_UNAVAILABLE`  
**Supervised Binary Classification Status**: `BLOCKED_CANNOT_TRAIN_ON_FABRICATED_NEGATIVES`

---

## 1. Reference Event Sources and Verification

The repository and configured real-data root contain two GOES reference sources:
1. **Authoritative NOAA/NCEI GOES-R XRS L2 Flare Report**:
   - Path: `E:\SolarFlareData\goes_official\sci_xrsf-l2-flrpt_geo_y2026_v1-0-1.csv`
   - Size: 429,599 bytes | SHA-256: `b782aa60847e7cf0862bc8db281bb833680c50f8adf71bd24b21414d7e490de8`
   - Schema Metadata: `sci_xrsf-l2-flrpt_geo_metadata.json` (`26269e7f3918d47aa0e4138821cc177a25690a8b197e245049610a8fc6cf6760`)
   - Authoritative User Documentation: `GOES_Flare_Report_ReadMe.pdf` and `GOES-R_XRS_L2_Data_Users_Guide.pdf` in `E:\SolarFlareData\source_docs\`
   - Required Columns: `time`, `start_time`, `end_time`, `flare_class`, `flare_id` (fully present and schema-valid)
2. **Historical Local GOES Excerpt**:
   - Path: `C:\Users\ASHUTOSH\Downloads\data\goes\goes_flares.csv`
   - Size: 8,915 bytes | SHA-256: `a19a8e8af74164af60a6ca74387d611f93cc0b6d8e47427ac79e60b7b5e76cb9`
   - Note: Unofficial schema (`start`, `peak`, `end`, `goes_class`, `is_MX`); lacks official NOAA column names. The official NCEI CSV above takes precedence.

---

## 2. Event Population During Observation Window

Over the 6-day Aditya-L1 observational window (`2026-01-31 00:00:00` to `2026-02-05 23:59:59 UTC`), the official NOAA flare report registers **123 confirmed solar flares**:
- **C-Class**: 50 events
- **M-Class**: 67 events
- **X-Class**: 6 events
- **Total Significant Flares ($M \ge 1.0$ or $X \ge 1.0$)**: 73 events

The temporal distribution is concentrated across active region passages during this 6-day period, providing rich positive candidate flare onsets.

---

## 3. Positive Label Association Rules

For a positive label ($Y = 1$) to be scientifically defensible:
1. **Observation Time Coordinate**: The observation timestamp $t_{obs}$ must be proven to be UTC seconds with documented accuracy.
2. **Exposure Validity**: The instrument must be actively in an open, pointed, valid collecting state (governed by documented GTI).
3. **Data Quality**: The observation sample must not be flagged by saturation, telemetry corruption, or dead-time invalidation.
4. **Target Time Window**: Given forecast horizon $H \in \{1\text{h}, 3\text{h}, 6\text{h}, 12\text{h}, 24\text{h}\}$, a verified event onset $t_{start}$ from the official catalogue must satisfy:
   $$t_{obs} < t_{start} \le t_{obs} + H$$

Under these strict terms, positive candidate matches can be computed hypothetically. However, positive matching alone is insufficient for supervised binary training.

---

## 4. Negative Label Defensibility Audit

Supervised binary flare forecasting models predict whether a flare will occur ($Y = 1$) or not ($Y = 0$) within horizon $H$. Establishing a defensible negative label ($Y = 0$) requires scientific proof of **quiescence**—that is, proof that no flare exceeding the target threshold occurred in $(t_{obs}, t_{obs} + H]$.

### 4.1 Catalogue Absence Is Not Proof of Quiescence
The NOAA GOES Flare Report is an automated detection catalogue. As documented in the NOAA Data User's Guide:
- The detection algorithm applies operational background-subtraction and threshold criteria on 1-minute GOES XRS data.
- It is designed to identify and catalogue discrete flare events.
- It does **not** certify that unlisted intervals represent quiet Sun or complete absence of solar activity.
- Periods with missing data, satellite occultations, spacecraft maneuvers, or corrupted telemetry in the reference instruments do not generate event records, but they are not quiet periods.

### 4.2 Instrument Incompleteness and Ambiguity
Neither SoLEXS nor HEL1OS has complete, continuous, uncorrupted observation:
- SoLEXS has 12 separate daily GTI files with undocumented endpoint closure rules and overlap semantics (Blocker `B-SLX-GTI-001`).
- HEL1OS has 11 data dumps with large gaps between orbits/dumps and uninterpreted HK saturation counters (Blockers `B-HL-TIME-004`, `B-HL-QUALITY-005`).
- Labeling an interval as negative when instrument observation is absent, ambiguous, or incomplete creates artificial "quiet" intervals.

### 4.3 Explicit Prohibition of Fabricated Negatives
The project instructions and scientific audit rules strictly prohibit:
1. **Inferring that a missing catalogue entry means no flare occurred.**
2. **Labeling unobserved, invalid, ambiguous, or incomplete intervals as negative.**
3. **Manufacturing labels to meet minimum class count requirements.**

When unobserved and uncertified intervals remain strictly unlabeled (`NaN` / unknown):
- **Defensible Positive Labels**: Present during verified event windows.
- **Defensible Negative Labels**: **0 (Zero)**.

---

## 5. Decision on Supervised Binary Classification

A supervised binary classification model (such as logistic regression, random forest, or a deep CNN-LSTM) cannot be fitted when the true negative class count is zero:
1. **Mathematical Degeneracy**: Fitting a binary loss function (e.g. Binary Cross-Entropy) with only positive samples collapses the gradient, driving predicted probabilities to 1.0 or causing undefined class-weight divisions.
2. **Scientific Invalidity of Fabricated Negatives**: If uncatalogued intervals were arbitrarily relabeled as $Y = 0$, the resulting model would learn spurious correlations with instrument downtime and detection thresholds, rather than true physical solar precursors.

**Conclusion**: In accordance with Step 3 instructions, because the available sources cannot establish trustworthy negatives, supervised binary classification **must not be run on fabricated negatives**. Real observational model training is blocked.
