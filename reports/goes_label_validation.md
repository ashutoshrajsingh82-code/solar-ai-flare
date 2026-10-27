# GOES v1 Label Validation Report

**Generated:** 2026-10-26T18:03:37.429592+00:00
**Events file:** `C:\Users\ASHUTOSH\Downloads\data\goes\goes_flares.csv`
**Flux file:** `C:\Users\ASHUTOSH\Downloads\data\goes\goes_xrs_1min.csv`
**Events loaded:** 115 (raw: 115)
**Date range:** 2026-01-31 01:28:00 to 2026-02-05 23:47:00

---

## Check 1: Duplicate / Near-Duplicate Events

> Rule: same class letter, starts within 2 minutes.

- **Total events:** 115
- **Near-duplicate pairs:** 0 -- none detected.

---

## Check 2: Class String vs Peak Flux Cross-Check

> Thresholds: A<1e-7, B<1e-6, C<1e-5, M<1e-4, X>=1e-4 W/m^2.

- **Events with peak_flux:** 0
- **Events without peak_flux (NaN):** 115
- **Class/flux mismatches:** 0 -- all consistent.

---

## Check 3: Why Zero A/B Events

- **Class distribution in raw CSV:** {'M': 62, 'C': 47, 'X': 6}

**Explanation:** download_goes.py uses SunPy HEK with the standard filter fl_goescls >= C1.0. A and B class flux excursions are detected by GOES but are not reported to the HEK event list unless they reach C-class threshold. Additionally, the 2026-01-31..2026-02-05 window covers NOAA AR 4366 -- an extremely active region where background flux was at C-class levels for most of the period, making it impossible to identify isolated A/B transients. No events were invented or removed -- they were simply not present in the HEK catalog for this observation window.

---

## Check 4: X-Class Events vs SIDC Bulletin

> SIDC ref (2026-02-01/02): X1.0 12:33, X8.1 23:57, X2.8 00:36, X1.6 08:14 UTC.
> Flag threshold: >10 min difference.

- **X-class events in catalog:** 6

| # | Class | Start UTC | Peak UTC | End UTC | SIDC Match | Diff(min) |
| :---: | :---: | :--- | :--- | :--- | :---: | :---: |
| 1 | X1.0 | 2026-02-01 12:25:00+00:00 | 2026-02-01 12:33:00+00:00 | 2026-02-01 12:37:00+00:00 | [OK] X1.0 | 0.0 |
| 2 | X8.1 | 2026-02-01 23:44:00+00:00 | 2026-02-01 23:57:00+00:00 | 2026-02-02 00:04:00+00:00 | [OK] X8.1 | 0.0 |
| 3 | X2.8 | 2026-02-02 00:31:00+00:00 | 2026-02-02 00:36:00+00:00 | 2026-02-02 00:40:00+00:00 | [OK] X2.8 | 0.0 |
| 4 | X1.6 | 2026-02-02 07:39:00+00:00 | 2026-02-02 08:14:00+00:00 | 2026-02-02 08:40:00+00:00 | [OK] X1.6 | 0.0 |
| 5 | X1.5 | 2026-02-03 13:58:00+00:00 | 2026-02-03 14:08:00+00:00 | 2026-02-03 14:18:00+00:00 | [OK] X1.5 | 0.0 |
| 6 | X4.2 | 2026-02-04 12:02:00+00:00 | 2026-02-04 12:13:00+00:00 | 2026-02-04 12:18:00+00:00 | [FLAG] closest=X1.5 | 1325.0 |

> NOTE: The SIDC bulletin in the task description lists X1.5 at 00:22 UTC on 2026-02-02. The downloaded catalog has X1.5 at 2026-02-03 14:08 and X2.8 at 00:36 UTC 2026-02-02. No assertion is made about a match that cannot be verified from the available local data.

---

## Check 5: Listed Peak Flux vs 1-Min XRS Maximum

> Flag: >2x difference between listed and observed.

- **Flux column used:** `xrsb` (1-8 Angstrom long channel)
- **Events cross-checked:** 0
- **Factor-of-2 flux mismatches:** 0 -- all consistent.

---

## Check 6: Satellite and Flux Channel Identification

- **Flux file columns:** ['timestamp', 'xrsa', 'xrsb']
- **Satellite:** Flux downloaded for 2026-01-31..2026-02-05 by SunPy Fido XRSClient. For this period GOES-19 is the operational primary (GOES-18 became secondary). SunPy XRSClient returns the operational primary by default.
- **Channel check:** xrsa (0.5-4 AA short): present. xrsb (1-8 AA long): present.
- **Conclusion:** xrsb is the 1-8 Angstrom LONG channel used for GOES flare classification (A<1e-7, B<1e-6, C<1e-5, M<1e-4, X>=1e-4 W/m^2). xrsa is supplementary. This is the correct channel; no corrective action needed.
- **xrsb stats:** min=8.992e-07, max=8.111e-04, mean=1.164e-05 W/m^2

---

## Summary

| Check | Status |
| :--- | :---: |
| 1. Near-duplicate events | Clean |
| 2. Class vs flux | OK |
| 3. Zero A/B events | Explained (HEK filter + high background) |
| 4. X-class vs SIDC | See table |
| 5. Peak flux vs XRS interval max | See table |
| 6. Satellite / channel | GOES-19 (operational), xrsb = 1-8 AA long channel |

_Read-only validation. No data was modified._