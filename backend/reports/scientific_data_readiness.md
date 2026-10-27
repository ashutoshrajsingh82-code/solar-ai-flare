# Scientific Data Readiness

- **Status:** BLOCKED
- **Training allowed:** False
- **Generated (UTC):** 2026-10-21T17:52:45.427295+00:00
- **Dataset:** demo_aditya_l1_solexs_helios.csv (source: demo)
- **Rows:** 14401
- **UTC period:** 2026-08-01T00:00:00+00:00 to 2026-08-10T23:59:00+00:00
- **Median cadence:** 60.0 seconds
- **Paired flux rows:** 14369

## GOES label counts

| Horizon | Known | Positive | Negative | Unknown | Invalid |
|---|---:|---:|---:|---:|---:|
| 1h | 0 | 0 | 0 | 14401 | 0 |
| 3h | 0 | 0 | 0 | 14401 | 0 |
| 6h | 0 | 0 | 0 | 14401 | 0 |
| 12h | 0 | 0 | 0 | 14401 | 0 |
| 24h | 0 | 0 | 0 | 14401 | 0 |

## Blockers

- Observation timestamps contain 1 duplicate(s).
- The active dataset/runtime is demo mode; synthetic observations cannot support scientific training.
- Instrument units, energy channels, and calibration are not verified for the loaded measurement columns.
- A verified instrument observation_source identifier is missing.
- Independent NOAA/NCEI GOES reference-label provenance is missing or unverified.
- No known GOES 1h labels are available.
- No known GOES 3h labels are available.
- No known GOES 6h labels are available.
- No known GOES 12h labels are available.
- No known GOES 24h labels are available.
- Chronological train split is unavailable or empty.
- Chronological validation split is unavailable or empty.
- Chronological test split is unavailable or empty.
- Required scientific validation evidence is missing: observation_provenance_verified.
- Required scientific validation evidence is missing: label_provenance_verified.
- Required scientific validation evidence is missing: timestamps_utc_verified.
- Required scientific validation evidence is missing: forecast_target_verified.
- Required scientific validation evidence is missing: features_prediction_time_verified.
- Required scientific validation evidence is missing: unknown_labels_excluded.
- Required scientific validation evidence is missing: chronological_embargo_verified.

## Warnings

- None

This report describes only the loaded files and recorded provenance. A ready status does not replace independent scientific review.
