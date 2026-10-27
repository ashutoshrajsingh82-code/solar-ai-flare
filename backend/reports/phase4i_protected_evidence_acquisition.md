# Phase 4I - Protected technical evidence and scientific readiness

Audit date: 2026-10-25 UTC

## Result

No new authoritative document was acquired.  The canonical evidence set remains two hash-validated instrument manuals.  No rule status was changed and the Phase 3C gate remains `BLOCKED_SCIENTIFIC_SEMANTICS_UNRESOLVED`.

## Official acquisition route and outcome

The required official route is ISSDC PRADAN **Other Downloads**:

- URL: `https://pradan.issdc.gov.in/al1/protected/miscDownloads.xhtml`
- Required artifacts: SoLEXS L1 SIS/GTI and provenance schema; SoLEXS CALDB/release and rate-to-flux procedure; HEL1OS L1 SIS/time-correlation and quality documentation; HEL1OS V111/V211 release/change note.
- Access outcome: **INCONCLUSIVE_LOCAL_BROWSER_UNAVAILABLE**.  The supported in-app browser connection failed before navigation with a local filesystem-permission error. No browser tab, credentials, cookies, or session data were read; no authentication was attempted and no access control was bypassed.

The public [ISSDC PRADAN FAQ](https://pradan1.issdc.gov.in/al1/faq.xhtml) remains a discovery source only: it directs HEL1OS technical details to the protected Aditya-L1 User Handbook/Other Downloads area.  The public [ALPPS User Guide V1.0](https://alpps.issdc.gov.in/web/extensions/Aditya-l1/resources/docs/ALPPS_UserGuide_V1.0.pdf) provides operational/POC context only.  Neither is a field-level FITS, GTI, quality, calibration, or version-precedence contract.

## Canonical documents retained

- `E:\SolarFlareData\source_docs\isro\solexs\SoLEXS-UserManual.pdf` - SoLEXS Data Analysis Guide Version 1.0, U. R. Rao Satellite Centre / ISRO; SHA-256 `d97c83618c85a0984e7dd0300ce5c16c4748ed5facde525949f902fe9f72abda`.
- `E:\SolarFlareData\source_docs\isro\hel1os\HEL1OS_UserManual.pdf` - HEL1OS Data Analysis User Manual Version 1.2 (January 2025), HEL1OS Team; SHA-256 `8efdf3aaf48d8a72cd9d85b567a8c0a6e2655ecf52c5a8d87f36915b741acb4d`.

These hashes match the Phase 4H readiness report. No duplicate or unverified file was registered.

## Rule-by-rule before/after adjudication

| # | Rule | Before | After | Evidence / required next artifact |
|---|---|---|---|---|
| 1 | SoLEXS RATE.TIME | VERIFIED | VERIFIED | Manual p.6 §2.2: Unix seconds since 1970-01-01 UTC. |
| 2 | SoLEXS RATE.COUNTS | VERIFIED | VERIFIED | Manual p.5 §2.1.2: L1 LC count-rate scope; not irradiance. |
| 3 | SoLEXS GTI START/STOP | PARTIAL | PARTIAL | Need L1 SIS/GTI dictionary for units, RATE.TIME relationship, closure, malformed/overlap behavior. |
| 4 | SoLEXS detector/path identity | VERIFIED | VERIFIED | Manual pp.5-6; need release record for any version-selection rule. |
| 5 | SoLEXS provenance/cadence | PARTIAL | PARTIAL | Need L1 schema defining CREATOR/DATE/OBS_ID and compatibility. |
| 6 | SoLEXS flux conversion | PARTIAL | PARTIAL | Manual p.7 §2.4 identifies ARF/RMF; need CALDB release and applicable conversion procedure. |
| 7 | HEL1OS MJD | PARTIAL | PARTIAL | Need L1 SIS/time-correlation contract for scale/reference and mismatch policy. |
| 8 | HEL1OS ISOT | PARTIAL | PARTIAL | Manual pp.46,51 label UTC ISO-T; need controlling-field/disagreement rule. |
| 9 | HEL1OS CTR | VERIFIED | VERIFIED | Manual pp.51-52: count rate in cts/sec; no flux conversion. |
| 10 | HEL1OS STAT_ERR | VERIFIED | VERIFIED | Manual p.51: count-rate uncertainty units. |
| 11 | HEL1OS detector/band extension | VERIFIED | VERIFIED | Manual pp.5,13,51-52. |
| 12 | HEL1OS L1 GTI use | VERIFIED | VERIFIED | Manual p.13: L1 GTIs span dump and may be ignored for L1; not an endpoint/Level-2 rule. |
| 13 | HEL1OS V111/V211 precedence | UNVERIFIED | UNVERIFIED | Need POC/ISSDC release note explicitly establishing supersession; 14 counterpart groups remain separate. |
| 14 | HEL1OS quality/HK/saturation | PARTIAL | PARTIAL | Need L1 quality/SIS and calibration/operations note with bit meanings, thresholds, and rejection scope. |

## Separate gate assessment

The Phase 4H readiness assessment remains current: document provenance and the 178-product structural inventory pass, while timestamps, GTI endpoint applicability, quality/saturation, precedence, calibration, valid coverage, label readiness, negative completeness, and chronological split feasibility are blocked.  Structural readability does not establish scientific eligibility.

No training, tuning, evaluation, checkpoint activity, production activation, GTI/quality filtering, V111/V211 merge, or negative-label promotion was performed.
