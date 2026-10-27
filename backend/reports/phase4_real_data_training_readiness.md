# Phase 4 — Real observational training readiness

Audit date: 2026-10-24  
Status: **BLOCKED — do not train**

## Decision

Real observational training is not eligible. Phase 3C verified 0 of 14 instrument-semantic rules and reports `BLOCKED_DOCUMENTATION_UNAVAILABLE`. It records zero verified joint windows; positive and negative eligible window counts are withheld (`null`) because instrument timing, GTI, quality/calibration semantics, and negative-label completeness are unresolved. Do not interpret candidate samples as eligible observations or assign uncatalogued periods as negatives. No split boundaries, model selection, evaluation, or training were performed for this audit. No real-data checkpoint exists.

## Files found and provenance

The configured data root `E:\SolarFlareData` contains the Phase 3A/3B/3C reports, the NOAA annual GOES event report and metadata, and four source PDFs: `ALPPS_UserGuide_V1.0.pdf` plus three GOES readme/guide documents. It does **not** contain the SoLEXS or HEL1OS source archives.

The unchanged instrument archives are in `C:\Users\ASHUTOSH\Downloads`, outside the configured data root:

- `solexs_2026Oct23T184402685.zip`, SHA-256 `34C39624194EE2DA95BBF8E3DFED5AC002C55A517A31BC4091BFDD0D2B4671C2`; 6 nested daily archives, `AL1_SLX_L1_20260131_v1.0.zip` through `AL1_SLX_L1_20260205_v1.0.zip`.
- `hel1os_2026Oct23T184518914.zip`, SHA-256 `17654C33AFF72313E796CEAFB31577224F66F9F6B366438F8D655518EC033E17`; 11 nested product archives spanning V111 and V211.

The hashes match the values recorded by Phase 3A, so the previous nested-product audit refers to the same archive bytes. The originals were not extracted or modified during this Phase 4 inspection.

GOES files found under `C:\Users\ASHUTOSH\Downloads\data\goes`:

- `goes_flares.csv`: 115 local events, Jan 31–Feb 5, 2026; C=47, M=62, X=6; SHA-256 `a19a8e8af74164af60a6ca74387d611f93cc0b6d8e47427ac79e60b7b5e76cb9`.
- `goes_xrs_1min.csv`: 8,640 minute rows; 8,471 finite values per channel and 169 missing per channel; no duplicate timestamps. The Phase 3A audit matched values to six daily GOES NetCDF files, which retain quality flags; the CSV itself does not.
- `E:\SolarFlareData\goes_official\sci_xrsf-l2-flrpt_geo_y2026_v1-0-1.csv`: 2,525 annual records, Jan 1–Oct 22, 2026. Its metadata identifies NOAA NCEI, algorithm version 1-0-1, and one-minute event resolution. In Jan 31–Feb 6, it contains 123 C/M/X records: C=50, M=67, X=6.

Phase 3B reconciled 107 local event identities with exact start/class matches, 8 with a unique start but a class-magnitude discrepancy, and 8 official-only records. This supports event reconciliation but does not establish that the event catalog is complete enough to label non-event periods negative.

## Instrument archive inventory and unresolved semantics

| Product | Files and candidate inventory | Coverage observed in archive | Still unverified |
|---|---|---|---|
| SoLEXS | 6 light curves, 518,400 raw rows, 12 GTI tables (84 table rows, including empty companion tables); 272,223 finite/nonnegative candidate rows; detector SDD2 | Jan 31 00:00:00–Feb 5 23:59:59 UTC as provisionally parsed | TIME has no column TUNIT; headers say `TIMEUNIT=s`, `TIMESYS=UTC`, `TIMEDEL=1`, but authoritative epoch mapping and GTI endpoint meaning are unresolved. COUNTS has no TUNIT; count/bin versus rate is unresolved. GTIs, quality rules, calibration/response, and product precedence are not applied/verified. |
| HEL1OS | 11 nested archives, 44 light curves and 44 GTI tables, 1,047,288 rows summed across products (not unique observations); 40 V111 and 4 V211 product instances | Feb 1 00:00:08.786–Feb 5 23:59:40.497 UTC as provisionally parsed | Four repeated detector/band extensions: CDTE1/CDTE2 at 5–20 keV and CZT1/CZT2 at 20–40 keV. MJD and ISOT agreed within 1 ms on 44,000 sampled rows under provisional UTC, which does not establish authoritative time scale. Time reference, GTI, quality/calibration, version precedence, and pairing rules remain unresolved. |

Candidate rows include overlapping versions, detector/band products, and rows not yet screened using verified scientific rules. They are not valid-sample counts. Phase 3C reports `verified_solexs_samples=null`, `verified_hel1os_samples=null`, and `verified_joint_windows=0`.

## Labels, partitions, and training

- Independently catalogued reference records in the window: 123 official NOAA reports. Zero event overlaps have been verified against scientifically valid instrument intervals because none are established.
- Eligible positives: **unknown / withheld**. Eligible negatives: **unknown / withheld**. Unknown-label count is not estimable as a window count before valid time alignment and coverage rules are established; unknown remains unknown.
- No chronological train/validation/test partitions were selected. With six calendar days and zero verified joint windows, defensible leakage-safe partitions and independent event support cannot be established.
- No real observational training, synthetic training, threshold tuning, or final-test evaluation was run for this Phase 4 request. No production registration was attempted.

## Required to resume

Retrieve the registered SoLEXS and HEL1OS L1 user manuals/SIS and supporting time-system, GTI, quality, calibration/response, and version-selection documentation through the ISSDC/PRADAN portal. Save the documents unmodified in `E:\SolarFlareData\source_docs\isro\solexs`, `hel1os`, and `general`, with portal identifiers and revisions. The public PRADAN FAQ supports broad instrument energy coverage, but it does not settle these L1 semantics. Then rerun Phase 3C and only proceed to valid coverage, event-window counts, and chronological partition assessment if every required rule is verified. The instrument ZIPs and local GOES CSVs currently reside under Downloads; preserve them and either use the audit's existing Downloads input path or place verified copies in a designated raw-data directory on E: before treating E: as the complete data root.

## Code and tests inspected

Relevant implementation inspected: `backend/ml/training/train.py`, `backend/ml/training/eligibility.py`, `backend/ml/features/windowing.py`, `backend/ml/features/engineering.py`, `backend/app/services/scientific_readiness.py`, `backend/app/services/model_service.py`, and the SoLEXS, HEL1OS, and GOES label adapters under `backend/app/services/data_sources/`. The existing scientific-readiness service blocks real training. SoLEXS currently assumes Unix seconds/counts-per-second and does not apply GTIs; HEL1OS remains fail-closed. No adapter semantics were changed.

Focused tests passed: 63 tests across Phase 3A/3B/3C audit, real data sources, chronology, feature engineering, scientific readiness, and Phase 2C eligibility. One attempted invocation from the repository root failed during collection because `app` and `ml` are importable from `backend`; rerunning from `backend` succeeded. The successful command was:

```powershell
python -m pytest tests/test_phase3a_data_audit.py tests/test_phase3b_authoritative_audit.py tests/test_phase3c_semantics_audit.py tests/test_real_data_sources.py tests/test_chronological_audit.py tests/test_feature_engineering_step9.py tests/test_scientific_readiness.py tests/test_phase2c_eligibility.py -q --basetemp=.pytest-phase4-real-readiness-20261024
```

Result: **63 passed**, 25 warnings (Pydantic/SQLAlchemy deprecations and pandas fragmentation performance warnings); no test failures in the successful run.

Next executable step after registered documents are available:

```powershell
python scripts/phase3c_semantics_audit.py --data-root E:\SolarFlareData --repo-root C:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main --report-root E:\SolarFlareData\reports
```

This reruns the documentation/semantics gate. It does not train a model.
