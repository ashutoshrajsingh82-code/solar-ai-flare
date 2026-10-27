# Phase 4C — Documentation blockers and real-data readiness

Audit date: 2026-10-24  
Readiness: **BLOCKED — real-data training disabled**

## Outcome

Phase 3C was rerun after preparing the requested directories. No SoLEXS or HEL1OS registered L1 manuals/SIS were present, and the refreshed gate remains `BLOCKED_DOCUMENTATION_UNAVAILABLE`: **0 of 14** semantics rules verified. The existing service gate remains in effect. No real-data training dataset exists, no chronological partitions were selected, and model fitting was **not executed**. No real observational checkpoint or training provenance was produced.

The original Phase 4 report and earlier Phase 3 audits were preserved. The current Phase 3C outputs are in `E:\SolarFlareData\reports`: `phase3c_document_register.json`, `phase3c_instrument_semantics_matrix.csv`, `phase3c_coverage_reassessment.csv`, `phase3c_readiness.md`, and `phase3c_audit_summary.json`. The full read-only inventory and provisional coverage are in `backend\reports\phase4b_real_data_inventory.json` and `backend\reports\phase4b_real_data_coverage.md`; the 14-rule action register is `backend\reports\phase4b_phase3c_blockers.csv`.

## Directories prepared

These directories now exist; no source data was moved:

- `E:\SolarFlareData\source_docs\isro\solexs`
- `E:\SolarFlareData\source_docs\isro\hel1os`
- `E:\SolarFlareData\source_docs\isro\general`
- `E:\SolarFlareData\reports`

## Inputs and current locations

Hashes were recomputed during this audit and match Phase 3A/3B provenance. The two instrument archives and local GOES CSVs remain in Downloads, outside the configured E: data root.

| Input | Current location | SHA-256 | Current status |
|---|---|---|---|
| SoLEXS archive | `C:\Users\ASHUTOSH\Downloads\solexs_2026Oct23T184402685.zip` | `34c39624194ee2da95bbf8e3dfed5ac002c55a517a31bc4091bfdd0d2b4671c2` | 6 nested daily archives; unchanged; not extracted |
| HEL1OS archive | `C:\Users\ASHUTOSH\Downloads\hel1os_2026Oct23T184518914.zip` | `17654c33aff72313e796ceafb31577224f66f9f6b366438f8d655518ec033e17` | 11 nested archives; 40 V111 and 4 V211 products; unchanged; not extracted |
| Local GOES events | `C:\Users\ASHUTOSH\Downloads\data\goes\goes_flares.csv` | `a19a8e8af74164af60a6ca74387d611f93cc0b6d8e47427ac79e60b7b5e76cb9` | 115 local event records; candidate reconciliation only |
| Local GOES measurements | `C:\Users\ASHUTOSH\Downloads\data\goes\goes_xrs_1min.csv` | `891e38c4798ec28ecbd674593ba1854bb9771331fb65da44b78d7bc35705be5e` | 8,640 rows, no duplicate timestamps, 169 missing values per channel; CSV has no quality flags |
| Official NOAA GOES events | `E:\SolarFlareData\goes_official\sci_xrsf-l2-flrpt_geo_y2026_v1-0-1.csv` | `b782aa60847e7cf0862bc8db281bb833680c50f8adf71bd24b21414d7e490de8` | 2,525 annual records; 123 C/M/X records in the candidate interval |
| Official NOAA metadata | `E:\SolarFlareData\goes_official\sci_xrsf-l2-flrpt_geo_metadata.json` | `26269e7f3918d47aa0e4138821cc177a25690a8b197e245049610a8fc6cf6760` | Product algorithm version 1-0-1 |

Available PDFs under `E:\SolarFlareData\source_docs` remain the general ALPPS guide and GOES documentation; none is an instrument-specific SoLEXS/HEL1OS L1 product contract. Their document entries and hashes remain in the Phase 3C document register.

## Instrument and label coverage

- SoLEXS: 6 light curves, 518,400 raw rows, 12 GTI tables; 272,223 rows pass only the provisional finite/nonnegative screen. Verified usable rows remain **unknown**. Candidate intervals use an unverified time conversion; GTIs were not applied.
- HEL1OS: 44 light curves, 44 GTI tables, 1,047,288 rows summed across detector/band/version products. This is not a unique-observation count; V111/V211 precedence is unresolved. Verified usable rows remain **unknown**.
- Structural duplicate checks reported no duplicate or non-monotonic parsed timestamps in the Phase 3A instrument arrays; the parse does not verify the authoritative time epoch/scale. Candidate segment breaks and finite-value exclusions are inventory evidence, not verified exposure or GTI gaps.
- GOES event reconciliation: 123 official event records (C=50, M=67, X=6; M/X=73); 107 local records match exact start/class, 8 have same-family magnitude discrepancies, and 8 official events have no local match. Twenty-one official records lack an end time. Negative-label completeness is not established.
- Provisional event-start occurrences across candidate instrument products: SoLEXS 75 and HEL1OS 284. These overlap across products/versions and are not independent events or verified windows. Verified instrument-event windows remain **0**.
- Eligible positive windows: **unknown**. Eligible negative windows: **unknown**. Unknown-label windows: **not estimable** until time alignment, quality, GTIs, and forecast coverage are documented. No final split feasibility decision or partition boundary was selected.

## Documentation checklist: 14 unresolved rules

The rule-by-rule Phase 3C matrix remains authoritative for observed headers and proposed interpretations. No document hash, version, section, or page can be assigned for the missing instrument documentation.

The manual acquisition checklist, including the exact destination directory and the evidence needed from each document, is `backend\reports\phase4c_document_acquisition_checklist.csv`.

| # | Rule | Manual/SIS evidence needed |
|---:|---|---|
| 1 | SoLEXS `RATE.TIME` | Epoch/reference, time scale, conversion, and meaning of `TIMEUNIT`/`TIMESYS`/`TSTART`/`TSTOP` |
| 2 | SoLEXS `RATE.COUNTS` | Column units and whether values are counts per bin or a rate; documented measurement/calibration meaning |
| 3 | SoLEXS GTI `START`, `STOP` | Units, time reference, interval closure, row validity, detector/product association |
| 4 | SoLEXS detector/`RATE` identity | Detector mapping and matching GTI/product rules |
| 5 | SoLEXS `CREATOR`, `DATE`, `OBS_ID`, `TIMEDEL` | Product provenance, cadence meaning, processing compatibility and version precedence |
| 6 | SoLEXS irradiance/flux conversion | Applicable response/calibration files, units, and documented conversion |
| 7 | HEL1OS `MJD` | Authoritative MJD time scale, reference, offset, and conversion |
| 8 | HEL1OS `ISOT` | Authoritative time scale/format and which time field controls |
| 9 | HEL1OS `CTR` | Rate units, detector/band meaning, quality and calibration interpretation |
| 10 | HEL1OS `STAT_ERR` | Uncertainty units, validity, propagation, and calibration relationship |
| 11 | HEL1OS extension `EXTNAME` | Detector/band mapping and response/calibration by extension |
| 12 | HEL1OS GTI `START`, `STOP` and time keywords | Units, time reference, closure, detector association, and filtering rules |
| 13 | HEL1OS V111/V211 version identity | Reprocessing precedence, cross-version compatibility, duplicate-selection rules |
| 14 | HEL1OS quality/sentinel/dead-time/saturation/telemetry | Quality flags and housekeeping relationships; invalid-row and dead-time policy |

For SoLEXS, obtain the registered Level-1 product manual and SIS plus GTI, quality, calibration/response and processing-version documentation. Save under `E:\SolarFlareData\source_docs\isro\solexs`. For HEL1OS, obtain the registered Level-1 product manual and SIS plus time-system, detector/band GTI, quality/dead-time/telemetry, calibration/response and V111/V211 precedence documentation. Save under `...\hel1os`. Save the Aditya-L1 User Handbook/instrument appendices and shared time-system documentation under `...\general`. Record the exact portal title/identifier, revision/effective date, and keep each file unmodified.

For defensible negatives, also require an authoritative completeness/coverage basis for the GOES event catalog and an explicit rule for gaps and incomplete event intervals. Do not turn uncatalogued windows into negatives by default. Do not resolve the 8 class-magnitude discrepancies by assumption.

ISRO’s public release says the payload user manuals are available through the portal after registration; the public instrument descriptions do not specify these Level-1 FITS/GTI semantics ([ISRO release](https://www.isro.gov.in/ISRO_EN/Second_Set_of_Aditya_L1_Science_Data.html), [PRADAN public page](https://pradan.issdc.gov.in/al1/)). No authentication was bypassed, no credentials were entered, and no document URLs were invented.

## Tests and next action

Phase 3C rerun result: `BLOCKED_DOCUMENTATION_UNAVAILABLE`, 0/14 rules verified. Focused repository tests: **63 passed, 25 warnings** (Pydantic/SQLAlchemy deprecations and pandas fragmentation performance warnings).

```powershell
python -m pytest tests/test_phase3a_data_audit.py tests/test_phase3b_authoritative_audit.py tests/test_phase3c_semantics_audit.py tests/test_real_data_sources.py tests/test_chronological_audit.py tests/test_feature_engineering_step9.py tests/test_scientific_readiness.py tests/test_phase2c_eligibility.py -q --basetemp=.pytest-phase4c-readiness-20261024
```

After manually obtaining and placing the registered documents, rerun:

```powershell
python scripts/phase3c_semantics_audit.py --data-root E:\SolarFlareData --repo-root C:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main --report-root E:\SolarFlareData\reports
```

Then inspect the refreshed matrix/document register before any adapter change or label-window construction. Real training remains disabled until every critical rule, negative-label eligibility, event/window counts, leakage-safe partitions, and the existing eligibility gate pass.
