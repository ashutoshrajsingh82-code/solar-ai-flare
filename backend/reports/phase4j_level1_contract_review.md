# Phase 4J — Targeted Level-1 Contract Resolution

## Result

All eleven registered canonical documents were hash-validated before review.  No new document was acquired and no rule status changed: 7 are verified, 6 are partially resolved, and 1 remains unverified.  This is a documentation refinement, not a scientific-data processing action.

The authoritative Level-1 manuals improve the direct provenance for SoLEXS GTI timing and HEL1OS light-curve time/HK fields.  They do not supply the four required missing contracts: SoLEXS endpoint/error/overlap rules; HEL1OS cross-field time-conflict policy; HEL1OS quality/saturation rejection policy; and V111/V211 supersession.

## Integrity and source review

The registered source paths and SHA-256 values were validated against the Phase 4I document register.  The relevant primary sources were SoLEXS Data Analysis Guide Version 1.0 (`SoLEXS-UserManual.pdf`, `d97c83618c85a0984e7dd0300ce5c16c4748ed5facde525949f902fe9f72abda`), HEL1OS Data Analysis User Manual Version 1.2, January 2025 (`HEL1OS_UserManual.pdf`, `8efdf3aaf48d8a72cd9d85b567a8c0a6e2655ecf52c5a8d87f36915b741acb4d`), and HEL1OS Release Notes (`HEL1OS_ReleaseNotes.pdf`, `f9955c56e66f0465a1dda77ca4ed416e35771a71911cf2a24a4937d2c61167a7`).

The registered SoLEXS release-update text and tool archive were reviewed.  They document release dates, timestamp updates, and packaged SDD2 ARF/RMF assets, but do not define GTI interval behavior or a light-curve physical-flux conversion.  The HEL1OS Level-2 guide/package, response-reader archives, flare catalogue, and SEP catalogue were reviewed only for scope.  They contain Level-2 or calibration/catalogue material and cannot establish an unstated Level-1 conflict or rejection contract.

## Workstream A — SoLEXS GTI

Direct evidence: the SoLEXS guide, PDF p.5 §2.1.1, defines an L1 GTI as intervals in which the detector is collecting valid data.  PDF p.6 §2.2 states that all SoLEXS time-related data are Unix seconds since 1970-01-01 00:00:00 UTC.  This supports the shared time coordinate of the documented L1 GTI interval values and `RATE.TIME` for the applicable L1 product format.

Interpretation retained: the observed `START`/`STOP` table columns are interval values, but their names are structural evidence, not a quoted SIS field dictionary.

Unresolved: neither source says whether an endpoint is included; how empty, reversed, duplicated, malformed, or overlapping intervals are handled; or whether a library convention applies.  Required artifact: a SoLEXS Level-1 FITS/SIS/data dictionary with GTI column definitions and interval-validation rules.  Rule remains `PARTIALLY_RESOLVED`.

## Workstream B — HEL1OS Level-1 time

Direct evidence: the HEL1OS User Manual PDF pp.51-52, Appendix B.3, labels the documented light-curve `MJD` as Modified Julian Date, `ISOT` as UTC ISO-T corresponding to MJD, and the `ISOT` unit as UT.  The same appendix expresses `TSTART`/`TSTOP` in MJD with UTC forms.  These citations replace the earlier Level-2-code basis for the documented field meanings.

Unresolved: the manual does not select MJD or ISOT as authoritative when they disagree, set an allowed precision/tolerance, or prescribe treatment for missing, malformed, or inconsistent values.  The Level-2 package was explicitly excluded as a Level-1 contract.  Required artifact: HEL1OS L1 SIS or time-correlation specification with stored-field authority and discrepancy policy.  Both MJD and ISOT rules remain `PARTIALLY_RESOLVED`.

## Workstream C — HEL1OS quality and saturation

Direct evidence: User Manual PDF p.13 §4 says Level-1 GTIs cover the whole dump and are ignored for Level-1 analysis.  Appendix B.4, PDF pp.53-55, names relevant HK status/counter fields: `fehkstat`, `czt1bunpxst`, `czt2bunpxst`, hot-pixel fields, `czt1satctr1`, `czt2satctr1`, `cdte1pilectr`, and `cdte2pilectr`.  This documents existence and the broad stated roles of HK/bunched-pixel/pile-up/saturation-counter data.

Unresolved: no field bit dictionary, numerical saturation threshold, sentinel definition, row-rejection criterion, or exposure-invalidation action is given.  A statement about Level-2 saturation processing is not a Level-1 filter rule.  Required artifact: a versioned HEL1OS L1 HK/quality/SIS specification defining bits, thresholds, and authorized action.  Rule remains `PARTIALLY_RESOLVED`; no filter was applied.

## Workstream D — V111/V211

Direct evidence: Release Notes PDF pp.4-5 address Release 2/3 history and a historical regeneration/gap notice.  They do not name V111 or V211.  The observed counterpart products therefore remain unadjudicated; neither row count, filename order, nor apparent content was used.

Required artifact: an issuer-authored V111/V211 release note, manifest, or supersession notice mapping both exact identifiers and prescribing treatment of overlapping counterparts.  Rule remains `STILL_UNVERIFIED`.

## Canonical consequence

`phase4j_level1_contract_adjudication.csv` retains an exact one-to-one, 14-rule mapping.  It preserves original classifications while adding page-level Level-1 citations and explicit missing artifacts.  The subsequent canonical audit must therefore remain blocked until the four missing contracts are registered and meet adjudication validation.

## Safety statement

No model training, tuning, evaluation, checkpoint activity, production-model activation, GTI/quality filtering, flux conversion, V111/V211 merge or selection, or label promotion occurred.
