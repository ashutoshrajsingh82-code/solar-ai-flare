<!-- phase3c_semantics_audit_v1 -->
# Phase 3C — Instrument semantics and documentation gate

- Readiness: **BLOCKED_SCIENTIFIC_SEMANTICS_UNRESOLVED**
- Audit time UTC: 2026-10-24T21:10:50.399946+00:00
- Training, tuning, evaluation, and final split selection: **not started**
- Local archives modified/extracted: **No**

## Documentation acquisition

Source-document registration and scientific adjudication are separate. Registered files and hashes are in the document register; any reviewed adjudication must be supplied explicitly and is accepted only when every cited hash matches a registered source file and every canonical rule is represented once.
Adjudication: **APPLIED**; verified/partial/unresolved-or-conflicting rules: **7/6/1**. The gate remains blocked until all required semantics, coverage, and label-completeness requirements are independently satisfied.

## What the local files establish

The archives contain 6 SoLEXS SDD2 daily light curves (518400 rows) and 12 GTI tables. The reviewed manual establishes Unix-time semantics for the documented SoLEXS L1 coordinate and count-rate scope, but does not establish the GTI endpoint contract, complete provenance compatibility, or a physical-flux conversion.
HEL1OS contains 44 light curves across 4 repeated detector/band extension patterns (40 V111, 4 V211 products) and 44 GTI tables. MJD and ISOT agree within 1 ms for 44000 of 44000 sampled rows under a provisional UTC comparison. Agreement does not establish the authoritative time scale.
Product fields, FITS layouts, nominal extension labels, file versions, and observed timestamps are evidence, not semantics. The canonical matrix carries only the reviewed statuses and page citations accepted by the provenance checks; it does not create a filtering policy.

## Adapter review and coverage

The SoLEXS adapter currently accepts missing column units, assumes TIME is Unix seconds, describes COUNTS as counts/s, assigns an energy band/cadence contract, and does not apply GTI. The reviewed sources support only the cited portions of those assumptions; they do not establish a safe GTI, quality, calibration, or version policy. The HEL1OS reader remains fail-closed. No production adapter changes were made.
GOES event records remain a separate reference source. NOAA report records in the interval: 123 (C=50, M=67, X=6); the report documentation does not establish population completeness, so negative labels are not allowed. Instrument event overlap, verified valid samples, and positive/negative window counts remain withheld; verified joint windows = 0. No final chronological split boundaries were chosen.

## Next minimal action

Sign in manually to PRADAN, retrieve the SoLEXS/HEL1OS registered L1 manuals, SIS and supporting GTI, quality, calibration/response and version-selection documentation, then save them unmodified in the requested folders. If the portal cannot provide them, place any officially downloaded copies in those folders with portal identifiers/revision metadata; do not treat unregistered reposts as authoritative. Rerun this audit before changing adapters or calculating scientific coverage.
