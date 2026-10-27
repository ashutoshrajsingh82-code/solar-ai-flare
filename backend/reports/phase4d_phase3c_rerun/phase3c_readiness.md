<!-- phase3c_semantics_audit_v1 -->
# Phase 3C — Instrument semantics and documentation gate

- Readiness: **BLOCKED_DOCUMENTATION_UNAVAILABLE**
- Audit time UTC: 2026-10-24T16:41:31.286333+00:00
- Training, tuning, evaluation, and final split selection: **not started**
- Local archives modified/extracted: **No**

## Documentation acquisition

The public PRADAN homepage loaded. Selecting **Other Downloads** redirected to the ISSDC sign-in form. No authenticated session was available; no credentials were entered and no documents were downloaded. The currently available ALPPS User Guide V1.0 is a general portal guide and does not establish SoLEXS/HEL1OS L1 field, time, GTI, quality, calibration, or version semantics.
The document register lists each requested manual/SIS category, its expected destination, and the exact access result. Please sign in to PRADAN in the browser tab left ready for handoff, download the registered documents into `E:\SolarFlareData\source_docs\isro\solexs`, `hel1os`, and `general`, and resume the audit.

## What the local files establish

The archives contain 6 SoLEXS SDD2 daily light curves (518400 rows) and 12 GTI tables. The RATE schema includes TIME and COUNTS but neither column declares a FITS unit; headers contain TIMESYS=UTC, TIMEUNIT=s and TIMEDEL=1, while an authoritative epoch mapping and GTI endpoint convention remain undocumented.
HEL1OS contains 44 light curves across 4 repeated detector/band extension patterns (40 V111, 4 V211 products) and 44 GTI tables. MJD and ISOT agree within 1 ms for 44000 of 44000 sampled rows under a provisional UTC comparison. Agreement does not establish the authoritative time scale.
Product fields, FITS layouts, nominal extension labels, file versions, and observed timestamps are recorded as evidence, not as verified semantics. The complete rule-by-rule matrix marks critical instrument rules UNVERIFIED because no matching registered manual/SIS is available.

## Adapter review and coverage

The SoLEXS adapter currently accepts missing column units, assumes TIME is Unix seconds, describes COUNTS as counts/s, assigns an energy band/cadence contract, and does not apply GTI. Those assumptions are not established by the registered sources available in this audit. The HEL1OS reader remains fail-closed. No production adapter changes were made without the source documents needed to define safe conversions and rules.
GOES event records remain a separate reference source. NOAA report records in the interval: 123 (C=50, M=67, X=6); the report documentation does not establish population completeness, so negative labels are not allowed. Instrument event overlap, verified valid samples, and positive/negative window counts remain withheld; verified joint windows = 0. No final chronological split boundaries were chosen.

## Next minimal action

Sign in manually to PRADAN, retrieve the SoLEXS/HEL1OS registered L1 manuals, SIS and supporting GTI, quality, calibration/response and version-selection documentation, then save them unmodified in the requested folders. If the portal cannot provide them, place any officially downloaded copies in those folders with portal identifiers/revision metadata; do not treat unregistered reposts as authoritative. Rerun this audit before changing adapters or calculating scientific coverage.
