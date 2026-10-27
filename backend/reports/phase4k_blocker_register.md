# Phase 4K blocker register

Canonical gate: `BLOCKED_SCIENTIFIC_SEMANTICS_UNRESOLVED`.  All six blockers are `REQUEST_PREPARED`; no provider response has been received and no request has been sent.

- `B-SLX-GTI-001` — rule R03 — SoLEXS GTI endpoint/error/overlap contract.  Requires a versioned L1 FITS SIS/data dictionary.
- `B-SLX-PROV-002` — rule R05 — SoLEXS metadata, cadence, and version compatibility.  Requires an L1 provenance/schema specification.
- `B-SLX-CAL-003` — rule R06 — SoLEXS CALDB applicability and RATE-to-flux conversion.  Requires an applicable response/conversion and uncertainty policy.
- `B-HL-TIME-004` — rules R07/R08 — HEL1OS MJD/ISOT authority and discrepancy handling.  Requires an L1 SIS/time-correlation specification.
- `B-HL-QUALITY-005` — rule R14 — HEL1OS quality/HK bits, thresholds, and authorized actions.  Requires a versioned L1 HK/quality specification.
- `B-HL-VERSION-006` — rule R13 — HEL1OS V111/V211 mapping and supersession.  Requires an official version manifest or supersession notice.

The authoritative machine-readable register is `phase4k_blocker_register.json`; its companion CSV is a compact routing view.  The complete inquiry wording is in the SoLEXS and HEL1OS draft files, and response handling is governed by the response-ingestion procedure.
