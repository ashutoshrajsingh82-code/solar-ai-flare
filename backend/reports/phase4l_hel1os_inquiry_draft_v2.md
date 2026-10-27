# HEL1OS Level-1 clarification request — revision 2

**Operational status:** `REQUEST_PREPARED`; unsent.  **Request ID:** `REQ-HL-002`.

**Mapped blockers/rules:** `B-HL-TIME-004` / R07–R08; `B-HL-QUALITY-005` / R14; `B-HL-VERSION-006` / R13.

Dear authorized HEL1OS instrument/data-product contact,

We need an authoritative clarification for observed HEL1OS Level-1 detector/band light curves and their V111/V211 counterparts.  Please identify, for every answer, the governing document title, revision/publication date, exact page/section/table/field, and the applicable Level-1 detector, extension, product version, and date range.  If a formal policy does not exist, please state that explicitly.

1. For L1 `MJD` and UTC `ISOT`, state the authoritative stored representation, time scale/reference, permitted precision/disagreement, and the required behavior for missing, malformed, or conflicting values.  State whether this governs the observed Level-1 light curves rather than only Level-2 processing.
2. Define the bit meanings and units for L1 quality/status and housekeeping fields, including `fehkstat`, bunched-pixel/hot-pixel status fields, pile-up counters, and saturation counters.  State detector/version scope.
3. State all saturation thresholds, sentinel values, and authorized row-rejection or exposure-invalidation actions.  Distinguish diagnostic counters from mandatory Level-1 quality actions.
4. Define V111 versus V211 for the observed overlapping counterparts: release mapping, differences, supersession/precedence, calibration compatibility, and permitted treatment of overlap.  Please state explicitly if no precedence policy exists.

Current bounded evidence: HEL1OS User Manual v1.2, PDF p.13 §4 and pp.51–55 (Appendices B.3–B.4), SHA-256 `8efdf3aaf48d8a72cd9d85b567a8c0a6e2655ecf52c5a8d87f36915b741acb4d`; and Release Notes pp.4–5, SHA-256 `f9955c56e66f0465a1dda77ca4ed416e35771a71911cf2a24a4937d2c61167a7`.  These do not define the requested Level-1 authority, actions, or version precedence.

This request does not authorize timestamp reconciliation, quality filtering, counterpart merging, or scientific status change.  Any reply will be preserved and independently adjudicated.
