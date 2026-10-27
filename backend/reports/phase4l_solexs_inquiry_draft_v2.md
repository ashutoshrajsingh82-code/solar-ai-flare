# SoLEXS Level-1 clarification request — revision 2

**Operational status:** `REQUEST_PREPARED`; unsent.  **Request ID:** `REQ-SLX-001`.

**Mapped blockers/rules:** `B-SLX-GTI-001` / R03; `B-SLX-PROV-002` / R05; `B-SLX-CAL-003` / R06.

Dear authorized SoLEXS instrument/data-product contact,

We need an authoritative clarification for observed SoLEXS SDD2 Level-1 daily light-curve and companion-GTI products.  Please identify, for every answer, the governing document title, revision/publication date, exact page/section/table/field, and the applicable Level-1 product, detector, archive/product version, pipeline version, and cadence.  If a formal policy does not exist, please state that explicitly.

1. For the L1 GTI `START` and `STOP` fields, state units, epoch/reference, time scale, and the exact relationship to the L1 `RATE.TIME` coordinate.
2. State whether each GTI endpoint is closed, open, or half-open; how adjacent/overlapping intervals are interpreted; and the authorized treatment of empty, reversed, duplicated, malformed, or overlapping intervals.  Please identify any permitted repair, discard, or rejection action.
3. State which combinations of L1 product/archive version, pipeline version, `CREATOR`, `DATE`, `OBS_ID`, and `TIMEDEL` are compatible for analysis, including cadence guarantees and incompatible combinations.
4. State which CALDB/response versions apply to these native SDD2 L1 light curves.  If an approved count-rate-to-physical-flux/irradiance conversion exists, provide the equation, target unit, response-selection rule, assumptions, and uncertainty treatment; otherwise state that no approved conversion exists.

Current bounded evidence: the SoLEXS Data Analysis Guide v1.0, PDF pp.5–6 (§§2.1.1–2.2), SHA-256 `d97c83618c85a0984e7dd0300ce5c16c4748ed5facde525949f902fe9f72abda`; release updates; and `solexs_tools-1.1`.  These do not define the requested GTI behaviors, full compatibility matrix, or native L1 flux conversion.

This request does not authorize a GTI filter, time conversion, calibration conversion, or scientific status change.  Any reply will be preserved and independently adjudicated.
