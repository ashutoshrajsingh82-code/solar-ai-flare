# Draft technical inquiry — SoLEXS Level-1 contract

**Status:** draft for human review; not sent.

Dear SoLEXS data-product team,

We are conducting a read-only scientific-readiness audit of the SoLEXS L1 products.  We seek an authoritative, citable answer for the observed SDD2 L1 format.  Please identify the governing document title, revision/date, exact page/section/table, and the product/version scope for each response.  If no policy exists, please state that explicitly.

1. For L1 GTI `START` and `STOP`, what are the units and time reference?  Does each value use precisely the same coordinate as L1 `RATE.TIME`?
2. Are GTIs closed, open, or half-open at each endpoint?  How are adjacent or overlapping intervals interpreted?
3. What is the prescribed handling of empty, reversed, duplicated, malformed, or overlapping GTI intervals?  Is any repair, discard, or rejection permitted?
4. Which L1 product versions, pipeline versions, cadence values, and `CREATOR`, `DATE`, `OBS_ID`, and `TIMEDEL` metadata are mutually compatible for analysis?
5. Which CALDB/response files apply to the observed native SDD2 L1 light curves?  Is there an approved count-rate-to-physical-flux/irradiance conversion, including equation, units, response selection, assumptions, and uncertainty propagation?

Current evidence is the SoLEXS Data Analysis Guide v1.0, pp.5–6 (§§2.1.1–2.2), SHA-256 `d97c83618c85a0984e7dd0300ce5c16c4748ed5facde525949f902fe9f72abda`, the release-updates file, and `solexs_tools-1.1`.  These are insufficient for endpoint/error policy, metadata compatibility, and L1 flux conversion.

We will preserve the original response and attachments, register hashes and provenance, and apply any answer only after the existing Phase 3C adjudication validates it.  No GTI filter or flux conversion will be applied from this inquiry alone.
