# Phase 4L response ingestion and adjudication procedure

1. Preserve the original provider message and attachments unchanged in a new, versioned intake location.  Record receipt time/timezone, sender/channel, issuer verification method, document title/revision, and relevant request/blocker/rule IDs.
2. Hash every original with SHA-256 and register canonical path, size, route, provenance, and revision separately from scientific adjudication.  Reject a missing file or hash mismatch.
3. Extract direct statements with page/section/table/field citations.  Record exact L1 product, detector, extension, version, date range, and scope.  Separately record interpretation, conflicts, and unaddressed questions.
4. Move an operational request to `RESPONSE_RECEIVED` only after preservation and provenance capture.  Move it to `UNDER_ADJUDICATION` only when a proposed 14-rule change is prepared.
5. Validate that the evidence governs the observed Level-1 product/version and satisfies the corresponding Phase 4K blocker criteria.  A provider response does not itself authorize conversion, filtering, or merge behavior.
6. Produce a new versioned, exact 14-row adjudication CSV with canonical paths/hashes and citations.  Run the existing Phase 3C validator; it must accept every evidence hash and surface conflicts.
7. Only a successful canonical Phase 3C adjudication and the recorded supporting evidence can move a request/blocker to `RESOLVED`.  Then rerun the separate data-readiness report.  Otherwise retain the current scientific rule status and record why.
8. Record every operational and scientific status change in both the request/blocker audit histories.  Do not overwrite the source response, Phase 4K baseline, or a prior adjudication.
