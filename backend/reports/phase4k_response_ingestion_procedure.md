# Phase 4K controlled provider-response ingestion

## Intake

1. Preserve the received message and each attachment in original form; do not edit, rename in place, or overwrite an existing canonical artifact.
2. Record receipt date/time, sender/contact route, issuer/organization, stated document title, revision/publication date, and each affected blocker ID.  A request draft or transmission receipt is not scientific evidence.
3. Calculate SHA-256 from the preserved original.  Register its canonical path, hash, size, acquisition route, and provenance separately from rule adjudication.

## Evidence review

4. Verify that the sender is an authorized instrument/data-product representative or that the attachment is an issuer-authored artifact.  Record verification method and limitations.
5. Extract only precise claims with page/section/table/field citations.  Record product level, detector, product version, date range, and any stated limitations.
6. Record contradictions with registered sources or among response statements as `CONFLICTING_EVIDENCE`; do not resolve them by preference or sample behavior.
7. Mark a blocker `RESPONSE_RECEIVED` when provenance is recorded.  It remains unresolved until the response meets every stated adjudication criterion.

## Canonical adjudication

8. Create a versioned 14-row adjudication CSV.  Retain the one-to-one mapping and cite only canonical, hash-validated documents.  Do not change unrelated rows.
9. Run `scripts/phase3c_semantics_audit.py --adjudication <versioned-file>`.  The existing validator rejects unregistered hashes and records conflicts.  A blocker may move to `RESOLVED` only after this run applies qualifying evidence; otherwise retain `RESPONSE_RECEIVED`, `REQUEST_PREPARED`, or `UNAVAILABLE`.
10. Re-run `backend/scripts/phase4h_data_readiness.py` against the versioned canonical output.  The readiness gate stays blocked while any required rule is partial, unverified, or conflicting.

## Prohibited shortcuts

Do not infer a contract from headers, sample agreement, Level-2 behavior, a reply without product/version scope, an undocumented engineering convention, row count, or a request itself.  Do not apply time conversions, GTI/quality filters, calibration conversions, V111/V211 merges, or label changes until a qualified rerun authorizes them.
