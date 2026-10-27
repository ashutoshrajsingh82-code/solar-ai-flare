# Draft technical inquiry — HEL1OS Level-1 contract

**Status:** draft for human review; not sent.

Dear HEL1OS data-product team,

We are conducting a read-only scientific-readiness audit of observed HEL1OS Level-1 light curves, including V111/V211 counterparts.  Please identify the governing document title, revision/date, exact page/section/table, and exact product/version applicability for every response.  If no policy exists, please state that explicitly.

1. For L1 light curves, what is the authoritative interpretation of stored `MJD` and UTC `ISOT`?  State time scale/reference, whether either field controls if they disagree, permitted disagreement/precision, and behavior for missing, malformed, or conflicting values.
2. Provide the definitions and bit meanings for L1 quality/status and housekeeping fields, including `fehkstat`, bunched-pixel/hot-pixel status fields, pile-up counters, and saturation counters.
3. State detector/version-specific saturation thresholds, sentinel values, and any authorized row-rejection or exposure-invalidation actions.  Please distinguish a diagnostic counter from a mandatory data-quality action.
4. Define the exact relationship between V111 and V211 for the observed Level-1 counterpart products: release mapping, changes, supersession/precedence, calibration compatibility, and the permitted treatment of overlap.

Current evidence is HEL1OS User Manual v1.2, p.13 §4 and pp.51–55 (Appendices B.3–B.4), SHA-256 `8efdf3aaf48d8a72cd9d85b567a8c0a6e2655ecf52c5a8d87f36915b741acb4d`, and Release Notes pp.4–5, SHA-256 `f9955c56e66f0465a1dda77ca4ed416e35771a71911cf2a24a4937d2c61167a7`.  They do not define conflict authority, quality actions, or V111/V211 precedence.

We will preserve the original response and attachments, register hashes and provenance, and apply any answer only after the existing Phase 3C adjudication validates it.  No time reconciliation, quality filtering, or version merge will be applied from this inquiry alone.
