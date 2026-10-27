<!-- phase4h_data_readiness_v1 -->
# Phase 4H - data-readiness gate

- Canonical readiness: **BLOCKED_SCIENTIFIC_SEMANTICS_UNRESOLVED**
- Rules (verified / partial / unresolved): **7 / 6 / 1**

## Gate assessment

| Item | Status | Rationale |
|---|---|---|
| Canonical source-document provenance | **PASS** | Only hash-matching local canonical documents may support adjudication. |
| Archive structural inventory | **PASS** | Structural readability is not scientific eligibility. |
| Valid observation timestamps and time reference | **BLOCKED** | No combined observation-time policy is established. |
| Applicable GTI intervals and endpoint semantics | **BLOCKED** | No GTI mask or endpoint convention may be applied. |
| Quality and saturation handling | **BLOCKED** | Missing metadata is not evidence of clean data. |
| Instrument/product/version precedence | **BLOCKED** | Products remain separate; row counts and apparent recency are not precedence evidence. |
| Calibration eligibility and physical-flux conversion | **BLOCKED** | Count rate is not promoted to physical flux. |
| Observation coverage and gaps | **BLOCKED** | File presence and raw rows do not establish usable coverage. |
| Verified flare-event labels and provenance | **BLOCKED** | Reference catalog provenance does not establish instrument-positive windows. |
| Completeness of negative labels | **BLOCKED** | Unobserved or uncatalogued intervals are not negative examples. |
| Chronological split feasibility without leakage | **BLOCKED** | No eligible observation/label population exists for a leakage assessment. |

## Official acquisition routes

| Source | Outcome | Scope |
|---|---|---|
| [PRADAN Aditya-L1 FAQ](https://pradan1.issdc.gov.in/al1/faq.xhtml) | PUBLICLY_ACCESSIBLE_REVIEWED | HEL1OS FAQ directs technical specifications to protected OtherDownloads; no field-level contract. |
| [Aditya-L1 Proposal Processing System User Guide V1.0](https://alpps.issdc.gov.in/web/extensions/Aditya-l1/resources/docs/ALPPS_UserGuide_V1.0.pdf) | PUBLICLY_ACCESSIBLE_REVIEWED | No GTI, time-field, quality-bit, response-conversion, or version-precedence contract. |
| [PRADAN Other Downloads](https://pradan.issdc.gov.in/al1/protected/miscDownloads.xhtml) | ACCESS_RESTRICTED | ISSDC authenticated session required; no access control was bypassed. |

## Safety

No training, tuning, evaluation, checkpoint activity, production activation, GTI/quality filtering, counterpart merge, or negative-label promotion was performed.
