# Phase 4B — Real data inventory and coverage

Generated: 2026-10-24T16:14:32.659387+00:00

## Gate status

**BLOCKED_DOCUMENTATION_UNAVAILABLE** — 0 of 14 rules verified. Real training is blocked; no checkpoint was created.

## Candidate observation coverage (unverified)

| Instrument | Product rows | Candidate observation rows | Candidate event-start occurrences | Verified coverage products |
|---|---:|---:|---:|---:|
| SoLEXS | 6 | 518,400 | 75 | 0 |
| HEL1OS | 44 | 1,047,288 | 284 | 0 |

Candidate event-start occurrences repeat across products/versions and are not unique flares or verified label windows. Observation timestamps and coverage remain provisional until the instrument semantics and quality/GTI rules are documented.

## GOES catalog reconciliation

The official NOAA report contains 123 records (C=50, M=67, X=6); 73 are M/X. The local event file has 115 candidate records; 107 exact start/class matches and 8 same-family magnitude discrepancies. Negative-label completeness is not established.

Eligible positive, negative, and unknown window counts are withheld. No partitions or training were started.

See `phase4b_real_data_inventory.json` for provenance, exact input hashes, product inventory, per-product candidate intervals, and per-rule status. See `phase4b_phase3c_blockers.csv` for the documentation and next-action register.
