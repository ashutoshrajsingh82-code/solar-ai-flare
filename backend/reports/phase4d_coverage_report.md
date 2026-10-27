# Phase 4D — Evidence-based semantics and label readiness

Audit timestamp: 2026-10-24T16:27:54.497180+00:00

## Decision

**BLOCKED — 0/14 SoLEXS/HEL1OS semantics verified. No verified real training dataset exists; no model fitting was run.** The authoritative Phase 3C gate remains `BLOCKED_DOCUMENTATION_UNAVAILABLE`. Instrument folders `source_docs/isro/solexs` and `source_docs/isro/hel1os` contain no files.

## Document inventory

See `phase4d_document_inventory.json` for every PDF/JSON under `E:\SolarFlareData\source_docs` and the GOES metadata directory, including byte size, SHA-256, PDF pages, and embedded metadata. The four available PDFs are the general ALPPS guide and three GOES documents; none is an instrument L1 contract. ALPPS User Guide PDF p.6 defines its scope and points instrument-specific details to instrument guides.

## Rule-by-rule evidence

`phase4d_semantics_evidence_matrix.csv` carries forward all 14 Phase 3C rules, their observed product evidence and unresolved questions. Each remains `UNVERIFIED`; the ALPPS scope statement is cited only to establish absence/scope, never to infer semantics. Required evidence includes SoLEXS RATE epoch/time scale, units, GTI endpoints, quality, calibration and version compatibility; HEL1OS MJD/ISOT authoritative time scale, CTR/STAT_ERR semantics, GTI endpoints, quality, calibration and V111/V211 product precedence. No adapter semantics were changed.

## GOES matching and negative labels (separate assessment)

The local `GOES_Flare_Report_ReadMe.pdf` documents report fields and composition: PDF pp.1-4, including one record per event, start/peak/end fields, and a four-minute plus irradiance-tolerance rule for reconciling observations between GOES satellites. That rule is internal to NOAA report composition; it does not establish direct identity matching against the local catalog. The local Phase 3B start-time candidate tolerance remains project policy, not a NOAA-authorized match rule. NOAA materials describe available-data/report inputs and caveats, but do not state that an absent record proves no flare. Thus catalog-unlisted intervals remain unknown. Existing reported audit counts remain provisional: 123 official events (C=50, M=67, X=6), 107 local exact start/class matches, 8 same-family magnitude discrepancies, 8 official-only events; 21 official rows lack event end. These are audit reconciliation counts, not independently verified instrument event windows.

## Coverage and training eligibility

Prior structural inventory: SoLEXS 518,400 raw rows across six light curves; HEL1OS 1,047,288 rows across 44 products, with V111/V211 overlap unresolved. These are raw parsed counts, not usable observations. Verified usable observations: **unknown**; verified positive/negative windows: **0 established**; negative completeness: **unverified**; independently verified event windows: **0**. No final chronological split feasibility can be established until time semantics, GTI/quality policy, paired-observation rules, event identity and label-completeness policy are resolved. No checkpoint was produced.

## Tests and scope

Existing focused tests cover provisional event matching, preserving class discrepancies, ambiguous candidates, unknown labels for incomplete coverage/catalog completeness, split-horizon purging and fail-closed instrument readers. Phase 4D adds no tests that assert unverified instrument semantics. See `phase4d_test_results.md`.

## Exact next step

Manually obtain the registered SoLEXS and HEL1OS Level-1 product manuals/SIS and applicable calibration/quality documents via authorized ISRO/ISSDC access, then place copies in the prepared instrument folders. After the documents are available, rerun:

```powershell
python scripts/phase3c_semantics_audit.py --data-root E:\SolarFlareData --repo-root C:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main --report-root E:\SolarFlareData\reports
```
