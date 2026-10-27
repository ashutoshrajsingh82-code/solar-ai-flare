# Read-only Aditya-L1 FITS structural audit

## Summary
- Candidate FITS products: 178
- Structurally inspected: 178
- Failed inspection: 0
- GTI-like tables: 56
- Instrument counts: {'SoLEXS': 24, 'HEL1OS': 154}
- Product/file counts: {'GTI': 56, 'LIGHTCURVE': 50, 'SPECTRA': 50, 'HOUSEKEEPING': 11, 'EVENTS': 11}
- GTI-like tables: 56; malformed/reversed/nonfinite GTI rows: 0
- Candidate quality/status fields: {'fehkstat': 11, 'czt1hotpixlgcstat': 11, 'czt2hotpixlgcstat': 11, 'cdte1pilectr': 11, 'cdte2pilectr': 11, 'czt1satctr1': 11, 'czt2satctr1': 11}
- Phase 3C evidence rows: 14; document-backed statuses: {'VERIFIED_FROM_AUTHORITATIVE_DOCUMENT': 7, 'PARTIALLY_RESOLVED': 6, 'STILL_UNVERIFIED': 1}.

## Interpretation limits
This is a structural audit. No timestamp conversion, GTI mask, or quality mask is applied. Missing GTI/quality metadata is not treated as valid exposure or clean data. Candidate values and headers do not establish scientific semantics. A documented rule may be verified only within the specific product scope cited in the evidence CSV.

## Verified source paths
- `C:\Users\ASHUTOSH\Downloads\solexs_2026Oct23T184402685.zip`
- `C:\Users\ASHUTOSH\Downloads\hel1os_2026Oct23T184518914.zip`
## Available inputs and evidence status
The audit inspected the explicit paths listed above, including nested archive members in memory. Extraction performed: `False`. See the per-file inventory for exact member names, hashes, and structural status.
Instrument documentation evidence is drawn from `C:\Users\ASHUTOSH\Downloads\solar-ai-flare-main\solar-ai-flare-main\backend\reports\phase4e_semantics_evidence_matrix.csv`. The per-rule dispositions are listed in `fits_document_evidence.csv`; statuses remain scoped to cited rules and do not establish overall training readiness.
No timestamp was converted or compared, no GTI endpoint rule was inferred, and no quality/status value was classified as good or bad. The reported rejection counts and final retained counts are null in audit mode because no masks were applied.

## Outputs
- fits_file_inventory.csv
- fits_hdu_inventory.csv
- fits_gti_validation.csv
- fits_quality_distribution.csv
- fits_document_evidence.csv
- fits_audit_summary.json
