# Read-only Aditya-L1 FITS structural audit

## Summary
- Candidate FITS products: 0
- Structurally inspected: 0
- Failed inspection: 0
- GTI-like tables: 0
- Phase 3C evidence rows: 14; structural observations do not resolve scientific rules.

## Interpretation limits
This is a structural audit. No timestamp conversion, GTI mask, or quality mask is applied. Missing GTI/quality metadata is not treated as valid exposure or clean data. Candidate values and headers do not establish scientific semantics.

## Available inputs and evidence status
No candidate FITS files or archives containing FITS products were found recursively under the configured data root at audit time. Consequently there are no successfully inspected or failed FITS files to report. The Phase 3A/3B inventory already stored in the reports directory describes archives located elsewhere; this run did not treat those records as current inputs under the configured root.
The local document set contains `ALPPS_UserGuide_V1.0.pdf` and three GOES PDFs. `SoLEXS-UserManual.pdf`, `HEL1OS_UserManual.pdf`, and instrument-specific product specifications were not found under the configured root. The current Phase 3C rerun remains `BLOCKED_DOCUMENTATION_UNAVAILABLE`, with 0/14 rules verified. See `fits_document_evidence.csv` for the 14 row-by-row disposition; each is STILL_UNVERIFIED because available documents and headers do not establish its scientific interpretation.
No timestamp was converted or compared, no GTI endpoint rule was inferred, and no quality/status value was classified as good or bad. The reported rejection counts and final retained counts are null in audit mode because no masks were applied.

## Outputs
- fits_file_inventory.csv
- fits_hdu_inventory.csv
- fits_gti_validation.csv
- fits_quality_distribution.csv
- fits_document_evidence.csv
- fits_audit_summary.json
