# Phase 4E: real-data location and FITS audit

Audit date: 2026-10-24. This report uses the verified Phase 4E inventory and the read-only nested-archive FITS audit. No archives were extracted, moved, renamed, or modified.

## Verified current inputs

| Source | Verified path | Size | SHA-256 | Status |
|---|---|---:|---|---|
| SoLEXS Level-1 nested archive | `C:\Users\ASHUTOSH\Downloads\solexs_2026Oct23T184402685.zip` | 29,334,261 bytes | `34c39624194ee2da95bbf8e3dfed5ac002c55a517a31bc4091bfdd0d2b4671c2` | Readable; 6 nested daily archives, 2026-01-31 to 2026-02-05 |
| HEL1OS Level-1 nested archive | `C:\Users\ASHUTOSH\Downloads\hel1os_2026Oct23T184518914.zip` | 1,066,706,121 bytes | `17654c33aff72313e796ceafb31577224f66f9f6b366438f8d655518ec033e17` | Readable; nested product archives; 11 data dumps |
| GOES event catalog | `C:\Users\ASHUTOSH\Downloads\data\goes\goes_flares.csv` | 8,915 bytes | `a19a8e8af74164af60a6ca74387d611f93cc0b6d8e47427ac79e60b7b5e76cb9` | Readable local catalog; completeness is not established by local presence |
| GOES 1-minute measurements | `C:\Users\ASHUTOSH\Downloads\data\goes\goes_xrs_1min.csv` | 413,137 bytes | `891e38c4798ec28ecbd674593ba1854bb9771331fb65da44b78d7bc35705be5e` | Readable |
| SoLEXS manual | `C:\Users\ASHUTOSH\Downloads\SoLEXS-UserManual.pdf` | 363,311 bytes | `d97c83618c85a0984e7dd0300ce5c16c4748ed5facde525949f902fe9f72abda` | Found outside configured root; evidence reconciled in Phase 4E matrix |
| HEL1OS manual | `C:\Users\ASHUTOSH\Downloads\HEL1OS_UserManual.pdf` | 1,540,669 bytes | `8efdf3aaf48d8a72cd9d85b567a8c0a6e2655ecf52c5a8d87f36915b741acb4d` | Found outside configured root; evidence reconciled in Phase 4E matrix |

The configured root `E:\SolarFlareData` has no `raw` or `extracted` directory and no direct FITS products. Original archives and manuals remain in Downloads. Nested archive members were inspected in memory; there was no extraction destination.

## Structural audit results

The audit was run against the two explicit archive paths, recursively inspecting nested ZIP members without extraction. It found 178 FITS products: 24 SoLEXS and 154 HEL1OS. All 178 were structurally inspected; 0 inspection failures. File associations: 56 GTI, 50 lightcurve, 50 spectra, 11 housekeeping, and 11 event products.

The audited tables included 84 populated SoLEXS GTI rows plus six empty SDD1 companion tables, and 44 HEL1OS GTI rows. No malformed, nonfinite, reversed, duplicate, or unsorted intervals were flagged by structural validation. Seven kinds of candidate HEL1OS housekeeping status/counter columns were observed. Their values remain uninterpreted as quality rules. No time conversion or GTI/quality filtering was applied; no records are certified science-ready by this audit.

The Phase 4E evidence matrix has 14 rows: 7 `VERIFIED_FROM_AUTHORITATIVE_DOCUMENT`, 6 `PARTIALLY_RESOLVED`, and 1 `STILL_UNVERIFIED`. The canonical Phase 3C audit rerun remains `BLOCKED_DOCUMENTATION_UNAVAILABLE` with 0/14 verified because its document scanner only registers documents in the configured source-document tree; the manuals were found in Downloads and are recorded with hashes and citations in the additive Phase 4E matrix.

## Training readiness

Real observational training remains blocked. Unresolved items include SoLEXS GTI time units and endpoint closure, SoLEXS quality/calibration requirements for the intended lightcurve task, HEL1OS authoritative time-scale authority, product-version precedence, and a defensible basis for negative-label completeness. The six-day overlap alone does not establish independently labelled events or leakage-safe train/validation/test feasibility. Unknown and uncatalogued intervals remain unknown.

No model was trained, no checkpoint was created, no production registry was changed, and no observational data were modified.

## Reproduction commands

From the backend directory:

```powershell
python scripts\phase4e_verified_inventory.py
python scripts\phase4e_document_reconciliation.py
python scripts\fits_semantics_audit.py --data-root E:\SolarFlareData --report-root reports\phase4e_real_fits_audit --input C:\Users\ASHUTOSH\Downloads\solexs_2026Oct23T184402685.zip --input C:\Users\ASHUTOSH\Downloads\hel1os_2026Oct23T184518914.zip
python -m pytest tests\test_fits_semantics_audit.py tests\test_phase3a_data_audit.py tests\test_phase3b_authoritative_audit.py tests\test_phase3c_semantics_audit.py tests\test_real_data_sources.py tests\test_chronological_audit.py tests\test_feature_engineering_step9.py tests\test_scientific_readiness.py tests\test_phase2c_eligibility.py -q
```

The Phase 4E structural outputs are in `backend/reports/phase4e_real_fits_audit/`; file-level provenance is in `phase4e_verified_data_inventory.json`, and documentary evidence is in `phase4e_semantics_evidence_matrix.csv` plus `phase4e_document_register.json`.
