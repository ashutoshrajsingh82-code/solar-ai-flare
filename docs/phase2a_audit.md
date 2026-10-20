# Phase 2A repository audit

Audit performed before implementation. No SoLEXS or HEL1OS product files are present in `backend/sample_data`; the only bundled observation file is a synthetic CSV.

## Existing data flow and contracts

- On startup, `DATA_MODE=demo` (the default) calls `load_demo_data`, loading or generating `demo_aditya_l1_solexs_helios.csv`. Other modes do not bootstrap data. `/api/data/load-demo` remains explicit.
- `/api/data/upload` currently accepts only CSV with `timestamp`, `solexs_flux`, and `helios_flux`; it supplies unknown labels (`state=0`, `flare_class='-'`) when absent. The general cleaner sorts timestamps and removes duplicates, but may clip negative values and interpolate short gaps. That policy is a synthetic/normalized-input policy, not established as valid for instrument counts.
- Alignment resamples both columns to a configured cadence. Feature engineering assumes both named columns. The transformer is fitted on the earliest chronological training portion and persisted; inference must reuse it.
- Existing state labels use 0=quiet, 1=pre-flare, 2=flare, 3=decay. Forecast targets are M/X event onsets within 1, 3, 6, 12, or 24 hours, inferred from `state` and `flare_class`. This is appropriate only when verified reference event labels already exist. No independent label adapter is present.
- Training uses chronological splits, but current stage order generates labels after the split and does not purge overlapping forecast windows at boundaries. This pipeline cannot be used for scientific real-data training until verified labels and leakage-safe split handling are added.
- Supported runtime dependencies include pandas and Astropy. `Dataset` records source, filename, time range, cadence, missing count, quality score, and channels; Phase 2 added disk artifacts and provenance-capable metadata.

## Product contract status

- Official URSC pages describe SoLEXS as a 2–22 keV soft-X-ray instrument and HEL1OS as a 10–150 keV hard-X-ray instrument. The URSC SoLEXS page reports 1 s spectra and 0.1 s broad temporal channels; those channels are different products and must not be conflated.
- ISRO says the product user manuals are available through the ISSDC/PRADAN portal after registration. No authenticated archive session or downloaded user manual/product is available in this workspace. Therefore mission-file extension/level details, exact column/header names, time reference keywords, unit/calibration semantics, product versions, quality flags, and HEL1OS schema are not independently verified here.
- The demo CSV is synthetic and is not a valid fixture for either mission product.

## Implementation boundary

Implement a fail-closed source adapter for only a product schema that can be stated and validated; preserve original units and source metadata. Do not map counts to calibrated physical flux, merge different energy bands, invent labels, or train/forecast from unlabelled observations. HEL1OS and paired-instrument support remain gated on its registered manual and a representative downloaded file if their exact schema cannot be verified from official public documentation.
