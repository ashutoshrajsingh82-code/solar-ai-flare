# Scientific data recovery audit

Audit date: 2026-10-21. This audit inspects the checked-out repository, the local `backend/solarflare.db`, and its persisted pipeline/model reports. It does not infer data availability from names alone, and it does not inspect or disclose environment-variable values.

## Root cause

The current trained forecasting interface is built around two normalized numeric columns, `solexs_flux` and `helios_flux`. Those are the only bundled observations with such names, but their file is explicitly a demo: `backend/sample_data/demo_aditya_l1_solexs_helios.csv`. The checked-in adapter for native SoLEXS L1 light curves emits `solexs_counts_per_second`, not calibrated flux. HEL1OS parsing raises `NotImplementedError` until the registered product manual and a representative file establish its product schema. No scientifically defensible mapping currently turns both instruments' products into the two model input columns.

The demo sample contains 14,401 one-minute rows from 2026-08-01 00:00 UTC through 2026-08-10 23:59 UTC. Repository scan SHA-256: `628e3d0425f79f90b66c85964162c232fdb54feb56ced49f0e8951ec22675d4f`. It includes 14,401 timestamps, 14,369 finite values per named channel, and one duplicate timestamp. The second sample, `backend/sample_data/demo_flare_events.csv`, is an index-based synthetic event fixture with 14 rows; SHA-256: `aee4c8eab689d9415281cb5440ba92b338feaaa1f14520186909bac32d6c6c47`. Neither is a satellite observation or GOES reference report.

The local database contains one dataset row: source `demo`, filename `demo_aditya_l1_solexs_helios.csv`, and 14,401 rows. The record has no raw-file hash or file-format metadata (it predates the new metadata fields); its source metadata fingerprint is for the loaded data frame. The persisted default pipeline manifest also identifies this dataset as `demo` and contains only its raw stage. No SoLEXS/HEL1OS FITS, NetCDF, product manuals, calibration files, or local GOES reference CSV were found in the repository scan. FITS files built during tests are generated test fixtures, not retained observations.

The `backend/reports/phase3_step11_demo/model_artifacts` directory contains 24 feature manifests, 15 `.pt` checkpoints, and classical-model artifacts. They sit under an explicitly demo-named run directory; their evaluation reports label results synthetic/demo and include held-out horizons with zero positive cases. No checkpoint produced from genuine instrument observations was found. These artifacts must not be cited as observational validation.

## Product and label contract

- **SoLEXS:** Official URSC documentation describes soft X-ray spectra over 2–22 keV, one-second spectral cadence, and broad temporal-channel data at 0.1 seconds. The current parser's `RATE.TIME`/`RATE.COUNTS` FITS table contract came from an unauthenticated manual copy and has only synthetic-FITS unit tests. The present repository has no genuine file with which to verify exact HDUs, column units, time reference, product version, quality/saturation flags, GTIs, or calibration state. It preserves native counts/s and does not apply GTI filtering or convert counts to irradiance.
- **HEL1OS:** Official URSC/PRADAN descriptions identify a hard X-ray instrument with CdTe 10–40 keV and CZT 20–150 keV detectors and Type-II spectra/light-curve products. The exact product-level schema, timestamps, detector/channel fields, cadence, quality rules, response/calibration products, and conversion to physical quantities remain unknown to this checkout. Its adapter correctly refuses ingestion instead of guessing.
- **GOES targets:** The local adapter reads NOAA/NCEI GOES XRS L2 Flare Report CSV; it uses the event `start_time` onset, `time` peak, `end_time`, and `flare_class`. Forecast targets are M/X onset in `(observation_time, observation_time + horizon]` for 1, 3, 6, 12, and 24 hours. A target is unknown unless the entire future window is inside declared report coverage. There is no genuine GOES report file in this checkout, only tests. The new parser provenance records accepted flare-class counts and the M/X event count.
- **Model input mismatch:** `backend/app/services/data_service.py` expects `timestamp`, `solexs_flux`, and `helios_flux`; feature names inherit the `flux` assumption. A SoLEXS count rate and a HEL1OS detector/spectral series are not interchangeable physical fluxes. The project has not established energy-response conversion or a matched physical measurement definition. The ratio feature remains disabled without compatible explicit metadata.

## Official references and access prerequisites

- URSC's [SoLEXS instrument page](https://www.ursc.gov.in/SoLEXS.jsp) describes the instrument range and broad cadences; it does not document the exact current FITS table schema.
- URSC's [HEL1OS instrument page](https://www.ursc.gov.in/hel1os.jsp) and the [ISRO PRADAN Aditya-L1 archive](https://pradan.issdc.gov.in/al1/) describe its detectors and science products. PRADAN is a login-based archive; the [ALPPS user guide](https://alpps.issdc.gov.in/web/extensions/Aditya-l1/resources/docs/ALPPS_UserGuide_V1.0.pdf) says default-mode science products are public without an exclusive lock-in period, with products archived/hosted through ISSDC.
- NOAA/NCEI's [GOES Flare Report ReadMe](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES_Flare_Report_ReadMe.pdf) documents the event report; NOAA's [EXIS/XRS product page](https://www.ncei.noaa.gov/products/goes-r-extreme-ultraviolet-xray-irradiance) links the L1b/L2 guides and data.

This environment has no authenticated PRADAN session and no downloaded product bundle or registered instrument manuals. To validate the current parser contract, provide one genuine SoLEXS L1 light-curve file plus its matching registered manual, version/manifest, GTI/quality companions, and calibration status; provide a corresponding HEL1OS observation bundle with its registered manual and all calibration/quality companions; and provide an overlapping NOAA/NCEI GOES report with explicit complete coverage. The manual must establish timestamp reference, units, channel/detector mapping, quality/saturation flags, missing-value sentinels, cadence, and response corrections. Actual training additionally requires enough independent GOES-labeled events for train/validation/test chronological periods; the project does not define a fixed minimum event count, so this audit does not invent one.

## Changes in this recovery step

- Added `GET /api/data/readiness`, which writes a machine-readable JSON report and human-readable Markdown report under the persistent pipeline artifact directory.
- The report measures loaded row count, UTC coverage, cadence, invalid/duplicate timestamps, present/finite input channels, paired-channel count, per-horizon known/positive/negative/unknown labels, GOES file hash/coverage/class counts, split sizes, and scientific-provenance evidence.
- Real-source training now generates the report first and fails with its blockers and report location if the evidence is insufficient. Explicit demo training remains demo-only under the existing model-status rules.
- GOES ingestion now records accepted flare-class counts and an explicit M/X event count in its ingestion report.
- Existing real-product ingestion remains fail-closed where format details are not verified. No demo data, fabricated labels, or calibrated flux conversion was added.

## Current readiness snapshot

The checked-out `backend/artifacts/pipeline` state is **BLOCKED**: it restores the demo dataset, has one duplicate timestamp, has no GOES labels, no verified measurement contract, and no train/validation/test splits. There are zero verified real observations available to evaluate and zero real-trained checkpoints found. Accordingly, no real-data training or held-out performance result is reported.

The generated snapshot is in `backend/reports/scientific_data_readiness.json` and `.md`. At runtime, the API regenerates the current snapshot at `PIPELINE_ARTIFACT_DIR/readiness_reports/scientific_data_readiness.json` and `.md`. Synthetic fixtures remain limited to automated tests.
