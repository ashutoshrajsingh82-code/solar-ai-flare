# Real SoLEXS and HEL1OS data ingestion contract

## Scope and verification status

The backend supports demo data unchanged and has an offline, local-file ingestion path for the documented SoLEXS Level-1 light curve FITS product. No downloaded SoLEXS or HEL1OS files or authenticated ISSDC user manuals were available in this workspace. Consequently, the parser is covered with synthetic FITS fixtures, not a real-product integration test. HEL1OS raw-product parsing is intentionally gated: its archive FAQ says HEL1OS always operates in event mode, but the registered user manual is needed to establish the event/listing and science-product fields. Do not point the HEL1OS adapter at an assumed FITS schema.

## Product contracts

| Product | Verified contract | Still unverified / limitations |
|---|---|---|
| SoLEXS L1 light curve (LC), detector-specific | SoLEXS instrument; L1; FITS with gzip compression is described in the SoLEXS user manual. `RATE` table has `TIME` (Unix seconds) and `COUNTS` (counts/s) for 2–22 keV, one-second cadence. Timestamp normalization is UTC. SoLEXS covers 2–22 keV; do not treat these counts as calibrated physical flux. | Product version, exact deployed FITS extension/header variants, quality flags, missing-value sentinels, calibration/deadtime status per file, and real-product behavior must be checked against the matching registered manual and file. The parser requires RATE/TIME/COUNTS and fails closed on a mismatch. GTI companion filtering is not applied yet. |
| SoLEXS L1 PI Type-II spectra | Manual describes one-second integrations; `SPECTRUM` has TSTART, TELAPSE, SPEC_NUM, CHANNEL, COUNTS, EXPOSURE. | Not ingested. Energy bin interpretation/calibration response requires the matching CALDB and manual; do not sum channels as though they were an approved light curve. |
| HEL1OS observations/products | Official PRADAN description: hard X-ray instrument, CdTe 10–40 keV and CZT 20–150 keV detector bands, Type-II spectra and light curves (10–150 keV); official archive FAQ says HEL1OS operates in event mode. | Product levels, file format/version, event fields, time reference, energy calibration, detector IDs, quality flags and cadence/aggregation are not established from accessible official documents. No parser or model normalization is enabled pending authenticated user manual and sample. |
| NOAA/NCEI GOES XRS L2 Flare Report (independent reference labels) | Official NCEI report, CSV or NetCDF, mission-wide/yearly files, current documented v1.0.1. CSV event columns include `start_time`, `time` (peak), `end_time`, `flare_class`, and `xrsb_irrad` (W/m²). CSV times are datetimes interpreted as UTC; NetCDF times are seconds since 1970-01-01 UTC neglecting leap seconds. One event per flare; source measurements are 1-minute XRS-B averages. The adapter supports CSV and uses `start_time` as onset; only M/X events are positive under the existing target definition. | File coverage must be explicitly supplied to distinguish true negatives from unknown periods. The report includes GOES-8 through GOES-19 for 1995–2025 in the documented release; confirm chosen version, period, and caveats. GOES classes are independent reference labels, not SoLEXS/HEL1OS measurements. |

Official sources: [ISRO Science Data Archive (PRADAN)](https://pradan.issdc.gov.in/al1/), [ISRO note that payload manuals require portal registration](https://www.isro.gov.in/ISRO_EN/Second_Set_of_Aditya_L1_Science_Data.html), [URSC SoLEXS page](https://www.ursc.gov.in/SoLEXS.jsp), [URSC HEL1OS page](https://www.ursc.gov.in/hel1os.jsp), [PRADAN FAQ](https://pradan1.issdc.gov.in/al1/faq.xhtml), and [NOAA/NCEI GOES Flare Report ReadMe](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES_Flare_Report_ReadMe.pdf). The detailed SoLEXS FITS table contract above is based on a publicly discoverable copy of the SoLEXS user manual, not an authenticated PRADAN download. Confirm it against the registered manual matching the downloaded product version before production use.

## API and local usage

- `DATA_MODE=demo` remains the default. Startup demo bootstrap remains unchanged.
- `POST /api/data/upload-real?instrument=solexs&product=lc` accepts a local-user-uploaded `.fits` or `.fits.gz` SoLEXS L1 LC file. It never substitutes demo data. An unsupported product or schema returns a clear 422 response.
- `POST /api/data/upload-real?instrument=hel1os&product=...` fails closed with a request for the authenticated manual/product contract; it does not interpret arbitrary columns as HEL1OS measurements.
- After loading a real observation dataset, `POST /api/data/upload-goes-labels?coverage_start=...&coverage_end=...` accepts an official NOAA/NCEI GOES Flare Report CSV and attaches forecast labels plus independent source provenance. Explicit UTC coverage bounds are mandatory; rows whose full horizon is outside coverage remain unknown (NaN), not negative.
- Legacy normalized CSV upload remains for compatible pre-normalized inputs, and the demo endpoint remains available.
- SoLEXS COUNTS are stored as `solexs_counts_per_second`; they are not renamed to `solexs_flux`. The current forecasting pipeline requires both `solexs_flux` and `helios_flux` plus independent verified labels. Therefore raw mission observations are not currently eligible for preprocessing/training/prediction through that pipeline. `/api/preprocess` reports this explicitly instead of silently changing scientific meaning.

## Validation and alignment policy

The SoLEXS LC reader checks file readability, FITS structure, required named fields, numeric values, finite/nonnegative counts, finite Unix times, nonempty input, duplicate timestamps, and chronological order. It reports rejected rows and missing intervals; out-of-order or duplicate times are rejected rather than silently sorted/deduplicated. Original Unix seconds and file/header provenance are retained. It does not apply an undocumented quality filter or fill gaps. A reusable alignment utility can aggregate already-normalized inputs independently using configured `mean`, `sum`, or `last` bins, outer-join on UTC timestamps, and retain NaNs; the appropriate aggregation must be selected from the product units. It reports source periods, overlap, source-only/missing bins, and actual input cadences, and does not interpolate. HEL1OS parsing and actual cross-instrument alignment remain disabled until its time/quality convention is verified.

## Labels, leakage and evaluation

The existing nowcast state labels are synthetic/event-catalogue fields: state 0–3. The independent GOES adapter supports only forecast onset targets, using official `start_time` and M/X classes over the existing 1, 3, 6, 12, and 24-hour horizons. GOES class does not label SoLEXS/HEL1OS spectral measurements as calibrated flux. Real training still requires a scientifically validated dual-instrument model adapter and a defensible nowcast target; the current pipeline refuses native counts and unknown state labels. Chronological training label generation is split-scoped and horizon tails are purged, but that does not itself establish scientific validity. No real-data training, forecast, or performance claim is made here.

## Exact files required for end-to-end validation

1. A sanitized SoLEXS PRADAN download bundle for one day, including the archive manifest/version and one detector’s L1 LC FITS.gz plus any GTI/quality companion and its matching registered user manual.
2. A sanitized HEL1OS PRADAN download bundle for a matching day, including product level/version, event or Type-II spectrum/light-curve files, all calibration/quality companions, and matching registered HEL1OS user manual.
3. An authoritative flare-event reference list covering both observation windows, with event start/peak/end times, class/category, time standard and uncertainty/quality fields, plus its published matching rules.

After obtaining these, verify the exact headers, units, time system, detector/quality semantics, overlap, and calibration status before enabling HEL1OS parsing, paired alignment, model adaptation, training, or evaluation.
