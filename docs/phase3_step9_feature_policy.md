# Phase 3 Step 9: Feature policy

## Schema and availability

The feature schema is causal-time-aware-v2 (registry feature-registry-v2).
Normalized inputs are solexs_flux (SoLEXS) and helios_flux (HEL1OS), plus
UTC-compatible timestamp. Raw measurements remain source columns; model
selection uses their train-fitted *_scaled values and causal derivatives.
The repository does not establish calibrated physical units or matched energy
bands for these normalized channels. Consequently hard_soft_ratio and its
derivatives are disabled unless the caller supplies compatible units, energy
bands, physical quantity, measurement convention, and explicit compatibility
metadata. No bundled synthetic channel metadata qualifies as instrument
documentation.

## Causal features

For each channel the registry includes: measurement and scaled value (where
available), missing indicator, elapsed minutes since the last valid sample,
one-observation difference and percent change, trailing elapsed-time mean,
median, standard deviation, minimum, maximum, variance, 90th percentile,
recent peak, elapsed-time lag difference and least-squares slope, rolling
missing fraction, and causal spike score/flag. Default rolling durations are
5, 10, 30, and 60 minutes. Rolling statistics use trailing windows including
the observation at the origin and require the larger of two samples or the
configured fraction of expected cadence samples.

The spike score is (current - prior-window median) / (1.4826 * prior-window
MAD), with a configurable trailing lookback, minimum prior valid samples, and
absolute score threshold (default 6). The current observation is compared only
to strictly earlier samples. A zero MAD or insufficient prior history yields
an unknown score/flag; no arbitrary epsilon is added.

Timestamp quality features report interval seconds, cadence deviation,
irregular sampling, and long gaps. Rolling windows use elapsed time, not row
counts. Duplicate timestamps are rejected at feature generation. Resampling
uses right-closed, right-labeled bins so a bin's mean only becomes available at
its right edge. Cleaning retains spikes, marks negative flux as missing, and
does not interpolate or blanket-fill.

## Missing values and preprocessing

Flux zero is retained as a valid measurement. Invalid negative measurements
become missing. Missing values remain NaN through feature generation. The
log transform uses log1p(x / training_median_positive) with each reference
scale learned exclusively from the training partition, then applies the
train-fitted scaler. Inference does not forward-fill or fabricate zeros; it
rejects a model request if required latest features or sequence values are
unknown. Deep inference also rejects shorter-than-required windows rather than
padding them.

## Instrument selection

solexs_only, helios_only, and fusion use registry provenance. A
single-instrument selection excludes the other instrument's direct,
derived, quality, missingness, and ratio features. Timestamp-only cadence
quality features are available to either single-instrument experiment.
Unregistered numeric features with unknown provenance fail safe when selecting
an instrument-specific schema.

## Reproducibility and migration

Each newly trained model has a feature_manifest.json artifact with the exact
ordered inputs, definitions/provenance, registry and schema version,
configuration, ratio validation report, data fingerprint, split boundaries,
preprocessing configuration, dependency versions, and seed where available.
Inference loads and validates the manifest, experiment configuration,
feature provenance, order, and required data. Existing rolling_features_v1
checkpoints do not match this schema and must be retrained; they are not
silently reused.

## Limitations

The available bundled dataset is synthetic. These deterministic tests verify
causality, selection, and artifact compatibility, not measurement accuracy or
forecast skill on observed solar data. Ratio features remain disabled without
product metadata, and source-provided quality flag semantics are not inferred.
