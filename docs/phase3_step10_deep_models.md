# Phase 3 Step 10: deep model updates

## Architecture audit

The application uses PyTorch sequence tensors shaped `[batch, time, channels]`. Deep models share the `MultiTaskHead`: four nowcast logits and five forecast logits (1 h, 3 h, 6 h, 12 h, 24 h). The existing CNN, LSTM, and CNN-LSTM encoders remain unchanged. The CNN-Transformer previously projected pooled CNN time steps directly to batch-first self-attention without an explicit time signal. Training already used weighted cross-entropy and masked `BCEWithLogitsLoss`; it did not restore the best epoch or stop early.

## CNN-Transformer

Sinusoidal positional vectors are added after the CNN's stride-2 max pool and linear projection, immediately before the batch-first Transformer encoder. Thus positions match the downsampled temporal axis and the transformer's `d_model`. The encoding buffer follows input device and dtype, and the model rejects lengths beyond its configured maximum. The encoding is fixed, adds no future observations, and the training builder configures the supported maximum from `sequence_length`.

## DLSTM and historical STL

`dlstm` is registered in the training API and deep-model factory. It consumes aligned `(trend, seasonal, residual)` triplets for each selected raw instrument and returns the shared multi-task logits. Hidden size and layer count are configurable. Each sample's STL is fit independently to exactly its lookback window ending at that sample's forecast origin. No whole-dataset decomposition is used. The documented transformation is `log1p(flux / training_median_positive)`, where each positive reference median is fit on training data and reused at inference. Negative, missing, nonfinite data, invalid scales, and windows shorter than two configured seasonal cycles are rejected explicitly; no value is imputed. STL's two-sided smoothing within the historical window makes its endpoint estimate less reliable than interior estimates, which is recorded in the manifest.

Precomputed samples use an in-memory cache keyed by source fingerprint, complete configuration/lookback/channel-scale fingerprint, UTC origin, and exact window bytes. This saves repeat work without sharing components between origins or configurations. STL remains a CPU cost for each unique origin.

## Training, thresholding, artifacts

Python, NumPy, PyTorch, and CUDA random generators are seeded. Deterministic PyTorch algorithms are enabled in warning mode where supported. Forecast targets must contain at least one positive in each trained horizon; training-only class weighting retains the existing weighted CE and positive-weighted masked BCE strategy. `ReduceLROnPlateau` and early stopping monitor finite validation loss; the best validation checkpoint is restored. Optimizer and scheduler states, epoch, training configuration, distributions, random settings, chronology, feature-manifest reference, target/horizons, and decomposition configuration are persisted.

Per-horizon max-TSS thresholds are selected from validation probabilities only (with an explicitly recorded 0.5 uncertain fallback when support is inadequate). The threshold and validation period are persisted before holdout prediction. The final report applies those frozen thresholds to the chronological test split. Older CNN-Transformer and DLSTM checkpoints without architecture version 2 are rejected with a retraining message.

## Scientific scope

Unit and synthetic tests validate tensor contracts, temporal encoding, window-limited decomposition, cache scoping, training mechanics, and threshold support handling. They do not establish real-world forecast skill. Model comparisons or performance claims require genuine, adequately populated, held-out observations and event-level support reporting.
