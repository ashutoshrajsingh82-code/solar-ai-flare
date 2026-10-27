# Phase 3 Step 11: integration and scientific verification

## Reproducible CPU smoke benchmark

From `backend/`, run:

```powershell
python -m ml.evaluation.phase3_benchmark --epochs 1 --batch-size 128 --sequence-length 60 --out-dir reports/phase3_step11_demo
```

The runner uses the bundled CSV by default, executes the application preprocessing pipeline, checks strict UTC split ordering, runs the configured representative baseline/deep-model set, persists each model through the production artifact catalog, and calls the production raw-window inference service repeatedly to check deterministic output and measure warm median latency. It writes `benchmark.json` and `benchmark.csv` with one row per model and forecast horizon. Model artifacts, the isolated SQLite artifact catalog, and pipeline artifacts are saved beside those reports. Use `--models name1,name2` for a subset or `--all-models` to attempt every registered model. `--csv` can select another input, but the real-data eligibility gate remains authoritative and blocks datasets without required provenance evidence.

The audit records observation boundaries, purge counts, training-only scaler policy, input-context length, exact window counts, per-horizon known/positive/negative/unknown window counts, and distinct event counts from the available event catalogue. Validation/test windows reuse only preceding observations as input context. Target horizons remain split-scoped and their split tails are purged.

## Integration findings

- Deep models used training and validation for checkpoint/threshold selection; classical training previously bypassed validation and evaluated at 0.5. Classical models now fit only on training rows, freeze per-horizon max-TSS thresholds from validation probabilities, and evaluate test rows once with those frozen thresholds. Thresholds are stored in the evaluation report, model record, and checkpoint.
- The split audit previously undercounted validation/test windows because it omitted permitted historical context. It now uses the shared `build_windows` implementation with the same earlier-row context policy as training and reports per-horizon window support.
- On Windows, `_atomic_save` attempted `fsync` on a read-only file descriptor. It now syncs the completed file through a read/write descriptor before atomically publishing it.
- Metrics now label F1 undefined when the test partition has no positive examples and explicitly expose false-alarm and missed-positive window counts. ROC-AUC, PR-AUC, recall and TSS remain undefined where support is absent.

## Scientific interpretation

The bundled example is synthetic. Its event counts, labels, and model scores are only end-to-end software smoke evidence. Forecast windows overlap and are temporally dependent; these reports do not claim independent-window uncertainty, event-warning skill, or superiority over baselines. Real M/X-class evaluation remains inconclusive until provenance-verified observations and independent GOES event labels provide adequate held-out distinct-event support. The default ratio feature remains disabled because compatible physical units and disjoint energy-band metadata are not documented for the bundled channels.
