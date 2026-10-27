# Phase 4 multi-instrument fusion experiment

## Status

**Blocked; no scientific comparison was run.** The scientific conclusion is that fusion benefit cannot yet be determined. No model configurations were trained for this comparison, and no skill scores, confidence intervals, or winner are reported. Completed seed runs: 0 of 5.

## Planned hypothesis and evaluation

The hypothesis is that calibrated HEL1OS hard X-ray measurements add predictive information beyond SoLEXS soft X-ray measurements for a coverage-aware flare-onset target. The intended comparisons are SoLEXS-only, HEL1OS-only, fused, persistence, and climatology. All configurations must use identical eligible paired examples, forecast origins, chronological train/validation/test boundaries, target labels, and five paired random seeds. Thresholds must be selected using validation data only. Event windows and unknown reference coverage must follow the label contract; unknown labels are excluded, never converted to quiet examples.

The eventual report must give the real dataset and product provenance, aligned cohort counts, label counts, split date boundaries, per-seed and aggregate TSS/HSS/POD/POFD/FAR/PR-AUC/Brier scores, paired uncertainty intervals, and explicit baseline results. This report contains none of those results because the required cohort is unavailable.

## Readiness findings

- The current SoLEXS reader exposes native counts per second. These are not calibrated flux values comparable to the dual-flux model input.
- The HEL1OS product reader fails closed pending its product manual and representative product files.
- There is no verified paired real-data cohort with independent coverage-aware reference labels available to this experiment.
- The demo dataset is synthetic and cannot support an observational conclusion.
- Consequently, shared split boundaries, paired cohort sizes, label prevalence, metrics, and confidence intervals are not available; persistence and climatology were not run.

## Requirements before execution

1. Verify instrument product documentation, calibration, units, timestamps, quality flags, cadence, and coverage for representative SoLEXS and HEL1OS files.
2. Build a shared paired cohort without future-nearest matching or channel-specific row selection.
3. Verify independent flare labels and coverage for each full forecast window.
4. Freeze a chronological split and five paired seeds; tune thresholds only on validation data.
5. Run all five configurations on the same test origins and publish paired metrics and uncertainty intervals.
