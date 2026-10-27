# Phase 4V: Findings and Next Steps

**Project:** Solar AI Flare  
**Phase:** Phase 4V — Multi-Epoch Real-Data Forecasting Pipeline and Gated Training  
**Status:** `PHASE4V_DEVELOPMENT_TRAINING_COMPLETED_HOLDOUT_PENDING`  
**Report Date:** 2026-10-26T16:36 UTC

---

## 1. Phase 4V Deliverables — Status

| Deliverable | Status |
|-------------|--------|
| Phase 4V readiness gate script | ✅ PASS (6/7 gates, GATE-4V-07 non-blocking) |
| Multi-epoch pipeline (`phase4v_pipeline.py`) | ✅ Completed |
| Dataset build report | ✅ Finalized |
| Split and leakage audit | ✅ Clean — no temporal leakage |
| Model and baseline protocol | ✅ Documented |
| Training readiness gate | ✅ Authorized |
| Development training run | ✅ Completed (596 s, 5 epochs) |
| Training results report | ✅ Finalized with actual metrics |
| Metrics CSV (per-cohort, per-horizon) | ✅ 302 rows, 2 epochs × 5 cohorts × 3 models × 10 horizons |
| Checkpoint + metadata | ✅ Saved with SHA-256 provenance |
| Test suite (`test_phase4v_pipeline.py`) | ✅ 52 tests passing |

---

## 2. Key Findings

### 2.1 Multi-Epoch Training Works for Low-Activity Regimes

The CNN-LSTM model trained on combined May 2024 + February 2026 data shows **genuine forecasting skill** in quiet/moderate-activity windows:

- **Feb 2026 ALL cohort:** +21% skill vs persistence at H=1, +20.7% at H=10
- **May 2024 Causal_REG_QUIET cohort:** +23% skill vs persistence at H=1, +24.3% at H=10
- **Feb 2026 Causal_REG_QUIET cohort:** +22.3% skill at H=1, +22.0% at H=10

### 2.2 High-Activity Surge Regime Remains Unresolved

- **May 2024 ALL cohort:** −51% vs persistence at H=1 (model worse than simple persistence)
- Root cause: The ~13K surge windows (mean ~3,885 c/s) dominate ALL-cohort MAE. The 60-second sliding-window representation cannot capture the dynamics of extreme flare-driven variability.
- This is consistent with Phase 4T audit findings about persistence dominance at high autocorrelation.

### 2.3 Cross-Epoch Generalization

The model trained on combined epochs generalizes across both activity regimes for the quiet cohort. The +23% quiet-cohort skill on May 2024 (which has much higher mean activity) suggests the model captures transferable low-variability dynamics.

### 2.4 Scaler Integrity

The `log10(x+1)` z-score scaler was fitted **strictly on the training partition** (mean=2.598, std=0.582). Validation and test data used the frozen scaler, confirming no information leakage through normalization.

---

## 3. Pipeline Architecture Summary

```
[Archive Verification] → [FITS Ingestion] → [Temporal Splitting (date-based)]
    → [Scaler Fitting (train only)] → [Sliding Window Extraction]
    → [Model Training (CNN-LSTM)] → [Per-Epoch Test Evaluation]
    → [Causal Cohort Metrics] → [Checkpoint + Metadata Save]
```

All steps are deterministic (seed=42) and reproducible from the archived FITS files.

---

## 4. Outstanding Scientific Blockers

| Blocker | Severity | Resolution Path |
|---------|----------|-----------------|
| October 2024 holdout not acquired | **Critical** | Requires ISRO PRADAN operator login |
| Surge-regime skill negative | Moderate | Needs architectural investigation (longer context, attention, multi-scale) |
| Only 2 observational epochs | Moderate | Acquire additional PRADAN archives |
| No GOES cross-calibration | Low | Not required for self-supervised count-rate forecasting |

---

## 5. Recommended Next Steps

### 5.1 Immediate — Holdout Acquisition (Phase 4W Prerequisite)

1. **Acquire October 2024 SoLEXS archive** from ISRO PRADAN via operator-assisted login
2. Verify archive integrity (SHA-256, FITS column validation)
3. Ingest as holdout epoch — model MUST NOT be retrained before holdout evaluation

### 5.2 Near-Term — Holdout Evaluation (Phase 4W)

1. Evaluate the frozen Phase 4V development checkpoint on October 2024 holdout
2. Report skill vs persistence per cohort
3. This is the first opportunity for a genuinely untouched-holdout skill claim

### 5.3 Medium-Term — Architectural Exploration

1. Investigate longer sequence contexts (120 s, 300 s) for surge-regime coverage
2. Evaluate attention-based architectures for variable-length context
3. Consider multi-scale input representations (1 s + 10 s + 60 s cadence)

### 5.4 Long-Term — Multi-Epoch Expansion

1. Acquire additional PRADAN observation periods covering different solar-cycle phases
2. Build a 5+ epoch dataset for robust cross-epoch validation
3. Establish formal train/val/test epoch governance

---

## 6. Decision

**`PHASE4V_DEVELOPMENT_TRAINING_COMPLETED_HOLDOUT_PENDING`**

Phase 4V has successfully delivered a reproducible multi-epoch self-supervised forecasting pipeline with genuine skill in quiet/moderate-activity regimes. The model is explicitly marked as development-only. Final scientific claims require holdout evaluation on an untouched observational epoch.
