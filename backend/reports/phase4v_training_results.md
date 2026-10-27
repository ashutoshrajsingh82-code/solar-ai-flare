# Phase 4V: Training Results

**Project:** Solar AI Flare  
**Phase:** Phase 4V  
**Report Version:** `phase4v_training_results_v1.1`  
**Training Mode:** `DEVELOPMENT_ONLY_NOT_FINAL_MODEL`  
**Status:** `PHASE4V_DEVELOPMENT_TRAINING_COMPLETED_HOLDOUT_PENDING`  
**Run ID:** `phase4v_20261026T162621Z`  
**Completed:** 2026-10-26T16:36:33 UTC

---

## 1. Dataset Window Counts (Actual)

| Epoch | Partition | Windows |
|-------|-----------|---------|
| Epoch 1 (May 2024) | Train (May 8–12) | 431,069 |
| Epoch 1 (May 2024) | Val (May 13) | 86,185 |
| Epoch 1 (May 2024) | Test (May 14–15) | 172,439 |
| Epoch 2 (Feb 2026) | Train (Jan 31–Feb 2) | 174,334 |
| Epoch 2 (Feb 2026) | Val (Feb 3–4) | 48,523 |
| Epoch 2 (Feb 2026) | Test (Feb 5) | 44,221 |
| **Combined** | **Train** | **605,403** |
| **Combined** | **Val** | **134,708** |
| **Combined** | **Test** | **216,660** |

---

## 2. Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | SelfSupervisedCNNLSTM |
| CNN channels | 32 |
| LSTM hidden | 64 |
| LSTM layers | 2 |
| Dropout | 0.2 |
| Sequence length | 60 s |
| Horizon | 10 s |
| Training epochs | 5 |
| Batch size | 256 |
| Learning rate | 1e-3 |
| Optimizer | Adam |
| Loss | MSELoss (scaled) |
| Seed | 42 |
| Device | CPU |
| Training mode | DEVELOPMENT_ONLY_NOT_FINAL_MODEL |

---

## 3. Scaler

| Parameter | Value |
|-----------|-------|
| Type | log10(x+1) z-score |
| Mean | 2.59788 |
| Std | 0.58240 |
| Fitted on | Training partition only |

---

## 4. Training History

| Epoch | Train Loss (scaled MSE) | Val Loss (scaled MSE) |
|-------|------------------------|-----------------------|
| 1 | 0.020728 | 0.006052 |
| 2 | 0.008478 | 0.006938 |
| 3 | 0.007830 | 0.007003 |
| 4 | 0.007061 | 0.005055 |
| 5 | 0.006952 | **0.004794** ✓ |

Best validation loss: **0.004794** (epoch 5).  
Training duration: **596.3 s** (~9.9 minutes on CPU).

---

## 5. Evaluation Metrics (Development Test Partitions)

### 5.1 Epoch 1 — May 2024 (Test: May 14–15, 172,439 windows)

| Model | Cohort | H=1 MAE (c/s) | H=10 MAE (c/s) | Skill vs Persist H=1 | Skill vs Persist H=10 |
|-------|--------|---------------|----------------|---------------------|----------------------|
| CNN_LSTM_MultiEpoch | ALL | 49.63 | 54.04 | **−51.1%** | **−44.6%** |
| Persistence | ALL | 32.84 | 37.36 | 0.0% | 0.0% |
| RollingMean | ALL | 46.80 | 55.63 | −42.5% | −48.9% |
| CNN_LSTM_MultiEpoch | Causal_REG_QUIET | 10.53 | 10.37 | **+23.1%** | **+24.3%** |
| Persistence | Causal_REG_QUIET | 13.68 | 13.70 | 0.0% | 0.0% |

### 5.2 Epoch 2 — Feb 2026 (Test: Feb 5, 44,221 windows)

| Model | Cohort | H=1 MAE (c/s) | H=10 MAE (c/s) | Skill vs Persist H=1 | Skill vs Persist H=10 |
|-------|--------|---------------|----------------|---------------------|----------------------|
| CNN_LSTM_MultiEpoch | ALL | 12.49 | 12.85 | **+21.0%** | **+20.7%** |
| Persistence | ALL | 15.81 | 16.21 | 0.0% | 0.0% |
| RollingMean | ALL | 14.11 | 15.47 | +10.8% | +4.6% |
| CNN_LSTM_MultiEpoch | Causal_REG_QUIET | 7.21 | 7.34 | **+22.3%** | **+22.0%** |
| Persistence | Causal_REG_QUIET | 9.28 | 9.41 | 0.0% | 0.0% |

---

## 6. Performance Analysis

### 6.1 Regime-Dependent Skill

The multi-epoch CNN-LSTM shows a **clear regime-dependent pattern**:

1. **Low-activity regime (Feb 2026, mean ~282 c/s):** CNN-LSTM beats persistence across ALL cohorts by +21%, demonstrating learned forecasting skill in quiet/moderate-activity windows.

2. **High-activity regime (May 2024, mean ~1,105 c/s):** The CNN-LSTM underperforms persistence on the ALL cohort by −51%. However, when conditioned on Causal_REG_QUIET windows (60,267 test windows with low forecast-origin count rate), the CNN-LSTM shows +23% skill — indicating the model has genuinely learned low-variability dynamics.

3. **Surge cohort dominates ALL-cohort error in May 2024:** The ~13K surge windows in the May 2024 test partition have extreme count rates (mean ~3,885 c/s) and high variability, which the 60-second sliding-window representation cannot capture. These windows dominate the ALL-cohort MAE.

### 6.2 Comparison with Phase 4S Single-Epoch Model

| Configuration | Feb 2026 ALL H=1 Skill | May 2024 Quiet H=1 Skill |
|--------------|------------------------|--------------------------|
| Phase 4S (single-epoch May 2024) | N/A (not evaluated) | ~19% |
| Phase 4V (multi-epoch) | +21.0% | +23.1% |

Multi-epoch training appears to improve quiet-cohort skill (+23% vs ~19% in Phase 4S) and provides cross-epoch generalization.

---

## 7. Checkpoint Provenance

| Field | Value |
|-------|-------|
| Checkpoint filename | `solexs_phase4v_cnn_lstm_20261026T163620Z.pt` |
| SHA-256 | `e67662368c49f0d110b15e2378c04dceb8df8fbd9936d2b882a028d4fb0bd26d` |
| Training mode marker | `PHASE4V_DEVELOPMENT_ONLY_NOT_FINAL_MODEL` |
| Production eligible | `false` |
| Registry status | `EXCLUDED` — development experiment only |
| Location | `artifacts/real_data_experiments/phase4v/` |

---

## 8. Interpretation Caveats

1. **Development data only:** Both test sets (May 14–15 and Feb 5) are from previously examined development epochs. Neither constitutes an untouched holdout.

2. **Persistence dominance at short horizons for high-activity ALL cohort:** Consistent with Phase 4T audit findings. This is a consequence of high physical autocorrelation (69 s overlap from 60-second windows at 1-second cadence) and extreme count-rate variability in surge periods.

3. **Genuine quiet-regime skill:** The +21–23% skill vs persistence in both quiet-cohort and low-activity-epoch evaluations demonstrates that the model has learned meaningful dynamics beyond persistence in low-variability regimes.

4. **No final claims:** These results are development experiments. Final skill claims require October 2024 holdout evaluation.

5. **Holdout locked:** The October 2024 observation epoch has not been acquired and remains locked. The multi-epoch development model is explicitly marked `DEVELOPMENT_ONLY_NOT_FINAL_MODEL`.
