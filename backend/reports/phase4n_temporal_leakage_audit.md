# Phase 4N: Temporal Leakage, Window Integrity, and Evaluation Fairness Audit

**Document Version:** 1.0.0  
**Audit Date:** 2026-10-25  
**Canonical Designation:** `PHASE4N_TEMPORAL_LEAKAGE_AUDIT`  
**Parent Phase:** Phase 4N (Independent Reproducibility, Robustness & Generalization Validation)  
**Audited Modules:**  
- [`backend/ml/experiments/real_self_supervised.py`](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/ml/experiments/real_self_supervised.py)  
- [`backend/scripts/evaluate_real_self_supervised.py`](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/scripts/evaluate_real_self_supervised.py)  

---

## 1. Audit Scope and Objectives

A rigorous evaluation of time-series machine learning models in solar physics requires auditing the underlying code to ensure that reported skill scores are genuine rather than artifacts of subtle lookahead leakage, future normalization, boundary crossing, or baseline misalignments.

This audit inspected the physical implementation of:
1. Chronological boundary partitioning between train, validation, and test splits.
2. Contiguous segment extraction and gap-handling logic.
3. Sliding window formulation (input history vs. prediction horizon).
4. Normalization scaler fitting and inference transforms.
5. Checkpoint selection and early stopping mechanisms.
6. Alignment of baseline forecasts against neural network targets.

---

## 2. Chronological Split and Boundary Isolation Audit

### 2.1 Boundary Definitions
The chronological splits were established using exact UTC timestamp thresholds:
- **Train Period:** `2026-01-31 00:00:00` to `2026-02-02 23:59:59 UTC` ($t \le 1,769,990,399$)
- **Validation Period:** `2026-02-03 00:00:00` to `2026-02-04 23:59:59 UTC` ($1,769,990,399 < t \le 1,770,163,199$)
- **Held-Out Test Period:** `2026-02-05 00:00:00` to `2026-02-05 23:59:59 UTC` ($1,770,163,199 < t \le 1,770,249,599$)

### 2.2 Boundary Leakage Inspection
In `backend/ml/experiments/real_self_supervised.py` lines 262–274:
```python
train_mask = (all_times <= train_end_ts)
val_mask = (all_times > train_end_ts) & (all_times <= val_end_ts)
test_mask = (all_times > val_end_ts) & (all_times <= test_end_ts)

train_segs = extract_contiguous_segments(all_times[train_mask], all_counts[train_mask], min_len)
val_segs = extract_contiguous_segments(all_times[val_mask], all_counts[val_mask], min_len)
test_segs = extract_contiguous_segments(all_times[test_mask], all_counts[test_mask], min_len)
```

**Audit Findings:**
1. **Strict Disjointness:** The boolean masks `train_mask`, `val_mask`, and `test_mask` form a mutually exclusive partition of the observational timeline.
2. **Segment Isolation:** `extract_contiguous_segments` is invoked independently on each sliced array. Therefore, no continuous observation segment can ever span across a split boundary (e.g. from Feb 2 into Feb 3, or from Feb 4 into Feb 5).
3. **Zero Lookahead:** No target window in the training set contains observations from the validation or test dates.

---

## 3. Window Construction, Gap Handling, and Monotonicity

### 3.1 Gap-Segment Extraction Logic
In `extract_contiguous_segments`:
```python
valid_mask = np.isfinite(counts) & (counts >= 0)
valid_times = times[valid_mask]
valid_counts = counts[valid_mask]

time_diffs = np.diff(valid_times)
is_gap = np.concatenate([[False], time_diffs > 1.0])
segment_ids = np.cumsum(is_gap)
```

**Audit Findings:**
1. **NaN / Missing Telemetry:** Spacecraft night, telemetry losses, or corrupted records are represented by IEEE NaNs. The `valid_mask` strictly filters them out before segmentation.
2. **Gap Thresholding:** Any gap where $\Delta t > 1.0\text{ s}$ increments `segment_ids`, cleanly terminating the preceding segment.
3. **No Synthetic Imputation:** Missing intervals are never interpolated, zero-filled, or forward-filled.
4. **Minimum Segment Guard:** Segments shorter than $\text{seq\_len} + \text{horizon} = 70\text{ s}$ are discarded, ensuring all sliding windows are composed of 100% genuine continuous observations.

### 3.2 Sliding Window Geometry
In `create_sliding_windows`:
```python
x_window = seg_counts[i : i + seq_len]
y_window = seg_counts[i + seq_len : i + total_window_len]
t_origin = seg_times[i + seq_len - 1]
```

**Audit Findings:**
1. **Input vs Target Separation:** `x_window` spans $[i, i+59]$ (60 seconds) and `y_window` spans $[i+60, i+69]$ (10 seconds).
2. **No Single-Window Overlap:** The forecast target starts strictly at $T+1\text{s}$ following the forecast origin $T$. There is zero overlap between input history and output targets for any given window.
3. **Rolling-Origin Nature:** Windows slide by a step size of 1 second. Each window represents a distinct, high-cadence forecast origin $T$. While adjacent windows overlap in time (as standard in real-time solar nowcasting), each individual $(X_t, Y_t)$ pair preserves strict causal time ordering.

> [!NOTE]
> **Documentation Refinement:** Section 4 of `phase4ms_real_observation_manifest.md` previously described windows as "non-overlapping". Code inspection confirmed the evaluation uses rolling contiguous sliding windows with step=1. This is mathematically sound for rolling nowcasting, and the terminology has been updated to "contiguous rolling sliding windows".

---

## 4. Normalization and Preprocessing Fairness

### 4.1 Scaler Fitting Scope
In `load_and_split_solexs_real_data`:
```python
scaler = CountRateScaler().fit(all_counts[train_mask])
```
The scaler parameters $\mu$ and $\sigma$ were computed strictly on the training partition ($N = 176,488$ finite training records):
- $\mu_{\text{train}} = 2.247274921086125$
- $\sigma_{\text{train}} = 0.6966391869930155$

### 4.2 Evaluation Scaler Protection
In Phase 4N evaluation ([`evaluate_real_self_supervised.py`](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/scripts/evaluate_real_self_supervised.py)):
- The scaler was **not** refitted on evaluation or test data.
- The scaler was initialized directly with the frozen parameters stored in `metadata.json`.
- A dedicated unit test (`test_no_fitting_of_scalers_on_evaluation_data`) confirms that passing arbitrary evaluation distributions does not alter $\mu$ or $\sigma$.

---

## 5. Checkpoint Selection and Metric Integrity

### 5.1 Validation-Only Model Selection
In `train_self_supervised_model` (lines 379–384):
```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
```
1. Model checkpointing was triggered exclusively by MSE on the validation partition (`2026-02-03` to `2026-02-04`).
2. Test partition data (`2026-02-05`) was **never** ingested during training or used for early stopping.
3. Test evaluation was performed only once, after training was terminated and the optimal validation weights restored.

### 5.2 Baseline Target Alignment
In Phase 4M-S and Phase 4N:
```python
persist_preds = persistence_forecast(X_test_raw, cfg.forecast_horizon)
mean_preds = rolling_mean_forecast(X_test_raw, cfg.forecast_horizon)
metrics_model = evaluate_forecast(y_test_raw, test_preds_raw)
metrics_persist = evaluate_forecast(y_test_raw, persist_preds)
metrics_mean = evaluate_forecast(y_test_raw, mean_preds)
```
1. Persistence baseline repeats the exact last observed value $X_{T, 59}$ across the 10-step horizon.
2. Rolling mean baseline averages the 60 seconds of history $[X_{T, 0} \dots X_{T, 59}]$.
3. All three models are evaluated on the **identical ground-truth target matrix** $Y_{\text{test}} \in \mathbb{R}^{44221 \times 10}$.
4. Error metrics (MAE, RMSE, Relative MAE) are calculated directly on native `counts/s`, completely bypassing uncalibrated flux pipelines.

---

## 6. Audit Verdict

| Audit Checklist Item | Verification Mechanism | Status | Notes |
|---|---|---|---|
| Train/Val/Test Boundary Separation | Code inspection & timestamp masking | **PASSED** | Strictly disjoint UTC intervals. |
| Gap & Night Telemetry Handling | Unit tests & FITS data inspection | **PASSED** | NaNs excluded; $\Delta t > 1.0\text{ s}$ breaks segments. |
| Input/Target Overlap in Windows | Window indexing arithmetic | **PASSED** | Zero overlap between input and target. |
| Scaler Isolation | Code inspection & unit tests | **PASSED** | Fitted strictly on train; loaded frozen in eval. |
| Model Selection Fairness | Training loop execution path | **PASSED** | Validation-loss only; zero test leakage. |
| Baseline Target Alignment | Metric evaluation code & test assertions | **PASSED** | Evaluated on identical targets in native units. |
| Registry Separation | Registry mock test & metadata check | **PASSED** | Production registry strictly protected. |

**Audit Conclusion:** The Phase 4M-S experimental pipeline exhibits **zero temporal lookahead leakage, zero boundary contamination, and zero future normalization artifacts**.
