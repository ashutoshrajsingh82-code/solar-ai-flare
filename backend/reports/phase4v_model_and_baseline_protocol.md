# Phase 4V: Model and Baseline Evaluation Protocol

**Project:** Solar AI Flare  
**Phase:** Phase 4V  
**Report Version:** `phase4v_model_baseline_protocol_v1.0`  
**Generated:** 2026-10-26T16:23:00Z

---

## 1. Evaluation Scope

Phase 4V evaluates three models on identical test window sets using the same forecast
origins, target timestamps, valid masks, horizons, and aggregation rules. All evaluation
is performed in **native count-rate units (counts/s)**, not in scaled space.

### Models Evaluated

| Model ID | Description |
|----------|-------------|
| `CNN_LSTM_MultiEpoch` | Direct multi-horizon SelfSupervisedCNNLSTM trained on Epochs 1+2 |
| `Persistence` | Last-observed count rate repeated for all horizons (zero-skill reference) |
| `RollingMean` | Mean of 60-second input window repeated for all horizons |

---

## 2. Forecast Origin and Target Alignment

For every test window `i`:

- **Input:** `X_i = [x_{t-59}, x_{t-58}, ..., x_t]` — 60-second native count rate history
- **Target:** `y_i = [x_{t+1}, x_{t+2}, ..., x_{t+10}]` — 10-second future count rate sequence
- **Forecast origin timestamp:** `t_origin = t` (Unix seconds)

All three models receive identical `X_i` and are evaluated against identical `y_i`.  
No model has access to any information beyond `X_i` at forecast time.

---

## 3. Persistence Baseline

$$\hat{y}_{t+h}^{\text{persist}} = x_t \quad \forall h \in \{1, \ldots, 10\}$$

Implementation: `persistence_forecast(X, horizon)` — uses `X[:, -1, 0]` (last observed sample).

This is the zero-skill reference model. A skill score of 0% vs persistence indicates a model
performs no better than naively repeating the last observation.

---

## 4. Rolling Mean Baseline

$$\hat{y}_{t+h}^{\text{rolling}} = \frac{1}{60} \sum_{i=0}^{59} x_{t-i} \quad \forall h \in \{1, \ldots, 10\}$$

Implementation: `rolling_mean_forecast(X, horizon)` — uses `np.mean(X[:, :, 0], axis=1)`.

Provides a smoothed reference that may perform better than persistence during rapid transients.

---

## 5. CNN-LSTM Inference

1. Input `X_i` (60 × 1 float32) transformed to scaled space using frozen training scaler.
2. Model produces 10-step prediction in scaled space.
3. Predictions inverted to native count rates: `x̂ = max(0, 10^(z·σ + μ) − 1)`.
4. All three models evaluated on the same native-count-rate targets.

Non-finite predictions are detected and flagged; a warning is issued and values clipped to
finite range for diagnostics (not silently ignored).

---

## 6. Evaluation Metrics

For each (epoch, cohort, model, horizon h):

### 6.1 Mean Absolute Error (MAE)

$$\text{MAE}_h = \frac{1}{N} \sum_{i=1}^N |y_{i,h} - \hat{y}_{i,h}|$$

Units: counts/s

### 6.2 Root Mean Squared Error (RMSE)

$$\text{RMSE}_h = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_{i,h} - \hat{y}_{i,h})^2}$$

Units: counts/s

### 6.3 Relative MAE (RelMAE)

$$\text{RelMAE}_h = \frac{\text{MAE}_h}{\bar{y}} \times 100\%$$

Denominator `ȳ`: mean of all target values in the cohort (across all samples and horizons).
This denominator is explicitly documented per cohort to prevent ambiguity.

### 6.4 Skill Score vs Persistence

$$\text{Skill}_{h} = \left(1 - \frac{\text{MAE}_{\text{model},h}}{\text{MAE}_{\text{persist},h}}\right) \times 100\%$$

A positive skill indicates the model outperforms persistence at that horizon.  
A negative skill indicates persistence is superior.

---

## 7. Causal Activity Cohorts

Cohorts are defined strictly on the observable count rate at forecast origin (`x_t = X[:, -1, 0]`).
Future target peaks are **never** used to assign cohorts (Phase 4T audit finding enforced).

| Cohort Name | Conditioning Rule | Physical Interpretation |
|-------------|------------------|------------------------|
| `ALL` | All windows | Global aggregate across all activity levels |
| `Causal_REG_QUIET` | `x_t < 200 c/s` | Quiescent coronal background |
| `Causal_REG_MOD` | `200 ≤ x_t ≤ 1000 c/s` | Moderate flares, active coronal loops |
| `Causal_REG_HIGH` | `x_t > 1000 c/s` | Major M/X-class solar eruptions |
| `Causal_SURGE_50` | `x_t − x_{t−10} > 50 c/s` | Rapid impulsive rise in recent 10 s |

**Minimum cohort sample count:** 10 windows required to report cohort metrics.  
Cohorts with fewer than 10 windows are excluded from reporting.

---

## 8. Per-Epoch Reporting Policy

Because Epoch 1 (May 2024) and Epoch 2 (Jan-Feb 2026) represent different solar activity
regimes and different NaN rates:

- **Metrics are reported separately per epoch** before any aggregate.
- The larger epoch (Epoch 1 with 2 test days) must not silently dominate the combined
  conclusion.
- Readers are explicitly cautioned that Feb 2026 test data has substantially fewer valid
  windows due to orbital eclipse patterns (~52% duty cycle on most days).

---

## 9. Statistical Considerations

Due to 69-second overlap between consecutive 70-second sliding windows (overlap ratio = 69/70 = 98.6%),
consecutive windows are highly correlated and cannot be treated as i.i.d. samples.

Full uncertainty quantification via time-aware block bootstrap (block length ≥ 300 s as
required by the Phase 4U contract) is recommended for final holdout evaluation. For this
development evaluation, point estimates are reported with sample counts documented for
each cohort to allow readers to assess statistical power.

---

## 10. Evaluation Exclusions

- **October 2024 holdout:** Excluded from all evaluation in this phase.
- **Validation partition:** Reported as model selection only; metrics not used for scientific claims.
- **NaN-containing bins:** Excluded from all windows at the segmentation stage.
- **Sub-minimum-length segments:** Excluded before windowing.
