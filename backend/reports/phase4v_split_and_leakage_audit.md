# Phase 4V: Split and Leakage Audit

**Project:** Solar AI Flare  
**Phase:** Phase 4V  
**Report Version:** `phase4v_split_leakage_audit_v1.0`  
**Generated:** 2026-10-26T16:21:00Z  
**Audit Outcome:** `NO_TEMPORAL_LEAKAGE_DETECTED`

---

## 1. Partition Assignment Summary

All 14 daily observation products are assigned to exactly one chronological partition. No
product appears in more than one partition. Training always precedes validation which
always precedes test within each epoch.

### Epoch 1 — May 2024 (AR 13664 Solar Maximum)

| Observation Date | Partition | Justification |
|-----------------|-----------|---------------|
| 2024-05-08 | **Training** | Earliest available; high-activity solar maximum conditions |
| 2024-05-09 | **Training** | Continued X-class flare activity |
| 2024-05-10 | **Training** | X-class activity including X3.9 event |
| 2024-05-11 | **Training** | X-class activity continuing |
| 2024-05-12 | **Training** | Moderate-to-high activity; training boundary |
| 2024-05-13 | **Validation** | Post-burst decay; model selection checkpoint |
| 2024-05-14 | **Test** | X8.7 event — highest recorded in Cycle 25 at time |
| 2024-05-15 | **Test** | Active decay following X8.7 |

Epoch 1 split: 5 training days / 1 validation day / 2 test days

### Epoch 2 — January–February 2026 (Quiet Sun to Moderate Flaring)

| Observation Date | Partition | Justification |
|-----------------|-----------|---------------|
| 2026-01-31 | **Training** | Quiet-sun baseline; low activity |
| 2026-02-01 | **Training** | Quiet-to-moderate transition |
| 2026-02-02 | **Training** | Moderate flaring including M-class events |
| 2026-02-03 | **Validation** | Low duty cycle (5.9%); provides limited but real validation data |
| 2026-02-04 | **Validation** | Moderate activity; validation boundary |
| 2026-02-05 | **Test** | Final available Feb 2026 day; examined test partition |

Epoch 2 split: 3 training days / 2 validation days / 1 test day

---

## 2. Temporal Isolation Checks

### 2.1 Chronological Ordering

- ✅ All Epoch 1 training dates (May 8–12) precede Epoch 1 validation date (May 13)
- ✅ Epoch 1 validation date (May 13) precedes Epoch 1 test dates (May 14–15)
- ✅ All Epoch 2 training dates (Jan 31–Feb 2) precede Epoch 2 validation dates (Feb 3–4)
- ✅ Epoch 2 validation dates (Feb 3–4) precede Epoch 2 test date (Feb 5)

### 2.2 Partition Non-Overlap

- ✅ `{Train} ∩ {Val} = ∅` for both epochs
- ✅ `{Val} ∩ {Test} = ∅` for both epochs
- ✅ `{Train} ∩ {Test} = ∅` for both epochs
- ✅ `{Epoch1 Dates} ∩ {Epoch2 Dates} = ∅` (May 2024 and Jan-Feb 2026 non-overlapping)

### 2.3 Window Boundary Enforcement

- ✅ Sliding windows are generated strictly within individual contiguous segments
- ✅ No window crosses a daily midnight UTC boundary
- ✅ No window crosses a NaN telemetry gap
- ✅ No window crosses an inter-partition boundary (train/val, val/test)
- ✅ Each epoch's partitions are processed independently; no cross-epoch window is generated

---

## 3. Scaler Isolation Audit

### Scaler Fitting Policy

The `CountRateScaler` applies `log₁₀(x + 1)` z-score standardization. Its parameters
(`μ_train`, `σ_train`) are fitted **strictly and exclusively** on finite records from the
combined training partition of both epochs.

**Training partition records used to fit scaler:**

| Epoch | Dates | Approximate Finite Records |
|-------|-------|---------------------------|
| Epoch 1 | May 8–12 | ~431,983 |
| Epoch 2 | Jan 31 – Feb 2 | ~176,488 |
| **Combined training** | — | **~608,471** |

**Records explicitly excluded from scaler fitting:**

| Category | Records | Reason |
|----------|---------|--------|
| Epoch 1 validation (May 13) | ~86,396 | Never touches `scaler.fit()` |
| Epoch 1 test (May 14–15) | ~172,793 | Quarantined from scaler |
| Epoch 2 validation (Feb 3–4) | ~50,426 | Never touches `scaler.fit()` |
| Epoch 2 test (Feb 5) | ~45,309 | Quarantined from scaler |
| October 2024 holdout | N/A | Not acquired; no records exist locally |

### Scaler Application

- Training data: fitted + transformed (correct)
- Validation data: transformed only using frozen training scaler (correct)
- Test data: transformed only using frozen training scaler (correct)
- Holdout data: **excluded from all operations** (correct)

**Scaler parameters recorded in `run_metadata.json`** for audit reproduction.

---

## 4. Holdout Quarantine Verification

### October 2024 Holdout (Epoch 3)

- **Status:** `ACQUISITION_PENDING_OPERATOR_PRADAN_LOGIN`
- **Local availability:** False — no October 2024 archive exists on disk
- **Pipeline access:** None — zero code paths in `phase4v_pipeline.py` reference October 2024 data
- **Gate condition:** GATE-4V-07 fails (non-blocking for development), but correctly recorded
- **Model decisions:** Architecture, hyperparameters, and training split all fixed before holdout access

**Verification:** A `grep` of `phase4v_pipeline.py` for `"202410"` returns zero matches,
confirming October 2024 dates are not referenced anywhere in the active pipeline.

---

## 5. Look-Ahead Bias Audit

### Causal Cohort Conditioning (Phase 4U Contract §8)

All activity cohort masks are computed using `x_t` (the final observed count rate at the
forecast origin), **never** using future target peaks `max(y_{t+1:t+10})`.

| Cohort | Conditioning Variable | Permitted? |
|--------|----------------------|-----------|
| `Causal_REG_QUIET` | `x_t < 200 c/s` | ✅ Causal |
| `Causal_REG_MOD` | `200 ≤ x_t ≤ 1000 c/s` | ✅ Causal |
| `Causal_REG_HIGH` | `x_t > 1000 c/s` | ✅ Causal |
| `Causal_SURGE_50` | `x_t - x_{t-10} > 50 c/s` | ✅ Causal |
| ~~`Regime_REG_HIGH_gt1000`~~ | ~~`max(y) > 1000`~~ | ❌ Prohibited (Phase 4S/4T pattern rejected) |

The Phase 4T audit identified look-ahead bias in the Phase 4S cohort definition that used
`peak_target = max(y_test, axis=1)`. Phase 4V strictly rejects this pattern.

---

## 6. Cross-Epoch Generalization Caveat

May 2024 and January–February 2026 data are separated by ~21 months. Although both constitute
"development data" under the Phase 4U governance policy, combining them for a single model
evaluation requires acknowledging distributional differences:

- **Activity level:** May 2024 median ~219–903 c/s (solar maximum); Feb 2026 median 62–1063 c/s
- **NaN rate:** May 2024 ~0.002% missing; Feb 2026 ~47–94% missing on some days (eclipse pattern)
- **Instrument evolution:** Detector bias and thermal state may have drifted across 21 months

Per-epoch metrics are reported separately to prevent the larger or more active epoch from
silently dominating aggregate conclusions.
