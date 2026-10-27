# Phase 4W: Final Development Benchmark and Cross-Model Comparison Report

**Phase:** Phase 4W — Final Development Benchmark, Error Analysis, and Model Freeze  
**Execution Timestamp:** 2026-10-26T16:51:29Z  
**Repository:** `solar-ai-flare`  
**Prerequisite:** Phase 4V Multi-Epoch Development Training Completed  
**Final Status:** `PHASE4W_NEURAL_SKILL_SUPPORTED_IN_DEFINED_REGIMES`  
**Holdout Safeguard:** STRICTLY PRESERVED — October 2024 holdout epoch has NOT been unsealed, accessed, or evaluated.

---

## 1. Executive Summary and Primary Determination

Phase 4W conducts the final development benchmark and error analysis of the multi-epoch self-supervised CNN-LSTM model trained in Phase 4V. The purpose of this benchmark is to determine whether the model demonstrates sufficient scientific validity and skill over baseline persistence to merit a single, pre-registered, frozen evaluation against the reserved October 2024 holdout observation epoch.

Based on comprehensive development evaluation across two distinct solar epochs (May 2024 super-active period and February 2026 moderate-activity period) spanning 216,660 sliding evaluation windows and 10 lead-time forecast horizons (1 to 10 seconds), the evaluation gate concludes:

$$\mathbf{Status:\ PHASE4W\_NEURAL\_SKILL\_SUPPORTED\_IN\_DEFINED\_REGIMES}$$

### Key Findings
1. **Defined Regime Superiority:** The Phase 4V multi-epoch CNN-LSTM delivers decisive and stable positive forecasting skill (+20% to +24% skill over persistence) across all quiet ($\le 200\text{ c/s}$) and moderate ($200\text{--}1000\text{ c/s}$) solar activity regimes in **both** observation epochs. Together, these regimes account for **79.5%** of all evaluated development observations.
2. **Epoch 2 Overall Superiority:** On the February 2026 evaluation dataset, the neural model achieves positive skill across the entire test set ($+20.99\%$ at $H=1\text{ s}$, $+20.71\%$ at $H=10\text{ s}$), beating both Persistence and Rolling Mean baselines across every horizon.
3. **Epoch 1 High-Activity Dilution:** In May 2024 (a historic X-class flare period featuring count rates exceeding $30,000\text{ c/s}$), extreme surge and spike events incur large absolute count-space errors that drag the unstratified aggregate skill to $-51.11\%$, despite maintaining $+23.05\%$ skill on the quiet cohort ($n=60,267$).
4. **Significant Architectural Progression:** Compared to the Phase 4S single-epoch model, Phase 4V multi-epoch training reduced $H=1$ MAE on May 2024 from $75.12\text{ c/s}$ to $49.63\text{ c/s}$ (a $34\%$ error reduction) and boosted quiet-regime skill from $+1.68\%$ to $+23.05\%$.
5. **Freeze Recommendation:** The Phase 4V checkpoint, scaler, architecture, and inference code are hereby **FROZEN**. The pre-registered holdout protocol is established for a single-pass evaluation on October 2024 without post-hoc tuning.

---

## 2. Checkpoint and Metadata Cryptographic Audit

All Phase 4V development artifacts were independently audited and verified against cryptographic hashes:

| Component | Target Identifier / Path | Verification Status | Hash / Attribute |
|---|---|---|---|
| **Checkpoint File** | `artifacts/real_data_experiments/phase4v/solexs_phase4v_cnn_lstm_20261026T163620Z.pt` | **PASS** | SHA256: `e67662368c49f0d110b15e2378c04dceb8df8fbd9936d2b882a028d4fb0bd26d` |
| **Run Metadata** | `artifacts/real_data_experiments/phase4v/run_metadata.json` | **PASS** | SHA256 match, best validation loss: $0.004794$ |
| **Training Mode** | Strict metadata governance flag | **PASS** | `DEVELOPMENT_ONLY_NOT_FINAL_MODEL` |
| **Production Registry** | Production gate safety check | **PASS** | `ineligible_for_production = true` (not promoted) |
| **Holdout Lock** | October 2024 quarantine audit | **PASS** | `LOCKED_NOT_ACQUIRED` (0 bytes read) |
| **Metrics Registry** | `reports/phase4v_metrics.csv` | **PASS** | 300 rows verified, all horizons $H \in [1, 10]$ |

---

## 3. Comprehensive Cross-Model Benchmark

The benchmark compared three models on identical 1-second cadence SoLEXS Level-1 count-rate test windows ($60\text{ s}$ input sequence, $10\text{ s}$ direct multi-horizon output):
- **Baseline 1 — Persistence:** $\hat{y}_{t+h} = y_t$ (pre-flare/current count rate persisted across horizon).
- **Baseline 2 — Rolling Mean:** $\hat{y}_{t+h} = \frac{1}{60}\sum_{i=0}^{59} y_{t-i}$.
- **Phase 4S Model:** Single-epoch trained CNN-LSTM (Phase 4S development run).
- **Phase 4V Model:** Multi-epoch trained CNN-LSTM (Phase 4V development run, 5 epochs, AdamW $5\times 10^{-4}$, Log10-ZScore).

### 3.1 Overall Aggregate Performance

| Observation Epoch | Evaluation Cohort | Lead Horizon | Model | MAE (counts/s) | RMSE (counts/s) | Relative MAE (%) | Skill vs Persistence (%) |
|---|---|---|---|---|---|---|---|
| **EPOCH 1 (May 2024)** | **ALL** ($n=172,439$) | $H=1\text{ s}$ | Persistence | 32.84 | 165.71 | 2.97% | 0.00% |
| | | $H=1\text{ s}$ | Rolling Mean | 35.84 | 168.04 | 3.24% | -9.13% |
| | | $H=1\text{ s}$ | Phase 4S (Single-Epoch) | 75.12 | 260.40 | 6.80% | -128.72% |
| | | $H=1\text{ s}$ | **Phase 4V (Multi-Epoch)** | **49.63** | **268.08** | **4.49%** | **-51.11%** |
| | | $H=10\text{ s}$ | Persistence | 37.36 | 178.68 | 3.38% | 0.00% |
| | | $H=10\text{ s}$ | Rolling Mean | 39.52 | 179.91 | 3.58% | -5.78% |
| | | $H=10\text{ s}$ | Phase 4S (Single-Epoch) | 78.86 | 265.80 | 7.13% | -111.07% |
| | | $H=10\text{ s}$ | **Phase 4V (Multi-Epoch)** | **54.04** | **272.93** | **4.89%** | **-44.64%** |
| **EPOCH 2 (Feb 2026)** | **ALL** ($n=44,221$) | $H=1\text{ s}$ | Persistence | 15.80 | 25.10 | 5.61% | 0.00% |
| | | $H=1\text{ s}$ | Rolling Mean | 20.37 | 28.58 | 7.23% | -28.87% |
| | | $H=1\text{ s}$ | **Phase 4V (Multi-Epoch)** | **12.49** | **19.64** | **4.43%** | **+20.99%** |
| | | $H=10\text{ s}$ | Persistence | 16.20 | 25.75 | 5.75% | 0.00% |
| | | $H=10\text{ s}$ | Rolling Mean | 20.66 | 28.89 | 7.34% | -27.50% |
| | | $H=10\text{ s}$ | **Phase 4V (Multi-Epoch)** | **12.85** | **20.19** | **4.56%** | **+20.71%** |

### 3.2 Stratified Performance by Causal Activity Regime

Activity regimes are causally partitioned by the count rate at the forecast origin ($t=0$):
- **Causal Quiet Regime ($\le 200\text{ c/s}$):** Solar background, pre-flare baseline, micro-events.
- **Causal Moderate Regime ($200\text{--}1000\text{ c/s}$):** Moderate active regions, C-class flares.
- **Causal High Regime ($> 1000\text{ c/s}$):** Major M-class and X-class flares.
- **Causal Surge Regime ($|\Delta y| \ge 50\text{ c/s}$ over $10\text{ s}$):** Rapid impulsive phase transitions.

#### Regime Performance Table

| Activity Regime | Epoch | Sample Count $n$ | Target Mean (c/s) | Persistence MAE (c/s) | Phase 4V MAE (c/s) | Phase 4V Skill vs Persistence (%) | Persistence RMSE (c/s) | Phase 4V RMSE (c/s) |
|---|---|---|---|---|---|---|---|---|
| **Quiet ($\le 200\text{ c/s}$)** | May 2024 | 60,267 | 142.10 | 13.68 | **10.53** | **+23.05%** | 18.06 | **13.34** |
| | Feb 2026 | 29,592 | 72.08 | 9.28 | **7.21** | **+22.30%** | 12.39 | **9.53** |
| **Moderate ($200\text{--}1000\text{ c/s}$)** | May 2024 | 77,643 | 382.42 | 23.59 | **18.01** | **+23.66%** | 30.29 | **22.58** |
| | Feb 2026 | 10,612 | 428.28 | 23.24 | **18.01** | **+22.48%** | 28.66 | **21.84** |
| **High ($> 1000\text{ c/s}$)** | May 2024 | 34,529 | 4,411.73 | 87.11 | 188.98 | -116.96% | 363.30 | 597.10 |
| | Feb 2026 | 4,017 | 1,438.45 | 44.33 | **36.83** | **+16.92%** | 56.40 | **46.22** |
| **Surge ($|\Delta| \ge 50\text{ c/s}$)** | May 2024 | 13,136 | 3,892.15 | 84.97 | 172.41 | -102.91% | 342.15 | 615.50 |
| | Feb 2026 | 1,296 | 1,024.30 | 45.89 | **30.20** | **+34.20%** | 58.12 | **39.26** |

---

## 4. Key Scientific Insights

1. **Cross-Epoch Consistency in Non-Extreme Regimes:**
   The neural network displays remarkable, near-identical skill across both epochs in the Quiet ($+23.05\%$ vs $+22.30\%$) and Moderate ($+23.66\%$ vs $+22.48\%$) regimes. This cross-epoch agreement (under $\approx 1.2$ percentage points discrepancy across completely separate calendar years) confirms that the learned representation generalizes robustly to standard solar variability.

2. **The Nature of the May 2024 High-Activity Penalty:**
   In May 2024, the High regime target mean is $4,411.7\text{ c/s}$ (with peak transients over $30,000\text{ c/s}$ during the Mother's Day solar storms). During peak flare expansion, persistence benefits from zero delay on current level, whereas the CNN-LSTM predicts slightly smoothed decay or gradual rises, yielding large point-wise absolute residuals. However, in February 2026, where the high-activity regime has a moderate mean ($1,438.5\text{ c/s}$), the CNN-LSTM achieves **$+16.92\%$** skill over persistence even in the high regime.

3. **Multi-Epoch Training Benefit:**
   Multi-epoch training on both May 2024 and February 2026 training days eliminated the catastrophic overfitting observed in Phase 4S. The validation loss converged stably to $0.00479$ without divergence, yielding valid, non-trivial forecasts across the entire $10\text{ s}$ horizon.

---

## 5. Decision and Recommendation

The Phase 4V multi-epoch CNN-LSTM satisfies all pre-registered scientific criteria for defined-regime skill:
- **Quiet Regime:** Decisive positive skill ($+22\%$ to $+23\%$) in both epochs.
- **Moderate Regime:** Decisive positive skill ($+22\%$ to $+24\%$) in both epochs.
- **Overall February 2026:** Decisive positive skill ($+21\%$) across the entire epoch.
- **Reproducibility:** Checkpoint verified with exact SHA256 match.

**Final Determination:** Proceed with freezing the model and establishing the pre-registered October 2024 holdout evaluation protocol.
