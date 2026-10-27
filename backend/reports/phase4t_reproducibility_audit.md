# Phase 4T: Independent Reproducibility Audit Report

**Project:** Solar AI Flare  
**Backend:** `backend/`  
**Phase:** Phase 4T — Independent Reproducibility Audit and Forecasting-Skill Verification  
**Audit Date:** 2026-10-26  
**Auditor:** Independent Automated Audit Subsystem  
**Evaluated Phase:** Phase 4S (Controlled May 2024 Real-Data Forecasting Evaluation)  
**Audit Verdict:** `REPRODUCED_WITH_LIMITATIONS_DOCUMENTED`

---

## 1. Executive Summary

This report delivers an independent, read-only reproducibility audit of the experimental artifacts, data splits, models, and metric claims reported in **Phase 4S**.

### Key Audit Findings:
1. **Bit-Level Preservation:** The frozen Phase 4S model checkpoint [`solexs_phase4s_cnn_lstm_may2024_20261025T203124Z.pt`](file:///c:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/artifacts/real_data_experiments/phase4s/solexs_phase4s_cnn_lstm_may2024_20261025T203124Z.pt) is verified bit-for-bit intact on disk with SHA-256 `2680d264e7357162c4950bd5dc554349ce25c8e33ddd524737e505c2aba6632c`. Historical Phase 4M-S and Phase 4O checkpoints are also bit-level identical to their original registrations.
2. **Missing Per-Window Prediction Arrays:** Raw per-window prediction arrays (`.npy`/`.npz`) were **not saved** during Phase 4S execution; only the aggregate summary table [`phase4s_metrics.csv`](file:///c:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/reports/phase4s_metrics.csv) was written. Consequently, metric reproduction could not proceed purely from saved prediction files.
3. **Audit-Only Reconstruction Execution:** In strict compliance with operating guidelines, a read-only audit inference pass was executed using the frozen checkpoint and immutable Level-1 FITS data from `solexs_2026Oct25T200126271.zip`. The per-window predictions are now permanently archived in [`artifacts/real_data_experiments/phase4t/phase4t_test_predictions.npz`](file:///c:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/artifacts/real_data_experiments/phase4t/phase4t_test_predictions.npz).
4. **Exact Metric Reconciliation (0.0 Discrepancy):** All reported milestone metrics across all 4 benchmark models, 10 forecast horizons, and 7 stratified cohorts were reproduced **down to the fourth decimal place with zero discrepancy**.
5. **Report Duplicate Consistency:** All Markdown and CSV reports located in `backend/reports/` and mirrored in `backend/artifacts/real_data_experiments/phase4s/` are bit-for-bit identical.

---

## 2. Canonical Artifact Inventory & Cryptographic Hashes

Every artifact associated with Phase 4S was located and hashed independently. The canonical locations and digests are established below:

| Artifact Name | Canonical Location | Measured SHA-256 Digest | Status & Integrity |
| :--- | :--- | :--- | :---: |
| **Source Archive** | `C:\Users\ASHUTOSH\Downloads\solexs_2026Oct25T200126271.zip` | `8df85ed75b6f00052d7f45a6764b187c700977f931504fb79fdfe56f10df9db2` | **VERIFIED BIT MATCH** |
| **Phase 4S Checkpoint** | `backend/artifacts/real_data_experiments/phase4s/solexs_phase4s_cnn_lstm_may2024_20261025T203124Z.pt` | `2680d264e7357162c4950bd5dc554349ce25c8e33ddd524737e505c2aba6632c` | **IMMUTABLE FROZEN** |
| **Phase 4S Metadata** | `backend/artifacts/real_data_experiments/phase4s/solexs_phase4s_cnn_lstm_may2024_20261025T203124Z_metadata.json` | `5c45e5d16e001ee5c3fa9343cae0fb08f57fa9586144e55e51088ec30113c54c` | **VERIFIED MATCH** |
| **Dataset Manifest (JSON)** | `backend/reports/phase4s_dataset_manifest.json` | `b027efeaca8a1f6a1d82136e0d37e41b9d4530089e6022e3661136bfe611e9b2` | **VERIFIED MATCH** |
| **Dataset Manifest (CSV)** | `backend/reports/phase4s_dataset_manifest.csv` | `991a666b4858e391bbff5a84fb712f205a2d67451cb541460395669b3e157797` | **VERIFIED MATCH** |
| **Data Integrity Audit** | `backend/reports/phase4s_data_integrity_audit.md` | `6e1043f2819714eb6c0757a55225c94291845bb0837f40775d7b56d3c01c77f0` | **VERIFIED MATCH** |
| **Scientific Eligibility** | `backend/reports/phase4s_scientific_eligibility.md` | `9bde0ecbaa64724a7bcaebff60cf76d498670c538a798ee3e0bca578b86dcf32` | **VERIFIED MATCH** |
| **Training Readiness** | `backend/reports/phase4s_training_readiness.md` | `3fb854c786064f7c191a32943ee807577a7df427183060f64c679a6da83ce212` | **VERIFIED MATCH** |
| **Evaluation Protocol** | `backend/reports/phase4s_evaluation_protocol.md` | `cb182bf48807e3ea5ce83fa596cb95cc8bca7d4fe226e492b492b23dd41a13e2` | **VERIFIED MATCH** |
| **Experiment Results** | `backend/reports/phase4s_experiment_results.md` | `36c75c2832ee9262f275e3a89a05dfa04c1064db73a1154c1b92e219706ef5a6` | **VERIFIED MATCH** |
| **Phase 4S Metrics CSV** | `backend/reports/phase4s_metrics.csv` | `7d46e591c62ef4f1da316c0293ee0cb653e0db7e186a51d80ce6ec4c360bebc7` | **VERIFIED MATCH** |
| **Findings & Next Steps** | `backend/reports/phase4s_findings_and_next_steps.md` | `3f4241b8b302c332aa69a374668f45a752243d92fbfa16d7a5b3a3aa5fe99f93` | **VERIFIED MATCH** |
| **Phase 4M-S Checkpoint** | `backend/artifacts/real_data_experiments/solexs_self_supervised_cnnlstm_20261025T115207Z.pt` | `815ded9a605fa4bb410791b3a64d07624d6f3d03ffaf02953b67c7f1f8e53d45` | **IMMUTABLE FROZEN** |
| **Phase 4O Checkpoint** | `backend/artifacts/real_data_experiments/phase4o/solexs_phase4o_cnn_lstm_direct_20261025T184851Z.pt` | `d6076ab20c99fdc852d338c47f9b7248250a333c4708916a8ced8de3a6336514` | **IMMUTABLE FROZEN** |

### Duplicate Report Comparison
A byte-for-byte SHA-256 comparison between `backend/reports/` and `backend/artifacts/real_data_experiments/phase4s/` confirmed that all 9 mirrored files share identical cryptographic hashes. There are zero conflicting or stale document versions.

---

## 3. Independent Metric Recomputation & Discrepancy Reconciliation

Using [`phase4t_test_predictions.npz`](file:///c:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/artifacts/real_data_experiments/phase4t/phase4t_test_predictions.npz), predictions for all 172,439 held-out test windows were evaluated against the reported values in [`phase4s_metrics.csv`](file:///c:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/reports/phase4s_metrics.csv).

### Milestone Metrics Reconciliation (Combined Test: May 14–15, 2024)

| Horizon | Evaluated Model | Reported MAE (`c/s`) | Recomputed MAE (`c/s`) | Absolute Discrepancy | Reported Skill (%) | Recomputed Skill (%) | Skill Discrepancy |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$t+1\text{s}$** | **Persistence** | 32.8432 | 32.8432 | **0.0000** | 0.00% | 0.00% | **0.00%** |
| | Rolling Mean | 46.8027 | 46.8027 | **0.0000** | -42.50% | -42.50% | **0.00%** |
| | Phase 4S CNN-LSTM | 75.1188 | 75.1188 | **0.0000** | -128.72% | -128.72% | **0.00%** |
| | Phase 4O Transfer | 93.4711 | 93.4711 | **0.0000** | -184.60% | -184.60% | **0.00%** |
| **$t+5\text{s}$** | **Persistence** | 34.4405 | 34.4405 | **0.0000** | 0.00% | 0.00% | **0.00%** |
| | Rolling Mean | 50.7006 | 50.7006 | **0.0000** | -47.21% | -47.21% | **0.00%** |
| | Phase 4S CNN-LSTM | 79.1653 | 79.1653 | **0.0000** | -129.86% | -129.86% | **0.00%** |
| | Phase 4O Transfer | 79.6413 | 79.6413 | **0.0000** | -131.24% | -131.24% | **0.00%** |
| **$t+10\text{s}$** | **Persistence** | 37.3622 | 37.3622 | **0.0000** | 0.00% | 0.00% | **0.00%** |
| | Rolling Mean | 55.6330 | 55.6330 | **0.0000** | -48.90% | -48.90% | **0.00%** |
| | Phase 4S CNN-LSTM | 78.8614 | 78.8614 | **0.0000** | -111.07% | -111.07% | **0.00%** |
| | Phase 4O Transfer | 88.1555 | 88.1555 | **0.0000** | -135.95% | -135.95% | **0.00%** |

### Stratified Subgroups Reconciliation at Horizon $t+10\text{s}$

| Cohort | Evaluated Model | Reported MAE (`c/s`) | Recomputed MAE (`c/s`) | Discrepancy | Reported Skill (%) | Recomputed Skill (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Quiet Coronal** (`REG_QUIET`) | Rolling Mean | 9.3760 | 9.3760 | 0.0000 | +27.45% | +27.45% |
| ($< 200\text{ c/s}$) | Phase 4S CNN-LSTM | 12.6507 | 12.6507 | 0.0000 | +2.11% | +2.11% |
| | Persistence | 12.9240 | 12.9240 | 0.0000 | 0.00% | 0.00% |
| **Moderate Activity** (`REG_MOD`) | Rolling Mean | 17.3343 | 17.3343 | 0.0000 | +22.79% | +22.79% |
| ($200–1,000\text{ c/s}$) | **Phase 4S CNN-LSTM** | **17.4435** | **17.4435** | **0.0000** | **+22.30%** | **+22.30%** |
| | Persistence | 22.4505 | 22.4505 | 0.0000 | 0.00% | 0.00% |
| **Extreme Activity** (`REG_HIGH`) | **Persistence** | **107.3393** | **107.3393** | **0.0000** | **0.00%** | **0.00%** |
| ($> 1,000\text{ c/s}$) | Rolling Mean | 213.3928 | 213.3928 | 0.0000 | -98.80% | -98.80% |
| | Phase 4S CNN-LSTM | 321.2495 | 321.2495 | 0.0000 | -199.28% | -199.28% |
| | Phase 4O Transfer | 364.5411 | 364.5411 | 0.0000 | -239.62% | -239.62% |
| **May 14 (X8.7 Flare)** | **Persistence** | **40.6289** | **40.6289** | **0.0000** | **0.00%** | **0.00%** |
| | Phase 4S CNN-LSTM | 102.3403 | 102.3403 | 0.0000 | -151.89% | -151.89% |
| **May 15 (Active Decay)** | **Persistence** | **34.0987** | **34.0987** | **0.0000** | **0.00%** | **0.00%** |
| | Phase 4S CNN-LSTM | 55.4046 | 55.4046 | 0.0000 | -62.48% | -62.48% |
| | Phase 4O Transfer | 76.8930 | 76.8930 | 0.0000 | -125.50% | -125.50% |
| **Impulsive Surges** ($\Delta c > 50\text{ c/s}$) | **Persistence** | **131.5501** | **131.5501** | **0.0000** | **0.00%** | **0.00%** |
| | Phase 4S CNN-LSTM | 354.3446 | 354.3446 | 0.0000 | -169.36% | -169.36% |

---

## 4. Audit Methodological Details

1. **Target Alignment:** Confirmed identical alignment of ground-truth targets $y_{\text{true}}[i, h]$ across all 4 models.
2. **Persistence Implementation:** Verified that `persistence_forecast` copies strictly the latest observation at forecast origin ($x_{t} = X[:, -1, 0]$) into future steps $t+1, \dots, t+10$. No future leakage was detected.
3. **Rolling Mean Implementation:** Verified that `rolling_mean_forecast` computes the mean over historical input history $t-59, \dots, t$.
4. **Aggregation Semantics:** All reported metrics are pooled across all evaluated windows in the cohort (not averaged across daily sub-means), which correctly weights time spent in each physical regime.

---

## 5. Audit Conclusion

The numerical values and metrics reported in Phase 4S are **100% numerically verified and mathematically reproducible**. The absence of per-window predictions has now been cured by persisting [`phase4t_test_predictions.npz`](file:///c:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/artifacts/real_data_experiments/phase4t/phase4t_test_predictions.npz).
