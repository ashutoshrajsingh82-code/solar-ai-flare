# Phase 4U: Test-Set Governance and Untouched Holdout Protocol

**Project:** Solar AI Flare  
**Backend:** `backend/`  
**Phase:** Phase 4U — Multi-Epoch Real-Data Expansion and Untouched Holdout Design  
**Governance Version:** `phase4u_governance_v1.0`  
**Status:** `ACTIVE_AND_ENFORCED`

---

## 1. Executive Summary

Scientific integrity in machine learning requires that model architectures, feature representations, hyperparameters, and preprocessing routines are never selected, tuned, or iterated against data used for final performance claims. 

This governance protocol establishes the strict quarantine, access limitations, pre-registration requirements, and statistical considerations governing the **reserved, untouched third observational epoch** (October 1–7, 2024).

---

## 2. Categorization of Observational Epochs

All observational periods within the Solar AI Flare project are formally partitioned into two distinct categories:

| Observational Epoch | Calendar Range | Product Count | Prior Project Usage | Governance Classification | Access Permission |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Epoch 1 (Solar Max)** | May 8–15, 2024 | 8 daily packages | Train/Val/Test in Phase 4S; fully audited in Phase 4T | **Examined Development Data** | Open for training, feature engineering, and cross-validation. |
| **Epoch 2 (Quiet Sun)** | Jan 31–Feb 5, 2026 | 6 daily packages | Train/Val/Test in Phase 4M-S; tested in Phase 4N, 4O, 4R | **Examined Development Data** | Open for training, feature engineering, and cross-validation. |
| **Epoch 3 (Reserved Holdout)** | **Oct 1–7, 2024** | **7 daily packages** | **Zero prior access (Unacquired)** | **QUARANTINED UNTOUCHED HOLDOUT** | **STRICTLY LOCKED. Zero training, tuning, or inference permitted.** |

> [!WARNING]
> Neither May 8–15, 2024 nor January 31–February 5, 2026 may ever be cited as an untouched or independent holdout in future project phases. Both datasets are thoroughly examined development data.

---

## 3. Pre-Registration & Decisions That Must Be Frozen Prior to Holdout Access

Before the reserved Epoch 3 holdout is unlocked for evaluation in Phase 4V, the following decisions must be permanently committed and frozen in a verifiable, machine-readable manifest:

1. **Neural Network Architecture:**
   - Layer dimensions, convolution kernel sizes, LSTM hidden sizes, activation functions, and dropout probabilities must be locked.
2. **Preprocessing & Scaling Parameters:**
   - The fitted [CountRateScaler](file:///c:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/ml/experiments/real_self_supervised.py#L51-L84) parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) must be frozen. No parameter updating based on Epoch 3 is permitted.
3. **Loss Function & Training Weights:**
   - The training checkpoint weights must be saved and cryptographically hashed with SHA-256 before holdout evaluation.
4. **Cohort Definitions:**
   - Activity cohort cutoffs ($200\text{ c/s}$, $1000\text{ c/s}$, surge threshold $50\text{ c/s}$) must be fixed.
5. **Baseline Algorithms:**
   - Persistence and Rolling Mean baseline implementations must be locked.

---

## 4. Single-Evaluation Rule & Access Restrictions

1. **Automated Blocker:**
   - No automated training pipeline, gradient descent optimizer, grid search routine, or hyperparameter tuner may receive data paths containing Epoch 3.
2. **Single Evaluation Pass:**
   - The holdout evaluation must be executed as a **single, non-iterative, read-only inference pass**.
   - If evaluation results reveal unexpected failure modes or poor skill, the model **cannot be modified and re-evaluated against Epoch 3**. The results must be reported as-is, and any subsequent model revisions must be deferred to a separate, fourth observational epoch.
3. **Prediction Archival:**
   - All ground truth targets, baseline predictions, and model predictions on Epoch 3 must be saved in full precision to prevent selective metric reporting.

---

## 5. Statistical Dependence & Generalization Considerations

Merely storing files in separate directories or selecting dates from different calendar months does not guarantee true statistical independence. The evaluation analysis must explicitly account for the following dependencies:

1. **Serial Autocorrelation:**
   - Contiguous 1-second count rates exhibit strong physical autocorrelation. Consecutive 70-second sliding windows overlap by 69 seconds. All statistical significance testing on Epoch 3 must utilize time-aware block bootstrapping ($L_{\text{block}} \ge 300\text{s}$) rather than i.i.d. sampling.
2. **Instrumental Drift & Detector State:**
   - Aditya-L1 orbital thermal cycles, detector bias voltages, and silicon drift degradation evolve over months. Differences in baseline noise or gain between May 2024, October 2024, and February 2026 reflect true instrument evolution that models must accommodate.
3. **Solar Cycle & Active Region Clustering:**
   - Solar flares are physically clustered in active regions governed by solar rotation (27-day synodic period). October 2024 captures Active Region 13842, presenting a genuinely different coronal magnetic topology than May 2024 (AR 13664).
