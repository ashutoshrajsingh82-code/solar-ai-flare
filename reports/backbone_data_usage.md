# Phase 4V Backbone Data Usage & Temporal Overlap Audit

**Date:** 2026-10-27  
**Scope:** Investigation into data partitions used for Phase 4V self-supervised backbone pretraining vs. downstream supervised splits (`split_v3_solexs.json` and `split_v3_fusion.json`).  
**Status:** COMPLETE — LEAKAGE RISKS IDENTIFIED & FORMAL RECOMMENDATIONS PROVIDED  

---

## 1. Executive Summary

Phase 4V trained a self-supervised CNN-LSTM autoencoder backbone (`SelfSupervisedCNNLSTM`) on all currently available local SoLEXS observation days across two epochs (May 2024 and Jan–Feb 2026). 

When evaluating potential downstream supervised flare prediction models on the newly constructed instrument-grounded splits (`split_v3_solexs` and `split_v3_fusion`):
- **All February 2026 test and validation days were directly exposed during Phase 4V backbone training, validation, or testing.**
- Specifically, `split_v3_solexs` validation days (`2026-02-01` and `2026-02-02`) were in the **Phase 4V training set**.
- `split_v3_solexs` test days (`2026-02-04` and `2026-02-05`) were in the **Phase 4V validation and test sets**.
- Consequently, evaluating a model initialized with the Phase 4V checkpoint on February 2026 test days constitutes **representation leakage**. Downstream test performance would be artificially inflated by pretraining exposure.

---

## 2. Phase 4V Backbone Partition Mapping

Source files audited:
- `backend/reports/phase4v_split_and_leakage_audit.md` (Lines 19–43)
- `backend/artifacts/real_data_experiments/phase4v/run_metadata_phase4v_20261026T162621Z.json` (Lines 28–67)

| Date Range | Instrument | Phase 4V Role | Audit Reference |
| :--- | :--- | :--- | :--- |
| `2024-05-08` .. `2024-05-12` | SoLEXS (AR 13664) | **Training** | `phase4v_split_and_leakage_audit.md:21-25`, `run_metadata...:31-37` |
| `2024-05-13` | SoLEXS (AR 13664) | **Validation** | `phase4v_split_and_leakage_audit.md:26`, `run_metadata...:38-40` |
| `2024-05-14` .. `2024-05-15` | SoLEXS (AR 13664) | **Test** | `phase4v_split_and_leakage_audit.md:27-28`, `run_metadata...:41-44` |
| `2026-01-31` .. `2026-02-02` | SoLEXS (AR 13664) | **Training** | `phase4v_split_and_leakage_audit.md:36-38`, `run_metadata...:52-56` |
| `2026-02-03` .. `2026-02-04` | SoLEXS (AR 13664) | **Validation** | `phase4v_split_and_leakage_audit.md:39-40`, `run_metadata...:57-60` |
| `2026-02-05` | SoLEXS (AR 13664) | **Test** | `phase4v_split_and_leakage_audit.md:41`, `run_metadata...:61-63` |
| `2024-10-01` .. `2024-10-31` | Both | **QUARANTINED** | Sealed Holdout — Never accessed or downloaded |

---

## 3. Intersection with `split_v3` Grounded Splits

### 3.1 `split_v3_solexs` Intersection

| `split_v3_solexs` Partition | Calendar Dates | Overlapping Phase 4V Role | Contamination Severity |
| :--- | :--- | :--- | :--- |
| **TRAIN** (8 days) | `2024-05-08` .. `2024-05-15` | May Train (5d), Val (1d), Test (2d) | Minor (fine-tuning on earlier self-supervised test data is acceptable if not used for final test) |
| **PURGE GAP** | `2024-05-16` .. `2026-01-31` | Contains Phase 4V Jan 31 Train | Cleanly excluded from `split_v3` |
| **VAL** (2 days) | `2026-02-01` .. `2026-02-02` | **Phase 4V TRAINING** | **HIGH**: Validation metrics will be overfitted because backbone saw raw time series during self-supervised pretraining |
| **PURGE GAP** | `2026-02-03` | Phase 4V Validation | Cleanly excluded from `split_v3` |
| **TEST** (2 days) | `2026-02-04` .. `2026-02-05` | **Phase 4V Validation (`02-04`) & Test (`02-05`)** | **CRITICAL**: Test evaluation violates clean-slate isolation |

### 3.2 `split_v3_fusion` Intersection

| `split_v3_fusion` Partition | Calendar Dates | Overlapping Phase 4V Role | Contamination Severity |
| :--- | :--- | :--- | :--- |
| **TRAIN** (1 day) | `2026-02-01` | **Phase 4V TRAINING** | Expected |
| **PURGE GAP** | `2026-02-02` | Phase 4V Training | Cleanly excluded |
| **VAL** (1 day) | `2026-02-03` | **Phase 4V Validation** | Medium |
| **PURGE GAP** | `2026-02-04` | Phase 4V Validation | Cleanly excluded |
| **TEST** (1 day) | `2026-02-05` | **Phase 4V Test** | **CRITICAL**: Identical test day |

---

## 4. Formal Recommendations for Supervised Modeling

1. **Do NOT Use Phase 4V Checkpoint for Evaluating Feb 2026 Test Partitions:**
   Evaluating a Phase 4V backbone fine-tuned model against `2026-02-04..2026-02-05` will yield misleading TSS/Brier skill scores due to prior temporal exposure during Phase 4V pretraining.

2. **Execute the PRADAN Download Plan to Unblock Clean Evaluation:**
   The PRADAN download plan (`data/pradan_plan.csv`) prioritizes:
   - High-activity flare periods from 2024 (e.g. March, April, June, July, August, September 2024)
   - 2025 observation windows
   - 2026 post-February dates
   Downloading $\ge 30$ fresh days from these uncontaminated periods will allow creating a truly isolated train/val/test split where the test set has **zero overlap** with any pretraining artifacts.

3. **Backbone Retraining Protocol:**
   When retraining or updating the self-supervised foundation model:
   - Restrict backbone pretraining strictly to pre-2025 dates (or designated pretraining months).
   - Enforce an ironclad calendar boundary between self-supervised pretraining data and any supervised test days.
