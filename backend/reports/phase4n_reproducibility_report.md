# Phase 4N: Independent Model Reproducibility and Verification Report

**Document Version:** 1.0.0  
**Audit Date:** 2026-10-25  
**Canonical Designation:** `PHASE4N_REPRODUCIBILITY_VERIFICATION_REPORT`  
**Parent Phase:** Phase 4N (Independent Reproducibility, Robustness & Generalization Validation)  
**Evaluated Checkpoint:** `backend/artifacts/real_data_experiments/solexs_self_supervised_cnnlstm_20261025T115207Z.pt`  
**Checkpoint Metadata:** `backend/artifacts/real_data_experiments/solexs_self_supervised_cnnlstm_20261025T115207Z_metadata.json`  

---

## 1. Executive Summary

This report documents the rigorous, independent reproduction of the Phase 4M-S self-supervised solar forecasting model trained on genuine Aditya-L1 SoLEXS Level-1 observations. The evaluation was conducted using a dedicated, read-only script ([`evaluate_real_self_supervised.py`](file:///C:/Users/ASHUTOSH/Downloads/solar-ai-flare-main/solar-ai-flare-main/backend/scripts/evaluate_real_self_supervised.py)) that strictly loads the frozen model checkpoint and saved normalization parameters without modifying weights, without fitting scalers on evaluation data, and without altering the production model registry.

### Core Finding
**Bit-for-bit exact numerical reproduction was achieved across all primary metrics on the held-out test day (2026-02-05, 44,221 forecast windows).** The absolute difference between reported Phase 4M-S metrics and independently reproduced Phase 4N metrics is **$0.00 \times 10^{-0}$ counts/s** down to full 64-bit IEEE floating-point precision.

---

## 2. Source-Data and Checkpoint Provenance Audit

### 2.1 Bitstream Hashes and File Properties

| Artifact | Canonical Path | Actual SHA-256 Checksum | Expected Checksum (Phase 4M-S Metadata) | Verification Status |
|---|---|---|---|---|
| **Observational Archive** | `C:\Users\ASHUTOSH\Downloads\solexs_2026Oct23T184402685.zip` | `34c39624194ee2da95bbf8e3dfed5ac002c55a517a31bc4091bfdd0d2b4671c2` | `34c39624194ee2da95bbf8e3dfed5ac002c55a517a31bc4091bfdd0d2b4671c2` | **VERIFIED (Exact Match)** |
| **Model Checkpoint** | `backend/artifacts/real_data_experiments/solexs_self_supervised_cnnlstm_20261025T115207Z.pt` | `815ded9a605fa4bb410791b3a64d07624d6f3d03ffaf02953b67c7f1f8e53d45` | `815ded9a605fa4bb410791b3a64d07624d6f3d03ffaf02953b67c7f1f8e53d45` | **VERIFIED (Exact Match)** |
| **Provenance Metadata** | `backend/artifacts/real_data_experiments/solexs_self_supervised_cnnlstm_20261025T115207Z_metadata.json` | `5c644efb3b18aa2c1143c68364b8bb26c31bf0ea4e2bf7618ae032ec302c019d` | N/A (Canonical Source) | **VERIFIED** |

> [!NOTE]
> **Documentation Discrepancy Resolved:** In `phase4ms_real_observation_manifest.md` line 27, the archive hash was recorded as `34c396e95bf1bfd8e41113ca22a6114ebc0a876a4ba251c6e1ff08a9f0e13636`. However, the physical archive and the machine-readable metadata file (`metadata.json` line 13) carry the true SHA-256 `34c39624194ee2da95bbf8e3dfed5ac002c55a517a31bc4091bfdd0d2b4671c2`. The archive physically present in `Downloads` matches the metadata hash bit-for-bit.

### 2.2 Model Architecture and Parameter Count Audit

The saved PyTorch state dictionary was independently inspected and structurally audited against the defined network architecture:

- **Network Class:** `SelfSupervisedCNNLSTM`
- **Input Channels:** 1 (native count rate $r_t$, 60s history)
- **1D Convolutional Layer:** `in_channels=1`, `out_channels=32`, `kernel_size=5`, `padding=2`
- **1D Batch Normalization:** `num_features=32`
- **LSTM Backbone:** `input_size=32`, `hidden_size=64`, `num_layers=2`, `batch_first=True`
- **Fully Connected Projection:** `Linear(64, 32) -> ReLU -> Linear(32, 10)`
- **Output Horizon:** 10 future time steps ($T+1 \dots T+10$ seconds)

```
Parameter Tensor Audit:
- conv.0.weight: torch.Size([32, 1, 5])      -> 160 parameters
- conv.0.bias:   torch.Size([32])            -> 32 parameters
- conv.1.weight: torch.Size([32])            -> 32 parameters
- conv.1.bias:   torch.Size([32])            -> 32 parameters
- conv.1.running_mean: torch.Size([32])      -> 32 buffer elements
- conv.1.running_var:  torch.Size([32])      -> 32 buffer elements
- conv.1.num_batches:  torch.Size([])        -> 1 buffer element
- lstm.weight_ih_l0: torch.Size([256, 32])   -> 8,192 parameters
- lstm.weight_hh_l0: torch.Size([256, 64])   -> 16,384 parameters
- lstm.bias_ih_l0:   torch.Size([256])       -> 256 parameters
- lstm.bias_hh_l0:   torch.Size([256])       -> 256 parameters
- lstm.weight_ih_l1: torch.Size([256, 64])   -> 16,384 parameters
- lstm.weight_hh_l1: torch.Size([256, 64])   -> 16,384 parameters
- lstm.bias_ih_l1:   torch.Size([256])       -> 256 parameters
- lstm.bias_hh_l1:   torch.Size([256])       -> 256 parameters
- fc.0.weight:       torch.Size([32, 64])    -> 2,048 parameters
- fc.0.bias:         torch.Size([32])        -> 32 parameters
- fc.2.weight:       torch.Size([10, 32])    -> 320 parameters
- fc.2.bias:         torch.Size([10])        -> 10 parameters

Total Trainable Weights: 61,034
Total State Elements (including BatchNorm buffers): 61,099
```

> [!NOTE]
> **Parameter Count Discrepancy Resolved:** The narrative text of `phase4ms_training_and_evaluation_report.md` (line 54) cited 43,146 parameters (corresponding to a 1-layer LSTM or kernel size 3). The physical state dictionary contains exactly **61,034 trainable parameters** and **61,099 total state elements**. The model loaded and executed flawlessly with 61,034 parameters.

---

## 3. Independent Numerical Reproduction

### 3.1 Numerical Tolerance and Ground-Truth Comparison

Evaluation was performed on the exact held-out test cohort (`2026-02-05 00:00:00` to `2026-02-05 23:59:59 UTC`, 44,221 valid contiguous windows) in native detector units (`counts/s`, 2.0–22.0 keV):

| Evaluation Metric | Phase 4M-S Reported | Phase 4N Reproduced | Absolute Difference | Relative Tolerance ($10^{-5}$) | Reproduction Status |
|---|---|---|---|---|---|
| **CNN-LSTM MAE** | `14.032015800476074` | `14.032015800476074` | `0.000000000000` | PASSED | **BIT-FOR-BIT IDENTICAL** |
| **CNN-LSTM RMSE** | `23.323683278165056` | `23.323683278165056` | `0.000000000000` | PASSED | **BIT-FOR-BIT IDENTICAL** |
| **CNN-LSTM Relative MAE** | `0.04982992989472233` (4.98%) | `0.04982992989472233` (4.98%) | `0.000000000000` | PASSED | **BIT-FOR-BIT IDENTICAL** |
| **Persistence MAE** | `15.927862167358398` | `15.927862167358398` | `0.000000000000` | PASSED | **BIT-FOR-BIT IDENTICAL** |
| **Persistence RMSE** | `24.46358477370945` | `24.46358477370945` | `0.000000000000` | PASSED | **BIT-FOR-BIT IDENTICAL** |
| **Persistence Relative MAE** | `0.05656238322831287` (5.66%) | `0.05656238322831287` (5.66%) | `0.000000000000` | PASSED | **BIT-FOR-BIT IDENTICAL** |
| **Rolling Mean MAE** | `14.774998664855957` | `14.774998664855957` | `0.000000000000` | PASSED | **BIT-FOR-BIT IDENTICAL** |
| **Rolling Mean RMSE** | `25.081308209189867` | `25.081308209189867` | `0.000000000000` | PASSED | **BIT-FOR-BIT IDENTICAL** |
| **Rolling Mean Relative MAE** | `0.05246838074679260` (5.25%) | `0.05246838074679260` (5.25%) | `0.000000000000` | PASSED | **BIT-FOR-BIT IDENTICAL** |
| **Skill Score vs Persistence** | `0.11902704499587880` (+11.90%) | `0.11902704499587880` (+11.90%) | `0.000000000000` | PASSED | **BIT-FOR-BIT IDENTICAL** |
| **Skill Score vs Rolling Mean** | `0.05028649282704534` (+5.03%) | `0.05028649282704534` (+5.03%) | `0.000000000000` | PASSED | **BIT-FOR-BIT IDENTICAL** |

### 3.2 Denominator Definition for Relative MAE
The relative MAE was audited to confirm mathematical consistency:
$$\text{RelMAE} = \frac{\text{MAE}}{\bar{y}_{\text{test}}} = \frac{14.0320158}{281.6026} = 0.0498299 \quad (4.98\%)$$
The denominator is explicitly defined as the empirical arithmetic mean of the genuine ground-truth observation targets across all 44,221 evaluated test windows ($\bar{y}_{\text{test}} = 281.60\text{ counts/s}$).

---

## 4. Evaluation-Only Immutability and State Safety

To guarantee evaluation fairness and reproducibility:
1. **Requires Grad:** Model was placed into `model.eval()` and all parameter tensors set to `requires_grad = False`.
2. **Zero In-Memory Weight Drift:** Cloned state dictionary tensors before evaluation were verified against state dictionary tensors after evaluation using `torch.equal()`. Zero drift detected.
3. **No Retraining or Parameter Fitting:** Scaler parameters ($\mu = 2.2472749, \sigma = 0.6966392$) were loaded directly from `metadata.json` rather than fitted on the test partition.
4. **Isolated Output Directory:** All Phase 4N evaluation outputs were written to `backend/artifacts/real_data_experiments/phase4n/` and `backend/reports/`. The original Phase 4M-S checkpoint and reports remain untouched.

---

## 5. Execution Environment and Determinism

- **Operating System:** Windows 11 Pro (win32)
- **Python Version:** 3.13.4
- **PyTorch Version:** 2.6.0+cpu
- **NumPy Version:** 2.2.3
- **Astropy Version:** 7.0.1
- **Device Used:** CPU (deterministic execution verified)
- **Random Seed:** 42
- **Command Used:** `python scripts/evaluate_real_self_supervised.py --deterministic`
