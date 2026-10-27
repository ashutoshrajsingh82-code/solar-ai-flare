"""Phase 4T: Independent Reproducibility Audit and Forecasting-Skill Verification.

This script executes a read-only, non-destructive audit of Phase 4S:
1. Compares file copies and computes SHA-256 hashes for all Phase 4S and historical artifacts.
2. Audits dataset and window reconciliation (78 indexed products, 14 eligible, exact window counts per segment).
3. Executes audit-only inference using frozen checkpoint solexs_phase4s_cnn_lstm_may2024_20261025T203124Z.pt
   and generates per-window predictions on held-out test data (May 14-15, 2024).
4. Evaluates all 4 models (Phase 4S CNN-LSTM, Persistence, Rolling Mean, Phase 4O Transfer) across all 10 horizons.
5. Reconciles exact metrics against phase4s_metrics.csv.
6. Audits cohort conditioning: compares target-based (future) vs origin-based (causal) regime assignment.
7. Computes time-aware block bootstrap confidence intervals on cohort skill scores.
8. Writes all required Phase 4T structured outputs:
   - phase4t_artifact_inventory.json
   - phase4t_dataset_and_window_reconciliation.csv
   - phase4t_independent_metrics.csv
   - phase4t_cohort_metrics.csv
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Any

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import numpy as np
import pandas as pd
import torch
from astropy.io import fits

from ml.experiments.real_self_supervised import (
    validate_solexs_fits_table,
    extract_contiguous_segments,
    create_sliding_windows,
    CountRateScaler,
    SelfSupervisedCNNLSTM,
    persistence_forecast,
    rolling_mean_forecast,
)

ARCHIVE_PATH = Path(r"C:\Users\ASHUTOSH\Downloads\solexs_2026Oct25T200126271.zip")
ARCHIVE_EXPECTED_SHA = "8df85ed75b6f00052d7f45a6764b187c700977f931504fb79fdfe56f10df9db2"

REPORTS_DIR = backend_dir / "reports"
ART_4S_DIR = backend_dir / "artifacts" / "real_data_experiments" / "phase4s"
ART_4T_DIR = backend_dir / "artifacts" / "real_data_experiments" / "phase4t"
ART_4T_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()


# ==============================================================================
# STAGE 1: Artifact Inventory and Comparison
# ==============================================================================
def stage1_artifact_inventory() -> Dict[str, Any]:
    print("\n--- STAGE 1: Verifying Artifact Inventory and SHA-256 Hashes ---")
    files_to_check = [
        "phase4s_dataset_manifest.json",
        "phase4s_dataset_manifest.csv",
        "phase4s_data_integrity_audit.md",
        "phase4s_scientific_eligibility.md",
        "phase4s_training_readiness.md",
        "phase4s_evaluation_protocol.md",
        "phase4s_experiment_results.md",
        "phase4s_metrics.csv",
        "phase4s_findings_and_next_steps.md",
    ]

    inventory = {
        "audit_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_archive": {
            "path": str(ARCHIVE_PATH),
            "exists": ARCHIVE_PATH.is_file(),
            "size_bytes": ARCHIVE_PATH.stat().st_size if ARCHIVE_PATH.is_file() else None,
            "sha256": sha256_file(ARCHIVE_PATH),
            "sha256_match": sha256_file(ARCHIVE_PATH) == ARCHIVE_EXPECTED_SHA,
        },
        "phase4s_checkpoint": {
            "canonical_path": str(ART_4S_DIR / "solexs_phase4s_cnn_lstm_may2024_20261025T203124Z.pt"),
            "exists": (ART_4S_DIR / "solexs_phase4s_cnn_lstm_may2024_20261025T203124Z.pt").is_file(),
            "sha256": sha256_file(ART_4S_DIR / "solexs_phase4s_cnn_lstm_may2024_20261025T203124Z.pt"),
            "size_bytes": (ART_4S_DIR / "solexs_phase4s_cnn_lstm_may2024_20261025T203124Z.pt").stat().st_size
            if (ART_4S_DIR / "solexs_phase4s_cnn_lstm_may2024_20261025T203124Z.pt").is_file() else None,
        },
        "phase4s_metadata": {
            "canonical_path": str(ART_4S_DIR / "solexs_phase4s_cnn_lstm_may2024_20261025T203124Z_metadata.json"),
            "exists": (ART_4S_DIR / "solexs_phase4s_cnn_lstm_may2024_20261025T203124Z_metadata.json").is_file(),
            "sha256": sha256_file(ART_4S_DIR / "solexs_phase4s_cnn_lstm_may2024_20261025T203124Z_metadata.json"),
        },
        "historical_reference_models": {
            "phase4m_s_checkpoint": {
                "path": "artifacts/real_data_experiments/solexs_self_supervised_cnnlstm_20261025T115207Z.pt",
                "sha256": sha256_file(backend_dir / "artifacts/real_data_experiments/solexs_self_supervised_cnnlstm_20261025T115207Z.pt"),
                "expected_sha256": "815ded9a605fa4bb410791b3a64d07624d6f3d03ffaf02953b67c7f1f8e53d45",
            },
            "phase4o_checkpoint": {
                "path": "artifacts/real_data_experiments/phase4o/solexs_phase4o_cnn_lstm_direct_20261025T184851Z.pt",
                "sha256": sha256_file(backend_dir / "artifacts/real_data_experiments/phase4o/solexs_phase4o_cnn_lstm_direct_20261025T184851Z.pt"),
                "expected_sha256": "d6076ab20c99fdc852d338c47f9b7248250a333c4708916a8ced8de3a6336514",
            },
        },
        "saved_predictions": {
            "exists_in_artifacts_phase4s": False,
            "exists_in_reports": False,
            "status": "NOT_SAVED_DURING_PHASE4S",
            "audit_action": "PERFORM_READ_ONLY_AUDIT_INFERENCE_FROM_FROZEN_CHECKPOINT",
        },
        "reports_vs_artifacts_comparison": [],
    }

    for fname in files_to_check:
        rep_p = REPORTS_DIR / fname
        art_p = ART_4S_DIR / fname
        sha_r = sha256_file(rep_p)
        sha_a = sha256_file(art_p)
        match = (sha_r == sha_a) and (sha_r != "MISSING")
        inventory["reports_vs_artifacts_comparison"].append({
            "filename": fname,
            "reports_location": str(rep_p),
            "reports_sha256": sha_r,
            "artifacts_location": str(art_p),
            "artifacts_sha256": sha_a,
            "identical": match,
        })
        print(f"  {fname:<38} | reports: {sha_r[:12]}... | artifacts: {sha_a[:12]}... | match: {match}")

    out_inv_path = ART_4T_DIR / "phase4t_artifact_inventory.json"
    with open(out_inv_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)
    # Mirror to reports
    with open(REPORTS_DIR / "phase4t_artifact_inventory.json", "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    return inventory


# ==============================================================================
# STAGE 2: Dataset and Window Reconciliation
# ==============================================================================
def stage2_dataset_and_window_reconciliation() -> pd.DataFrame:
    print("\n--- STAGE 2: Auditing Dataset and Reconciling Sliding Windows ---")
    manifest_path = REPORTS_DIR / "phase4s_dataset_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        man_data = json.load(f)

    inv = man_data["inventory"]
    total_indexed = man_data["total_products_indexed"]
    eligible_count = man_data["eligible_for_forecasting"]
    print(f"  Total Indexed Products: {total_indexed} (Claimed in report: 78)")
    print(f"  Eligible Products: {eligible_count} (Claimed in report: 14)")

    # Breakdown of items by epoch and product type
    may_items = [x for x in inv if x.get("observation_date", "").startswith("202405")]
    jan_feb_items = [x for x in inv if x.get("observation_date", "").startswith("2026")]
    print(f"  May 2024 Items Indexed: {len(may_items)}")
    print(f"  Jan-Feb 2026 Items Indexed: {len(jan_feb_items)}")

    # Unpack May 8-15 from archive and reconcile exact daily numbers
    dates = ["20240508", "20240509", "20240510", "20240511", "20240512", "20240513", "20240514", "20240515"]
    reconciliation_rows = []

    split_map = {
        "20240508": "train", "20240509": "train", "20240510": "train",
        "20240511": "train", "20240512": "train",
        "20240513": "val",
        "20240514": "test", "20240515": "test",
    }

    total_windows_by_split = {"train": 0, "val": 0, "test": 0}

    with zipfile.ZipFile(ARCHIVE_PATH) as outer:
        for d in dates:
            split_name = split_map[d]
            inner_name = f"AL1_SLX_L1_{d}_v1.0.zip"
            inner_bytes = outer.read(inner_name)
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                lc_member = f"AL1_SLX_L1_{d}_v1.0/SDD2/AL1_SOLEXS_{d}_SDD2_L1.lc.gz"
                decomp = gzip.decompress(inner.read(lc_member))
                with fits.open(io.BytesIO(decomp)) as hdul:
                    times, counts, t_unit, c_unit = validate_solexs_fits_table(hdul)
                    
                    total_records = len(times)
                    nan_records = int(np.isnan(counts).sum())
                    finite_records = int(np.isfinite(counts).sum())
                    min_val = float(np.nanmin(counts))
                    median_val = float(np.nanmedian(counts))
                    max_val = float(np.nanmax(counts))

                    # Check monotonicity and cadence
                    time_diffs = np.diff(times)
                    cadence_is_1s = bool(np.all(time_diffs == 1.0))
                    is_monotonic = bool(np.all(time_diffs > 0))

                    # Contiguous segmentation
                    segs = extract_contiguous_segments(times, counts, min_segment_length=70)
                    segment_count = len(segs)
                    
                    # Compute sliding windows for this date
                    X_d, y_d, t_d = create_sliding_windows(segs, seq_len=60, horizon=10)
                    window_count = len(X_d)
                    total_windows_by_split[split_name] += window_count

                    # Segment lengths detail
                    seg_lengths = [len(s[0]) for s in segs]
                    # Verify window count formula: sum(len(s) - 70 + 1 for s in segs)
                    calc_windows = sum(len(s[0]) - 70 + 1 for s in segs)
                    assert calc_windows == window_count

                    row = {
                        "observation_date": d,
                        "partition_split": split_name,
                        "total_records": total_records,
                        "finite_records": finite_records,
                        "nan_records": nan_records,
                        "duty_cycle_pct": round(finite_records / total_records * 100.0, 4),
                        "monotonic_timestamps": is_monotonic,
                        "cadence_1s_uniform": cadence_is_1s,
                        "contiguous_segments_count": segment_count,
                        "segment_lengths": str(seg_lengths),
                        "computed_sliding_windows": window_count,
                        "min_count_rate": min_val,
                        "median_count_rate": median_val,
                        "peak_count_rate": max_val,
                        "units_time": t_unit,
                        "units_rate": c_unit,
                    }
                    reconciliation_rows.append(row)
                    print(f"  {d} ({split_name:<5}): {finite_records:,} finite, {nan_records} NaNs, "
                          f"{segment_count} segs -> {window_count:,} windows (Peak: {max_val:,.1f} c/s)")

    print(f"\n  Reconciled Split Window Totals:")
    print(f"    Training   (May 8-12):  {total_windows_by_split['train']:,} (Reported: 431,069) -> Match: {total_windows_by_split['train'] == 431069}")
    print(f"    Validation (May 13):     {total_windows_by_split['val']:,} (Reported:  86,185) -> Match: {total_windows_by_split['val'] == 86185}")
    print(f"    Test       (May 14-15): {total_windows_by_split['test']:,} (Reported: 172,439) -> Match: {total_windows_by_split['test'] == 172439}")
    print(f"    Grand Total:            {sum(total_windows_by_split.values()):,} (Reported: 689,693)")

    df_recon = pd.DataFrame(reconciliation_rows)
    df_recon.to_csv(ART_4T_DIR / "phase4t_dataset_and_window_reconciliation.csv", index=False)
    df_recon.to_csv(REPORTS_DIR / "phase4t_dataset_and_window_reconciliation.csv", index=False)
    return df_recon


# ==============================================================================
# STAGE 3: Audit-Only Inference & Per-Window Metric Recomputation
# ==============================================================================
def stage3_reproduce_metrics() -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\n--- STAGE 3: Independent Metric Recomputation via Audit-Only Inference ---")
    print("  Note: Re-evaluating frozen models on already-examined held-out test set (May 14-15, 2024).")
    
    # 1. Ingest Data for Training Scaler and Test Windows
    train_dates = ["20240508", "20240509", "20240510", "20240511", "20240512"]
    test_dates = ["20240514", "20240515"]

    train_counts_list = []
    test_records = {}

    with zipfile.ZipFile(ARCHIVE_PATH) as outer:
        for d in train_dates:
            inner_name = f"AL1_SLX_L1_{d}_v1.0.zip"
            with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
                lc_member = f"AL1_SLX_L1_{d}_v1.0/SDD2/AL1_SOLEXS_{d}_SDD2_L1.lc.gz"
                decomp = gzip.decompress(inner.read(lc_member))
                with fits.open(io.BytesIO(decomp)) as hdul:
                    times, counts, _, _ = validate_solexs_fits_table(hdul)
                    train_counts_list.append(counts)

        for d in test_dates:
            inner_name = f"AL1_SLX_L1_{d}_v1.0.zip"
            with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
                lc_member = f"AL1_SLX_L1_{d}_v1.0/SDD2/AL1_SOLEXS_{d}_SDD2_L1.lc.gz"
                decomp = gzip.decompress(inner.read(lc_member))
                with fits.open(io.BytesIO(decomp)) as hdul:
                    times, counts, _, _ = validate_solexs_fits_table(hdul)
                    test_records[d] = (times, counts)

    # 2. Scaler Verification
    train_counts_all = np.concatenate(train_counts_list)
    scaler = CountRateScaler().fit(train_counts_all)
    print(f"  Independently Fitted Scaler: mean={scaler.mean:.6f}, std={scaler.std:.6f}")
    
    # Check Phase 4S saved metadata scaler
    meta_p4s_path = ART_4S_DIR / "solexs_phase4s_cnn_lstm_may2024_20261025T203124Z_metadata.json"
    with open(meta_p4s_path, "r", encoding="utf-8") as f:
        meta_p4s = json.load(f)
    print(f"  Phase 4S Metadata Scaler:    mean={meta_p4s['scaler_mean']:.6f}, std={meta_p4s['scaler_std']:.6f}")
    assert np.isclose(scaler.mean, meta_p4s["scaler_mean"], atol=1e-5)
    assert np.isclose(scaler.std, meta_p4s["scaler_std"], atol=1e-5)

    # 3. Build Test Windows
    test_X_list, test_y_list, test_t_list, test_d_list = [], [], [], []
    for d in test_dates:
        t_d, c_d = test_records[d]
        segs = extract_contiguous_segments(t_d, c_d, min_segment_length=70)
        X_d, y_d, t_orig = create_sliding_windows(segs, seq_len=60, horizon=10)
        test_X_list.append(X_d)
        test_y_list.append(y_d)
        test_t_list.append(t_orig)
        test_d_list.extend([d] * len(X_d))

    X_test = np.concatenate(test_X_list, axis=0)
    y_test = np.concatenate(test_y_list, axis=0)
    t_test = np.concatenate(test_t_list, axis=0)
    dates_test = np.array(test_d_list)
    N_test = len(X_test)
    print(f"  Test Windows Loaded: {N_test:,} (May 14: {(dates_test == '20240514').sum():,}, May 15: {(dates_test == '20240515').sum():,})")

    # 4. Generate Predictions for all 4 Models
    # A. Persistence
    preds_persist = persistence_forecast(X_test, 10)

    # B. Rolling Mean
    preds_rolling = rolling_mean_forecast(X_test, 10)

    # C. Phase 4S CNN-LSTM
    ckpt_p4s = ART_4S_DIR / "solexs_phase4s_cnn_lstm_may2024_20261025T203124Z.pt"
    model_p4s = SelfSupervisedCNNLSTM(1, 10, 32, 64, 2, 0.2)
    model_p4s.load_state_dict(torch.load(ckpt_p4s, map_location="cpu", weights_only=True))
    model_p4s.eval()

    X_test_scaled = scaler.transform(X_test.reshape(-1)).reshape(N_test, 60, 1)
    preds_scaled_list = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, N_test, batch_size):
            bx = torch.from_numpy(X_test_scaled[i : i + batch_size])
            out = model_p4s(bx)
            preds_scaled_list.append(out.numpy())
    preds_scaled_all = np.concatenate(preds_scaled_list, axis=0)
    preds_p4s = scaler.inverse_transform(preds_scaled_all)

    # D. Phase 4O Transfer Model
    ckpt_p4o = backend_dir / "artifacts/real_data_experiments/phase4o/solexs_phase4o_cnn_lstm_direct_20261025T184851Z.pt"
    meta_p4o_path = backend_dir / "artifacts/real_data_experiments/phase4o/solexs_phase4o_cnn_lstm_direct_20261025T184851Z_metadata.json"
    with open(meta_p4o_path, "r", encoding="utf-8") as f:
        meta_p4o = json.load(f)
    scaler_p4o = CountRateScaler()
    scaler_p4o.mean = float(meta_p4o["scaler_parameters"]["mean"])
    scaler_p4o.std = float(meta_p4o["scaler_parameters"]["std"])
    scaler_p4o.fitted = True

    model_p4o = SelfSupervisedCNNLSTM(1, 10, 32, 64, 2, 0.2)
    model_p4o.load_state_dict(torch.load(ckpt_p4o, map_location="cpu", weights_only=True))
    model_p4o.eval()

    X_test_p4o_scaled = scaler_p4o.transform(X_test.reshape(-1)).reshape(N_test, 60, 1)
    preds_p4o_scaled_list = []
    with torch.no_grad():
        for i in range(0, N_test, batch_size):
            bx = torch.from_numpy(X_test_p4o_scaled[i : i + batch_size])
            out = model_p4o(bx)
            preds_p4o_scaled_list.append(out.numpy())
    preds_p4o_scaled_all = np.concatenate(preds_p4o_scaled_list, axis=0)
    preds_p4o = scaler_p4o.inverse_transform(preds_p4o_scaled_all)

    # Save per-window predictions to artifacts/phase4t
    np.savez_compressed(
        ART_4T_DIR / "phase4t_test_predictions.npz",
        timestamps=t_test,
        dates=dates_test,
        y_true=y_test,
        X_last=X_test[:, -1, 0],
        preds_p4s=preds_p4s,
        preds_persist=preds_persist,
        preds_rolling=preds_rolling,
        preds_p4o=preds_p4o,
    )
    print("  Saved per-window predictions array to: artifacts/real_data_experiments/phase4t/phase4t_test_predictions.npz")

    # 5. Cohort Definitions
    # (a) Original Phase 4S definition: TARGET PEAK
    peak_target = np.max(y_test, axis=1)
    # (b) Causal definition: FORECAST ORIGIN COUNT RATE
    origin_rate = X_test[:, -1, 0]
    # Input surge trend
    input_trend = X_test[:, -1, 0] - X_test[:, -10, 0]

    cohorts_dict = {
        # Standard partitions
        "Combined_Test_May14_15": np.ones(N_test, dtype=bool),
        "Day7_May14_X8_7_Flare": (dates_test == "20240514"),
        "Day8_May15_Active_Decay": (dates_test == "20240515"),
        # Phase 4S Target-conditioned regimes (Future information used)
        "Regime_REG_QUIET_lt200": (peak_target < 200.0),
        "Regime_REG_MOD_200_1000": (peak_target >= 200.0) & (peak_target <= 1000.0),
        "Regime_REG_HIGH_gt1000": (peak_target > 1000.0),
        # Historical surge definition
        "Rising_Surge_DeltaC_gt50": (input_trend > 50.0),
        # Causal (Origin-Conditioned) Regimes for Comparison
        "Causal_REG_QUIET_lt200": (origin_rate < 200.0),
        "Causal_REG_MOD_200_1000": (origin_rate >= 200.0) & (origin_rate <= 1000.0),
        "Causal_REG_HIGH_gt1000": (origin_rate > 1000.0),
    }

    # Recompute metrics for all cohorts and horizons
    all_rows = []
    models_dict = {
        "Phase4S_CNN_LSTM_May2024": preds_p4s,
        "Persistence_Baseline": preds_persist,
        "Rolling_Mean_Baseline": preds_rolling,
        "Phase4O_CNN_LSTM_Transfer": preds_p4o,
    }

    # Load Phase 4S reported metrics for direct row-by-row discrepancy check
    p4s_metrics_df = pd.read_csv(REPORTS_DIR / "phase4s_metrics.csv")

    for c_name, mask in cohorts_dict.items():
        if not np.any(mask):
            continue
        y_c = y_test[mask]
        mean_y = float(np.mean(y_c))
        n_samples = len(y_c)

        for h in range(1, 11):
            idx = h - 1
            y_true_h = y_c[:, idx]
            
            p_dict_h = {m_name: p[mask, idx] for m_name, p in models_dict.items()}
            mae_dict = {m_name: float(np.mean(np.abs(y_true_h - p_dict_h[m_name]))) for m_name in models_dict}
            rmse_dict = {m_name: float(np.sqrt(np.mean((y_true_h - p_dict_h[m_name]) ** 2))) for m_name in models_dict}

            mae_persist = mae_dict["Persistence_Baseline"]

            for m_name in models_dict:
                mae_val = mae_dict[m_name]
                rmse_val = rmse_dict[m_name]
                rel_mae = (mae_val / mean_y * 100.0) if mean_y > 0 else 0.0
                skill_persist = ((1.0 - mae_val / max(mae_persist, 1e-6)) * 100.0) if m_name != "Persistence_Baseline" else 0.0

                # Check against Phase 4S reported if present
                rep_match = p4s_metrics_df[
                    (p4s_metrics_df["cohort"] == c_name) &
                    (p4s_metrics_df["model_name"] == m_name) &
                    (p4s_metrics_df["horizon_step"] == h)
                ]
                reported_mae = float(rep_match["mae"].values[0]) if len(rep_match) else None
                mae_diff = round(mae_val - reported_mae, 4) if reported_mae is not None else None

                all_rows.append({
                    "cohort": c_name,
                    "model_name": m_name,
                    "horizon_step": h,
                    "sample_count": n_samples,
                    "recomputed_mae": round(mae_val, 4),
                    "recomputed_rmse": round(rmse_val, 4),
                    "recomputed_rel_mae_pct": round(rel_mae, 2),
                    "recomputed_skill_vs_persist_pct": round(skill_persist, 2),
                    "reported_mae": reported_mae,
                    "mae_discrepancy": mae_diff,
                    "conditioning_type": "ORIGIN_CAUSAL" if c_name.startswith("Causal_") else ("TARGET_FUTURE" if "Regime_REG_" in c_name else "TEMPORAL"),
                })

    df_indep = pd.DataFrame(all_rows)
    df_indep.to_csv(ART_4T_DIR / "phase4t_independent_metrics.csv", index=False)
    df_indep.to_csv(REPORTS_DIR / "phase4t_independent_metrics.csv", index=False)

    # Filter cohort-level summary at horizon 10
    h10_df = df_indep[df_indep["horizon_step"] == 10].copy()
    h10_df.to_csv(ART_4T_DIR / "phase4t_cohort_metrics.csv", index=False)
    h10_df.to_csv(REPORTS_DIR / "phase4t_cohort_metrics.csv", index=False)

    # Print verification of combined test numbers cited in prompt
    print("\n  Reconciliation of Key Milestone Metrics (Combined Test May 14-15):")
    for h in [1, 5, 10]:
        sub_p = df_indep[(df_indep["cohort"] == "Combined_Test_May14_15") & (df_indep["horizon_step"] == h) & (df_indep["model_name"] == "Persistence_Baseline")]
        sub_c = df_indep[(df_indep["cohort"] == "Combined_Test_May14_15") & (df_indep["horizon_step"] == h) & (df_indep["model_name"] == "Phase4S_CNN_LSTM_May2024")]
        print(f"    Horizon t+{h}s:")
        print(f"      Persistence MAE: {sub_p['recomputed_mae'].values[0]} c/s (Reported: {sub_p['reported_mae'].values[0]}) -> Diff: {sub_p['mae_discrepancy'].values[0]}")
        print(f"      CNN-LSTM    MAE: {sub_c['recomputed_mae'].values[0]} c/s (Reported: {sub_c['reported_mae'].values[0]}) -> Diff: {sub_c['mae_discrepancy'].values[0]}")

    return df_indep, h10_df


# ==============================================================================
# STAGE 4: Time-Aware Block Bootstrap & Stability Analysis
# ==============================================================================
def stage4_block_bootstrap_and_causal_audit(df_indep: pd.DataFrame) -> Dict[str, Any]:
    print("\n--- STAGE 4: Time-Aware Block Bootstrap & Stability Analysis ---")
    data = np.load(ART_4T_DIR / "phase4t_test_predictions.npz")
    y_true = data["y_true"]
    preds_p4s = data["preds_p4s"]
    preds_persist = data["preds_persist"]
    dates = data["dates"]
    X_last = data["X_last"]

    # Target-conditioned vs Origin-conditioned Moderate Cohorts at h=10
    h_idx = 9  # horizon 10
    y_10 = y_true[:, h_idx]
    p_cnn_10 = preds_p4s[:, h_idx]
    p_per_10 = preds_persist[:, h_idx]

    # Target-conditioned mask (Phase 4S definition: max future y in [200, 1000])
    target_max = np.max(y_true, axis=1)
    mask_target_mod = (target_max >= 200.0) & (target_max <= 1000.0)
    
    # Causal origin-conditioned mask (origin count rate in [200, 1000])
    mask_origin_mod = (X_last >= 200.0) & (X_last <= 1000.0)

    # Evaluate across days for target-conditioned moderate
    for d in ["20240514", "20240515"]:
        d_mask = mask_target_mod & (dates == d)
        mae_c = float(np.mean(np.abs(y_10[d_mask] - p_cnn_10[d_mask])))
        mae_p = float(np.mean(np.abs(y_10[d_mask] - p_per_10[d_mask])))
        skill = (1.0 - mae_c / mae_p) * 100.0
        print(f"  Target-Conditioned Moderate on {d} (N={d_mask.sum():,}): CNN={mae_c:.2f}, Persist={mae_p:.2f}, Skill={skill:+.2f}%")

    # Evaluate causal origin-conditioned moderate across days
    for d in ["20240514", "20240515"]:
        d_mask = mask_origin_mod & (dates == d)
        mae_c = float(np.mean(np.abs(y_10[d_mask] - p_cnn_10[d_mask])))
        mae_p = float(np.mean(np.abs(y_10[d_mask] - p_per_10[d_mask])))
        skill = (1.0 - mae_c / mae_p) * 100.0
        print(f"  Causal Origin-Conditioned Moderate on {d} (N={d_mask.sum():,}): CNN={mae_c:.2f}, Persist={mae_p:.2f}, Skill={skill:+.2f}%")

    # Time-Aware Block Bootstrap (Block length = 300 seconds to respect autocorrelation)
    print("\n  Executing Block Bootstrap (B=500 iterations, Block Size=300 seconds):")
    block_size = 300
    N = len(y_10)
    n_blocks = N // block_size
    
    # Pre-aggregate blocks for target-conditioned moderate
    block_indices = [np.arange(b * block_size, min((b + 1) * block_size, N)) for b in range(n_blocks)]
    
    rng = np.random.RandomState(42)
    B = 500
    boot_skills_target_mod = []
    boot_skills_causal_mod = []
    boot_skills_combined = []

    for _ in range(B):
        sample_block_ids = rng.choice(n_blocks, size=n_blocks, replace=True)
        sample_indices = np.concatenate([block_indices[b] for b in sample_block_ids])
        
        # Combined test
        mae_c_comb = np.mean(np.abs(y_10[sample_indices] - p_cnn_10[sample_indices]))
        mae_p_comb = np.mean(np.abs(y_10[sample_indices] - p_per_10[sample_indices]))
        boot_skills_combined.append((1.0 - mae_c_comb / mae_p_comb) * 100.0)

        # Target mod
        m_tm = mask_target_mod[sample_indices]
        if m_tm.sum() > 50:
            mc = np.mean(np.abs(y_10[sample_indices][m_tm] - p_cnn_10[sample_indices][m_tm]))
            mp = np.mean(np.abs(y_10[sample_indices][m_tm] - p_per_10[sample_indices][m_tm]))
            boot_skills_target_mod.append((1.0 - mc / mp) * 100.0)

        # Causal mod
        m_cm = mask_origin_mod[sample_indices]
        if m_cm.sum() > 50:
            mc = np.mean(np.abs(y_10[sample_indices][m_cm] - p_cnn_10[sample_indices][m_cm]))
            mp = np.mean(np.abs(y_10[sample_indices][m_cm] - p_per_10[sample_indices][m_cm]))
            boot_skills_causal_mod.append((1.0 - mc / mp) * 100.0)

    comb_ci = np.percentile(boot_skills_combined, [2.5, 50.0, 97.5])
    tmod_ci = np.percentile(boot_skills_target_mod, [2.5, 50.0, 97.5])
    cmod_ci = np.percentile(boot_skills_causal_mod, [2.5, 50.0, 97.5])

    print(f"    Combined Test Skill 95% CI:             [{comb_ci[0]:.2f}%, {comb_ci[1]:.2f}%, {comb_ci[2]:.2f}%]")
    print(f"    Target-Conditioned Moderate 95% CI:     [{tmod_ci[0]:.2f}%, {tmod_ci[1]:.2f}%, {tmod_ci[2]:.2f}%]")
    print(f"    Causal Origin-Conditioned Moderate 95% CI:[{cmod_ci[0]:.2f}%, {cmod_ci[1]:.2f}%, {cmod_ci[2]:.2f}%]")

    bootstrap_results = {
        "block_size_seconds": block_size,
        "n_blocks": n_blocks,
        "bootstrap_iterations": B,
        "combined_test_skill_median": round(float(comb_ci[1]), 2),
        "combined_test_skill_95ci": [round(float(comb_ci[0]), 2), round(float(comb_ci[2]), 2)],
        "target_mod_skill_median": round(float(tmod_ci[1]), 2),
        "target_mod_skill_95ci": [round(float(tmod_ci[0]), 2), round(float(tmod_ci[2]), 2)],
        "causal_mod_skill_median": round(float(cmod_ci[1]), 2),
        "causal_mod_skill_95ci": [round(float(cmod_ci[0]), 2), round(float(cmod_ci[2]), 2)],
    }
    return bootstrap_results


def main():
    print("=" * 80)
    print("Phase 4T: Independent Reproducibility Audit & Forecasting-Skill Verification")
    print("=" * 80)

    t0 = time.time()
    inv = stage1_artifact_inventory()
    df_recon = stage2_dataset_and_window_reconciliation()
    df_indep, df_h10 = stage3_reproduce_metrics()
    boot_res = stage4_block_bootstrap_and_causal_audit(df_indep)

    print(f"\nAudit completed in {time.time() - t0:.2f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
