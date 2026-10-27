"""Phase 4S: Controlled Chronological Training and Evaluation on May 2024 SoLEXS Observations.

Pipeline stages:
1. Verify May 2024 archive integrity and gate clearance.
2. Ingest and segment genuine SoLEXS SDD2 Level-1 light curves for May 8-15, 2024.
3. Form chronological partitions:
   - Training: May 8-12, 2024 (5 days)
   - Validation: May 13, 2024 (1 day)
   - Final Test: May 14-15, 2024 (2 days, quarantined)
4. Fit CountRateScaler strictly on training partition.
5. Train direct-forecasting SelfSupervisedCNNLSTM model.
6. Evaluate against Persistence and Rolling Mean baselines on identical test targets.
7. Perform transfer evaluation of Phase 4O reference model.
8. Stratify results across regimes (Quiet, Mod, High) and rising-rate intervals.
9. Save checkpoints, metadata, and comprehensive CSV metrics.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from astropy.io import fits

from ml.experiments.real_self_supervised import (
    validate_solexs_fits_table,
    extract_contiguous_segments,
    create_sliding_windows,
    CountRateScaler,
    SelfSupervisedCNNLSTM,
    persistence_forecast,
    rolling_mean_forecast,
    evaluate_forecast,
)
from scripts.phase4s_training_gate import evaluate_phase4s_gate, EXPECTED_ARCHIVE_PATH, EXPECTED_ARCHIVE_SHA256


@dataclass
class Phase4SConfig:
    sequence_length: int = 60          # 60s input history
    forecast_horizon: int = 10         # 10s forecast horizon
    batch_size: int = 256
    epochs: int = 5
    learning_rate: float = 1e-3
    random_seed: int = 42
    device: str = "cpu"
    train_dates: List[str] = None
    val_dates: List[str] = None
    test_dates: List[str] = None

    def __post_init__(self):
        if self.train_dates is None:
            self.train_dates = ["20240508", "20240509", "20240510", "20240511", "20240512"]
        if self.val_dates is None:
            self.val_dates = ["20240513"]
        if self.test_dates is None:
            self.test_dates = ["20240514", "20240515"]


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_partition_data(archive_path: Path, dates: List[str], seq_len: int = 60, horizon: int = 10):
    """Load and extract contiguous sequence windows for specific observation dates."""
    records_by_date = {}
    with zipfile.ZipFile(archive_path) as outer:
        for d in dates:
            inner_name = f"AL1_SLX_L1_{d}_v1.0.zip"
            inner_bytes = outer.read(inner_name)
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                lc_member = f"AL1_SLX_L1_{d}_v1.0/SDD2/AL1_SOLEXS_{d}_SDD2_L1.lc.gz"
                raw_gz = inner.read(lc_member)
                decomp = gzip.decompress(raw_gz)
                with fits.open(io.BytesIO(decomp)) as hdul:
                    times, counts, tu, cu = validate_solexs_fits_table(hdul)
                    records_by_date[d] = (times, counts)

    # Segment each date and generate windows
    all_X, all_y, all_t, all_dates = [], [], [], []
    date_windows_map = {}

    for d in dates:
        times, counts = records_by_date[d]
        segs = extract_contiguous_segments(times, counts, min_segment_length=seq_len + horizon)
        X_d, y_d, t_d = create_sliding_windows(segs, seq_len, horizon)
        date_windows_map[d] = (X_d, y_d, t_d)
        if len(X_d) > 0:
            all_X.append(X_d)
            all_y.append(y_d)
            all_t.append(t_d)
            all_dates.extend([d] * len(X_d))

    if all_X:
        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)
        t_combined = np.concatenate(all_t, axis=0)
    else:
        X_combined = np.empty((0, seq_len, 1), dtype=np.float32)
        y_combined = np.empty((0, horizon), dtype=np.float32)
        t_combined = np.empty((0,), dtype=np.float64)

    return X_combined, y_combined, t_combined, all_dates, records_by_date, date_windows_map


def compute_metrics_for_cohort(y_true: np.ndarray, y_pred: np.ndarray, model_name: str,
                               cohort_name: str, horizon: int = 10) -> List[Dict[str, Any]]:
    """Compute per-horizon metrics (MAE, RMSE, RelError)."""
    rows = []
    mean_target = float(np.mean(y_true)) if len(y_true) else 1.0
    for h in range(1, horizon + 1):
        idx = h - 1
        true_h = y_true[:, idx]
        pred_h = y_pred[:, idx]
        mae = float(np.mean(np.abs(true_h - pred_h)))
        rmse = float(np.sqrt(np.mean((true_h - pred_h) ** 2)))
        rel_mae = float(mae / mean_target * 100.0) if mean_target > 0 else 0.0

        rows.append({
            "model_name": model_name,
            "cohort": cohort_name,
            "horizon_step": h,
            "horizon_seconds": h,
            "sample_count": len(true_h),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "rel_mae_pct": round(rel_mae, 2),
            "target_mean": round(mean_target, 2),
        })
    return rows


def run_experiment():
    print("=" * 80)
    print("Phase 4S: Controlled May 2024 Real-Data Ingestion & Forecasting Evaluation")
    print("=" * 80)

    cfg = Phase4SConfig()
    set_seed(cfg.random_seed)

    # 1. Gate Evaluation
    manifest_path = backend_dir / "reports" / "phase4s_dataset_manifest.json"
    gate_res = evaluate_phase4s_gate(manifest_path, EXPECTED_ARCHIVE_PATH)
    print(f"Data Readiness Gate Status: {gate_res['status']}")
    if not gate_res["passed"]:
        print(f"ERROR: Training Gate Failed: {gate_res['reason']}")
        return 1

    print("Gate Passed. Ingesting May 2024 observations across chronological splits...")

    # 2. Ingest Data
    t0 = time.time()
    X_train, y_train, t_train, _, train_raw, _ = load_partition_data(
        EXPECTED_ARCHIVE_PATH, cfg.train_dates, cfg.sequence_length, cfg.forecast_horizon
    )
    X_val, y_val, t_val, _, val_raw, _ = load_partition_data(
        EXPECTED_ARCHIVE_PATH, cfg.val_dates, cfg.sequence_length, cfg.forecast_horizon
    )
    X_test, y_test, t_test, test_dates_list, test_raw, test_by_date = load_partition_data(
        EXPECTED_ARCHIVE_PATH, cfg.test_dates, cfg.sequence_length, cfg.forecast_horizon
    )

    print(f"   Training Partition   (May 8-12): {len(X_train):,} sliding windows")
    print(f"   Validation Partition (May 13)  : {len(X_val):,} sliding windows")
    print(f"   Final Test Partition (May 14-15): {len(X_test):,} sliding windows")
    print(f"   Ingestion completed in {time.time() - t0:.2f}s")

    # 3. Fit Scaler Strictly on Training Partition
    print("\nFitting CountRateScaler strictly on Training Partition (May 8-12)...")
    train_counts_all = np.concatenate([train_raw[d][1] for d in cfg.train_dates])
    scaler = CountRateScaler().fit(train_counts_all)
    print(f"   Scaler parameters: mean={scaler.mean:.4f}, std={scaler.std:.4f}")

    # Scale Inputs & Targets
    N_tr = len(X_train)
    X_train_scaled = scaler.transform(X_train.reshape(-1)).reshape(N_tr, cfg.sequence_length, 1)
    y_train_scaled = scaler.transform(y_train.reshape(-1)).reshape(N_tr, cfg.forecast_horizon)

    N_val = len(X_val)
    X_val_scaled = scaler.transform(X_val.reshape(-1)).reshape(N_val, cfg.sequence_length, 1)
    y_val_scaled = scaler.transform(y_val.reshape(-1)).reshape(N_val, cfg.forecast_horizon)

    # 4. Initialize & Train CNN-LSTM
    print("\nTraining direct-forecasting SelfSupervisedCNNLSTM...")
    model = SelfSupervisedCNNLSTM(
        in_features=1,
        horizon=cfg.forecast_horizon,
        cnn_channels=32,
        lstm_hidden=64,
        lstm_layers=2,
        dropout=0.2
    ).to(cfg.device)

    art_dir = backend_dir / "artifacts" / "real_data_experiments" / "phase4s"
    art_dir.mkdir(parents=True, exist_ok=True)
    existing_ckpts = sorted(art_dir.glob("solexs_phase4s_cnn_lstm_may2024_*.pt"))

    if existing_ckpts:
        ckpt_path = existing_ckpts[-1]
        print(f"\nFound existing trained checkpoint: {ckpt_path.name}")
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device, weights_only=True))
        model.eval()
        ckpt_sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        print(f"Loaded weights successfully (SHA-256: {ckpt_sha256})")
    else:
        train_ds = TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(y_train_scaled))
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

        val_ds = TensorDataset(torch.from_numpy(X_val_scaled), torch.from_numpy(y_val_scaled))
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        best_val_loss = float("inf")
        best_weights = None
        train_history = []

        t_train_start = time.time()
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running_loss = 0.0
            n_batches = 0
            for bx, by in train_loader:
                bx, by = bx.to(cfg.device), by.to(cfg.device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1
            epoch_tr_loss = running_loss / max(n_batches, 1)

            # Validation loss
            model.eval()
            running_val = 0.0
            n_v_batches = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(cfg.device), by.to(cfg.device)
                    out = model(bx)
                    loss = criterion(out, by)
                    running_val += loss.item()
                    n_v_batches += 1
            epoch_val_loss = running_val / max(n_v_batches, 1)

            train_history.append({"epoch": epoch, "train_loss": epoch_tr_loss, "val_loss": epoch_val_loss})
            print(f"   Epoch {epoch}/{cfg.epochs} - Train Loss: {epoch_tr_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        train_duration = time.time() - t_train_start
        print(f"Training completed in {train_duration:.2f}s. Restoring best weights (Val Loss: {best_val_loss:.6f})...")
        model.load_state_dict(best_weights)
        model.eval()

        # 5. Save Checkpoint & Metadata
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        ckpt_path = art_dir / f"solexs_phase4s_cnn_lstm_may2024_{timestamp_str}.pt"
        torch.save(model.state_dict(), ckpt_path)
        ckpt_sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

        meta_path = art_dir / f"solexs_phase4s_cnn_lstm_may2024_{timestamp_str}_metadata.json"
        metadata = {
            "model_architecture": "SelfSupervisedCNNLSTM",
            "task": "native_count_rate_forecasting_may2024",
            "instrument": "SoLEXS",
            "detector": "SDD2",
            "measurement_column": "COUNTS",
            "measurement_units": "counts/s",
            "energy_band_kev": "2-22",
            "input_sequence_length_seconds": cfg.sequence_length,
            "forecast_horizon_seconds": cfg.forecast_horizon,
            "source_archive": EXPECTED_ARCHIVE_PATH.name,
            "source_archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "training_dates": cfg.train_dates,
            "validation_dates": cfg.val_dates,
            "final_test_dates": cfg.test_dates,
            "scaler_mean": scaler.mean,
            "scaler_std": scaler.std,
            "best_val_loss_scaled_mse": best_val_loss,
            "training_duration_seconds": round(train_duration, 2),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": ckpt_sha256,
            "production_registry_eligible": False,
            "experimental_marker": "PHASE4S_MAY2024_SOLAR_MAX_RESEARCH_ONLY",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved Checkpoint: {ckpt_path.name} (SHA-256: {ckpt_sha256})")

    # 6. Evaluation on Held-Out Test Set (May 14-15, 2024)
    print("\nEvaluating models on Quarantined Final Test Partition (May 14-15)...")

    # Predict with trained CNN-LSTM
    N_test = len(X_test)
    X_test_scaled = scaler.transform(X_test.reshape(-1)).reshape(N_test, cfg.sequence_length, 1)

    preds_scaled = []
    with torch.no_grad():
        for i in range(0, N_test, cfg.batch_size):
            bx = torch.from_numpy(X_test_scaled[i : i + cfg.batch_size]).to(cfg.device)
            out = model(bx)
            preds_scaled.append(out.cpu().numpy())
    preds_scaled = np.concatenate(preds_scaled, axis=0)
    preds_counts = scaler.inverse_transform(preds_scaled)

    # Baselines
    preds_persist = persistence_forecast(X_test, cfg.forecast_horizon)
    preds_rolling = rolling_mean_forecast(X_test, cfg.forecast_horizon)

    # Phase 4O reference model transfer evaluation
    p4o_ckpt_path = backend_dir / "artifacts" / "real_data_experiments" / "phase4o" / "solexs_phase4o_cnn_lstm_direct_20261025T184851Z.pt"
    p4o_meta_path = backend_dir / "artifacts" / "real_data_experiments" / "phase4o" / "solexs_phase4o_cnn_lstm_direct_20261025T184851Z_metadata.json"
    
    preds_p4o_transfer = None
    if p4o_ckpt_path.exists() and p4o_meta_path.exists():
        with open(p4o_meta_path, "r", encoding="utf-8") as f:
            p4o_meta = json.load(f)
        scaler_p = p4o_meta.get("scaler_parameters", {})
        p4o_scaler = CountRateScaler()
        p4o_scaler.mean = float(scaler_p.get("mean", p4o_meta.get("scaler_mean", 2.247274921086125)))
        p4o_scaler.std = float(scaler_p.get("std", p4o_meta.get("scaler_std", 0.6966391869930155)))
        p4o_scaler.fitted = True

        p4o_model = SelfSupervisedCNNLSTM(1, 10, 32, 64, 2, 0.2)
        p4o_model.load_state_dict(torch.load(p4o_ckpt_path, map_location="cpu", weights_only=True))
        p4o_model.eval()

        X_test_p4o_scaled = p4o_scaler.transform(X_test.reshape(-1)).reshape(N_test, cfg.sequence_length, 1)
        p4o_preds_scaled = []
        with torch.no_grad():
            for i in range(0, N_test, cfg.batch_size):
                bx = torch.from_numpy(X_test_p4o_scaled[i : i + cfg.batch_size])
                p4o_preds_scaled.append(p4o_model(bx).numpy())
        p4o_preds_scaled = np.concatenate(p4o_preds_scaled, axis=0)
        preds_p4o_transfer = p4o_scaler.inverse_transform(p4o_preds_scaled)

    # Compile Evaluation Cohorts
    cohorts_eval = {
        "Combined_Test_May14_15": np.ones(N_test, dtype=bool),
        "Day7_May14_X8_7_Flare": (np.array(test_dates_list) == "20240514"),
        "Day8_May15_Active_Decay": (np.array(test_dates_list) == "20240515"),
    }

    # Add Regime Cohorts
    peak_target = np.max(y_test, axis=1)
    cohorts_eval["Regime_REG_QUIET_lt200"] = (peak_target < 200.0)
    cohorts_eval["Regime_REG_MOD_200_1000"] = (peak_target >= 200.0) & (peak_target <= 1000.0)
    cohorts_eval["Regime_REG_HIGH_gt1000"] = (peak_target > 1000.0)

    # Add Rising Trend Cohort (delta c > 50 c/s in last 10s of input)
    input_trend = X_test[:, -1, 0] - X_test[:, -10, 0]
    cohorts_eval["Rising_Surge_DeltaC_gt50"] = (input_trend > 50.0)

    all_metric_rows = []
    for c_name, mask in cohorts_eval.items():
        if not np.any(mask):
            continue
        y_c = y_test[mask]
        p_cnn = preds_counts[mask]
        p_per = preds_persist[mask]
        p_rol = preds_rolling[mask]

        rows_cnn = compute_metrics_for_cohort(y_c, p_cnn, "Phase4S_CNN_LSTM_May2024", c_name)
        rows_per = compute_metrics_for_cohort(y_c, p_per, "Persistence_Baseline", c_name)
        rows_rol = compute_metrics_for_cohort(y_c, p_rol, "Rolling_Mean_Baseline", c_name)

        # Compute skill score
        for r_c, r_p, r_r in zip(rows_cnn, rows_per, rows_rol):
            r_c["skill_vs_persistence_pct"] = round((1.0 - r_c["mae"] / max(r_p["mae"], 1e-6)) * 100.0, 2)
            r_c["skill_vs_rolling_mean_pct"] = round((1.0 - r_c["mae"] / max(r_r["mae"], 1e-6)) * 100.0, 2)
            r_p["skill_vs_persistence_pct"] = 0.0
            r_p["skill_vs_rolling_mean_pct"] = round((1.0 - r_p["mae"] / max(r_r["mae"], 1e-6)) * 100.0, 2)
            r_r["skill_vs_persistence_pct"] = round((1.0 - r_r["mae"] / max(r_p["mae"], 1e-6)) * 100.0, 2)
            r_r["skill_vs_rolling_mean_pct"] = 0.0

        all_metric_rows.extend(rows_cnn)
        all_metric_rows.extend(rows_per)
        all_metric_rows.extend(rows_rol)

        if preds_p4o_transfer is not None:
            p_p4o = preds_p4o_transfer[mask]
            rows_p4o = compute_metrics_for_cohort(y_c, p_p4o, "Phase4O_CNN_LSTM_Transfer", c_name)
            for r_t, r_p, r_r in zip(rows_p4o, rows_per, rows_rol):
                r_t["skill_vs_persistence_pct"] = round((1.0 - r_t["mae"] / max(r_p["mae"], 1e-6)) * 100.0, 2)
                r_t["skill_vs_rolling_mean_pct"] = round((1.0 - r_t["mae"] / max(r_r["mae"], 1e-6)) * 100.0, 2)
            all_metric_rows.extend(rows_p4o)

    df_metrics = pd.DataFrame(all_metric_rows)
    csv_report_path = backend_dir / "reports" / "phase4s_metrics.csv"
    csv_art_path = art_dir / "phase4s_metrics.csv"
    df_metrics.to_csv(csv_report_path, index=False)
    df_metrics.to_csv(csv_art_path, index=False)
    print(f"\nSaved Comprehensive Metrics to: {csv_report_path.name}")

    # Summary Print
    print("\n" + "=" * 80)
    print("PHASE 4S EXPERIMENTAL BENCHMARK SUMMARY (Combined Test: May 14-15, 2024)")
    print("=" * 80)
    df_sub = df_metrics[(df_metrics["cohort"] == "Combined_Test_May14_15") & (df_metrics["horizon_step"].isin([1, 5, 10]))]
    pivot_mae = df_sub.pivot(index="horizon_step", columns="model_name", values="mae")
    pivot_skill = df_sub.pivot(index="horizon_step", columns="model_name", values="skill_vs_persistence_pct")
    print("MAE (counts/s):")
    print(pivot_mae)
    print("\nSkill Score vs Persistence (%):")
    print(pivot_skill)
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(run_experiment())
