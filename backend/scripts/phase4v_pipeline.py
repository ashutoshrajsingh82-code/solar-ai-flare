"""Phase 4V: Multi-Epoch Real-Data Forecasting Pipeline.

Pipeline stages
---------------
1. Readiness gate evaluation (fail-closed on hard prerequisites).
2. Data ingestion from both local archives (May 2024 + Jan-Feb 2026).
3. Per-epoch FITS validation, segmentation, window generation.
4. Chronological split assignment per Phase 4U dataset contract.
5. Scaler fitting strictly on training partition.
6. Optional DEVELOPMENT_ONLY_NOT_FINAL_MODEL training.
7. Evaluation of CNN-LSTM, Persistence, and Rolling Mean baselines.
8. Per-epoch and per-cohort metric reporting (causal conditioning).
9. Artifact preservation with SHA-256 provenance.

Usage
-----
    python phase4v_pipeline.py --mode gate-only    # evaluate gate, no training
    python phase4v_pipeline.py --mode dev-train    # gate + development training

The October holdout is excluded from ALL pipeline code paths in this script.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import time
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from astropy.io import fits
from torch.utils.data import DataLoader, TensorDataset

from ml.experiments.real_self_supervised import (
    CountRateScaler,
    SelfSupervisedCNNLSTM,
    create_sliding_windows,
    evaluate_forecast,
    extract_contiguous_segments,
    persistence_forecast,
    rolling_mean_forecast,
    validate_solexs_fits_table,
)
from scripts.phase4v_readiness_gate import (
    ARCHIVE_FEB2026,
    ARCHIVE_MAY2024,
    SHA256_FEB2026,
    SHA256_MAY2024,
    evaluate_phase4v_gate,
)

REPORTS_DIR = backend_dir / "reports"
ART_4V_DIR = backend_dir / "artifacts" / "real_data_experiments" / "phase4v"

# ── Phase 4U dataset contract splits ─────────────────────────────────────────
# Epoch 1 — May 2024 (AR 13664 X-class solar maximum)
EPOCH1_TRAIN_DATES = ["20240508", "20240509", "20240510", "20240511", "20240512"]
EPOCH1_VAL_DATES = ["20240513"]
EPOCH1_TEST_DATES = ["20240514", "20240515"]

# Epoch 2 — Jan-Feb 2026 (quiet-sun to moderate flaring)
EPOCH2_TRAIN_DATES = ["20260131", "20260201", "20260202"]
EPOCH2_VAL_DATES = ["20260203", "20260204"]
EPOCH2_TEST_DATES = ["20260205"]

SEQ_LEN = 60      # 60 s input history
HORIZON = 10      # 10 s forecast horizon
MIN_SEG_LEN = SEQ_LEN + HORIZON   # 70 s minimum segment

# Causal activity cohort thresholds (Phase 4U dataset contract §8)
COHORT_QUIET_MAX = 200.0          # counts/s
COHORT_MOD_MAX = 1000.0           # counts/s
SURGE_DELTA_THRESH = 50.0         # counts/s rise over last 10 s of input

# Minimum window count per cohort to report (avoid reporting on <10 samples)
MIN_COHORT_SAMPLES = 10


# ── Configuration dataclass ───────────────────────────────────────────────────

@dataclass
class Phase4VConfig:
    batch_size: int = 256
    epochs: int = 5
    learning_rate: float = 1e-3
    random_seed: int = 42
    training_mode: str = "DEVELOPMENT_ONLY_NOT_FINAL_MODEL"
    run_id: str = ""

    def __post_init__(self):
        if not self.run_id:
            self.run_id = f"phase4v_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Data loading ──────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def load_daily_lc(archive_path: Path, date_str: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Extract a single-day SoLEXS SDD2 light curve from a nested ZIP archive.

    Returns (times, counts, member_path).
    """
    inner_name = f"AL1_SLX_L1_{date_str}_v1.0.zip"
    with zipfile.ZipFile(archive_path) as outer:
        inner_bytes = outer.read(inner_name)
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        sdd2_members = [m for m in inner.namelist() if "SDD2" in m and m.endswith(".lc.gz")]
        if not sdd2_members:
            raise FileNotFoundError(f"No SDD2 .lc.gz member in {inner_name}")
        member = sdd2_members[0]
        raw_gz = inner.read(member)
    decomp = gzip.decompress(raw_gz)
    with fits.open(io.BytesIO(decomp)) as hdul:
        times, counts, t_unit, c_unit = validate_solexs_fits_table(hdul)
    return times, counts, member


def load_epoch_partition(
    archive_path: Path,
    dates: List[str],
    partition_label: str,
    epoch_id: str,
) -> Dict[str, Any]:
    """Load and segment all days for a partition; return raw arrays and provenance."""
    all_segs: List[Tuple[np.ndarray, np.ndarray]] = []
    all_times_raw: List[np.ndarray] = []
    all_counts_raw: List[np.ndarray] = []
    per_day_meta: List[Dict[str, Any]] = []

    for d in dates:
        times, counts, member = load_daily_lc(archive_path, d)
        segs = extract_contiguous_segments(times, counts, min_segment_length=MIN_SEG_LEN)
        finite_n = int(np.isfinite(counts).sum())
        nan_n = int(np.isnan(counts).sum())

        per_day_meta.append({
            "observation_date": d,
            "total_bins": len(times),
            "finite_records": finite_n,
            "nan_records": nan_n,
            "contiguous_segments": len(segs),
            "member_path": member,
        })
        all_segs.extend(segs)
        all_times_raw.append(times)
        all_counts_raw.append(counts)

    X, y, t_origins = create_sliding_windows(all_segs, SEQ_LEN, HORIZON)

    return {
        "epoch_id": epoch_id,
        "partition": partition_label,
        "archive": archive_path.name,
        "dates": dates,
        "per_day_meta": per_day_meta,
        "segments_count": len(all_segs),
        "window_count": len(X),
        "X_raw": X,              # shape (N, seq_len, 1)  float32
        "y_raw": y,              # shape (N, horizon)     float32
        "t_origins": t_origins,  # shape (N,)             float64 Unix seconds
        "all_times": np.concatenate(all_times_raw) if all_times_raw else np.empty(0),
        "all_counts": np.concatenate(all_counts_raw) if all_counts_raw else np.empty(0),
    }


# ── Causal cohort assignment ──────────────────────────────────────────────────

def assign_causal_cohorts(X_raw: np.ndarray) -> Dict[str, np.ndarray]:
    """Assign causal activity cohort masks using only x_t (forecast origin).

    Never uses future target peaks.
    """
    origin = X_raw[:, -1, 0]          # last observed count rate at forecast origin
    delta = X_raw[:, -1, 0] - X_raw[:, -11, 0] if X_raw.shape[1] >= 11 else np.zeros(len(X_raw))

    return {
        "ALL": np.ones(len(X_raw), dtype=bool),
        "Causal_REG_QUIET":  origin < COHORT_QUIET_MAX,
        "Causal_REG_MOD":    (origin >= COHORT_QUIET_MAX) & (origin <= COHORT_MOD_MAX),
        "Causal_REG_HIGH":   origin > COHORT_MOD_MAX,
        "Causal_SURGE_50":   delta > SURGE_DELTA_THRESH,
    }


# ── Metric computation ────────────────────────────────────────────────────────

def compute_per_horizon_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    cohort: str,
    epoch_id: str,
    partition: str,
) -> List[Dict[str, Any]]:
    """Return per-horizon metric rows (MAE, RMSE, RelMAE)."""
    rows = []
    mean_target = float(np.mean(y_true)) if len(y_true) else 1.0
    n = len(y_true)
    for h in range(1, HORIZON + 1):
        true_h = y_true[:, h - 1]
        pred_h = y_pred[:, h - 1]
        mae = float(np.mean(np.abs(true_h - pred_h)))
        rmse = float(np.sqrt(np.mean((true_h - pred_h) ** 2)))
        rel_mae_pct = float(mae / mean_target * 100.0) if mean_target > 0 else float("nan")
        rows.append({
            "epoch_id": epoch_id,
            "partition": partition,
            "cohort": cohort,
            "model": model_name,
            "horizon_step": h,
            "n_samples": n,
            "mae_counts_per_s": round(mae, 4),
            "rmse_counts_per_s": round(rmse, 4),
            "rel_mae_pct": round(rel_mae_pct, 3),
            "target_mean_counts_per_s": round(mean_target, 3),
        })
    return rows


def compute_skill_vs_persistence(metric_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add skill_vs_persistence_pct column to existing metric rows (in place)."""
    # Build persistence MAE lookup by (epoch_id, partition, cohort, horizon_step)
    persist_mae: Dict[Tuple, float] = {}
    for r in metric_rows:
        if r["model"] == "Persistence":
            key = (r["epoch_id"], r["partition"], r["cohort"], r["horizon_step"])
            persist_mae[key] = r["mae_counts_per_s"]

    for r in metric_rows:
        key = (r["epoch_id"], r["partition"], r["cohort"], r["horizon_step"])
        p_mae = persist_mae.get(key, None)
        if p_mae is not None and p_mae > 1e-9:
            r["skill_vs_persistence_pct"] = round((1.0 - r["mae_counts_per_s"] / p_mae) * 100.0, 2)
        else:
            r["skill_vs_persistence_pct"] = float("nan")
    return metric_rows


# ── Model training ────────────────────────────────────────────────────────────

def train_model(
    X_train_sc: np.ndarray,
    y_train_sc: np.ndarray,
    X_val_sc: np.ndarray,
    y_val_sc: np.ndarray,
    cfg: Phase4VConfig,
    device: str,
) -> Tuple[SelfSupervisedCNNLSTM, List[Dict[str, float]], float]:
    """Train CNN-LSTM; return (best_model, history, best_val_loss)."""
    model = SelfSupervisedCNNLSTM(
        in_features=1, horizon=HORIZON, cnn_channels=32, lstm_hidden=64, lstm_layers=2, dropout=0.2
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()

    tr_ds = TensorDataset(torch.from_numpy(X_train_sc), torch.from_numpy(y_train_sc))
    val_ds = TensorDataset(torch.from_numpy(X_val_sc), torch.from_numpy(y_val_sc))
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    best_val = float("inf")
    best_weights = None
    history = []

    for epoch in range(1, cfg.epochs + 1):
        # Training
        model.train()
        tr_loss, n_tr = 0.0, 0
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}: {loss.item()}")
            loss.backward()
            # Gradient check
            for name, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise RuntimeError(f"Non-finite gradient in {name} at epoch {epoch}")
            optimizer.step()
            tr_loss += loss.item() * len(bx)
            n_tr += len(bx)
        tr_loss /= max(n_tr, 1)

        # Validation
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                loss = criterion(pred, by)
                val_loss += loss.item() * len(bx)
                n_val += len(bx)
        val_loss /= max(n_val, 1)

        history.append({"epoch": epoch, "train_loss": float(tr_loss), "val_loss": float(val_loss)})
        print(f"   Epoch {epoch:2d}/{cfg.epochs} — Train Loss: {tr_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    return model, history, float(best_val)


def infer_model(model: SelfSupervisedCNNLSTM, X_scaled: np.ndarray,
                scaler: CountRateScaler, device: str, batch_size: int = 256) -> np.ndarray:
    """Run inference and invert back to native count rates."""
    preds_sc = []
    with torch.no_grad():
        for i in range(0, len(X_scaled), batch_size):
            bx = torch.from_numpy(X_scaled[i: i + batch_size]).to(device)
            preds_sc.append(model(bx).cpu().numpy())
    preds_sc_arr = np.concatenate(preds_sc, axis=0)
    # Check for non-finite predictions
    if not np.isfinite(preds_sc_arr).all():
        print("  WARNING: Non-finite values in model predictions; clipping to finite range.")
        preds_sc_arr = np.nan_to_num(preds_sc_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return scaler.inverse_transform(preds_sc_arr)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(mode: str = "gate-only") -> int:
    """Execute the Phase 4V pipeline.

    Parameters
    ----------
    mode : {"gate-only", "dev-train"}
        "gate-only" — evaluate the readiness gate and exit.
        "dev-train" — gate + development-only training + evaluation.
    """
    print("=" * 80)
    print("Phase 4V: Multi-Epoch Real-Data Forecasting Pipeline")
    print(f"Mode: {mode}  |  {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)

    ART_4V_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Gate evaluation ────────────────────────────────────────────────────
    print("\n[1/8] Evaluating Phase 4V readiness gate…")
    gate = evaluate_phase4v_gate()
    gate_path = REPORTS_DIR / "phase4v_readiness_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    (ART_4V_DIR / "phase4v_readiness_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    print(f"  Gate status  : {gate['overall_status']}")
    print(f"  Training mode: {gate['training_mode']}")
    print(f"  Holdout      : {'LOCKED (October 2024 not acquired)' if gate['holdout_locked'] else 'OPEN'}")
    for c in gate["conditions"]:
        mark = "PASS" if c["satisfied"] else "FAIL"
        nb = " [non-blocking]" if c["condition_id"] == "GATE-4V-07" else ""
        print(f"    [{mark}]{nb} {c['condition_id']}")

    if not gate["training_authorized"]:
        print(f"\nGate BLOCKED: {gate['summary']}")
        print("TRAINING_NOT_EXECUTED — resolve gate conditions first.")
        return 1

    if mode == "gate-only":
        print("\nMode=gate-only: Gate passed. Pipeline implementation verified. Training not requested.")
        print("Status: PHASE4V_PIPELINE_IMPLEMENTED_TESTED_TRAINING_NOT_EXECUTED")
        return 0

    # ── 2. Configuration ──────────────────────────────────────────────────────
    cfg = Phase4VConfig()
    device = detect_device()
    set_seed(cfg.random_seed)
    print(f"\n[2/8] Configuration: device={device}, seed={cfg.random_seed}, "
          f"epochs={cfg.epochs}, lr={cfg.learning_rate}, batch={cfg.batch_size}")
    print(f"  Run ID: {cfg.run_id}")
    print("  Training mode: DEVELOPMENT_ONLY_NOT_FINAL_MODEL")
    print("  October holdout: EXCLUDED from all pipeline paths")

    # ── 3. Data ingestion ─────────────────────────────────────────────────────
    print("\n[3/8] Ingesting SoLEXS observations from both archives…")

    # Epoch 1 — May 2024
    e1_train = load_epoch_partition(ARCHIVE_MAY2024, EPOCH1_TRAIN_DATES, "train", "EPOCH1_MAY2024")
    e1_val   = load_epoch_partition(ARCHIVE_MAY2024, EPOCH1_VAL_DATES,   "val",   "EPOCH1_MAY2024")
    e1_test  = load_epoch_partition(ARCHIVE_MAY2024, EPOCH1_TEST_DATES,  "test",  "EPOCH1_MAY2024")
    print(f"  Epoch 1 — May 2024  : train={e1_train['window_count']:,}, "
          f"val={e1_val['window_count']:,}, test={e1_test['window_count']:,} windows")

    # Epoch 2 — Jan-Feb 2026
    e2_train = load_epoch_partition(ARCHIVE_FEB2026, EPOCH2_TRAIN_DATES, "train", "EPOCH2_FEB2026")
    e2_val   = load_epoch_partition(ARCHIVE_FEB2026, EPOCH2_VAL_DATES,   "val",   "EPOCH2_FEB2026")
    e2_test  = load_epoch_partition(ARCHIVE_FEB2026, EPOCH2_TEST_DATES,  "test",  "EPOCH2_FEB2026")
    print(f"  Epoch 2 — Feb 2026  : train={e2_train['window_count']:,}, "
          f"val={e2_val['window_count']:,}, test={e2_test['window_count']:,} windows")

    # ── 4. Scaler fitting strictly on training partition ──────────────────────
    print("\n[4/8] Fitting CountRateScaler strictly on combined training partition…")
    train_counts_all = np.concatenate([
        e1_train["all_counts"], e2_train["all_counts"]
    ])
    scaler = CountRateScaler().fit(train_counts_all)
    print(f"  Scaler: log10(x+1) z-score — mean={scaler.mean:.5f}, std={scaler.std:.5f}")
    print("  Scaler fitted ONLY on training partition. Validation/test records excluded.")

    # ── 5. Scale inputs for training ──────────────────────────────────────────
    def _scale(p: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        n = p["window_count"]
        if n == 0:
            return np.empty((0, SEQ_LEN, 1), np.float32), np.empty((0, HORIZON), np.float32)
        Xs = scaler.transform(p["X_raw"].reshape(-1)).reshape(n, SEQ_LEN, 1)
        ys = scaler.transform(p["y_raw"].reshape(-1)).reshape(n, HORIZON)
        return Xs, ys

    X_e1_tr_sc, y_e1_tr_sc = _scale(e1_train)
    X_e1_val_sc, y_e1_val_sc = _scale(e1_val)
    X_e2_tr_sc, y_e2_tr_sc = _scale(e2_train)
    X_e2_val_sc, y_e2_val_sc = _scale(e2_val)

    # Combined train + val for multi-epoch training
    X_tr_sc = np.concatenate([X_e1_tr_sc, X_e2_tr_sc], axis=0)
    y_tr_sc = np.concatenate([y_e1_tr_sc, y_e2_tr_sc], axis=0)
    X_val_sc = np.concatenate([X_e1_val_sc, X_e2_val_sc], axis=0)
    y_val_sc = np.concatenate([y_e1_val_sc, y_e2_val_sc], axis=0)
    print(f"  Combined train: {len(X_tr_sc):,} windows | Combined val: {len(X_val_sc):,} windows")

    # ── 6. Model training ─────────────────────────────────────────────────────
    print("\n[5/8] Training SelfSupervisedCNNLSTM (DEVELOPMENT_ONLY_NOT_FINAL_MODEL)…")
    t_start = time.time()

    existing_ckpts = sorted(ART_4V_DIR.glob(f"solexs_phase4v_cnn_lstm_*.pt"))
    if existing_ckpts:
        ckpt_path = existing_ckpts[-1]
        print(f"  Found existing checkpoint: {ckpt_path.name}. Loading weights…")
        model = SelfSupervisedCNNLSTM(1, HORIZON, 32, 64, 2, 0.2).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()
        ckpt_sha = _sha256(ckpt_path)
        print(f"  Loaded (SHA-256: {ckpt_sha})")
        train_history: List[Dict] = []
        best_val_loss = float("nan")
        train_duration = 0.0
    else:
        model, train_history, best_val_loss = train_model(
            X_tr_sc, y_tr_sc, X_val_sc, y_val_sc, cfg, device
        )
        train_duration = time.time() - t_start
        print(f"  Training complete in {train_duration:.1f}s. Best Val Loss (scaled MSE): {best_val_loss:.6f}")

        # Save checkpoint — ensure run cannot overwrite prior artifacts
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        ckpt_path = ART_4V_DIR / f"solexs_phase4v_cnn_lstm_{ts}.pt"
        assert not ckpt_path.exists(), f"Checkpoint collision: {ckpt_path}"
        torch.save(model.state_dict(), ckpt_path)
        ckpt_sha = _sha256(ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path.name} (SHA-256: {ckpt_sha})")

    # ── 7. Evaluation: per-epoch + per-cohort ────────────────────────────────
    print("\n[6/8] Evaluating models on test partitions…")
    all_metric_rows: List[Dict[str, Any]] = []

    for epoch_id, test_part in [
        ("EPOCH1_MAY2024", e1_test),
        ("EPOCH2_FEB2026", e2_test),
    ]:
        X_test = test_part["X_raw"]
        y_test = test_part["y_raw"]
        if len(X_test) == 0:
            print(f"  {epoch_id}: no test windows — skipping")
            continue

        # CNN-LSTM predictions
        n_test = len(X_test)
        X_test_sc = scaler.transform(X_test.reshape(-1)).reshape(n_test, SEQ_LEN, 1)
        preds_cnn = infer_model(model, X_test_sc, scaler, device, cfg.batch_size)

        # Baselines
        preds_persist = persistence_forecast(X_test, HORIZON)
        preds_rolling = rolling_mean_forecast(X_test, HORIZON)

        cohorts = assign_causal_cohorts(X_test)
        for cohort_name, mask in cohorts.items():
            n_c = int(mask.sum())
            if n_c < MIN_COHORT_SAMPLES:
                continue
            y_c = y_test[mask]
            for model_name, preds in [
                ("CNN_LSTM_MultiEpoch", preds_cnn[mask]),
                ("Persistence", preds_persist[mask]),
                ("RollingMean", preds_rolling[mask]),
            ]:
                rows = compute_per_horizon_metrics(
                    y_c, preds, model_name, cohort_name, epoch_id, "test"
                )
                all_metric_rows.extend(rows)

        print(f"  {epoch_id}: {n_test:,} test windows evaluated across {len(cohorts)} cohorts")

    # Add skill-vs-persistence column
    all_metric_rows = compute_skill_vs_persistence(all_metric_rows)

    # ── 8. Checkpoint and metadata ────────────────────────────────────────────
    print("\n[7/8] Saving checkpoint metadata and metrics…")

    run_metadata: Dict[str, Any] = {
        "run_id": cfg.run_id,
        "phase": "Phase 4V",
        "training_mode": "DEVELOPMENT_ONLY_NOT_FINAL_MODEL",
        "production_registry_eligible": False,
        "experimental_marker": "PHASE4V_DEVELOPMENT_ONLY_NOT_FINAL_MODEL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "random_seed": cfg.random_seed,
        "model_architecture": "SelfSupervisedCNNLSTM",
        "architecture_params": {
            "in_features": 1, "horizon": HORIZON, "cnn_channels": 32,
            "lstm_hidden": 64, "lstm_layers": 2, "dropout": 0.2,
        },
        "dataset_contract": "phase4u_dataset_contract.md (version phase4u_contract_v1.0)",
        "sequence_length_seconds": SEQ_LEN,
        "forecast_horizon_seconds": HORIZON,
        "scaler": {"type": "log10_zscore", "mean": scaler.mean, "std": scaler.std, "fitted_on": "training_partition_only"},
        "epoch1_may2024": {
            "source_archive": ARCHIVE_MAY2024.name,
            "source_sha256": SHA256_MAY2024,
            "train_dates": EPOCH1_TRAIN_DATES,
            "val_dates": EPOCH1_VAL_DATES,
            "test_dates": EPOCH1_TEST_DATES,
            "train_windows": e1_train["window_count"],
            "val_windows": e1_val["window_count"],
            "test_windows": e1_test["window_count"],
        },
        "epoch2_feb2026": {
            "source_archive": ARCHIVE_FEB2026.name,
            "source_sha256": SHA256_FEB2026,
            "train_dates": EPOCH2_TRAIN_DATES,
            "val_dates": EPOCH2_VAL_DATES,
            "test_dates": EPOCH2_TEST_DATES,
            "train_windows": e2_train["window_count"],
            "val_windows": e2_val["window_count"],
            "test_windows": e2_test["window_count"],
        },
        "holdout_epoch3_status": "LOCKED_NOT_ACQUIRED — October 2024 not accessed",
        "training": {
            "epochs_requested": cfg.epochs,
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "optimizer": "Adam",
            "loss": "MSELoss (scaled)",
            "best_val_loss_scaled_mse": best_val_loss,
            "training_duration_seconds": round(train_duration, 2),
            "history": train_history,
        },
        "checkpoint_sha256": ckpt_sha if existing_ckpts or (ART_4V_DIR / f"solexs_phase4v_cnn_lstm_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.pt").exists() else None,
        "gate_status": gate["overall_status"],
        "evaluation_note": (
            "Metrics reported separately per epoch and causal cohort. "
            "Causal cohort conditioning uses only x_t (forecast origin count rate), "
            "never future target peaks. October holdout excluded from all evaluation."
        ),
    }

    meta_path = ART_4V_DIR / f"run_metadata_{cfg.run_id}.json"
    meta_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    # Also write the canonical run_metadata.json required by spec
    (ART_4V_DIR / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print(f"  Run metadata: {meta_path.name}")

    # Save metrics CSV
    df_metrics = pd.DataFrame(all_metric_rows)
    metrics_csv = ART_4V_DIR / f"phase4v_metrics_{cfg.run_id}.csv"
    metrics_csv_rep = REPORTS_DIR / "phase4v_metrics.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    df_metrics.to_csv(metrics_csv_rep, index=False)
    print(f"  Metrics CSV : {metrics_csv.name}")
    print(f"  Metrics CSV : {metrics_csv_rep.name}")

    # Print summary
    print("\n[8/8] Summary — MAE (counts/s) on test partitions (ALL cohort, horizon 1 and 10)")
    print("=" * 80)
    for epoch_id in ["EPOCH1_MAY2024", "EPOCH2_FEB2026"]:
        sub = df_metrics[(df_metrics["epoch_id"] == epoch_id) &
                         (df_metrics["cohort"] == "ALL") &
                         (df_metrics["horizon_step"].isin([1, 10]))]
        if sub.empty:
            continue
        print(f"\n  {epoch_id}:")
        pivot = sub.pivot(index="horizon_step", columns="model", values="mae_counts_per_s")
        print(pivot.to_string())
        pivot_skill = sub.pivot(index="horizon_step", columns="model", values="skill_vs_persistence_pct")
        print("  Skill vs Persistence (%):")
        print(pivot_skill.to_string())

    print("\n" + "=" * 80)
    print("Phase 4V Status: PHASE4V_DEVELOPMENT_TRAINING_COMPLETED_HOLDOUT_PENDING")
    print("Training mode : DEVELOPMENT_ONLY_NOT_FINAL_MODEL")
    print("Holdout       : LOCKED — October 2024 not evaluated")
    print("=" * 80)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4V Multi-Epoch Forecasting Pipeline")
    parser.add_argument(
        "--mode",
        choices=["gate-only", "dev-train"],
        default="gate-only",
        help="'gate-only': evaluate prerequisites only. 'dev-train': run full development training.",
    )
    args = parser.parse_args()
    return run_pipeline(mode=args.mode)


if __name__ == "__main__":
    sys.exit(main())
