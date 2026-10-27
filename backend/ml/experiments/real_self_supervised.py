"""Experimental self-supervised count-rate forecasting on real SoLEXS L1 observations.

This module implements a self-supervised autoregressive forecasting task predicting future
native count rates from past native count rates. It operates purely on verified physical
measurements without creating binary flare labels, without assuming quiet Sun on unobserved
intervals, and without routing uncalibrated counts through physical-flux pipelines.
"""
from __future__ import annotations

import io
import gzip
import json
import zipfile
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from astropy.io import fits


SOLEXS_ARCHIVE_SHA256 = "34c39624194ee2da95bbf8e3dfed5ac002c55a517a31bc4091bfdd0d2b4671c2"
MARKER_EXPERIMENTAL = "EXPERIMENTAL_SELF_SUPERVISED_NOT_FOR_FLARE_CLASSIFICATION"


@dataclass
class SelfSupervisedConfig:
    sequence_length: int = 60         # 60 seconds history
    forecast_horizon: int = 10        # next 10 seconds future count rate
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    random_seed: int = 42
    device: str = "cpu"
    instrument: str = "SoLEXS"
    detector: str = "SDD2"
    measurement_column: str = "COUNTS"
    measurement_units: str = "counts/s"
    energy_band: str = "2-22 keV"
    train_end_utc: str = "2026-02-02 23:59:59"
    val_end_utc: str = "2026-02-04 23:59:59"
    test_end_utc: str = "2026-02-05 23:59:59"


class CountRateScaler:
    """Log10(x + 1) transform with z-score standardization fitted strictly on training data."""

    def __init__(self):
        self.mean: float = 0.0
        self.std: float = 1.0
        self.fitted: bool = False

    def fit(self, values: np.ndarray) -> "CountRateScaler":
        finite = values[np.isfinite(values) & (values >= 0)]
        if len(finite) == 0:
            raise ValueError("Cannot fit scaler on empty or non-finite count rates")
        log_v = np.log10(finite + 1.0)
        self.mean = float(np.mean(log_v))
        self.std = float(np.std(log_v))
        if self.std < 1e-6:
            self.std = 1.0
        self.fitted = True
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("CountRateScaler must be fitted before transform")
        clipped = np.clip(values, 0.0, None)
        log_v = np.log10(clipped + 1.0)
        return (log_v - self.mean) / self.std

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("CountRateScaler must be fitted before inverse_transform")
        log_v = (scaled * self.std) + self.mean
        counts = np.power(10.0, log_v) - 1.0
        return np.clip(counts, 0.0, None)


class SelfSupervisedCNNLSTM(nn.Module):
    """1D-CNN + LSTM sequence model for autoregressive native count-rate forecasting."""

    def __init__(self, in_features: int = 1, horizon: int = 10, cnn_channels: int = 32,
                 lstm_hidden: int = 64, lstm_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.horizon = horizon
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1) -> transpose for conv1d: (batch, 1, seq_len)
        x_conv = x.transpose(1, 2)
        feat = self.conv(x_conv)  # (batch, cnn_channels, seq_len // 2)
        feat_lstm = feat.transpose(1, 2)  # (batch, seq_len // 2, cnn_channels)
        out, _ = self.lstm(feat_lstm)
        last_hidden = out[:, -1, :]  # (batch, lstm_hidden)
        preds = self.fc(last_hidden)  # (batch, horizon)
        return preds


def verify_file_hash(path: Path, expected_hash: str) -> bool:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower() == expected_hash.lower()


def validate_solexs_fits_table(hdul: fits.HDUList) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Validate FITS table structure, column names, units, and temporal ordering."""
    rate_hdu = next((h for h in hdul if str(h.name).upper() == "RATE"), None)
    if rate_hdu is None or rate_hdu.data is None:
        raise ValueError("SoLEXS FITS is missing a populated RATE binary table")

    colnames = {str(c).upper(): c for c in (rate_hdu.columns.names or [])}
    if not {"TIME", "COUNTS"}.issubset(colnames):
        raise ValueError(f"SoLEXS RATE table lacks required columns. Found: {list(colnames.keys())}")

    header = rate_hdu.header
    time_col = rate_hdu.columns[colnames["TIME"]]
    count_col = rate_hdu.columns[colnames["COUNTS"]]

    col_t_unit = str(time_col.unit or "").strip().lower()
    hdr_t_unit = str(header.get("TIMEUNIT", "")).strip().lower()
    time_unit = col_t_unit or hdr_t_unit or "s"

    col_c_unit = str(count_col.unit or "").strip().lower().replace(" ", "")
    hdr_c_unit = str(header.get("HDUCLAS3", "")).strip().lower().replace(" ", "")
    count_unit = col_c_unit or ("counts/s" if hdr_c_unit == "counts" else "") or "counts/s"

    if col_t_unit and col_t_unit not in {"s", "sec", "second", "seconds"}:
        raise ValueError(f"Invalid SoLEXS TIME unit '{col_t_unit}'; expected 's' (Unix seconds)")
    if hdr_t_unit and hdr_t_unit not in {"s", "sec", "second", "seconds"}:
        raise ValueError(f"Invalid SoLEXS header TIMEUNIT '{hdr_t_unit}'; expected 's'")

    if col_c_unit and col_c_unit not in {"count/s", "counts/s", "countsec-1", "countssec-1", "ct/s"}:
        raise ValueError(f"Invalid SoLEXS COUNTS unit '{col_c_unit}'; expected 'counts/s'")

    times = np.asarray(rate_hdu.data[colnames["TIME"]], dtype=float).reshape(-1)
    counts = np.asarray(rate_hdu.data[colnames["COUNTS"]], dtype=float).reshape(-1)

    if not np.isfinite(times).all():
        raise ValueError("SoLEXS TIME column contains non-finite values")
    if len(np.unique(times)) != len(times):
        raise ValueError("SoLEXS TIME column contains duplicate timestamps")
    if np.any(np.diff(times) < 0):
        raise ValueError("SoLEXS TIME column is not monotonically increasing")

    return times, counts, time_unit, count_unit


def extract_contiguous_segments(times: np.ndarray, counts: np.ndarray,
                                min_segment_length: int = 70) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Segment data at gaps where dt > 1.0s or counts are NaN/negative."""
    valid_mask = np.isfinite(counts) & (counts >= 0)
    valid_times = times[valid_mask]
    valid_counts = counts[valid_mask]

    if len(valid_times) == 0:
        return []

    time_diffs = np.diff(valid_times)
    is_gap = np.concatenate([[False], time_diffs > 1.0])
    segment_ids = np.cumsum(is_gap)

    segments = []
    for sid in np.unique(segment_ids):
        mask = (segment_ids == sid)
        seg_times = valid_times[mask]
        seg_counts = valid_counts[mask]
        if len(seg_times) >= min_segment_length:
            segments.append((seg_times, seg_counts))
    return segments


def create_sliding_windows(segments: List[Tuple[np.ndarray, np.ndarray]],
                           seq_len: int = 60, horizon: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (X, y, timestamp) windows strictly within individual contiguous segments."""
    x_list, y_list, t_list = [], [], []
    total_window_len = seq_len + horizon

    for seg_times, seg_counts in segments:
        n_windows = len(seg_counts) - total_window_len + 1
        if n_windows <= 0:
            continue
        for i in range(n_windows):
            x_window = seg_counts[i : i + seq_len]
            y_window = seg_counts[i + seq_len : i + total_window_len]
            t_origin = seg_times[i + seq_len - 1]
            x_list.append(x_window)
            y_list.append(y_window)
            t_list.append(t_origin)

    if not x_list:
        return np.empty((0, seq_len, 1)), np.empty((0, horizon)), np.empty((0,))

    X = np.asarray(x_list, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.asarray(y_list, dtype=np.float32)
    timestamps = np.asarray(t_list, dtype=np.float64)
    return X, y, timestamps


def load_and_split_solexs_real_data(archive_path: str | Path,
                                    cfg: SelfSupervisedConfig) -> Dict[str, Any]:
    """Load real SoLEXS L1 data from archive, partition chronologically, and extract windows."""
    arch_path = Path(archive_path)
    if not arch_path.is_file():
        raise FileNotFoundError(f"SoLEXS archive not found at {arch_path}")

    archive_hash = hashlib.sha256(arch_path.read_bytes()).hexdigest()
    if archive_hash != SOLEXS_ARCHIVE_SHA256:
        raise ValueError(f"Archive SHA-256 mismatch: {archive_hash} != {SOLEXS_ARCHIVE_SHA256}")

    daily_records: List[Tuple[str, np.ndarray, np.ndarray]] = []
    with zipfile.ZipFile(arch_path) as outer:
        for inner_zip_name in sorted(outer.namelist()):
            if not inner_zip_name.endswith(".zip"):
                continue
            inner_bytes = outer.read(inner_zip_name)
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                lc_members = [m for m in inner.namelist() if "SDD2" in m and m.endswith(".lc.gz")]
                if not lc_members:
                    continue
                member = lc_members[0]
                decompressed = gzip.decompress(inner.read(member))
                with fits.open(io.BytesIO(decompressed)) as hdul:
                    times, counts, t_unit, c_unit = validate_solexs_fits_table(hdul)
                    date_str = inner_zip_name.split("_")[3]  # e.g. '20260131'
                    daily_records.append((date_str, times, counts))

    all_times = np.concatenate([r[1] for r in daily_records])
    all_counts = np.concatenate([r[2] for r in daily_records])

    # Convert split boundary strings to Unix seconds
    train_end_ts = pd.to_datetime(cfg.train_end_utc, utc=True).timestamp()
    val_end_ts = pd.to_datetime(cfg.val_end_utc, utc=True).timestamp()
    test_end_ts = pd.to_datetime(cfg.test_end_utc, utc=True).timestamp()

    # Partition continuous data strictly by boundary
    train_mask = (all_times <= train_end_ts)
    val_mask = (all_times > train_end_ts) & (all_times <= val_end_ts)
    test_mask = (all_times > val_end_ts) & (all_times <= test_end_ts)

    min_len = cfg.sequence_length + cfg.forecast_horizon

    train_segs = extract_contiguous_segments(all_times[train_mask], all_counts[train_mask], min_len)
    val_segs = extract_contiguous_segments(all_times[val_mask], all_counts[val_mask], min_len)
    test_segs = extract_contiguous_segments(all_times[test_mask], all_counts[test_mask], min_len)

    X_train, y_train, t_train = create_sliding_windows(train_segs, cfg.sequence_length, cfg.forecast_horizon)
    X_val, y_val, t_val = create_sliding_windows(val_segs, cfg.sequence_length, cfg.forecast_horizon)
    X_test, y_test, t_test = create_sliding_windows(test_segs, cfg.sequence_length, cfg.forecast_horizon)

    # Fit scaler strictly on training inputs and targets
    scaler = CountRateScaler().fit(all_counts[train_mask])

    return {
        "X_train": X_train, "y_train": y_train, "t_train": t_train,
        "X_val": X_val, "y_val": y_val, "t_val": t_val,
        "X_test": X_test, "y_test": y_test, "t_test": t_test,
        "scaler": scaler,
        "source_archive": arch_path.name,
        "source_sha256": archive_hash,
        "total_records": len(all_times),
        "finite_records": int(np.isfinite(all_counts).sum()),
        "nan_records": int(np.isnan(all_counts).sum()),
        "split_counts": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
        },
        "split_boundaries_utc": {
            "train": ("2026-01-31 00:00:00", cfg.train_end_utc),
            "val": ("2026-02-03 00:00:00", cfg.val_end_utc),
            "test": ("2026-02-05 00:00:00", cfg.test_end_utc),
        }
    }


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MSE, RMSE, MAE, and Relative MAE on physical count rates."""
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    mean_true = float(np.mean(y_true))
    rel_mae = float(mae / mean_true) if mean_true > 0 else 0.0
    return {"mse": mse, "rmse": rmse, "mae": mae, "rel_mae": rel_mae}


def persistence_forecast(X: np.ndarray, horizon: int) -> np.ndarray:
    """Standard physical baseline: future count rates match the last observed count rate."""
    last_val = X[:, -1, 0]  # shape (batch,)
    return np.repeat(last_val[:, np.newaxis], horizon, axis=1)


def rolling_mean_forecast(X: np.ndarray, horizon: int) -> np.ndarray:
    """Historical mean baseline: future count rates match the window's mean."""
    mean_val = np.mean(X[:, :, 0], axis=1)  # shape (batch,)
    return np.repeat(mean_val[:, np.newaxis], horizon, axis=1)


def train_self_supervised_model(data_bundle: Dict[str, Any],
                                cfg: SelfSupervisedConfig) -> Tuple[SelfSupervisedCNNLSTM, Dict[str, Any]]:
    """Train the CNN-LSTM autoregressive model on real SoLEXS training data."""
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    scaler: CountRateScaler = data_bundle["scaler"]
    X_train_raw = data_bundle["X_train"]
    y_train_raw = data_bundle["y_train"]
    X_val_raw = data_bundle["X_val"]
    y_val_raw = data_bundle["y_val"]

    # Transform inputs and targets to scaled space
    X_train_sc = scaler.transform(X_train_raw)
    y_train_sc = scaler.transform(y_train_raw)
    X_val_sc = scaler.transform(X_val_raw)
    y_val_sc = scaler.transform(y_val_raw)

    train_ds = TensorDataset(torch.from_numpy(X_train_sc), torch.from_numpy(y_train_sc))
    val_ds = TensorDataset(torch.from_numpy(X_val_sc), torch.from_numpy(y_val_sc))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = SelfSupervisedCNNLSTM(in_features=1, horizon=cfg.forecast_horizon)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = None
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                preds = model(bx)
                val_loss += criterion(preds, by).item() * len(bx)
        val_loss /= len(val_loader.dataset)

        history.append({"epoch": epoch, "train_loss": float(train_loss), "val_loss": float(val_loss)})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Evaluate on held-out test set
    X_test_raw = data_bundle["X_test"]
    y_test_raw = data_bundle["y_test"]
    X_test_sc = scaler.transform(X_test_raw)

    model.eval()
    with torch.no_grad():
        test_preds_sc = model(torch.from_numpy(X_test_sc)).numpy()
    test_preds_raw = scaler.inverse_transform(test_preds_sc)

    # Baselines on test set
    persist_preds = persistence_forecast(X_test_raw, cfg.forecast_horizon)
    mean_preds = rolling_mean_forecast(X_test_raw, cfg.forecast_horizon)

    metrics_model = evaluate_forecast(y_test_raw, test_preds_raw)
    metrics_persist = evaluate_forecast(y_test_raw, persist_preds)
    metrics_mean = evaluate_forecast(y_test_raw, mean_preds)

    # Skill score vs persistence: (1 - MAE_model / MAE_persist)
    skill_score = 1.0 - (metrics_model["mae"] / metrics_persist["mae"]) if metrics_persist["mae"] > 0 else 0.0

    eval_report = {
        "history": history,
        "best_val_loss_scaled": float(best_val_loss),
        "test_metrics_native_units": {
            "cnn_lstm": metrics_model,
            "persistence_baseline": metrics_persist,
            "rolling_mean_baseline": metrics_mean,
            "skill_score_vs_persistence": float(skill_score),
        }
    }
    return model, eval_report


def save_experimental_checkpoint(model: SelfSupervisedCNNLSTM,
                                 data_bundle: Dict[str, Any],
                                 eval_report: Dict[str, Any],
                                 cfg: SelfSupervisedConfig,
                                 output_dir: str | Path) -> Tuple[Path, Path]:
    """Save model weights and full provenance metadata under experimental directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_name = f"solexs_self_supervised_cnnlstm_{timestamp}"
    weights_path = out_dir / f"{model_name}.pt"
    meta_path = out_dir / f"{model_name}_metadata.json"

    torch.save(model.state_dict(), weights_path)

    metadata = {
        "experiment_id": model_name,
        "experiment_designation": MARKER_EXPERIMENTAL,
        "production_registry_eligible": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_description": "Autoregressive 10-second count-rate forecasting from 60-second native history",
        "instrument": cfg.instrument,
        "detector": cfg.detector,
        "measurement_column": cfg.measurement_column,
        "measurement_units": cfg.measurement_units,
        "energy_band": cfg.energy_band,
        "source_archive": data_bundle["source_archive"],
        "source_archive_sha256": data_bundle["source_sha256"],
        "configuration": asdict(cfg),
        "dataset_statistics": {
            "total_records_in_archive": data_bundle["total_records"],
            "finite_observational_records": data_bundle["finite_records"],
            "unobserved_nan_records": data_bundle["nan_records"],
            "train_samples": data_bundle["split_counts"]["train_samples"],
            "val_samples": data_bundle["split_counts"]["val_samples"],
            "test_samples": data_bundle["split_counts"]["test_samples"],
        },
        "chronological_splits": data_bundle["split_boundaries_utc"],
        "evaluation_metrics": eval_report["test_metrics_native_units"],
        "scaler_parameters": {
            "type": "log10_zscore",
            "mean": data_bundle["scaler"].mean,
            "std": data_bundle["scaler"].std,
            "fitted_on_split": "train_only"
        },
        "exclusions_preserved": {
            "HEL1OS_L1": "Excluded: Unresolved V111/V211 counterpart ambiguity (B-HL-VERSION-006) and missing HK saturation thresholds (B-HL-QUALITY-005)",
            "SoLEXS_SDD1": "Excluded: SDD1 GTI tables contain 0 intervals; detector inactive in this phase",
            "SoLEXS_PI": "Excluded: Spectral histograms (PI) are not continuous sequence light curves",
            "Flare_Labels": "Excluded: Zero flare labels used; self-supervised forecasting avoids label fabrication"
        }
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return weights_path, meta_path
