"""
Training pipeline.

Handles:
  - Classical models (LogReg, RF, SVM) on engineered features
  - Deep models (CNN, LSTM, CNN-LSTM, CNN-Transformer) on raw sequence windows
  - Class imbalance: class_weight='balanced' for classical models;
    weighted CrossEntropy + BCEWithLogitsLoss(pos_weight=...) for deep models
  - Multi-task loss: total = alpha * CE(nowcast) + beta * BCE(forecast)
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from dataclasses import dataclass, field
from typing import List, Callable, Optional

from ml.models.deep_models import DEEP_MODEL_BUILDERS
from ml.training.validation import validate_training_data


@dataclass
class TrainConfig:
    model_type: str = "cnn_lstm_fusion"   # key into DEEP_MODEL_BUILDERS, or classical model name
    sequence_length: int = 60
    batch_size: int = 64
    epochs: int = 15
    learning_rate: float = 1e-3
    dropout: float = 0.3
    alpha: float = 1.0     # nowcast loss weight
    beta: float = 1.0      # forecast loss weight
    class_weighting: bool = True
    n_channels: int = 2
    device: str = "cpu"
    random_seed: int = 42


def _limit_torch_threads(max_threads: int = 4) -> None:
    """In CPU-limited containers (Docker/Render) torch defaults to the HOST's
    core count, so it spawns far more threads than the cgroup quota allows and
    gets throttled to a crawl. Cap to the cores this process may actually use."""
    try:
        usable = len(os.sched_getaffinity(0))
    except AttributeError:  # non-Linux
        usable = os.cpu_count() or 1
    n = max(1, min(max_threads, usable))
    try:
        torch.set_num_threads(n)
    except Exception:
        pass


def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_pos_weights(y_forecast: np.ndarray) -> torch.Tensor:
    """Compute per-horizon positive weights from known training labels only."""
    y_forecast = np.asarray(y_forecast, dtype=float)
    known = np.isfinite(y_forecast)
    pos = np.sum((y_forecast == 1) & known, axis=0).astype(float)
    neg = np.sum((y_forecast == 0) & known, axis=0).astype(float)
    if np.any(pos == 0):
        raise ValueError("Each training forecast horizon needs known positive examples.")
    return torch.tensor(neg / pos, dtype=torch.float32)


def _masked_forecast_loss(logits, targets, pos_weights=None):
    """BCE averaged over known targets only; NaN labels contribute no loss."""
    known = torch.isfinite(targets)
    if not torch.any(known):
        return logits.sum() * 0.0
    safe_targets = torch.where(known, targets, torch.zeros_like(targets))
    element_loss = nn.functional.binary_cross_entropy_with_logits(
        logits, safe_targets, reduction="none", pos_weight=pos_weights
    )
    return element_loss[known].mean()


def train_deep_model(
    X_train, y_now_train, y_fc_train,
    X_val, y_now_val, y_fc_val,
    config: TrainConfig = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
):
    """
    Trains a PyTorch multi-task model. Returns (model, history_list, final_metrics).
    progress_callback(epoch_record) is invoked after each epoch (used to stream
    live training progress to the frontend / DB).
    should_stop() -> bool allows cooperative cancellation.
    """
    cfg = config or TrainConfig()
    validate_training_data(X_train, y_now_train, y_fc_train, X_val, y_now_val, y_fc_val)
    if cfg.batch_size <= 0 or cfg.epochs <= 0 or cfg.learning_rate <= 0:
        raise ValueError("Training batch_size, epochs, and learning_rate must be positive.")
    if cfg.n_channels <= 0 or np.asarray(X_train).shape[2] != cfg.n_channels:
        raise ValueError("Training sequence channel count does not match the model configuration.")
    if cfg.sequence_length <= 0 or np.asarray(X_train).shape[1] != cfg.sequence_length:
        raise ValueError("Training sequence length does not match the model configuration.")
    if not 0 <= cfg.dropout < 1 or cfg.alpha < 0 or cfg.beta < 0 or cfg.alpha + cfg.beta == 0:
        raise ValueError("Training dropout and loss weights are outside their valid ranges.")
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    _limit_torch_threads()
    device = torch.device(cfg.device)

    model_fn = DEEP_MODEL_BUILDERS.get(cfg.model_type)
    if model_fn is None:
        raise ValueError(f"Unknown deep model type: {cfg.model_type}")
    model = model_fn(n_channels=cfg.n_channels, dropout=cfg.dropout).to(device)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_now_train, dtype=torch.long),
        torch.tensor(y_fc_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_now_val, dtype=torch.long),
        torch.tensor(y_fc_val, dtype=torch.float32),
    )
    loader_generator = torch.Generator().manual_seed(cfg.random_seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=loader_generator)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    if cfg.class_weighting and len(y_now_train) > 0:
        class_weights = compute_class_weights(y_now_train, n_classes=4).to(device)
        pos_weights = compute_pos_weights(y_fc_train).to(device)
    else:
        class_weights = None
        pos_weights = None

    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    history = []
    for epoch in range(cfg.epochs):
        if should_stop and should_stop():
            break
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        for xb, yb_now, yb_fc in train_loader:
            xb, yb_now, yb_fc = xb.to(device), yb_now.to(device), yb_fc.to(device)
            optimizer.zero_grad()
            now_logits, fc_logits = model(xb)
            loss_now = ce_loss_fn(now_logits, yb_now)
            loss_fc = _masked_forecast_loss(fc_logits, yb_fc, pos_weights)
            loss = cfg.alpha * loss_now + cfg.beta * loss_fc
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * xb.size(0)
        train_loss = train_loss_sum / max(1, len(train_ds))

        model.eval()
        val_loss_sum = 0.0
        all_now_pred, all_now_true = [], []
        all_fc_prob, all_fc_true = [], []
        with torch.no_grad():
            for xb, yb_now, yb_fc in val_loader:
                xb, yb_now, yb_fc = xb.to(device), yb_now.to(device), yb_fc.to(device)
                now_logits, fc_logits = model(xb)
                loss_now = ce_loss_fn(now_logits, yb_now)
                loss_fc = _masked_forecast_loss(fc_logits, yb_fc, pos_weights)
                loss = cfg.alpha * loss_now + cfg.beta * loss_fc
                val_loss_sum += loss.item() * xb.size(0)

                all_now_pred.append(torch.argmax(now_logits, dim=1).cpu().numpy())
                all_now_true.append(yb_now.cpu().numpy())
                all_fc_prob.append(torch.sigmoid(fc_logits).cpu().numpy())
                all_fc_true.append(yb_fc.cpu().numpy())

        val_loss = val_loss_sum / max(1, len(val_ds))

        from ml.evaluation.metrics import evaluate_binary_forecast
        tss_list, hss_list = [], []
        if all_fc_prob:
            fc_prob = np.concatenate(all_fc_prob, axis=0)
            fc_true = np.concatenate(all_fc_true, axis=0)
            for h in range(fc_prob.shape[1]):
                known = np.isfinite(fc_true[:, h])
                if not known.any():
                    continue
                m = evaluate_binary_forecast(fc_true[known, h], fc_prob[known, h])
                if m["tss"] is not None:
                    tss_list.append(m["tss"])
                if m["hss"] is not None:
                    hss_list.append(m["hss"])

        record = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_tss_mean": float(np.mean(tss_list)) if tss_list else None,
            "val_hss_mean": float(np.mean(hss_list)) if hss_list else None,
            "duration_sec": round(time.time() - t0, 3),
        }
        history.append(record)
        if progress_callback:
            progress_callback(record)

    return model, history
