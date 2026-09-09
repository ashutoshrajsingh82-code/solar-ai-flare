"""
Training pipeline.

Handles:
  - Classical models (LogReg, RF, SVM) on engineered features
  - Deep models (CNN, LSTM, CNN-LSTM, CNN-Transformer) on raw sequence windows
  - Class imbalance: class_weight='balanced' for classical models;
    weighted CrossEntropy + BCEWithLogitsLoss(pos_weight=...) for deep models
  - Multi-task loss: total = alpha * CE(nowcast) + beta * BCE(forecast)
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from dataclasses import dataclass, field
from typing import List, Callable, Optional

from ml.models.deep_models import DEEP_MODEL_BUILDERS


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


def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_pos_weights(y_forecast: np.ndarray) -> torch.Tensor:
    """pos_weight per horizon for BCEWithLogitsLoss, for rare positive class."""
    pos = y_forecast.sum(axis=0)
    neg = y_forecast.shape[0] - pos
    pos = np.clip(pos, 1, None)
    weights = neg / pos
    return torch.tensor(weights, dtype=torch.float32)


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
    device = torch.device(cfg.device)

    model_fn = DEEP_MODEL_BUILDERS.get(cfg.model_type)
    if model_fn is None:
        raise ValueError(f"Unknown deep model type: {cfg.model_type}")
    model = model_fn(n_channels=cfg.n_channels).to(device)

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
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    if cfg.class_weighting and len(y_now_train) > 0:
        class_weights = compute_class_weights(y_now_train, n_classes=4).to(device)
        pos_weights = compute_pos_weights(y_fc_train).to(device)
    else:
        class_weights = None
        pos_weights = None

    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

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
            loss_fc = bce_loss_fn(fc_logits, yb_fc)
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
                loss_fc = bce_loss_fn(fc_logits, yb_fc)
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
                m = evaluate_binary_forecast(fc_true[:, h], fc_prob[:, h])
                tss_list.append(m["tss"])
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
