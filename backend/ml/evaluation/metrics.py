"""
Evaluation metrics for imbalanced flare-prediction problems.

Accuracy is intentionally NOT used as the primary metric (flares are rare
events; a trivial "always quiet" model scores near-perfect accuracy).
TSS and HSS are the headline metrics, alongside POD/Recall, FAR, F1,
ROC-AUC, PR-AUC, and calibration (Brier score, reliability diagram).
"""
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, brier_score_loss, roc_curve, precision_recall_curve
)


def binary_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return dict(tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def true_skill_statistic(tp, fp, tn, fn) -> float:
    """TSS = POD - FAR_rate = TP/(TP+FN) - FP/(FP+TN). Range: [-1, 1], 0 = no skill."""
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pofd = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return float(pod - pofd)


def heidke_skill_score(tp, fp, tn, fn) -> float:
    """HSS: skill relative to random forecast. Range: (-inf, 1], 0 = no skill over chance."""
    numerator = 2 * (tp * tn - fp * fn)
    denominator = ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def false_alarm_rate(tp, fp, tn, fn) -> float:
    """FAR = FP / (FP + TP)  -- fraction of positive predictions that were wrong."""
    denom = (fp + tp)
    return float(fp / denom) if denom > 0 else 0.0


def probability_of_detection(tp, fp, tn, fn) -> float:
    denom = (tp + fn)
    return float(tp / denom) if denom > 0 else 0.0


def evaluate_binary_forecast(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Full metric suite for one binary forecast horizon."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    counts = binary_confusion_counts(y_true, y_pred)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]

    metrics = {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": float((tp + tn) / max(1, (tp + fp + tn + fn))),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "pod": probability_of_detection(tp, fp, tn, fn),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_alarm_rate": false_alarm_rate(tp, fp, tn, fn),
        "tss": true_skill_statistic(tp, fp, tn, fn),
        "hss": heidke_skill_score(tp, fp, tn, fn),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
    return metrics


def roc_curve_points(y_true, y_prob, max_points=100):
    if len(np.unique(y_true)) < 2:
        return {"fpr": [], "tpr": [], "thresholds": []}
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    idx = np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).astype(int)
    return {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist(), "thresholds": thr[idx].tolist()}


def pr_curve_points(y_true, y_prob, max_points=100):
    if len(np.unique(y_true)) < 2:
        return {"precision": [], "recall": [], "thresholds": []}
    precision, recall, thr = precision_recall_curve(y_true, y_prob)
    idx = np.linspace(0, len(precision) - 1, min(max_points, len(precision))).astype(int)
    return {"precision": precision[idx].tolist(), "recall": recall[idx].tolist(),
            "thresholds": (thr[np.clip(idx, 0, len(thr) - 1)].tolist() if len(thr) else [])}


def reliability_diagram(y_true, y_prob, n_bins=10):
    """Bins predicted probabilities and computes observed frequency per bin."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    mean_predicted, observed_freq, counts = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            mean_predicted.append(None)
            observed_freq.append(None)
            counts.append(0)
        else:
            mean_predicted.append(float(y_prob[mask].mean()))
            observed_freq.append(float(y_true[mask].mean()))
            counts.append(int(mask.sum()))
    return {"bin_edges": bins.tolist(), "mean_predicted": mean_predicted,
            "observed_frequency": observed_freq, "counts": counts}


def multiclass_confusion_matrix(y_true, y_pred, n_classes=4):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    return cm.tolist()
