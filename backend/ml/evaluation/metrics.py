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
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be non-empty one-dimensional arrays of equal length.")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("y_true and y_pred must contain only binary values 0 and 1.")
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return dict(tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def true_skill_statistic(tp, fp, tn, fn):
    """TSS = POD - POFD; undefined if either observed class is absent."""
    if tp + fn == 0 or fp + tn == 0:
        return None
    return float(tp / (tp + fn) - fp / (fp + tn))


def heidke_skill_score(tp, fp, tn, fn):
    """HSS: skill relative to random forecast. Range: (-inf, 1], 0 = no skill over chance."""
    numerator = 2 * (tp * tn - fp * fn)
    denominator = ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    if denominator == 0:
        return None
    return float(numerator / denominator)


def false_alarm_rate(tp, fp, tn, fn):
    """FAR = FP / (FP + TP)  -- fraction of positive predictions that were wrong."""
    denom = (fp + tp)
    return float(fp / denom) if denom > 0 else None


def probability_of_detection(tp, fp, tn, fn):
    denom = (tp + fn)
    return float(tp / denom) if denom > 0 else None


def brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray, y_ref_prob) -> float:
    """BSS = 1 - Brier(model) / Brier(reference). > 0 beats the reference (use the
    train-split climatological flare rate as the reference); 0 = no better; < 0 = worse."""
    y_true = np.asarray(y_true).astype(float)
    ref = np.broadcast_to(np.asarray(y_ref_prob, dtype=float), y_true.shape)
    b_ref = float(np.mean((ref - y_true) ** 2))
    if b_ref == 0.0:
        return None
    b_model = float(np.mean((np.asarray(y_prob, dtype=float) - y_true) ** 2))
    return float(1.0 - b_model / b_ref)

def _validate_binary_inputs(y_true, y_prob, threshold):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("y_true and y_prob must be one-dimensional.")

    if len(y_true) == 0:
        raise ValueError("Evaluation inputs must not be empty.")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have equal lengths.")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1.")

    if not np.all(np.isfinite(y_prob)):
        raise ValueError("Probabilities must be finite.")

    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("Probabilities must be between 0 and 1.")

    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be finite and between 0 and 1.") from exc
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between 0 and 1.")

    return y_true.astype(int), y_prob

def evaluate_binary_forecast(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Full metric suite for one binary forecast horizon."""
    y_true, y_prob = _validate_binary_inputs(
        y_true, y_prob, threshold
    )
    y_pred = (y_prob >= threshold).astype(int)

    counts = binary_confusion_counts(y_true, y_pred)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]

    reasons = {}
    total = int(len(y_true))
    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))
    if tp + fn == 0:
        reasons["pod"] = "No positive cases in the evaluated sample."
    if tp + fp == 0:
        reasons["far"] = "No positive predictions at the configured threshold."
    if tp + fn == 0:
        reasons["tss"] = "TSS requires at least one positive and one negative case."
    elif fp + tn == 0:
        reasons["tss"] = "TSS requires at least one positive and one negative case."
    if fp + tn == 0:
        reasons["pofd"] = "POFD requires at least one actual negative case."
    if (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn) == 0:
        reasons["hss"] = "HSS denominator is zero for this contingency table."
    if 2 * tp + fp + fn == 0:
        reasons["recall"] = "Recall requires at least one positive case."
        reasons["f1"] = "F1 denominator is zero because there are no positive cases or positive predictions."
    if tp + fp == 0:
        reasons["precision"] = "Precision requires at least one positive prediction."
        reasons["far"] = "FAR requires at least one positive prediction."

    metrics = {
        "sample_count": total,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_prevalence": float(positive_count / total),
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "false_alarms": fp, "missed_positive_windows": fn,
        "accuracy": float((tp + tn) / max(1, (tp + fp + tn + fn))),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if tp + fp else None,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if positive_count else None,
        "pod": probability_of_detection(tp, fp, tn, fn) if tp + fn else None,
        "pofd": float(fp / (fp + tn)) if fp + tn else None,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None,
        "false_alarm_rate": false_alarm_rate(tp, fp, tn, fn) if tp + fp else None,
        "far": false_alarm_rate(tp, fp, tn, fn) if tp + fp else None,
        "tss": true_skill_statistic(tp, fp, tn, fn) if tp + fn and fp + tn else None,
        "hss": heidke_skill_score(tp, fp, tn, fn) if not ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn) == 0) else None,
        "undefined_metric_reasons": reasons,
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        reasons["roc_auc"] = "ROC-AUC requires both positive and negative cases."
        reasons["pr_auc"] = "PR-AUC is undefined when the holdout contains no positive cases." if positive_count == 0 else "PR-AUC requires both classes in this evaluation contract."

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
    """Fixed-width [low, high) bins (last bin includes 1); empty bins omitted."""
    y_true, y_prob = _validate_binary_inputs(y_true, y_prob, 0.5)
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or not 1 <= n_bins <= 50:
        raise ValueError("n_bins must be an integer between 1 and 50.")
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ids = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    bins = []
    for idx in range(n_bins):
        mask = ids == idx
        if not mask.any():
            continue
        bins.append({
            "lower_bound": float(edges[idx]),
            "upper_bound": float(edges[idx + 1]),
            "upper_inclusive": idx == n_bins - 1,
            "sample_count": int(mask.sum()),
            "mean_predicted_probability": float(y_prob[mask].mean()),
            "observed_event_frequency": float(y_true[mask].mean()),
        })
    minimum = max(20, 2 * int(n_bins))
    if len(y_true) < minimum:
        status = "insufficient_sample_count"
        reason = f"At least {minimum} labeled out-of-sample predictions are required for this {n_bins}-bin display."
    elif np.unique(y_true).size < 2:
        status = "insufficient_class_diversity"
        reason = "Reliability is not displayed because the evaluated sample contains only one label class."
    else:
        status, reason = "available", None
    return {"bin_strategy": "fixed_width_equal_probability_intervals",
            "bin_edges": edges.tolist(), "bins": bins, "sample_count": int(len(y_true)),
            "minimum_samples_for_display": minimum, "status": status, "reason": reason}


def multiclass_confusion_matrix(y_true, y_pred, n_classes=4):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    return cm.tolist()
