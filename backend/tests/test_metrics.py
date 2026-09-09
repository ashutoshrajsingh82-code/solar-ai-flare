import pytest
import numpy as np
from ml.evaluation.metrics import (
    true_skill_statistic,
    heidke_skill_score,
    evaluate_binary_forecast,
    multiclass_confusion_matrix,
    binary_confusion_counts,
)


def test_tss_perfect():
    # Perfect predictions: TP=10, FN=0, TN=10, FP=0 -> TSS = 1 - 0 = 1.0
    tss = true_skill_statistic(tp=10, fp=0, tn=10, fn=0)
    assert pytest.approx(tss, rel=1e-5) == 1.0


def test_tss_all_zeros():
    # Trivial model predicting no flares: TP=0, FN=10, TN=10, FP=0 -> TSS = 0
    tss = true_skill_statistic(tp=0, fp=0, tn=10, fn=10)
    assert tss == 0.0


def test_hss_formula():
    # Heidke Skill Score on known contingency table
    hss = heidke_skill_score(tp=10, fp=2, tn=20, fn=3)
    assert 0.0 < hss < 1.0


def test_evaluate_binary_forecast_structure():
    y_true = np.array([1, 0, 1, 0, 0, 1])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.4, 0.85])
    metrics = evaluate_binary_forecast(y_true, y_prob, threshold=0.5)

    assert "tss" in metrics
    assert "hss" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "false_alarm_rate" in metrics
    assert "brier_score" in metrics
    assert metrics["brier_score"] < 0.25


def test_multiclass_nowcast_matrix():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    matrix = multiclass_confusion_matrix(y_true, y_pred, n_classes=4)

    assert len(matrix) == 4
    assert matrix[0][0] == 2
    assert matrix[1][1] == 2
    assert matrix[2][2] == 2
    assert matrix[3][3] == 2
