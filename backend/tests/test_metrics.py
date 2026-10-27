import pytest
import numpy as np

from ml.evaluation.metrics import (
    true_skill_statistic,
    heidke_skill_score,
    evaluate_binary_forecast,
    multiclass_confusion_matrix,
    binary_confusion_counts,
    reliability_diagram,
)


def test_tss_perfect():
    # Perfect predictions: TP=10, FN=0, TN=10, FP=0 -> TSS = 1.0
    tss = true_skill_statistic(tp=10, fp=0, tn=10, fn=0)
    assert pytest.approx(tss, rel=1e-5) == 1.0


def test_tss_all_zeros():
    # Predicting no flares: TP=0, FN=10, TN=10, FP=0 -> TSS = 0
    tss = true_skill_statistic(tp=0, fp=0, tn=10, fn=10)
    assert tss == 0.0


def test_hss_formula():
    # Heidke Skill Score on a known contingency table
    hss = heidke_skill_score(tp=10, fp=2, tn=20, fn=3)
    assert 0.0 < hss < 1.0


def test_evaluate_binary_forecast_structure():
    y_true = np.array([1, 0, 1, 0, 0, 1])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.4, 0.85])

    metrics = evaluate_binary_forecast(
        y_true, y_prob, threshold=0.5
    )

    assert "tss" in metrics
    assert "hss" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "false_alarm_rate" in metrics
    assert "brier_score" in metrics
    assert metrics["brier_score"] < 0.25


def test_no_positive_test_cases_leave_unsupported_scores_undefined():
    metrics = evaluate_binary_forecast(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
    assert metrics["positive_count"] == 0
    assert metrics["negative_count"] == 3
    assert metrics["precision"] is None
    assert metrics["recall"] is None
    assert metrics["f1"] is None
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["tss"] is None
    assert metrics["false_alarms"] == metrics["fp"]
    assert "f1" in metrics["undefined_metric_reasons"]


def test_multiclass_nowcast_matrix():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    matrix = multiclass_confusion_matrix(
        y_true, y_pred, n_classes=4
    )

    assert len(matrix) == 4
    assert matrix[0][0] == 2
    assert matrix[1][1] == 2
    assert matrix[2][2] == 2
    assert matrix[3][3] == 2


def test_binary_confusion_counts():
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0])

    counts = binary_confusion_counts(y_true, y_pred)

    assert counts == {
        "tp": 2,
        "fp": 1,
        "tn": 2,
        "fn": 1,
    }


@pytest.mark.parametrize(
    "y_true, y_prob",
    [
        ([0, 1], [0.2]),                  # Mismatched lengths
        ([], []),                         # Empty input
        ([0, 1], [0.2, float("nan")]),    # NaN probability
        ([0, 1], [0.2, float("inf")]),    # Infinite probability
        ([0, 1], [-0.1, 0.8]),            # Probability below zero
        ([0, 1], [0.2, 1.1]),             # Probability above one
        ([0, 2], [0.2, 0.8]),             # Invalid binary label
    ],
)
def test_invalid_forecast_inputs_are_rejected(y_true, y_prob):
    with pytest.raises(ValueError):
        evaluate_binary_forecast(y_true, y_prob)


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError):
        evaluate_binary_forecast(
            [0, 1], [0.2, 0.8], threshold=1.5
        )


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_out_of_range_or_nonfinite_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        evaluate_binary_forecast(
            [0, 1], [0.2, 0.8], threshold=threshold
        )


def test_single_class_targets_return_none_for_auc_metrics():
    # AUC is undefined when the evaluation data contains one class.
    metrics = evaluate_binary_forecast(
        np.array([0, 0, 0, 0]),
        np.array([0.1, 0.2, 0.3, 0.4]),
    )

    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert np.isfinite(metrics["brier_score"])


def test_undefined_event_metrics_are_null_with_reasons():
    metrics = evaluate_binary_forecast([0, 0, 0], [0.1, 0.2, 0.3])
    assert metrics["pod"] is None
    assert metrics["tss"] is None
    assert metrics["recall"] is None
    assert metrics["far"] is None
    assert metrics["hss"] is None
    assert metrics["positive_prevalence"] == 0
    assert metrics["undefined_metric_reasons"]["pod"]
    assert metrics["undefined_metric_reasons"]["tss"]


def test_far_and_brier_use_the_documented_definitions():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.8, 0.7, 0.4, 0.1])
    metrics = evaluate_binary_forecast(y_true, y_prob, threshold=0.5)
    # TP=1, FP=1, so FAR=FP/(TP+FP)=0.5.
    assert metrics["far"] == pytest.approx(0.5)
    assert metrics["brier_score"] == pytest.approx(np.mean((y_prob - y_true) ** 2))


def test_reliability_fixed_width_bins_report_counts_and_event_rates():
    result = reliability_diagram([0, 1, 1, 0], [0.0, 0.2, 0.99, 1.0], n_bins=2)
    assert result["status"] == "insufficient_sample_count"
    assert result["bins"] == [
        {"lower_bound": 0.0, "upper_bound": 0.5, "upper_inclusive": False,
         "sample_count": 2, "mean_predicted_probability": 0.1, "observed_event_frequency": 0.5},
        {"lower_bound": 0.5, "upper_bound": 1.0, "upper_inclusive": True,
         "sample_count": 2, "mean_predicted_probability": 0.995, "observed_event_frequency": 0.5},
    ]


def test_threshold_changes_confusion_metrics_but_not_brier_score():
    labels, probs = [1, 1, 0, 0], [0.9, 0.4, 0.8, 0.1]
    low = evaluate_binary_forecast(labels, probs, threshold=0.3)
    high = evaluate_binary_forecast(labels, probs, threshold=0.7)
    assert low["brier_score"] == high["brier_score"] == pytest.approx(0.255)
    assert low["tp"] == 2 and low["fp"] == 1
    assert high["tp"] == 1 and high["fp"] == 1
