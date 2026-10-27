import numpy as np
import pandas as pd
import pytest

from ml.training.dataset_eligibility import assess_dataset_eligibility, require_eligible_dataset


def _frame(start, labels):
    return pd.DataFrame({"timestamp": pd.date_range(start, periods=len(labels), freq="h", tz="UTC"),
                         "forecast_1h": labels})


def _meta():
    evidence = {key: True for key in (
        "observation_provenance_verified", "label_provenance_verified", "timestamps_utc_verified",
        "forecast_target_verified", "features_prediction_time_verified", "unknown_labels_excluded",
        "chronological_embargo_verified")}
    return {"source": "observational", "observation_source": "SoLEXS/HEL1OS", "label_source": "GOES",
            "source_hashes": {"instrument": "a" * 64, "labels": "b" * 64}, "canonical_semantics_status": "VERIFIED",
            "time_contract_verified": True, "observation_exposure_verified": True, "quality_saturation_verified": True,
            "version_calibration_verified": True, "phase2c_evidence": evidence, "minimum_class_counts": {"positive": 1, "negative": 1},
            "feature_label_temporal_separation_verified": True}


def test_unknown_scientific_contracts_are_unresolved_not_negative_labels():
    report = assess_dataset_eligibility({"source": "observational"}, {})
    assert report["status"] == "UNRESOLVED"
    assert any(row["code"] == "SCIENTIFIC_SEMANTICS_UNRESOLVED" for row in report["rejection_reasons"])
    assert report["no_training_or_filtering_performed"] is True


def test_eligible_fixture_requires_provenance_split_order_and_class_counts_for_each_horizon():
    labels = [0, 1] * 3
    splits = {"train": _frame("2026-01-01", labels), "validation": _frame("2026-01-02", labels), "test": _frame("2026-01-03", labels)}
    for frame in splits.values():
        for horizon in ("3h", "6h", "12h", "24h"):
            frame[f"forecast_{horizon}"] = labels
    report = assess_dataset_eligibility(_meta(), splits)
    assert report["status"] == "ELIGIBLE"
    assert report["training_allowed"] is True


def test_ineligible_split_order_and_invalid_labels_fail_closed():
    splits = {"train": _frame("2026-01-03", [0, 1]), "validation": _frame("2026-01-02", [0, 2]), "test": _frame("2026-01-04", [0, 1])}
    for frame in splits.values():
        for horizon in ("3h", "6h", "12h", "24h"):
            frame[f"forecast_{horizon}"] = [0, 1]
    report = assess_dataset_eligibility(_meta(), splits)
    assert report["status"] == "INELIGIBLE"
    assert {row["code"] for row in report["rejection_reasons"]} >= {"SPLIT_LEAKAGE_OR_ORDER", "INVALID_VALIDATION_FORECAST_1H"}
    with pytest.raises(ValueError, match="not eligible"):
        require_eligible_dataset(_meta(), splits)
