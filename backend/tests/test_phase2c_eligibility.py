from ml.training.eligibility import is_verified_real_training_data, real_training_blockers


def test_demo_and_unverified_datasets_cannot_be_called_observationally_trained():
    assert not is_verified_real_training_data({"source": "demo"})
    assert real_training_blockers({"source": "uploaded_csv"}) == [
        "Verified observation_source and label_source provenance identifiers are required."
    ]


def test_complete_explicit_evidence_is_required_for_real_training():
    evidence = {
        "observation_provenance_verified": True,
        "label_provenance_verified": True,
        "timestamps_utc_verified": True,
        "forecast_target_verified": True,
        "features_prediction_time_verified": True,
        "unknown_labels_excluded": True,
        "chronological_embargo_verified": True,
    }
    assert is_verified_real_training_data({"source": "verified_observations", "observation_source": "SoLEXS", "label_source": "GOES",
                                           "phase2c_evidence": evidence})
    evidence["unknown_labels_excluded"] = False
    assert not is_verified_real_training_data({"source": "verified_observations", "observation_source": "SoLEXS", "label_source": "GOES",
                                                "phase2c_evidence": evidence})
