"""Fail-closed gate for observational Phase 2C training."""


def real_training_blockers(dataset_meta):
    """Return missing evidence needed before a dataset can be called real/verified."""
    meta = dataset_meta or {}
    if meta.get("source") == "demo":
        return ["Dataset is demonstration data."]

    evidence = meta.get("phase2c_evidence")
    if not meta.get("observation_source") or not meta.get("label_source"):
        return ["Verified observation_source and label_source provenance identifiers are required."]
    required = (
        "observation_provenance_verified",
        "label_provenance_verified",
        "timestamps_utc_verified",
        "forecast_target_verified",
        "features_prediction_time_verified",
        "unknown_labels_excluded",
        "chronological_embargo_verified",
    )
    if not isinstance(evidence, dict):
        return ["Phase 2C provenance and validation evidence is absent."]
    missing = [name for name in required if evidence.get(name) is not True]
    return [f"Required Phase 2C evidence is missing: {name}." for name in missing]


def is_verified_real_training_data(dataset_meta):
    return not real_training_blockers(dataset_meta)
