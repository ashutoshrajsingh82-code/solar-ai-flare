"""Fail-closed, non-training eligibility assessment for real datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


REQUIRED_EVIDENCE = (
    "observation_provenance_verified", "label_provenance_verified",
    "timestamps_utc_verified", "forecast_target_verified",
    "features_prediction_time_verified", "unknown_labels_excluded",
    "chronological_embargo_verified",
)
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class EligibilityConfig:
    horizon_columns: tuple[str, ...] = ("forecast_1h", "forecast_3h", "forecast_6h", "forecast_12h", "forecast_24h")


def _reason(code: str, category: str, detail: str, status: str = "UNRESOLVED") -> dict:
    return {"code": code, "category": category, "status": status, "detail": detail}


def _timestamps(frame: pd.DataFrame | None):
    if frame is None or "timestamp" not in frame:
        return None
    return pd.to_datetime(frame["timestamp"], utc=True, errors="coerce", format="mixed")


def assess_dataset_eligibility(metadata: Mapping | None, splits: Mapping[str, pd.DataFrame | None] | None,
                               config: EligibilityConfig | None = None) -> dict:
    """Return ELIGIBLE only when every required contract is explicit and valid.

    This function never creates labels, converts timestamps, filters samples, or
    invokes a trainer.  Missing evidence is deliberately UNRESOLVED.
    """
    meta, parts, cfg = dict(metadata or {}), dict(splits or {}), config or EligibilityConfig()
    reasons: list[dict] = []
    if meta.get("source") == "demo":
        reasons.append(_reason("DEMO_DATASET", "provenance", "Synthetic/demo data cannot be eligible for real training.", "INELIGIBLE"))
    if not meta.get("observation_source") or not meta.get("label_source"):
        reasons.append(_reason("MISSING_SOURCE_PROVENANCE", "provenance", "Observation and label source identifiers are required."))
    hashes = meta.get("source_hashes")
    if not isinstance(hashes, dict) or not hashes or not all(isinstance(v, str) and len(v) == 64 for v in hashes.values()):
        reasons.append(_reason("MISSING_IMMUTABLE_SOURCE_HASH", "provenance", "Immutable SHA-256 hashes for all input sources are required."))
    if meta.get("canonical_semantics_status") != "VERIFIED":
        reasons.append(_reason("SCIENTIFIC_SEMANTICS_UNRESOLVED", "scientific_semantics", "Canonical scientific semantics must be VERIFIED; no assumption is substituted."))
    for key in ("time_contract_verified", "observation_exposure_verified", "quality_saturation_verified", "version_calibration_verified"):
        if meta.get(key) is not True:
            reasons.append(_reason(f"MISSING_{key.upper()}", "instrument_contract", f"{key} is not explicitly verified."))
    evidence = meta.get("phase2c_evidence")
    for key in REQUIRED_EVIDENCE:
        if not isinstance(evidence, dict) or evidence.get(key) is not True:
            reasons.append(_reason(f"MISSING_{key.upper()}", "labels_and_leakage", f"Required evidence {key} is missing."))
    minimums = meta.get("minimum_class_counts")
    if not isinstance(minimums, dict) or not all(isinstance(minimums.get(k), int) and minimums[k] > 0 for k in ("positive", "negative")):
        reasons.append(_reason("MISSING_MINIMUM_CLASS_POLICY", "split", "Explicit positive/negative minimum class-count policy is required."))
        minimums = {"positive": float("inf"), "negative": float("inf")}

    boundaries = []
    for name in SPLITS:
        frame = parts.get(name)
        if frame is None or frame.empty:
            reasons.append(_reason(f"MISSING_{name.upper()}_SPLIT", "split", f"{name} split is missing or empty."))
            continue
        stamps = _timestamps(frame)
        if stamps is None or stamps.isna().any() or stamps.duplicated().any() or not stamps.is_monotonic_increasing:
            reasons.append(_reason(f"INVALID_{name.upper()}_TIMESTAMPS", "split", f"{name} timestamps must be valid, unique, and ordered."))
        else:
            boundaries.append((name, stamps.min(), stamps.max()))
        for horizon in cfg.horizon_columns:
            if horizon not in frame:
                reasons.append(_reason(f"MISSING_{name.upper()}_{horizon.upper()}", "labels_and_leakage", f"{name} lacks {horizon}."))
                continue
            values = pd.to_numeric(frame[horizon], errors="coerce")
            known = values.isin([0, 1])
            if (values.notna() & ~known).any():
                reasons.append(_reason(f"INVALID_{name.upper()}_{horizon.upper()}", "labels_and_leakage", f"{name} has non-binary known labels for {horizon}.", "INELIGIBLE"))
            if (values[known] == 1).sum() < minimums["positive"] or (values[known] == 0).sum() < minimums["negative"]:
                reasons.append(_reason(f"INSUFFICIENT_{name.upper()}_{horizon.upper()}_CLASSES", "split", f"{name} does not meet explicit class counts for {horizon}."))
    if len(boundaries) == 3:
        _, _, train_end = boundaries[0]; _, val_start, val_end = boundaries[1]; _, test_start, _ = boundaries[2]
        if not (train_end < val_start <= val_end < test_start):
            reasons.append(_reason("SPLIT_LEAKAGE_OR_ORDER", "split", "Chronological split boundaries overlap or are unordered.", "INELIGIBLE"))
    if meta.get("feature_label_temporal_separation_verified") is not True:
        reasons.append(_reason("MISSING_FEATURE_LABEL_SEPARATION", "features", "Prediction-time-only feature/label temporal separation is not verified."))

    statuses = {item["status"] for item in reasons}
    status = "INELIGIBLE" if "INELIGIBLE" in statuses else ("UNRESOLVED" if reasons else "ELIGIBLE")
    return {"schema_version": "dataset_eligibility_v1", "status": status, "training_allowed": status == "ELIGIBLE",
            "rejection_reasons": reasons, "assessed_splits": list(SPLITS), "no_training_or_filtering_performed": True}


def require_eligible_dataset(metadata, splits, config: EligibilityConfig | None = None) -> dict:
    report = assess_dataset_eligibility(metadata, splits, config)
    if not report["training_allowed"]:
        codes = ", ".join(item["code"] for item in report["rejection_reasons"])
        raise ValueError(f"Dataset is not eligible for real training: {report['status']} ({codes})")
    return report
