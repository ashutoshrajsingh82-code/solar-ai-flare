"""Evidence-based readiness reports for observational model training.

Reports describe the loaded dataset and its provenance without promoting demo
or unverified measurements to scientific observations. Files are written
atomically beneath the persistent pipeline-artifact directory.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import settings
from app.services.pipeline_state import pipeline_state
from ml.training.eligibility import real_training_blockers

HORIZONS = ("1h", "3h", "6h", "12h", "24h")
PHASE2C_EVIDENCE = (
    "observation_provenance_verified",
    "label_provenance_verified",
    "timestamps_utc_verified",
    "forecast_target_verified",
    "features_prediction_time_verified",
    "unknown_labels_excluded",
    "chronological_embargo_verified",
)


def _artifact_root() -> Path:
    root = Path(settings.pipeline_artifact_dir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    return root.resolve() / "readiness_reports"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".readiness-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _utc_series(frame: pd.DataFrame) -> pd.Series | None:
    if frame is None or frame.empty or "timestamp" not in frame:
        return None
    values = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce", format="mixed")
    return values


def _class_counts(frame: pd.DataFrame | None) -> dict:
    counts = {}
    if frame is None:
        return counts
    for horizon in HORIZONS:
        column = f"forecast_{horizon}"
        if column not in frame:
            counts[horizon] = {"known": 0, "positive": 0, "negative": 0,
                               "unknown": int(len(frame)), "invalid": 0}
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        known = values.isin([0, 1])
        invalid = values.notna() & ~known
        counts[horizon] = {
            "known": int(known.sum()),
            "positive": int((values[known] == 1).sum()),
            "negative": int((values[known] == 0).sum()),
            "unknown": int(values.isna().sum()),
            "invalid": int(invalid.sum()),
        }
    return counts


def build_scientific_readiness_report(
    dataset_meta: dict | None = None,
    observations: pd.DataFrame | None = None,
    splits: dict[str, pd.DataFrame | None] | None = None,
) -> dict:
    """Assess real-data provenance, channels, labels, coverage, and split balance."""
    meta = dict(dataset_meta if dataset_meta is not None else pipeline_state.dataset_meta or {})
    frame = observations if observations is not None else pipeline_state.raw_df
    split_frames = splits if splits is not None else {
        "train": pipeline_state.train_df,
        "validation": pipeline_state.val_df,
        "test": pipeline_state.test_df,
    }
    blockers = []
    warnings = []
    columns = list(frame.columns) if frame is not None else []

    if frame is None or frame.empty:
        blockers.append("No observation dataset is currently loaded.")
        row_count = 0
        period = {"start_utc": None, "end_utc": None}
        cadence = None
        invalid_timestamps = 0
        duplicate_timestamps = 0
    else:
        row_count = int(len(frame))
        timestamps = _utc_series(frame)
        if timestamps is None:
            blockers.append("Observation data has no timestamp column.")
            period = {"start_utc": None, "end_utc": None}
            cadence = None
            invalid_timestamps = row_count
            duplicate_timestamps = 0
        else:
            invalid_timestamps = int(timestamps.isna().sum())
            valid_times = timestamps.dropna()
            duplicate_timestamps = int(valid_times.duplicated().sum())
            if invalid_timestamps:
                blockers.append(f"Observation timestamps contain {invalid_timestamps} invalid value(s).")
            if duplicate_timestamps:
                blockers.append(f"Observation timestamps contain {duplicate_timestamps} duplicate(s).")
            if not valid_times.is_monotonic_increasing:
                blockers.append("Observation timestamps are not in chronological order.")
            period = {
                "start_utc": valid_times.min().isoformat() if len(valid_times) else None,
                "end_utc": valid_times.max().isoformat() if len(valid_times) else None,
            }
            cadence_values = valid_times.diff().dt.total_seconds().dropna()
            cadence = float(cadence_values.median()) if len(cadence_values) else None

    if meta.get("source") == "demo" or settings.data_mode == "demo":
        blockers.append("The active dataset/runtime is demo mode; synthetic observations cannot support scientific training.")
    elif not meta:
        blockers.append("Dataset provenance metadata is missing.")

    required_channels = {"solexs_flux", "helios_flux"}
    channel_presence = {name: name in columns for name in sorted(required_channels)}
    if not all(channel_presence.values()):
        missing = sorted(name for name, present in channel_presence.items() if not present)
        blockers.append("Required forecast input channel(s) are absent: " + ", ".join(missing) + ".")

    if "solexs_counts_per_second" in columns:
        blockers.append(
            "SoLEXS input is native COUNTS/s, not calibrated physical flux; no validated response conversion is available."
        )
    contract = meta.get("measurement_contract") or {}
    contract_verified = meta.get("measurement_contract_verified") is True and isinstance(contract, dict)
    if not contract_verified:
        blockers.append(
            "Instrument units, energy channels, and calibration are not verified for the loaded measurement columns."
        )
    if not meta.get("observation_source"):
        blockers.append("A verified instrument observation_source identifier is missing.")
    product_integration_verified = meta.get("real_product_integration_verified") is True
    if meta.get("source") != "demo" and not product_integration_verified:
        blockers.append(
            "The exact instrument product version has not passed integration validation against its registered manual and a genuine sample file."
        )

    reference = meta.get("reference_labels") or {}
    reference_source = str(reference.get("source") or "")
    event_report = reference
    label_report = reference.get("label_report") or {}
    if not reference_source.startswith("NOAA/NCEI GOES") or not reference.get("file_sha256"):
        blockers.append("Independent NOAA/NCEI GOES reference-label provenance is missing or unverified.")

    label_counts = _class_counts(frame)
    for horizon, counts in label_counts.items():
        if counts["known"] == 0:
            blockers.append(f"No known GOES {horizon} labels are available.")
        if counts["invalid"]:
            blockers.append(f"GOES {horizon} targets contain {counts['invalid']} non-binary value(s).")
    split_summaries = {
        name: {"rows": int(len(value)) if value is not None else 0,
               "label_counts": _class_counts(value)}
        for name, value in split_frames.items()
    }
    for name in ("train", "validation", "test"):
        split = split_frames.get(name)
        if split is None or split.empty:
            blockers.append(f"Chronological {name} split is unavailable or empty.")
    training_counts = split_summaries.get("train", {}).get("label_counts", {})
    for horizon in HORIZONS:
        counts = training_counts.get(horizon, {})
        if counts and counts.get("positive", 0) == 0:
            blockers.append(f"Training split has no positive examples for {horizon}.")
        if counts and counts.get("negative", 0) == 0:
            blockers.append(f"Training split has no negative examples for {horizon}.")

    evidence = meta.get("phase2c_evidence")
    missing_evidence = [key for key in PHASE2C_EVIDENCE
                        if not isinstance(evidence, dict) or evidence.get(key) is not True]
    if missing_evidence:
        blockers.extend(f"Required scientific validation evidence is missing: {key}." for key in missing_evidence)

    if frame is not None and not frame.empty and all(channel_presence.values()):
        finite = np.isfinite(frame[["solexs_flux", "helios_flux"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))
        pair_count = int(finite.all(axis=1).sum())
        channel_valid = {
            channel: int(finite[:, index].sum())
            for index, channel in enumerate(("solexs_flux", "helios_flux"))
        }
        if pair_count == 0:
            warnings.append("There are no timestamps with both instrument inputs simultaneously measured.")
    else:
        pair_count = 0
        channel_valid = {"solexs_flux": 0, "helios_flux": 0}

    report = {
        "schema_version": 1,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "status": "READY" if not blockers else "BLOCKED",
        "training_allowed": not blockers,
        "dataset": {
            "id": meta.get("dataset_id"),
            "name": meta.get("filename"),
            "source": meta.get("source"),
            "observation_source": meta.get("observation_source"),
            "label_source": meta.get("label_source"),
            "raw_file_sha256": meta.get("raw_file_sha256"),
            "file_format": meta.get("file_format"),
            "instrument_product": meta.get("product"),
            "rows": row_count,
            "period": period,
            "median_cadence_seconds": cadence,
            "invalid_timestamps": invalid_timestamps,
            "duplicate_timestamps": duplicate_timestamps,
            "columns": columns,
            "channel_valid_rows": channel_valid,
            "paired_channel_rows": pair_count,
        },
        "measurement_contract": {
            "verified": contract_verified,
            "details": contract if contract_verified else None,
            "required_model_columns": sorted(required_channels),
            "physical_flux_columns_present": all(channel_presence.values()),
            "native_counts_preserved": "solexs_counts_per_second" in columns,
            "reported_measurement_unit": (meta.get("ingestion_report") or {}).get("measurement_unit"),
            "reported_energy_band": (meta.get("ingestion_report") or {}).get("energy_band"),
            "quality_filter_applied": (meta.get("ingestion_report") or {}).get("quality_filter_applied"),
            "real_product_integration_verified": product_integration_verified,
        },
        "reference_labels": {
            "verified_goes_report": bool(reference_source.startswith("NOAA/NCEI GOES")
                                         and reference.get("file_sha256")),
            "source": reference.get("source"),
            "report_sha256": reference.get("file_sha256"),
            "target_definition": (reference.get("label_report") or {}).get("target_event"),
            "significant_event_count": (reference.get("label_report") or {}).get("significant_event_count"),
            "flare_class_counts": event_report.get("flare_class_counts"),
            "significant_mx_event_count": event_report.get("significant_mx_event_count"),
            "coverage_start_utc": event_report.get("coverage_start"),
            "coverage_end_utc": event_report.get("coverage_end"),
            "forecast_window": label_report.get("forecast_window"),
            "horizon_counts": label_counts,
        },
        "splits": split_summaries,
        "phase2c_evidence": {key: isinstance(evidence, dict) and evidence.get(key) is True
                             for key in PHASE2C_EVIDENCE},
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": warnings,
        "minimum_sample_policy": {
            "source": "current model validation contract",
            "training_requires_both_binary_classes_per_horizon": True,
            "independent_test_set_required": True,
            "fixed minimum event counts": "not configured; no threshold is invented by this report",
        },
    }
    return report


def render_readiness_markdown(report: dict) -> str:
    dataset = report["dataset"]
    lines = [
        "# Scientific Data Readiness",
        "",
        f"- **Status:** {report['status']}",
        f"- **Training allowed:** {report['training_allowed']}",
        f"- **Generated (UTC):** {report['generated_at_utc']}",
        f"- **Dataset:** {dataset.get('name') or 'none'} (source: {dataset.get('source') or 'unknown'})",
        f"- **Rows:** {dataset['rows']}",
        f"- **UTC period:** {dataset['period']['start_utc']} to {dataset['period']['end_utc']}",
        f"- **Median cadence:** {dataset['median_cadence_seconds']} seconds",
        f"- **Paired flux rows:** {dataset['paired_channel_rows']}",
        "",
        "## GOES label counts",
        "",
        "| Horizon | Known | Positive | Negative | Unknown | Invalid |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for horizon, counts in report["reference_labels"]["horizon_counts"].items():
        lines.append(f"| {horizon} | {counts['known']} | {counts['positive']} | {counts['negative']} | {counts['unknown']} | {counts['invalid']} |")
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- {item}" for item in report["blockers"])
    if not report["blockers"]:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in report["warnings"])
    if not report["warnings"]:
        lines.append("- None")
    lines.extend([
        "",
        "This report describes only the loaded files and recorded provenance. A ready status does not replace independent scientific review.",
        "",
    ])
    return "\n".join(lines)


def generate_scientific_readiness_report() -> dict:
    """Build and persist both machine-readable JSON and human-readable Markdown."""
    report = build_scientific_readiness_report()
    root = _artifact_root()
    json_path = root / "scientific_data_readiness.json"
    markdown_path = root / "scientific_data_readiness.md"
    _atomic_write(json_path, json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _atomic_write(markdown_path, render_readiness_markdown(report))
    report["report_files"] = {
        "json": "readiness_reports/scientific_data_readiness.json",
        "markdown": "readiness_reports/scientific_data_readiness.md",
    }
    return report
