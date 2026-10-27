"""Reproducible CPU smoke benchmark through the application pipeline.

Run from backend/: python -m ml.evaluation.phase3_benchmark
The default bundled dataset is synthetic and results are smoke-test evidence,
not real solar-flare skill. Pass --csv only for a compatible, provenance-verified
dataset; unverified external uploads are intentionally blocked by the app gate.
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sklearn
import statsmodels
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.database.db import Base
from app.schemas.schemas import PreprocessRequest, TrainRequest
from app.services import artifact_catalog, model_service, predict_service, preprocess_service
from app.services.pipeline_state import PipelineState
from ml.features.manifest import data_fingerprint
from ml.features.engineering import FeatureConfig, select_feature_columns
from ml.models.baseline import MODEL_BUILDERS
from ml.models.deep_models import DEEP_MODEL_BUILDERS
from ml.models.registry import ModelRegistry


DEFAULT_MODELS = [
    "climatology", "flux_rule", "logistic_regression",
    "cnn_1d", "lstm", "cnn_lstm_fusion", "cnn_transformer", "dlstm",
]
HORIZONS = ["1h", "3h", "6h", "12h", "24h"]


def _hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_json(value):
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    return value


def _prediction_input(raw, sequence_length, featured=None, feature_columns=None):
    values = raw[["solexs_flux", "helios_flux"]].apply(pd.to_numeric, errors="coerce")
    complete = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    if featured is not None and feature_columns:
        finite_model_inputs = featured[feature_columns].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        eligible_times = pd.to_datetime(featured.loc[finite_model_inputs, "timestamp"], utc=True, format="mixed")
        raw_times = pd.to_datetime(raw.timestamp, utc=True, errors="coerce", format="mixed")
        raw_positions = pd.Series(np.arange(len(raw)), index=raw_times)
        ends = np.array([raw_positions.get(stamp, -1) for stamp in eligible_times], dtype=int)
        ends = ends[ends >= 0][::-1]
    else:
        ends = np.flatnonzero(complete)[::-1]
    for end in ends:
        start = end - sequence_length + 1
        if start >= 0 and complete[start:end + 1].all():
            frame = raw.iloc[start:end + 1][["timestamp", "solexs_flux", "helios_flux"]]
            return frame.to_dict(orient="records")
    raise ValueError(f"No complete {sequence_length}-row, two-instrument inference window exists.")


def run(args):
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "model_artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir = out_dir / "pipeline_artifacts"
    db_path = out_dir / "artifact_catalog.sqlite"

    if args.csv:
        source_path = Path(args.csv).resolve()
        raw = pd.read_csv(source_path)
        source = "upload"
        dataset_name = source_path.name
    else:
        source_path = Path(settings.sample_data_dir) / "demo_aditya_l1_solexs_helios.csv"
        if not source_path.is_absolute():
            source_path = (Path.cwd() / source_path).resolve()
        raw = pd.read_csv(source_path, parse_dates=["timestamp"])
        source = "demo"
        dataset_name = source_path.name
    if "timestamp" in raw:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce", format="mixed")
    if not {"timestamp", "solexs_flux", "helios_flux", "state"}.issubset(raw.columns):
        raise ValueError("Benchmark input needs timestamp, both flux channels, and state labels.")

    state = PipelineState(artifact_dir=pipeline_dir, restore=False)
    state.raw_df = raw.copy()
    state.dataset_meta = {
        "source": source, "filename": dataset_name, "synthetic": source == "demo",
        "event_metadata_verified": False,
        "scientific_status": "synthetic_demo_smoke" if source == "demo" else "unverified_external_input",
    }
    state.set_stage("raw", raw.copy(), {"input_rows": len(raw), "output_rows": len(raw)}, 0.0)

    # Isolate all generated models, manifests, pipeline artifacts, and catalog rows.
    settings.model_dir = str(model_dir)
    registry = ModelRegistry(str(model_dir))
    model_service.registry = registry
    predict_service.registry = registry
    preprocess_service.pipeline_state = state
    model_service.pipeline_state = state
    predict_service.pipeline_state = state
    engine = create_engine(f"sqlite:///{db_path.as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    artifact_catalog.SessionLocal = sessionmaker(autoflush=False, autocommit=False, bind=engine)

    preprocess_result = preprocess_service.run_full_pipeline(PreprocessRequest(
        target_cadence="1min", train_fraction=0.70, validation_fraction=0.15,
        preflare_minutes=30,
    ))
    audit = preprocess_result["split_audit"]
    if [row["partition"] for row in audit["partitions"]] != ["train", "validation", "test"]:
        raise RuntimeError("Split audit does not contain ordered train/validation/test partitions.")
    train_end = pd.to_datetime(state.train_df.timestamp, utc=True).max()
    val_start = pd.to_datetime(state.val_df.timestamp, utc=True).min()
    val_end = pd.to_datetime(state.val_df.timestamp, utc=True).max()
    test_start = pd.to_datetime(state.test_df.timestamp, utc=True).min()
    if not train_end < val_start <= val_end < test_start:
        raise RuntimeError("Chronological partition boundaries are invalid.")

    model_names = args.models.split(",") if args.models else list(DEFAULT_MODELS)
    if args.all_models:
        model_names = list(dict.fromkeys([*MODEL_BUILDERS, *DEEP_MODEL_BUILDERS]))
    unknown = sorted(set(model_names) - (set(MODEL_BUILDERS) | set(DEEP_MODEL_BUILDERS)))
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}")

    records = []
    # One extra sample spans the engineered 60-minute elapsed window.
    feature_config = FeatureConfig(**state.stage_reports.get("featured", {}).get("feature_config", {}))
    benchmark_features = select_feature_columns(state.featured_df, "fusion", feature_config)
    inference_rows = _prediction_input(
        state.raw_df, max(args.sequence_length, 61), state.featured_df, benchmark_features,
    )
    for model_name in model_names:
        model_config = {
            "model_type": model_name, "input_configuration": "fusion",
            "sequence_length": args.sequence_length, "epochs": args.epochs,
            "batch_size": args.batch_size, "learning_rate": 1e-3,
            "random_seed": args.seed, "deterministic": True, "device": "cpu",
            "early_stopping_patience": 2,
            "decomposition_period_samples": 12,
            "dlstm_hidden_size": 16, "dlstm_num_layers": 1,
        }
        try:
            start = time.perf_counter()
            trained = model_service.train_model(TrainRequest(**model_config))
            duration = time.perf_counter() - start
            model_id = trained["model_id"]
            record = registry.get(model_id)
            manifest_path = record.get("params", {}).get("feature_manifest_path")
            manifest_hash = _hash_file(manifest_path) if manifest_path and os.path.isfile(manifest_path) else None
            latency = None
            inference_verified = False
            try:
                first = predict_service.predict_from_raw_window(inference_rows, model_id=model_id)
                timings = []
                for _ in range(3):
                    before = time.perf_counter()
                    later = predict_service.predict_from_raw_window(inference_rows, model_id=model_id)
                    timings.append((time.perf_counter() - before) * 1000.0)
                inference_verified = (
                    first["nowcast"] == later["nowcast"] and first["forecast"] == later["forecast"]
                    and all(0.0 <= float(p) <= 1.0 for p in first["forecast"].values())
                )
                latency = float(statistics.median(timings))
            except Exception as exc:
                inference_error = f"{type(exc).__name__}: {exc}"
            else:
                inference_error = None

            metrics = record.get("metrics", {})
            audit_by_split = {row["partition"]: row for row in audit["partitions"]}
            for horizon in HORIZONS:
                horizon_metric = metrics.get("forecast_per_horizon", {}).get(horizon, {})
                records.append({
                    "status": "completed" if inference_verified else "trained_inference_issue",
                    "model": model_name, "model_id": model_id, "model_version": record.get("version"),
                    "model_configuration": model_config,
                    "feature_manifest_sha256": manifest_hash,
                    "dataset_fingerprint": audit.get("provenance", {}).get("dataset_fingerprint"),
                    "dataset_source": source, "synthetic": source == "demo",
                    "train_start_utc": audit_by_split["train"]["start_timestamp_utc"],
                    "train_end_utc": audit_by_split["train"]["end_timestamp_utc"],
                    "validation_start_utc": audit_by_split["validation"]["start_timestamp_utc"],
                    "validation_end_utc": audit_by_split["validation"]["end_timestamp_utc"],
                    "test_start_utc": audit_by_split["test"]["start_timestamp_utc"],
                    "test_end_utc": audit_by_split["test"]["end_timestamp_utc"],
                    "forecast_horizon": horizon,
                    "random_seed": args.seed,
                    "training_samples": record.get("params", {}).get("training_samples"),
                    "validation_samples": record.get("params", {}).get("validation_samples"),
                    "test_samples": record.get("params", {}).get("test_samples"),
                    "train_positive_windows": audit_by_split["train"]["forecast_window_counts"].get(f"forecast_{horizon}", {}).get("positive"),
                    "validation_positive_windows": (record.get("params", {}).get("threshold_selection", {}).get(horizon, {})
                                                     .get("validation_positive_windows")),
                    "test_positive_windows": horizon_metric.get("positive_count"),
                    "test_negative_windows": horizon_metric.get("negative_count"),
                    "distinct_test_flare_events": audit_by_split["test"]["distinct_observed_flare_events"],
                    "test_m_class_events": audit_by_split["test"]["m_class_events"],
                    "test_x_class_events": audit_by_split["test"]["x_class_events"],
                    "rare_event_status": audit_by_split["test"]["rare_event_evaluation_status"],
                    "threshold": horizon_metric.get("threshold"),
                    "threshold_selection": record.get("params", {}).get("threshold_selection", {}).get(horizon),
                    "test_metrics": horizon_metric,
                    "train_duration_sec": record.get("params", {}).get("train_duration_sec", round(duration, 3)),
                    "inference_median_warm_ms": latency,
                    "inference_verified_deterministic": inference_verified,
                    "inference_error": inference_error,
                    "peak_memory_mb": None,
                    "device": "cpu",
                    "python_version": platform.python_version(),
                    "numpy_version": np.__version__, "pandas_version": pd.__version__,
                    "torch_version": torch.__version__, "scikit_learn_version": sklearn.__version__,
                    "statsmodels_version": statsmodels.__version__,
                    "checkpoint_path": record.get("checkpoint_path"),
                    "warnings": ["Synthetic/demo benchmark; not observational evidence."] if source == "demo" else
                                ["External dataset was not provenance-verified."],
                })
            print(f"{model_name}: trained {model_id}; inference {'ok' if inference_verified else 'unverified'}")
        except Exception as exc:
            records.append({"status": "failed", "model": model_name,
                            "model_configuration": model_config,
                            "dataset_source": source, "synthetic": source == "demo",
                            "error": f"{type(exc).__name__}: {exc}",
                            "warnings": ["No benchmark score was generated for this model."]})
            print(f"{model_name}: failed: {type(exc).__name__}: {exc}")

    payload = {
        "report_version": 1,
        "experiment": "phase3_end_to_end_cpu_smoke",
        "scientific_status": "synthetic_smoke_not_scientific_evaluation" if source == "demo" else "external_source_provenance_required",
        "source_file": str(source_path), "dataset_fingerprint": data_fingerprint(raw),
        "split_audit": audit,
        "feature_engineering": state.stage_reports.get("featured", {}),
        "preprocessing": {"config": {"target_cadence": "1min", "train_fraction": 0.70,
                                       "validation_fraction": 0.15},
                          "transformer_fit_partition": "train_only"},
        "environment": {"platform": platform.platform(), "processor": platform.processor(),
                        "cuda_available": torch.cuda.is_available(), "device": "cpu",
                        "python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "scikit_learn": sklearn.__version__, "statsmodels": statsmodels.__version__},
        "model_records": records,
        "limitations": [
            "Bundled labels are synthetic and cannot establish solar-flare forecast skill.",
            "Overlapping forecast windows are dependent; window metrics are not independent-event uncertainty estimates.",
            "Event-level warning matching and confidence intervals are not claimed in this smoke report.",
            "M/X counts are synthetic fixture counts, not observed mission events.",
        ],
    }
    json_path = out_dir / "benchmark.json"
    csv_path = out_dir / "benchmark.csv"
    json_path.write_text(json.dumps(_clean_json(payload), indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame([_clean_json(row) for row in records]).to_csv(csv_path, index=False)
    print(f"JSON report: {json_path}")
    print(f"CSV report:  {csv_path}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="Compatible dataset with timestamp, flux, state, and flare_class columns")
    parser.add_argument("--models", help="Comma-separated model identifiers; default is a representative smoke set")
    parser.add_argument("--all-models", action="store_true", help="Try every registered baseline and deep architecture")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="reports/phase3_step11_demo")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
