"""Isolated synthetic-only training smoke path; never registers production models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.evaluation.metrics import evaluate_binary_forecast
from ml.features.windowing import WindowConfig, build_windows, chronological_split
from ml.models.deep_models import DEEP_MODEL_BUILDERS
from ml.preprocessing.transform import FluxTransformer
from ml.training.eligibility import real_training_blockers
from ml.training.train import TrainConfig, train_deep_model

SYNTHETIC_TAG = "SYNTHETIC_NOT_FOR_PRODUCTION"
HORIZON_HOURS = (1, 3, 6, 12, 24)
HORIZON_COLS = [f"forecast_{hours}h" for hours in HORIZON_HOURS]


def require_observational_readiness(dataset_meta: dict, readiness_report: dict) -> None:
    """Fail closed unless both existing provenance and scientific gates pass."""
    blockers = real_training_blockers(dataset_meta)
    if not readiness_report or readiness_report.get("training_allowed") is not True:
        report_blockers = (readiness_report or {}).get("blockers") or [
            "Phase 3C/scientific-readiness report is absent or not READY."
        ]
        blockers.extend(f"Scientific readiness: {reason}" for reason in report_blockers)
    if blockers:
        raise ValueError("Real observational training is blocked: " + " ".join(blockers))


def make_synthetic_observations(seed: int = 42, rows: int = 9000) -> pd.DataFrame:
    """Create reproducible hourly toy fluxes, state labels, and an event timeline."""
    if rows < 100:
        raise ValueError("Synthetic fixture generation requires at least 100 hourly rows.")
    rng = np.random.default_rng(seed)
    index = np.arange(rows)
    # A deliberately regular synthetic event schedule, not a downloaded or
    # user-provided event catalog. The 4-state labels encode toy event phases.
    event_centres = np.arange(36, rows, 72)
    event = np.zeros(rows, dtype=bool)
    state = np.zeros(rows, dtype=np.int64)
    for centre in event_centres:
        if centre - 2 >= 0:
            state[centre - 2:centre] = 1  # synthetic pre-event
        state[centre:min(rows, centre + 2)] = 2  # synthetic event
        state[centre + 2:min(rows, centre + 4)] = 3  # synthetic decay
        event[centre:min(rows, centre + 2)] = True
    soft = np.maximum(0.05, 1.0 + 0.04 * rng.normal(size=rows) + 0.8 * (state > 0))
    hard = np.maximum(0.05, 0.7 + 0.05 * rng.normal(size=rows) + 1.1 * (state >= 2))
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=rows, freq="h", tz="UTC"),
        "solexs_flux": soft,
        "helios_flux": hard,
        "state": state,
        "_synthetic_event": event,
        "_synthetic_row": index,
    })
    return frame


def _targets_within_partition(partition: pd.DataFrame) -> pd.DataFrame:
    """Label only when the full future horizon remains within this partition."""
    result = partition.copy()
    events = result["_synthetic_event"].to_numpy(dtype=bool)
    for hours, column in zip(HORIZON_HOURS, HORIZON_COLS):
        labels = np.full(len(result), np.nan, dtype=float)
        for i in range(len(result)):
            end = i + hours
            if end < len(result):
                labels[i] = float(events[i + 1:end + 1].any())
        result[column] = labels
    return result


def _window_partition(partition: pd.DataFrame, transformer: FluxTransformer,
                      sequence_length: int):
    transformed = transformer.transform(partition)
    return build_windows(
        transformed,
        WindowConfig(sequence_length=sequence_length, channel_cols=[
            "solexs_flux_scaled", "helios_flux_scaled"
        ], horizon_cols=HORIZON_COLS),
    )


def run_synthetic_smoke(output: str | Path, seed: int = 42, rows: int = 9000,
                        epochs: int = 2) -> dict:
    """Train/evaluate/save/reload a toy model without touching the production registry."""
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing synthetic checkpoint: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    synthetic = make_synthetic_observations(seed=seed, rows=rows)
    raw_train, raw_val, raw_test = chronological_split(synthetic)
    # Fit preprocessing using training rows only. Future partitions only call transform.
    transformer = FluxTransformer().fit(raw_train)
    train = _targets_within_partition(raw_train)
    val = _targets_within_partition(raw_val)
    test = _targets_within_partition(raw_test)
    x_train, y_now_train, y_fc_train, train_times = _window_partition(train, transformer, 24)
    x_val, y_now_val, y_fc_val, val_times = _window_partition(val, transformer, 24)
    config = TrainConfig(
        model_type="lstm", sequence_length=24, batch_size=128, epochs=epochs,
        n_channels=2,
        random_seed=seed, deterministic=True, shuffle_train_windows=False,
        early_stopping_patience=0, device="cpu",
    )
    model, history = train_deep_model(
        x_train, y_now_train, y_fc_train, x_val, y_now_val, y_fc_val, config=config,
    )

    # The final chronological partition is first accessed after training and
    # validation-based checkpoint selection have finished.
    x_test, y_now_test, y_fc_test, test_times = _window_partition(test, transformer, 24)
    model.eval()
    with torch.no_grad():
        now_logits, forecast_logits = model(torch.as_tensor(x_test, dtype=torch.float32))
        forecast_prob = torch.sigmoid(forecast_logits).cpu().numpy()
        now_predictions = torch.argmax(now_logits, dim=1).cpu().numpy()
    metrics = {}
    for horizon_index, horizon in enumerate(HORIZON_HOURS):
        known = np.isfinite(y_fc_test[:, horizon_index])
        if known.any():
            metrics[f"{horizon}h"] = evaluate_binary_forecast(
                y_fc_test[known, horizon_index], forecast_prob[known, horizon_index]
            )

    metadata = {
        "dataset_status": SYNTHETIC_TAG,
        "model_status": SYNTHETIC_TAG,
        "production_registry_eligible": False,
        "source": "explicitly_generated_synthetic_only",
        "scientific_forecasting_performance_claim": False,
        "feature_channels": ["solexs_flux_scaled", "helios_flux_scaled"],
        "feature_causality": "ordered historical input windows only",
        "split_policy": "chronological 70/15/15; labels computed within each partition; horizon tails unknown",
        "preprocessing_fit": "training partition only",
        "preprocessor": {
            "columns": list(transformer.columns),
            "reference_scale": dict(transformer.reference_scale),
            "scaler_type": transformer.config.scaler_type,
            "log_transform": transformer.config.log_transform,
            "scaler_mean": transformer.scaler.mean_.tolist(),
            "scaler_scale": transformer.scaler.scale_.tolist(),
        },
        "seed": seed,
        "config": vars(config),
        "best_epoch": model._training_metadata["best_epoch"],
        "training_metadata": model._training_metadata,
        "partition_windows": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
        "partition_time_bounds": {
            "train": [str(train_times[0]), str(train_times[-1])],
            "validation": [str(val_times[0]), str(val_times[-1])],
            "test": [str(test_times[0]), str(test_times[-1])],
        },
        "test_metrics_synthetic_only": metrics,
        "nowcast_prediction_count": int(len(now_predictions)),
    }
    checkpoint = {
        "model_type": config.model_type,
        "n_channels": config.n_channels,
        "architecture_version": 1,
        "model_config": {"dropout": config.dropout},
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    # Verify a candidate reload before publishing the file, then write once.
    loaded_model = DEEP_MODEL_BUILDERS[config.model_type](
        n_channels=config.n_channels,
        dropout=config.dropout,
    )
    loaded_model.load_state_dict(checkpoint["state_dict"])
    loaded_model.eval()
    with torch.no_grad():
        loaded_now, loaded_forecast = loaded_model(torch.as_tensor(x_test[:2], dtype=torch.float32))
    if not torch.allclose(loaded_now, now_logits[:2].cpu(), rtol=1e-6, atol=1e-7) or not torch.allclose(loaded_forecast, forecast_logits[:2].cpu(), rtol=1e-6, atol=1e-7):
        raise RuntimeError("Synthetic checkpoint reload did not reproduce inference outputs.")
    torch.save(checkpoint, output)
    saved_checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    disk_model = DEEP_MODEL_BUILDERS[saved_checkpoint["model_type"]](
        n_channels=saved_checkpoint["n_channels"], dropout=saved_checkpoint["model_config"]["dropout"]
    )
    disk_model.load_state_dict(saved_checkpoint["state_dict"])
    disk_model.eval()
    with torch.no_grad():
        disk_now, disk_forecast = disk_model(torch.as_tensor(x_test[:2], dtype=torch.float32))
    if not torch.allclose(disk_now, now_logits[:2].cpu(), rtol=1e-6, atol=1e-7) or not torch.allclose(disk_forecast, forecast_logits[:2].cpu(), rtol=1e-6, atol=1e-7):
        raise RuntimeError("Saved synthetic checkpoint failed disk reload inference verification.")
    return {"checkpoint": str(output), "metadata": metadata,
            "metrics": metrics, "history": history}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New checkpoint path; existing files are never overwritten")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows", type=int, default=9000)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args(argv)
    result = run_synthetic_smoke(args.output, seed=args.seed, rows=args.rows, epochs=args.epochs)
    metadata = result["metadata"]
    summary = {
        "checkpoint": result["checkpoint"],
        "dataset_status": metadata["dataset_status"],
        "model_status": metadata["model_status"],
        "production_registry_eligible": metadata["production_registry_eligible"],
        "seed": metadata["seed"],
        "best_epoch": metadata["best_epoch"],
        "partition_windows": metadata["partition_windows"],
        "partition_time_bounds": metadata["partition_time_bounds"],
        "test_metrics_synthetic_only": result["metrics"],
        "scientific_forecasting_performance_claim": False,
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
