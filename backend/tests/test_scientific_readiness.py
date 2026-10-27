import json

import numpy as np
import pandas as pd

from app.core.config import settings
from app.schemas.schemas import TrainRequest
from app.services import model_service, scientific_readiness


def test_native_solexs_counts_cannot_pass_physical_flux_readiness():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=4, freq="s", tz="UTC"),
        "solexs_counts_per_second": [10.0, 11.0, 12.0, 13.0],
    })
    report = scientific_readiness.build_scientific_readiness_report(
        dataset_meta={
            "source": "solexs_real", "observation_source": "SoLEXS",
            "ingestion_report": {"measurement_unit": "counts/s"},
        },
        observations=frame,
        splits={"train": None, "validation": None, "test": None},
    )

    assert report["status"] == "BLOCKED"
    assert report["training_allowed"] is False
    assert report["dataset"]["rows"] == 4
    assert report["dataset"]["median_cadence_seconds"] == 1.0
    assert report["measurement_contract"]["native_counts_preserved"] is True
    assert any("native COUNTS/s" in blocker for blocker in report["blockers"])
    assert any("helios_flux" in blocker for blocker in report["blockers"])
    assert any("calibrated physical flux" in blocker for blocker in report["blockers"])


def test_readiness_report_writes_json_and_markdown_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_mode", "real")
    monkeypatch.setattr(settings, "pipeline_artifact_dir", str(tmp_path / "pipeline"))
    monkeypatch.setattr(scientific_readiness.pipeline_state, "dataset_meta", {})
    monkeypatch.setattr(scientific_readiness.pipeline_state, "raw_df", None)
    monkeypatch.setattr(scientific_readiness.pipeline_state, "train_df", None)
    monkeypatch.setattr(scientific_readiness.pipeline_state, "val_df", None)
    monkeypatch.setattr(scientific_readiness.pipeline_state, "test_df", None)

    report = scientific_readiness.generate_scientific_readiness_report()
    root = tmp_path / "pipeline" / "readiness_reports"
    json_path = root / "scientific_data_readiness.json"
    markdown_path = root / "scientific_data_readiness.md"

    stored = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["status"] == "BLOCKED"
    assert stored["training_allowed"] is False
    assert "not configured" in stored["minimum_sample_policy"]["fixed minimum event counts"]
    assert "No observation dataset is currently loaded." in markdown_path.read_text(encoding="utf-8")
    assert not list(root.glob(".readiness-*.tmp"))


def test_goes_label_counts_are_reported_without_filling_unknowns():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="min", tz="UTC"),
        "forecast_1h": [1.0, 0.0, np.nan],
    })
    report = scientific_readiness.build_scientific_readiness_report(
        dataset_meta={"source": "fixture"}, observations=frame,
        splits={"train": None, "validation": None, "test": None},
    )
    assert report["reference_labels"]["horizon_counts"]["1h"] == {
        "known": 2, "positive": 1, "negative": 1, "unknown": 1, "invalid": 0,
    }


def test_unverified_real_training_fails_with_persisted_readiness_diagnostic(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_mode", "real")
    monkeypatch.setattr(settings, "pipeline_artifact_dir", str(tmp_path / "pipeline"))
    monkeypatch.setattr(scientific_readiness.pipeline_state, "dataset_meta", {"source": "upload"})
    monkeypatch.setattr(scientific_readiness.pipeline_state, "raw_df", pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=2, freq="min", tz="UTC"),
        "solexs_flux": [1.0, 2.0], "helios_flux": [1.0, 2.0],
    }))

    try:
        model_service._train_model_locked(TrainRequest(model_type="logistic_regression"))
    except ValueError as exc:
        assert "Real-data training is blocked by the scientific readiness gate" in str(exc)
        assert "readiness_reports/scientific_data_readiness.json" in str(exc)
    else:
        raise AssertionError("Unverified real observations must not enter model training")
    assert (tmp_path / "pipeline" / "readiness_reports" / "scientific_data_readiness.json").is_file()
