import json

import pandas as pd

from app.services.pipeline_state import PipelineState
from ml.preprocessing.transform import FluxTransformer


def test_pipeline_frames_metadata_transformer_and_splits_restore(tmp_path):
    state = PipelineState(tmp_path)
    raw = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=4, freq="min"),
                        "solexs_flux": [1., 2., 3., 4.], "helios_flux": [4., 3., 2., 1.]})
    state.dataset_meta = {"filename": "small.csv", "source": "test"}
    state.cadence_seconds = 60
    state.feature_columns = ["feature_a"]
    state.set_stage("raw", raw, {"input_rows": 4, "output_rows": 4, "warnings": []}, 0.01)
    original_fingerprint = state.snapshot_fingerprint()
    state.train_df, state.val_df, state.test_df = raw.iloc[:2], raw.iloc[2:3], raw.iloc[3:]
    state.transformer = FluxTransformer().fit(raw.iloc[:2])
    state.transformed_df = state.transformer.transform(raw)
    state.set_stage("transformed", state.transformed_df, {"input_rows": 4, "output_rows": 4, "warnings": []}, 0.01)
    state.set_stage("featured", state.transformed_df.copy(), {"input_rows": 4, "output_rows": 4, "warnings": []}, 0.01)
    assert state.snapshot_fingerprint() != original_fingerprint
    state.set_stage("labeled", state.transformed_df.copy(), {"input_rows": 4, "output_rows": 4, "warnings": []}, 0.01)
    state.persist()

    restored = PipelineState(tmp_path)
    assert restored.dataset_meta["filename"] == "small.csv"
    assert restored.cadence_seconds == 60
    assert restored.feature_columns == ["feature_a"]
    assert restored.stage_reports["transformed"]["status"] == "complete"
    assert len(restored.raw_df) == 4
    assert len(restored.train_df) == 2 and len(restored.val_df) == 1 and len(restored.test_df) == 1
    assert restored.transformer.fitted
    assert restored.snapshot_fingerprint() == state.snapshot_fingerprint()
    assert restored.is_ready_for_training()
    assert restored.transformer.transform(raw)["solexs_flux_scaled"].tolist() == state.transformer.transform(raw)["solexs_flux_scaled"].tolist()


def test_missing_corrupt_and_path_traversal_artifacts_are_reported(tmp_path):
    state = PipelineState(tmp_path)
    df = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=2, freq="min"), "v": [1, 2]})
    state.set_stage("raw", df, {"warnings": []}, 0.01)
    manifest = json.loads(state.manifest_path.read_text())
    artifact = tmp_path / manifest["artifacts"]["raw"]["file"]
    artifact.write_text("corrupt")
    restored = PipelineState(tmp_path)
    assert restored.raw_df is None
    assert restored.pipeline_status()[0]["status"] == "unavailable"
    assert "integrity" in restored.pipeline_status()[0]["warnings"][0]

    manifest["artifacts"]["raw"]["file"] = "../outside.csv"
    state.manifest_path.write_text(json.dumps(manifest))
    unsafe = PipelineState(tmp_path)
    assert unsafe.raw_df is None
    assert unsafe.pipeline_status()[0]["status"] == "unavailable"


def test_failed_manifest_update_preserves_last_good_artifact(tmp_path, monkeypatch):
    state = PipelineState(tmp_path)
    original = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=2, freq="min"), "v": [1, 2]})
    state.set_stage("raw", original, {"output_rows": 2, "warnings": []}, 0.01)
    original_file = state._artifacts["raw"]["file"]

    def fail_manifest():
        raise OSError("simulated manifest write failure")

    monkeypatch.setattr(state, "_write_manifest", fail_manifest)
    try:
        state.set_stage("raw", original.assign(v=[8, 9]), {"output_rows": 2, "warnings": []}, 0.02)
    except OSError:
        pass
    else:
        raise AssertionError("write failure should propagate")

    restored = PipelineState(tmp_path)
    assert restored._artifacts["raw"]["file"] == original_file
    assert restored.raw_df["v"].tolist() == [1, 2]
