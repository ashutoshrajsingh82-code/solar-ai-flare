"""Executable script for Phase 4M-S self-supervised real observational data training."""
import sys
import json
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from ml.experiments.real_self_supervised import (
    SelfSupervisedConfig,
    load_and_split_solexs_real_data,
    train_self_supervised_model,
    save_experimental_checkpoint,
    SOLEXS_ARCHIVE_SHA256,
)


def main():
    print("=" * 80)
    print("Phase 4M-S: Real-Data Self-Supervised Training on SoLEXS L1 Observations")
    print("=" * 80)

    archive_path = Path(r"C:\Users\ASHUTOSH\Downloads\solexs_2026Oct23T184402685.zip")
    if not archive_path.exists():
        print(f"Error: Archive not found at {archive_path}")
        return 1

    cfg = SelfSupervisedConfig(
        sequence_length=60,
        forecast_horizon=10,
        batch_size=128,
        epochs=8,
        learning_rate=1e-3,
        random_seed=42,
        device="cpu",
    )

    print(f"1. Loading and validating real FITS observations from {archive_path.name}...")
    try:
        data_bundle = load_and_split_solexs_real_data(archive_path, cfg)
    except Exception as exc:
        print(f"Data loading failed: {type(exc).__name__}: {exc}")
        return 2

    print(f"   - Verified Archive SHA-256: {data_bundle['source_sha256']}")
    print(f"   - Total Observation Records: {data_bundle['total_records']}")
    print(f"   - Finite Valid Records: {data_bundle['finite_records']}")
    print(f"   - Unobserved/NaN Records: {data_bundle['nan_records']}")
    print(f"   - Chronological Splits:")
    print(f"       Train: {data_bundle['split_boundaries_utc']['train']} -> {len(data_bundle['X_train'])} windows")
    print(f"       Validation: {data_bundle['split_boundaries_utc']['val']} -> {len(data_bundle['X_val'])} windows")
    print(f"       Held-out Test: {data_bundle['split_boundaries_utc']['test']} -> {len(data_bundle['X_test'])} windows")

    print("\n2. Training Self-Supervised CNN-LSTM on real count rates...")
    model, eval_report = train_self_supervised_model(data_bundle, cfg)

    print("\n3. Training History:")
    for h in eval_report["history"]:
        print(f"   Epoch {h['epoch']:02d}: train_loss={h['train_loss']:.4f}, val_loss={h['val_loss']:.4f}")

    print("\n4. Evaluation on Held-Out Test Set (2026-02-05, native counts/s):")
    test_metrics = eval_report["test_metrics_native_units"]
    cnn_lstm = test_metrics["cnn_lstm"]
    persist = test_metrics["persistence_baseline"]
    mean_bs = test_metrics["rolling_mean_baseline"]
    skill = test_metrics["skill_score_vs_persistence"]

    print(f"   Model (CNN-LSTM)        : MAE = {cnn_lstm['mae']:.2f} counts/s, RMSE = {cnn_lstm['rmse']:.2f} counts/s, RelMAE = {cnn_lstm['rel_mae']*100:.2f}%")
    print(f"   Persistence Baseline    : MAE = {persist['mae']:.2f} counts/s, RMSE = {persist['rmse']:.2f} counts/s, RelMAE = {persist['rel_mae']*100:.2f}%")
    print(f"   Rolling Mean Baseline   : MAE = {mean_bs['mae']:.2f} counts/s, RMSE = {mean_bs['rmse']:.2f} counts/s, RelMAE = {mean_bs['rel_mae']*100:.2f}%")
    print(f"   Skill Score vs Persist  : {skill * 100:+.2f}%")

    out_dir = backend_dir / "artifacts" / "real_data_experiments"
    print(f"\n5. Saving checkpoint to {out_dir}...")
    weights_path, meta_path = save_experimental_checkpoint(model, data_bundle, eval_report, cfg, out_dir)
    print(f"   - Checkpoint: {weights_path.name}")
    print(f"   - Metadata  : {meta_path.name}")

    print("\n" + "=" * 80)
    print("SUCCESS: Real observational self-supervised training completed successfully.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
