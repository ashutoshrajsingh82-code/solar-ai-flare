import pytest
import pandas as pd
import numpy as np
from ml.data.synthetic_generator import generate_synthetic_dataset
from ml.preprocessing.cleaning import clean_dataframe
from ml.preprocessing.alignment import align_and_resample
from ml.preprocessing.transform import FluxTransformer
from ml.features.engineering import add_rolling_features, get_feature_columns
from ml.features.labels import generate_forecast_labels
from ml.features.windowing import chronological_split


def test_synthetic_data_generator():
    df, events = generate_synthetic_dataset(n_days=1, cadence_seconds=60, n_flares=2, seed=42)
    assert len(df) > 1000
    assert "timestamp" in df.columns
    assert "solexs_flux" in df.columns
    assert "helios_flux" in df.columns
    assert "flare_class" in df.columns
    assert "state" in df.columns

    # Verify positive flux values
    assert (df["solexs_flux"].dropna() > 0).all()
    assert (df["helios_flux"].dropna() > 0).all()


def test_cleaning_and_alignment():
    df, _ = generate_synthetic_dataset(n_days=1, cadence_seconds=60, n_flares=1, seed=42)
    cleaned_df, report = clean_dataframe(df)
    assert len(cleaned_df) > 0

    aligned_df, align_report = align_and_resample(cleaned_df)
    assert len(aligned_df) > 0


def test_feature_engineering_hard_to_soft_ratio():
    df, _ = generate_synthetic_dataset(n_days=1, cadence_seconds=60, n_flares=1, seed=42)
    cleaned_df, _ = clean_dataframe(df)
    aligned_df, _ = align_and_resample(cleaned_df)

    scaler = FluxTransformer()
    transformed_df = scaler.fit_transform(aligned_df)

    featured_df = add_rolling_features(transformed_df)
    feat_cols = get_feature_columns(featured_df)

    # Verify Hard-to-Soft Ratio is computed
    assert "hard_soft_ratio" in featured_df.columns
    assert any("hard_soft_ratio" in col for col in feat_cols)


def test_forecast_labels():
    df, _ = generate_synthetic_dataset(n_days=1, cadence_seconds=60, n_flares=2, seed=42)
    labeled_df = generate_forecast_labels(df)

    for h in [1, 3, 6, 12, 24]:
        assert f"forecast_{h}h" in labeled_df.columns
        assert set(labeled_df[f"forecast_{h}h"].unique()).issubset({0, 1})


def test_chronological_split_no_leakage():
    n = 1000
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-09-01", periods=n, freq="min"),
        "val": np.arange(n),
    })

    train_df, val_df, test_df = chronological_split(df, train_frac=0.7, val_frac=0.15)
    assert len(train_df) == 700
    assert len(val_df) == 150
    assert len(test_df) == 150

    # Ensure strictly monotonic timestamps (no shuffling)
    assert train_df["timestamp"].max() < val_df["timestamp"].min()
    assert val_df["timestamp"].max() < test_df["timestamp"].min()
