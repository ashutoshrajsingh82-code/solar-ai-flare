import time
from app.services.pipeline_state import pipeline_state
from app.schemas.schemas import PreprocessRequest

from ml.preprocessing.cleaning import clean_dataframe, CleaningConfig
from ml.preprocessing.alignment import align_and_resample, AlignmentConfig
from ml.preprocessing.transform import FluxTransformer, TransformConfig
from ml.features.engineering import add_rolling_features, FeatureConfig, get_feature_columns
from ml.features.labels import generate_forecast_labels, apply_preflare_window, LabelConfig
from ml.features.windowing import chronological_split

CADENCE_SECONDS_MAP = {"1s": 1, "10s": 10, "1min": 60, "5min": 300}


def run_full_pipeline(req: PreprocessRequest) -> dict:
    if pipeline_state.raw_df is None:
        raise ValueError("No dataset loaded. Load demo data or upload a dataset first.")

    # --- Cleaning ---
    t0 = time.time()
    cleaning_cfg = CleaningConfig(spike_zscore_threshold=req.spike_zscore_threshold)
    cleaned_df, clean_report = clean_dataframe(pipeline_state.raw_df, cleaning_cfg)
    pipeline_state.set_stage("cleaned", cleaned_df, clean_report, time.time() - t0)

    # --- Alignment / resampling ---
    t0 = time.time()
    align_cfg = AlignmentConfig(target_cadence=req.target_cadence)
    aligned_df, align_report = align_and_resample(cleaned_df, align_cfg)
    pipeline_state.set_stage("aligned", aligned_df, align_report, time.time() - t0)
    cadence_seconds = CADENCE_SECONDS_MAP[req.target_cadence]
    pipeline_state.cadence_seconds = cadence_seconds

    # --- Chronological split BEFORE fitting scaler (never fit on test data) ---
    train_df, val_df, test_df = chronological_split(aligned_df)
    pipeline_state.train_df, pipeline_state.val_df, pipeline_state.test_df = train_df, val_df, test_df

    # --- Transform (log + scale), fit on TRAIN only ---
    t0 = time.time()
    transformer = FluxTransformer(TransformConfig(scaler_type=req.scaler_type))
    transformer.fit(train_df)
    transformed_train = transformer.transform(train_df)
    transformed_val = transformer.transform(val_df) if len(val_df) else val_df
    transformed_test = transformer.transform(test_df) if len(test_df) else test_df
    import pandas as pd
    transformed_df = pd.concat([transformed_train, transformed_val, transformed_test], ignore_index=True)
    pipeline_state.transformer = transformer
    transform_report = {
        "input_rows": len(aligned_df), "output_rows": len(transformed_df),
        "scaler_type": req.scaler_type, "fit_on": "train_split_only",
        "train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df),
        "warnings": [],
    }
    pipeline_state.set_stage("transformed", transformed_df, transform_report, time.time() - t0)

    # --- Feature engineering ---
    t0 = time.time()
    feat_cfg = FeatureConfig(cadence_seconds=cadence_seconds, compute_stl=req.compute_stl)
    featured_df = add_rolling_features(transformed_df, feat_cfg)
    feature_cols = get_feature_columns(featured_df)
    pipeline_state.feature_columns = feature_cols
    feature_report = {
        "input_rows": len(transformed_df), "output_rows": len(featured_df),
        "n_features_generated": len(feature_cols), "feature_names_sample": feature_cols[:15],
        "warnings": [],
    }
    pipeline_state.set_stage("featured", featured_df, feature_report, time.time() - t0)

    # --- Label generation ---
    t0 = time.time()
    label_cfg = LabelConfig(cadence_seconds=cadence_seconds, preflare_minutes=req.preflare_minutes)
    labeled_df = apply_preflare_window(featured_df, label_cfg)
    labeled_df = generate_forecast_labels(labeled_df, label_cfg)
    label_report = {
        "input_rows": len(featured_df), "output_rows": len(labeled_df),
        "preflare_minutes": req.preflare_minutes,
        "forecast_horizons": [1, 3, 6, 12, 24],
        "positive_rate_1h": float(labeled_df["forecast_1h"].mean()) if len(labeled_df) else 0.0,
        "positive_rate_24h": float(labeled_df["forecast_24h"].mean()) if len(labeled_df) else 0.0,
        "warnings": [],
    }
    pipeline_state.set_stage("labeled", labeled_df, label_report, time.time() - t0)

    return {
        "status": "complete",
        "stages": pipeline_state.pipeline_status(),
        "feature_columns_count": len(feature_cols),
        "train_date_range": [str(train_df["timestamp"].min()), str(train_df["timestamp"].max())] if len(train_df) else None,
        "val_date_range": [str(val_df["timestamp"].min()), str(val_df["timestamp"].max())] if len(val_df) else None,
        "test_date_range": [str(test_df["timestamp"].min()), str(test_df["timestamp"].max())] if len(test_df) else None,
        "note": "Chronological split (earliest 70% train / next 15% val / latest 15% test) prevents future information leaking into training. Scaler fit on train split only.",
    }
