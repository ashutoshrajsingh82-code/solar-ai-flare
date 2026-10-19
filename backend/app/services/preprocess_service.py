
import time

import pandas as pd

from app.services.pipeline_state import pipeline_state
from app.schemas.schemas import PreprocessRequest

from ml.preprocessing.cleaning import clean_dataframe, CleaningConfig
from ml.preprocessing.alignment import align_and_resample, AlignmentConfig
from ml.preprocessing.transform import FluxTransformer, TransformConfig
from ml.features.engineering import (
    add_rolling_features,
    FeatureConfig,
    get_feature_columns,
)
from ml.features.labels import (
    generate_forecast_labels,
    apply_preflare_window,
    LabelConfig,
)
from ml.features.windowing import chronological_split


CADENCE_SECONDS_MAP = {
    "1s": 1,
    "10s": 10,
    "1min": 60,
    "5min": 300,
}


def run_full_pipeline(req: PreprocessRequest, stage_callback=None) -> dict:
    # PipelineState is process-local; serialize writers/readers that consume it.
    with pipeline_state.operation_lock:
        return _run_full_pipeline(req, stage_callback)


def _run_full_pipeline(req: PreprocessRequest, stage_callback=None) -> dict:
    if pipeline_state.raw_df is None:
        raise ValueError(
            "No dataset loaded. Load demo data or upload a dataset first."
        )

    required_flux = {"timestamp", "solexs_flux", "helios_flux"}
    if not required_flux.issubset(pipeline_state.raw_df.columns):
        source = pipeline_state.dataset_meta.get("source", "unknown")
        raise ValueError(
            f"The loaded {source} observations do not match the existing dual-flux pipeline contract. "
            "Raw SoLEXS/HEL1OS counts cannot be relabeled as flux; verified instrument adapters and reference flare labels are required."
        )

    if "state" not in pipeline_state.raw_df.columns or pipeline_state.raw_df["state"].isna().any():
        raise ValueError(
            "Training labels are missing or unknown. Provide verified reference labels and their provenance; "
            "unlabeled observations must not be treated as quiet intervals."
        )

    def report_stage(stage):
        if stage_callback:
            stage_callback(stage)

    # --- Cleaning ---
    report_stage("cleaning")
    t0 = time.time()

    cleaning_cfg = CleaningConfig(
        spike_zscore_threshold=req.spike_zscore_threshold
    )

    cleaned_df, clean_report = clean_dataframe(
        pipeline_state.raw_df,
        cleaning_cfg,
    )

    pipeline_state.set_stage(
        "cleaned",
        cleaned_df,
        clean_report,
        time.time() - t0,
    )

    # --- Alignment / resampling ---
    report_stage("alignment")
    t0 = time.time()

    align_cfg = AlignmentConfig(
        target_cadence=req.target_cadence
    )

    aligned_df, align_report = align_and_resample(
        cleaned_df,
        align_cfg,
    )

    pipeline_state.set_stage(
        "aligned",
        aligned_df,
        align_report,
        time.time() - t0,
    )

    cadence_seconds = CADENCE_SECONDS_MAP[req.target_cadence]
    pipeline_state.cadence_seconds = cadence_seconds

    # --- Chronological split BEFORE fitting the scaler ---
    train_df, val_df, test_df = chronological_split(aligned_df)

    pipeline_state.train_df = train_df
    pipeline_state.val_df = val_df
    pipeline_state.test_df = test_df

    # --- Transform: fit scaler on TRAIN only ---
    report_stage("transforming")
    t0 = time.time()

    transformer = FluxTransformer(
        TransformConfig(scaler_type=req.scaler_type)
    )

    transformer.fit(train_df)

    transformed_train = transformer.transform(train_df)
    transformed_val = (
        transformer.transform(val_df) if len(val_df) else val_df
    )
    transformed_test = (
        transformer.transform(test_df) if len(test_df) else test_df
    )

    transformed_df = pd.concat(
        [transformed_train, transformed_val, transformed_test],
        ignore_index=True,
    )

    pipeline_state.transformer = transformer

    transform_report = {
        "input_rows": len(aligned_df),
        "output_rows": len(transformed_df),
        "scaler_type": req.scaler_type,
        "fit_on": "train_split_only",
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "warnings": [],
    }

    pipeline_state.set_stage(
        "transformed",
        transformed_df,
        transform_report,
        time.time() - t0,
    )

    # --- Feature engineering ---
    report_stage("feature_engineering")
    t0 = time.time()

    feat_cfg = FeatureConfig(
        cadence_seconds=cadence_seconds,
        compute_stl=req.compute_stl,
    )

    featured_df = add_rolling_features(
        transformed_df,
        feat_cfg,
    )

    feature_cols = get_feature_columns(featured_df)
    pipeline_state.feature_columns = feature_cols

    feature_report = {
        "input_rows": len(transformed_df),
        "output_rows": len(featured_df),
        "n_features_generated": len(feature_cols),
        "feature_names_sample": feature_cols[:15],
        "warnings": [],
    }

    pipeline_state.set_stage(
        "featured",
        featured_df,
        feature_report,
        time.time() - t0,
    )

    # --- Label generation ---
    report_stage("labeling")
    t0 = time.time()

    label_cfg = LabelConfig(
        cadence_seconds=cadence_seconds,
        preflare_minutes=req.preflare_minutes,
    )

    labeled_df, boundary_purged = generate_split_scoped_labels(
        featured_df,
        {"train": train_df, "validation": val_df, "test": test_df},
        label_cfg,
    )

    # --- Count positive labels in each chronological split ---
    forecast_columns = [
        "forecast_1h",
        "forecast_3h",
        "forecast_6h",
        "forecast_12h",
        "forecast_24h",
    ]

    positive_counts_by_split = {}

    split_dataframes = {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }

    for split_name, split_df in split_dataframes.items():
        split_timestamps = split_df["timestamp"]

        split_labels = labeled_df[
            labeled_df["timestamp"].isin(split_timestamps)
        ]

        positive_counts_by_split[split_name] = {
            column: int(split_labels[column].sum())
            for column in forecast_columns
        }

    # Count all rows and positive labels for each horizon.
    positive_totals = {
        column: int(labeled_df[column].sum())
        for column in forecast_columns
    }

    label_report = {
        "input_rows": len(featured_df),
        "output_rows": len(labeled_df),
        "preflare_minutes": req.preflare_minutes,
        "forecast_horizons": [1, 3, 6, 12, 24],
        "positive_rate_1h": (
            float(labeled_df["forecast_1h"].mean())
            if len(labeled_df)
            else 0.0
        ),
        "positive_rate_24h": (
            float(labeled_df["forecast_24h"].mean())
            if len(labeled_df)
            else 0.0
        ),
        "positive_counts_by_split": positive_counts_by_split,
        "positive_totals": positive_totals,
        "boundary_purged_rows_by_split_and_horizon": boundary_purged,
        "warnings": [],
    }

    pipeline_state.set_stage(
        "labeled",
        labeled_df,
        label_report,
        time.time() - t0,
    )

    # Persist the complete split and transformer snapshot after a successful run.
    report_stage("persisting_artifacts")
    pipeline_state.persist(include_frames=False)
    report_stage("completed")

    return {
        "status": "complete",
        "stages": pipeline_state.pipeline_status(),
        "feature_columns_count": len(feature_cols),
        "train_date_range": (
            [
                str(train_df["timestamp"].min()),
                str(train_df["timestamp"].max()),
            ]
            if len(train_df)
            else None
        ),
        "val_date_range": (
            [
                str(val_df["timestamp"].min()),
                str(val_df["timestamp"].max()),
            ]
            if len(val_df)
            else None
        ),
        "test_date_range": (
            [
                str(test_df["timestamp"].min()),
                str(test_df["timestamp"].max()),
            ]
            if len(test_df)
            else None
        ),
        "note": (
            "Chronological split (earliest 70% train / next 15% validation / latest 15% test); "
            "forecast labels are generated within each split and horizon tails are purged. "
            "Scaler fit on train split only."
        ),
    }


def generate_split_scoped_labels(featured_df, split_dataframes, label_cfg):
    """Generate event labels within each chronological split and purge unknown tails.

    An event in validation/test cannot create a positive training label. Rows
    whose forecast horizon extends past their split boundary are marked NaN so
    model training drops them instead of treating an unobserved future as quiet.
    """
    output = featured_df.copy().reset_index(drop=True)
    horizons = list(label_cfg.horizons_hours)
    columns = [f"forecast_{h}h" for h in horizons]
    output[columns] = float("nan")
    purged = {}

    for split_name, split_df in split_dataframes.items():
        timestamps = set(split_df["timestamp"].tolist())
        indices = output.index[output["timestamp"].isin(timestamps)].tolist()
        split_part = output.loc[indices].copy().reset_index(drop=True)
        if split_part.empty:
            purged[split_name] = {column: 0 for column in columns}
            continue
        split_part = apply_preflare_window(split_part, label_cfg)
        split_part = generate_forecast_labels(split_part, label_cfg)
        for column in columns:
            output.loc[indices, column] = split_part[column].to_numpy()
        output.loc[indices, "state"] = split_part["state"].to_numpy()

        split_purged = {}
        for horizon, column in zip(horizons, columns):
            steps = int(horizon * 3600 / label_cfg.cadence_seconds)
            n_purge = min(steps, len(indices))
            if n_purge:
                output.loc[indices[-n_purge:], column] = float("nan")
            split_purged[column] = n_purge
        purged[split_name] = split_purged

    return output, purged
