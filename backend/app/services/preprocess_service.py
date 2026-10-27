
import time
import json
import os

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
    get_feature_registry,
)
from ml.features.labels import (
    generate_forecast_labels,
    apply_preflare_window,
    LabelConfig,
)
from ml.features.windowing import chronological_split, build_split_audit
from ml.features.manifest import data_fingerprint


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
            f"The loaded {source} observations do not match the existing "
            "dual-flux pipeline contract. Raw SoLEXS/HEL1OS counts cannot "
            "be relabeled as flux; verified instrument adapters and "
            "reference flare labels are required."
        )

    if (
        "state" not in pipeline_state.raw_df.columns
        or pipeline_state.raw_df["state"].isna().any()
    ):
        raise ValueError(
            "Training labels are missing or unknown. Provide verified "
            "reference labels and their provenance; unlabeled observations "
            "must not be treated as quiet intervals."
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
    train_df, val_df, test_df = chronological_split(
        aligned_df,
        train_frac=req.train_fraction,
        val_frac=req.validation_fraction,
        train_end=req.train_end_utc,
        val_end=req.validation_end_utc,
    )

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
        "feature_config": feat_cfg.__dict__,
        "feature_schema_version": featured_df.attrs.get("feature_schema_version"),
        "feature_registry_version": "feature-registry-v2",
        "feature_registry": get_feature_registry(featured_df, feat_cfg),
        "warnings": [],
    }

    pipeline_state.set_stage(
        "featured",
        featured_df,
        feature_report,
        time.time() - t0,
    )

    # --- Label generation / reference-target preservation ---
    report_stage("labeling")
    t0 = time.time()

    label_cfg = LabelConfig(
        cadence_seconds=cadence_seconds,
        preflare_minutes=req.preflare_minutes,
    )

    # Metadata identifies that independent reference labels were attached.
    # This flag alone is not scientific verification; the Phase 2C evidence
    # gate must still validate provenance, coverage, timestamps and features.
    has_reference_labels = (
        pipeline_state.dataset_meta.get("source") != "demo"
        and bool(pipeline_state.dataset_meta.get("reference_labels"))
    )

    labeled_df, boundary_purged = generate_split_scoped_labels(
        featured_df,
        {
            "train": train_df,
            "validation": val_df,
            "test": test_df,
        },
        label_cfg,
        preserve_reference_labels=has_reference_labels,
    )

    forecast_columns = [
        f"forecast_{h}h" for h in label_cfg.horizons_hours
    ]

    split_dataframes = {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }

    positive_counts_by_split = {}
    target_counts_by_split = {}

    for split_name, split_df in split_dataframes.items():
        split_timestamps = split_df["timestamp"]

        split_labels = labeled_df[
            labeled_df["timestamp"].isin(split_timestamps)
        ]

        positive_counts_by_split[split_name] = {}
        target_counts_by_split[split_name] = {}

        for column in forecast_columns:
            if column not in split_labels.columns:
                continue

            target = pd.to_numeric(
                split_labels[column],
                errors="coerce",
            )
            known = target.isin([0, 1])

            positive_counts_by_split[split_name][column] = int(
                (target == 1).sum()
            )
            target_counts_by_split[split_name][column] = {
                "known": int(known.sum()),
                "positive": int((target == 1).sum()),
                "negative": int((target == 0).sum()),
                "unknown": int((~known).sum()),
            }

    positive_totals = {
        column: int(
            pd.to_numeric(labeled_df[column], errors="coerce")
            .eq(1)
            .sum()
        )
        for column in forecast_columns
        if column in labeled_df.columns
    }

    label_report = {
        "input_rows": len(featured_df),
        "output_rows": len(labeled_df),
        "label_mode": (
            "independent_reference_labels"
            if has_reference_labels
            else "demo_event_derived"
        ),
        "preflare_minutes": req.preflare_minutes,
        "forecast_horizons": list(label_cfg.horizons_hours),
        "positive_rate_1h": _known_positive_rate(
            labeled_df, "forecast_1h"
        ),
        "positive_rate_24h": _known_positive_rate(
            labeled_df, "forecast_24h"
        ),
        "positive_counts_by_split": positive_counts_by_split,
        "target_counts_by_split": target_counts_by_split,
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

    split_config = {
        "method": "explicit_utc_boundaries" if req.train_end_utc is not None else "chronological_proportions",
        "train_fraction": None if req.train_end_utc is not None else req.train_fraction,
        "validation_fraction": None if req.train_end_utc is not None else req.validation_fraction,
        "test_fraction": None if req.train_end_utc is not None else 1.0 - req.train_fraction - req.validation_fraction,
        "train_end_utc": req.train_end_utc,
        "validation_end_utc": req.validation_end_utc,
        "target_policy": "labels generated within each target partition; horizon tail purged",
        "input_context_policy": "earlier observations may be used as historical input only",
    }
    observed_class = labeled_df.get("flare_class", pd.Series("-", index=labeled_df.index)).fillna("-").astype(str).str.upper().str.extract(r"^\s*([BCMX])", expand=False)
    observed_flare = pd.to_numeric(labeled_df.get("state"), errors="coerce").eq(2) & observed_class.isin(["C", "M", "X"])
    event_onsets = observed_flare & ~observed_flare.shift(1, fill_value=False)
    event_catalog = [
        {"timestamp": timestamp, "flare_class_group": flare_class}
        for timestamp, flare_class in zip(labeled_df.loc[event_onsets, "timestamp"], observed_class.loc[event_onsets])
    ]
    raw_times = pd.to_datetime(pipeline_state.raw_df["timestamp"], utc=True, errors="coerce", format="mixed")
    ordered_raw_times = raw_times.dropna().reset_index(drop=True)
    timestamp_quality = {
        "invalid_or_missing_timestamps": int(raw_times.isna().sum()),
        "duplicate_timestamp_rows": int(ordered_raw_times.duplicated().sum()),
        "out_of_order_adjacent_pairs": int(ordered_raw_times.diff().lt(pd.Timedelta(0)).sum()),
    }
    audit_partitions = {
        "train": labeled_df[labeled_df.timestamp.isin(train_df.timestamp)],
        "validation": labeled_df[labeled_df.timestamp.isin(val_df.timestamp)],
        "test": labeled_df[labeled_df.timestamp.isin(test_df.timestamp)],
    }
    audit_contexts = {}
    for name, frame in audit_partitions.items():
        if frame.empty:
            audit_contexts[name] = frame.iloc[:0].copy()
        else:
            start = pd.to_datetime(frame.timestamp, utc=True).min()
            prior = pd.to_datetime(labeled_df.timestamp, utc=True) < start
            audit_contexts[name] = labeled_df.loc[prior].tail(59).copy()
    split_audit = build_split_audit(
        audit_partitions,
        forecast_columns, split_config,
        {**pipeline_state.dataset_meta, "synthetic": pipeline_state.dataset_meta.get("source") == "demo",
         "purged": boundary_purged, "raw_timestamp_quality": timestamp_quality,
         "dataset_fingerprint": data_fingerprint(pipeline_state.raw_df)},
        sequence_length=60,
        channel_cols=["solexs_flux_scaled", "helios_flux_scaled"],
        event_catalog=event_catalog, context_frames=audit_contexts,
    )
    audit_dir = pipeline_state.artifact_dir
    os.makedirs(audit_dir, exist_ok=True)
    audit_json_path = os.path.join(audit_dir, "split_audit.json")
    audit_csv_path = os.path.join(audit_dir, "split_audit.csv")
    with open(audit_json_path, "w", encoding="utf-8") as stream:
        json.dump(split_audit, stream, indent=2, default=str)
    pd.DataFrame(split_audit["partitions"]).to_csv(audit_csv_path, index=False)

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
            "Chronological split (earliest 70% train / next 15% validation / "
            "latest 15% test); forecast horizon tails are purged per split. "
            "Scaler fit on train split only. Forecast targets are either "
            "preserved from attached reference labels or generated for demo "
            "data; see label_mode and target_counts_by_split."
        ),
        "split_audit": split_audit,
        "split_audit_json": audit_json_path,
        "split_audit_csv": audit_csv_path,
    }


def _known_positive_rate(df: pd.DataFrame, column: str):
    """Return positive fraction among known binary targets, or None if undefined."""
    if column not in df.columns:
        return None

    target = pd.to_numeric(df[column], errors="coerce")
    known = target.isin([0, 1])

    if not known.any():
        return None

    return float((target[known] == 1).mean())


def generate_split_scoped_labels(
    featured_df,
    split_dataframes,
    label_cfg,
    preserve_reference_labels=False,
):
    """Generate split-scoped demo labels or preserve independent reference labels.

    An event in validation/test cannot create a positive training label. Rows
    whose forecast horizon extends past their split boundary are marked NaN
    so training drops them rather than treating an unobserved future as quiet.

    Reference-label mode never regenerates targets from the state column.
    """
    output = featured_df.copy().reset_index(drop=True)
    horizons = list(label_cfg.horizons_hours)
    columns = [f"forecast_{h}h" for h in horizons]

    if preserve_reference_labels:
        missing = [
            column for column in columns
            if column not in output.columns
        ]
        if missing:
            raise ValueError(
                "Reference-label mode requires all forecast target columns; "
                f"missing: {missing}. Refusing to generate replacement labels."
            )

        # Preserve independent targets; coerce invalid/missing values to unknown.
        for column in columns:
            values = pd.to_numeric(output[column], errors="coerce")
            invalid = values.notna() & ~values.isin([0, 1])

            if invalid.any():
                raise ValueError(
                    f"Reference target {column} contains non-binary values."
                )

            output[column] = values
    else:
        # Demo/event-derived labels retain the existing behavior.
        output[columns] = float("nan")

    purged = {}

    for split_name, split_df in split_dataframes.items():
        timestamps = set(split_df["timestamp"].tolist())
        indices = output.index[
            output["timestamp"].isin(timestamps)
        ].tolist()

        split_part = output.loc[indices].copy().reset_index(drop=True)

        if split_part.empty:
            purged[split_name] = {
                column: 0 for column in columns
            }
            continue

        if not preserve_reference_labels:
            split_part = apply_preflare_window(split_part, label_cfg)
            split_part = generate_forecast_labels(split_part, label_cfg)

            for column in columns:
                output.loc[indices, column] = split_part[column].to_numpy()

            output.loc[indices, "state"] = split_part["state"].to_numpy()

        split_purged = {}

        for horizon, column in zip(horizons, columns):
            steps = int(
                horizon * 3600 / label_cfg.cadence_seconds
            )
            n_purge = min(steps, len(indices))

            if n_purge:
                output.loc[indices[-n_purge:], column] = float("nan")

            split_purged[column] = n_purge

        purged[split_name] = split_purged

    return output, purged
