import os
import time
import hashlib
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.models import Dataset
from app.services.pipeline_state import pipeline_state
from app.services.data_sources.solexs import read_solexs_l1_light_curve
from app.services.data_sources.hel1os import read_hel1os_product
from app.services.data_sources.goes_labels import read_goes_flare_report_csv, label_observations_from_goes
from ml.data.synthetic_generator import save_demo_dataset, generate_synthetic_dataset
from ml.preprocessing.cleaning import clean_dataframe, CleaningConfig

REQUIRED_COLUMNS = {"timestamp", "solexs_flux", "helios_flux"}


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    canonical = df.reset_index(drop=True).copy()
    if "timestamp" in canonical:
        canonical["timestamp"] = pd.to_datetime(canonical["timestamp"], utc=True, errors="coerce").astype(str)
    for column in canonical.columns:
        if column == "timestamp":
            continue
        numeric = pd.to_numeric(canonical[column], errors="coerce")
        if numeric.notna().any() or canonical[column].isna().all():
            canonical[column] = numeric.astype("float64")
        else:
            canonical[column] = canonical[column].fillna("").astype(str)
    return hashlib.sha256(pd.util.hash_pandas_object(canonical, index=False).values.tobytes()).hexdigest()


def load_demo_data(db: Session) -> dict:
    """Generates (if absent) and loads the synthetic demo dataset into pipeline_state."""
    os.makedirs(settings.sample_data_dir, exist_ok=True)
    csv_path = os.path.join(settings.sample_data_dir, "demo_aditya_l1_solexs_helios.csv")

    if not os.path.exists(csv_path):
        csv_path, _, df, _ = save_demo_dataset(settings.sample_data_dir)
    else:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    return _ingest_dataframe(db, df, filename=os.path.basename(csv_path), source="demo")


def load_uploaded_csv(db: Session, filepath: str, filename: str, column_mapping: dict = None) -> dict:
    df = pd.read_csv(filepath)
    if column_mapping:
        df = df.rename(columns=column_mapping)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Uploaded file is missing required columns: {missing}. "
                          f"Detected columns: {list(df.columns)}. Provide a column_mapping if names differ.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if "state" not in df.columns:
        df["state"] = np.nan  # unknown ground truth must not be mislabeled as quiet
    if "flare_class" not in df.columns:
        df["flare_class"] = "-"

    return _ingest_dataframe(db, df, filename=filename, source="upload")


def load_real_product(db: Session, filepath: str, instrument: str, product: str, filename: str = None) -> dict:
    """Load a documented local mission product; unknown contracts fail closed."""
    instrument = instrument.lower().strip()
    product = product.lower().strip()
    if instrument == "hel1os":
        read_hel1os_product(filepath)
    if instrument != "solexs" or product not in {"lc", "light_curve"}:
        raise ValueError("Supported real product is instrument=solexs, product=lc")

    result = read_solexs_l1_light_curve(filepath)
    df = result.data
    report = result.report
    if filename:
        report["filename"] = os.path.basename(filename)
    cadence_seconds = max(1, int(round(report["cadence_seconds_median"] or 1)))
    meta = {
        **report,
        "source": "solexs",
        "channels_detected": ["solexs_counts_per_second"],
        "data_quality_score": round(len(df) / max(1, report["input_rows"]), 3),
        "content_sha256": hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest(),
    }
    # Publish a dataset record only after the file has passed adapter validation.
    record = Dataset(
        filename=report["filename"], source="solexs_real", n_rows=len(df),
        date_range_start=meta["data_period_start"], date_range_end=meta["data_period_end"],
        cadence_seconds=cadence_seconds, missing_values=report["rejected_rows"],
        data_quality_score=meta["data_quality_score"], channels_detected=meta["channels_detected"],
        filepath=report["filename"], source_metadata=meta,
    )
    with pipeline_state.operation_lock:
        pipeline_state.reset_for_dataset()
        try:
            pipeline_state.raw_df = df.copy()
            pipeline_state.dataset_meta = meta
            pipeline_state.cadence_seconds = cadence_seconds
            pipeline_state.set_stage("raw", df.copy(), {
                "input_rows": report["input_rows"], "output_rows": len(df),
                "warnings": ["Native SoLEXS counts are not normalized flux and cannot enter the existing dual-flux ML pipeline."],
                "ingestion_report": report,
            }, 0.0)
            db.add(record)
            db.flush()
            meta["dataset_id"] = record.id
            pipeline_state.dataset_meta = meta
            pipeline_state.persist(include_frames=False)
            db.commit()
            db.refresh(record)
        except Exception:
            db.rollback()
            pipeline_state.reset_for_dataset()
            raise
    return meta


def attach_goes_reference_labels(db: Session, filepath: str, filename: str,
                                 coverage_start: str, coverage_end: str) -> dict:
    """Attach independent NOAA event-derived horizon labels to a loaded real dataset."""
    if pipeline_state.raw_df is None or not pipeline_state.dataset_meta:
        raise ValueError("Load a real dataset before attaching reference flare labels")
    if pipeline_state.dataset_meta.get("source") == "demo":
        raise ValueError("NOAA reference labels are only attachable to an explicitly loaded real dataset")
    events = read_goes_flare_report_csv(filepath, coverage_start, coverage_end)
    label_frame, label_report = label_observations_from_goes(
        pipeline_state.raw_df, events.data, coverage_start, coverage_end,
    )
    source_meta = dict(pipeline_state.dataset_meta)
    provenance = {**events.report, "label_report": label_report}
    source_meta["reference_labels"] = provenance
    labeled_raw = pipeline_state.raw_df.copy()
    for column in label_frame.columns:
        if column != "timestamp":
            labeled_raw[column] = label_frame[column].to_numpy()

    with pipeline_state.operation_lock:
        dataset_id = source_meta.get("dataset_id")
        record = db.get(Dataset, dataset_id) if dataset_id is not None else None
        if dataset_id is not None and record is None:
            raise ValueError("Dataset database record is unavailable; reference-label provenance was not committed")
        old_meta = pipeline_state.dataset_meta
        old_raw = pipeline_state.raw_df.copy()
        previous_report = pipeline_state.stage_reports.get("raw", {})
        try:
            pipeline_state.dataset_meta = source_meta
            pipeline_state.set_stage("raw", labeled_raw, {
                **previous_report,
                "output_rows": len(labeled_raw),
                "warnings": list(previous_report.get("warnings", [])),
                "reference_label_report": provenance,
            }, 0.0)
            pipeline_state.raw_df = labeled_raw
            pipeline_state.persist(include_frames=False)
            if record is not None:
                record.source_metadata = {**(record.source_metadata or {}), "reference_labels": provenance}
            db.commit()
        except Exception:
            db.rollback()
            pipeline_state.dataset_meta = old_meta
            pipeline_state.set_stage("raw", old_raw, previous_report, 0.0)
            pipeline_state.raw_df = old_raw
            pipeline_state.persist(include_frames=False)
            raise
    return {
        "dataset_id": source_meta.get("dataset_id"),
        "source_used": "NOAA/NCEI GOES XRS L2 Flare Report",
        "is_demo": False,
        "label_report": label_report,
        "event_report": events.report,
        "note": "GOES labels are independent reference events; they do not establish SoLEXS/HEL1OS flux equivalence or train a forecast model by themselves.",
    }


def _ingest_dataframe(db: Session, df: pd.DataFrame, filename: str, source: str) -> dict:
    with pipeline_state.operation_lock:
        return _ingest_dataframe_locked(db, df, filename, source)


def _ingest_dataframe_locked(db: Session, df: pd.DataFrame, filename: str, source: str) -> dict:
    t0 = time.time()
    n_rows = len(df)
    missing_values = int(df[["solexs_flux", "helios_flux"]].isna().sum().sum())
    quality_score = round(max(0.0, 1.0 - (missing_values / max(1, n_rows * 2))), 3)

    if n_rows >= 2:
        cadence_seconds = int(np.median(np.diff(df["timestamp"].sort_values().values)).astype("timedelta64[s]").astype(int))
    else:
        cadence_seconds = 60
    cadence_seconds = max(1, cadence_seconds)

    meta = {
        "filename": filename,
        "n_rows": n_rows,
        "date_range_start": str(df["timestamp"].min()),
        "date_range_end": str(df["timestamp"].max()),
        "cadence_seconds": cadence_seconds,
        "channels_detected": ["solexs_flux", "helios_flux"],
        "missing_values": missing_values,
        "data_quality_score": quality_score,
        "source": source,
        "content_sha256": _dataset_fingerprint(df),
    }
    record = Dataset(
        filename=filename, source=source, n_rows=n_rows,
        date_range_start=meta["date_range_start"], date_range_end=meta["date_range_end"],
        cadence_seconds=cadence_seconds, missing_values=missing_values,
        data_quality_score=quality_score, channels_detected=meta["channels_detected"],
        filepath=filename, source_metadata=meta,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    meta["dataset_id"] = record.id
    same_dataset = (
        pipeline_state.dataset_meta.get("content_sha256") == meta["content_sha256"]
        and pipeline_state.dataset_meta.get("source") == source
    )
    if not same_dataset:
        pipeline_state.reset_for_dataset()
    pipeline_state.raw_df = df.copy()
    pipeline_state.dataset_meta = meta
    pipeline_state.cadence_seconds = cadence_seconds
    if not same_dataset or pipeline_state.stage_reports.get("raw") is None:
        pipeline_state.set_stage("raw", df.copy(), {
            "input_rows": len(df), "output_rows": len(df), "warnings": [],
        }, time.time() - t0)
    pipeline_state.persist(include_frames=not same_dataset)
    return meta


def preview_dataset(n: int = 50) -> dict:
    if pipeline_state.raw_df is None:
        return {"rows": [], "columns": []}
    df = pipeline_state.raw_df.head(n).copy()
    df["timestamp"] = df["timestamp"].astype(str)
    return {"rows": df.to_dict(orient="records"), "columns": list(df.columns)}


def dataset_summary_stats() -> dict:
    if pipeline_state.raw_df is None:
        return {}
    df = pipeline_state.raw_df
    measurement_columns = [column for column in df.select_dtypes(include=[np.number]).columns
                           if column != "state" and not column.endswith("_unix_seconds")]
    stats = df[measurement_columns].describe().to_dict() if measurement_columns else {}
    class_dist = df["flare_class"].value_counts().to_dict() if "flare_class" in df.columns else {}
    state_dist = df["state"].value_counts().to_dict() if "state" in df.columns else {}
    return {
        "source": pipeline_state.dataset_meta.get("source", "unknown"),
        "is_demo": pipeline_state.dataset_meta.get("source") == "demo",
        "summary_statistics": stats,
        "flare_class_distribution": class_dist,
        "state_distribution": {str(k): v for k, v in state_dist.items()},
    }
