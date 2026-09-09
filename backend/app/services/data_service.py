import os
import time
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.models import Dataset
from app.services.pipeline_state import pipeline_state
from ml.data.synthetic_generator import save_demo_dataset, generate_synthetic_dataset
from ml.preprocessing.cleaning import clean_dataframe, CleaningConfig

REQUIRED_COLUMNS = {"timestamp", "solexs_flux", "helios_flux"}


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
        df["state"] = 0  # unknown ground truth for uploaded real data; user data may lack labels
    if "flare_class" not in df.columns:
        df["flare_class"] = "-"

    return _ingest_dataframe(db, df, filename=filename, source="upload")


def _ingest_dataframe(db: Session, df: pd.DataFrame, filename: str, source: str) -> dict:
    t0 = time.time()
    pipeline_state.raw_df = df.copy()
    pipeline_state.stage_reports["raw"] = {
        "input_rows": len(df),
        "output_rows": len(df),
        "warnings": [],
    }

    n_rows = len(df)
    missing_values = int(df[["solexs_flux", "helios_flux"]].isna().sum().sum())
    quality_score = round(max(0.0, 1.0 - (missing_values / max(1, n_rows * 2))), 3)

    if n_rows >= 2:
        cadence_seconds = int(np.median(np.diff(df["timestamp"].sort_values().values)).astype("timedelta64[s]").astype(int))
    else:
        cadence_seconds = 60
    pipeline_state.cadence_seconds = max(1, cadence_seconds)

    meta = {
        "filename": filename,
        "n_rows": n_rows,
        "date_range_start": str(df["timestamp"].min()),
        "date_range_end": str(df["timestamp"].max()),
        "cadence_seconds": pipeline_state.cadence_seconds,
        "channels_detected": ["solexs_flux", "helios_flux"],
        "missing_values": missing_values,
        "data_quality_score": quality_score,
        "source": source,
    }
    pipeline_state.dataset_meta = meta

    record = Dataset(
        filename=filename, source=source, n_rows=n_rows,
        date_range_start=meta["date_range_start"], date_range_end=meta["date_range_end"],
        cadence_seconds=pipeline_state.cadence_seconds, missing_values=missing_values,
        data_quality_score=quality_score, channels_detected=meta["channels_detected"],
        filepath=filename,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    meta["dataset_id"] = record.id
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
    stats = df[["solexs_flux", "helios_flux"]].describe().to_dict()
    class_dist = df["flare_class"].value_counts().to_dict() if "flare_class" in df.columns else {}
    state_dist = df["state"].value_counts().to_dict() if "state" in df.columns else {}
    return {
        "summary_statistics": stats,
        "flare_class_distribution": class_dist,
        "state_distribution": {str(k): v for k, v in state_dist.items()},
    }
