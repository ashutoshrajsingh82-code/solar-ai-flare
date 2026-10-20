"""Validation and UTC normalization for raw inference observations."""
import numpy as np
import pandas as pd


def normalize_prediction_rows(rows):
    frame = pd.DataFrame(rows)
    required = {"timestamp", "solexs_flux", "helios_flux"}
    if not required.issubset(frame.columns) or len(frame) == 0:
        raise ValueError("Prediction rows must include timestamp, solexs_flux, and helios_flux.")
    for column in ("solexs_flux", "helios_flux"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[["solexs_flux", "helios_flux"]].to_numpy(dtype=float)).all():
        raise ValueError("Prediction flux values must be finite numbers.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce", format="mixed")
    if frame["timestamp"].isna().any():
        raise ValueError("Prediction timestamps must be valid.")
    if frame["timestamp"].duplicated().any():
        raise ValueError("Prediction rows must not contain duplicate timestamps.")
    return frame.sort_values("timestamp").reset_index(drop=True)
