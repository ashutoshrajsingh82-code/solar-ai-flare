"""Validation and UTC normalization for raw inference observations."""
import numpy as np
import pandas as pd


def normalize_prediction_rows(rows, expected_cadence_seconds=None):
    frame = pd.DataFrame(rows)
    if "timestamp" not in frame.columns or len(frame) == 0:
        raise ValueError("Prediction rows must include timestamp and at least one flux channel.")
    for column in ("solexs_flux", "helios_flux"):
        if column not in frame:
            frame[column] = np.nan
        original = frame[column]
        frame[column] = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & frame[column].isna()
        if invalid.any():
            raise ValueError(f"Prediction field '{column}' contains non-numeric values.")
    values = frame[["solexs_flux", "helios_flux"]].to_numpy(dtype=float)
    if not np.isfinite(values).any():
        raise ValueError("Prediction rows need at least one finite flux value.")
    if np.isinf(values).any():
        raise ValueError("Prediction flux values cannot be infinite.")
    if not frame[["solexs_flux", "helios_flux"]].notna().any(axis=1).all():
        raise ValueError("Each prediction timestamp must contain at least one measured flux channel.")
    if (frame[["solexs_flux", "helios_flux"]] < 0).any().any():
        raise ValueError("Prediction flux values cannot be negative.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce", format="mixed")
    if frame["timestamp"].isna().any():
        raise ValueError("Prediction timestamps must be valid.")
    if frame["timestamp"].duplicated().any():
        raise ValueError("Prediction rows must not contain duplicate timestamps.")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Prediction timestamps must be strictly increasing; rows are not reordered automatically.")
    warnings = []
    if frame[["solexs_flux", "helios_flux"]].isna().any().any():
        warnings.append("Some instrument measurements are missing; no values were imputed.")
    if expected_cadence_seconds and len(frame) > 1:
        intervals = frame["timestamp"].diff().dt.total_seconds().iloc[1:].to_numpy(dtype=float)
        tolerance = max(1.0, expected_cadence_seconds * 0.25)
        irregular = np.abs(intervals - expected_cadence_seconds) > tolerance
        if irregular.any():
            warnings.append(
                f"{int(irregular.sum())} timestamp interval(s) differ from the expected "
                f"{expected_cadence_seconds}s cadence by more than 25%."
            )
    frame.attrs["data_quality_warnings"] = warnings
    frame.attrs["cadence_seconds"] = expected_cadence_seconds
    return frame.reset_index(drop=True)
