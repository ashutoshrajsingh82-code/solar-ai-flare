"""Versioned, checksummed feature contracts shared by training and inference."""
import hashlib
import importlib.metadata
import json
import os
import tempfile

import numpy as np
import pandas as pd

from ml.features.engineering import (
    FEATURE_REGISTRY_VERSION,
    FEATURE_SCHEMA_VERSION,
    FeatureConfig,
    get_feature_registry,
)

MANIFEST_VERSION = 2
SUPPORTED_MANIFEST_VERSIONS = {1, MANIFEST_VERSION}


def data_fingerprint(frame: pd.DataFrame) -> str:
    """Stable SHA-256 over ordered timestamp and source values, when available."""
    selected = [c for c in ("timestamp", "solexs_flux", "helios_flux") if c in frame]
    if not selected:
        return hashlib.sha256(b"").hexdigest()
    canonical = frame[selected].copy()
    if "timestamp" in canonical:
        canonical["timestamp"] = pd.to_datetime(canonical.timestamp, utc=True, errors="coerce", format="mixed").astype(str)
    hashes = pd.util.hash_pandas_object(canonical, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def _period(frame):
    if frame is None or frame.empty or "timestamp" not in frame:
        return {"start_utc": None, "end_utc": None, "rows": 0}
    ts = pd.to_datetime(frame.timestamp, utc=True, errors="coerce", format="mixed").dropna()
    return {"start_utc": ts.min().isoformat() if len(ts) else None,
            "end_utc": ts.max().isoformat() if len(ts) else None, "rows": int(len(frame))}


def create_feature_manifest(feature_frame, feature_names, input_configuration,
                            feature_config=None, source_frame=None,
                            split_frames=None, preprocessing=None, random_seed=None,
                            decomposition=None):
    cfg = feature_config or FeatureConfig()
    registry = get_feature_registry(feature_frame, cfg)
    names = list(feature_names)
    if decomposition:
        channels = decomposition.get("raw_channels", [])
        valid_names = [f"{channel}::{component}" for channel in channels
                       for component in ("trend", "seasonal", "residual")]
        if names != valid_names:
            raise ValueError("DLSTM manifest feature order must match its raw channel/component configuration.")
        definitions = [{
            "name": name,
            "description_formula": (
                f"{name.split('::', 1)[1]} component from STL fitted only to the preceding "
                f"{decomposition.get('lookback_steps')} samples ending at forecast origin; "
                f"input transform={decomposition.get('log_transform')}."
            ),
            "source_columns": [name.split("::", 1)[0]],
            "source_instruments": [{"solexs_flux": "SoLEXS", "helios_flux": "HEL1OS"}[name.split("::", 1)[0]]],
            "physical_quantity": None, "units": None, "energy_band_kev": None,
            "rolling_window_minutes": None, "minimum_history_samples": decomposition.get("minimum_history_samples"),
            "availability": "forecast-origin endpoint from historical-only window",
            "missing_value_policy": "exclude sample if history contains invalid or missing measurements",
            "causal": True, "feature_version": "origin-window-stl-v1",
            "provenance_known": True, "allowed_experiments": [decomposition.get("instrument_names", {}).get(name.split("::", 1)[0])],
        } for name in names]
    else:
        missing = [name for name in names if name not in registry]
        if missing:
            raise ValueError(f"Cannot create feature manifest; selected features lack registry entries: {missing}")
        definitions = [registry[name] for name in names]
    return {
        "manifest_version": MANIFEST_VERSION,
        "feature_registry_version": FEATURE_REGISTRY_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "input_configuration": input_configuration,
        "ordered_feature_names": names,
        "features": definitions,
        "feature_config": {**cfg.__dict__, "channel_metadata": cfg.channel_metadata,
                           "quality_columns": cfg.quality_columns},
        "instrument_selection": input_configuration,
        "ratio_metadata": (feature_frame.attrs.get("feature_quality_report") or {}),
        "source_data_fingerprint": data_fingerprint(source_frame) if source_frame is not None else None,
        "training_period": _period((split_frames or {}).get("train")),
        "validation_period": _period((split_frames or {}).get("validation")),
        "test_period": _period((split_frames or {}).get("test")),
        "preprocessing": preprocessing or {},
        "label_contract": {
            "nowcast_class_order": ["quiet", "pre_flare", "flare", "decay"],
            "forecast_target": "Binary M/X-class flare-onset indicator within (forecast_origin, forecast_origin + horizon]; labels are unknown when reference coverage is insufficient.",
            "forecast_horizons": ["1h", "3h", "6h", "12h", "24h"],
            "calibration_validated": False,
            "exact_event_time_prediction": False,
        },
        "decomposition": decomposition,
        "random_seed": random_seed,
        "dependencies": {
            name: _version(name) for name in ("numpy", "pandas", "scikit-learn", "statsmodels")
        },
    }


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def save_feature_manifest(path, manifest):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".feature-manifest-", suffix=".tmp", dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_feature_manifest(path):
    with open(path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("manifest_version") not in SUPPORTED_MANIFEST_VERSIONS:
        raise ValueError("Unsupported feature manifest version.")
    if manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Model feature schema is incompatible with this runtime; retraining is required.")
    names = manifest.get("ordered_feature_names")
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError("Feature manifest has an invalid or duplicate ordered feature list.")
    definitions = manifest.get("features")
    if not isinstance(definitions, list) or [item.get("name") for item in definitions] != names:
        raise ValueError("Feature manifest definitions do not match the ordered feature list.")
    return manifest


def validate_feature_manifest(manifest, feature_frame, input_configuration=None):
    if manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Feature schema version mismatch; retrain or use the matching runtime.")
    if input_configuration is not None and manifest.get("input_configuration") != input_configuration:
        raise ValueError("Inference instrument configuration does not match the trained model manifest.")
    names = manifest.get("ordered_feature_names", [])
    decomposition = manifest.get("decomposition")
    if decomposition:
        expected = [f"{channel}::{component}" for channel in decomposition.get("raw_channels", [])
                    for component in ("trend", "seasonal", "residual")]
        if names != expected:
            raise ValueError("DLSTM manifest has an incompatible channel/component ordering.")
        missing_sources = [channel for channel in decomposition.get("raw_channels", [])
                           if channel not in feature_frame.columns]
        if missing_sources:
            raise ValueError(f"DLSTM inference is missing raw source channels: {missing_sources}")
        return feature_frame.loc[:, decomposition["raw_channels"]]
    missing = [name for name in names if name not in feature_frame.columns]
    if missing:
        raise ValueError(f"Inference is missing required model features: {missing}")
    registry = get_feature_registry(feature_frame, FeatureConfig(**manifest.get("feature_config", {})))
    for item in manifest.get("features", []):
        current = registry.get(item.get("name"))
        if current is None or any(
            current.get(key) != item.get(key)
            for key in ("source_columns", "source_instruments", "units", "energy_band_kev",
                        "physical_quantity", "causal", "feature_version")
        ):
            raise ValueError(f"Inference feature provenance is incompatible: {item.get('name')}")
    return feature_frame.loc[:, names]
