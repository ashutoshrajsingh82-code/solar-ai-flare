"""Held-out replay eligibility, provenance, and database persistence."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.exc import IntegrityError

from app.core.config import settings
from app.database.db import SessionLocal
from app.database.models import AlertRecord, Dataset, InferenceRecord, SimulationReplayRecord
from app.services import alert_service, predict_service
from app.services.pipeline_state import pipeline_state
from ml.features.manifest import data_fingerprint, load_feature_manifest
from ml.training.eligibility import real_training_blockers
from ml.inference.simulator import RAW_COLUMNS


class ReplayNotReadyError(ValueError):
    def __init__(self, blockers):
        self.blockers = list(blockers)
        super().__init__("Historical held-out replay is unavailable.")


def assess_replay_readiness(model_id=None):
    blockers = []
    meta = pipeline_state.dataset_meta or {}
    if settings.data_mode == "demo" or meta.get("source") == "demo":
        blockers.append("Demo mode or synthetic data is active; synthetic observations are not eligible held-out replay.")
    if not meta:
        blockers.append("No loaded dataset provenance is available.")
    if meta.get("source") in {"solexs", "solexs_real"}:
        blockers.append("SoLEXS-only native counts are not compatible with the trained dual-flux model inputs.")
    blockers.extend(real_training_blockers(meta))
    if not meta.get("reference_labels"):
        blockers.append("Independent reference-label provenance is missing.")
    if meta.get("source") not in (None, "demo"):
        db = SessionLocal()
        try:
            dataset = db.get(Dataset, meta.get("dataset_id")) if meta.get("dataset_id") is not None else None
            if dataset is None or dataset.processing_status != "complete":
                blockers.append("Persistent source dataset record is missing or incomplete.")
            else:
                root = Path(settings.upload_storage_dir)
                if not root.is_absolute():
                    root = Path(__file__).resolve().parents[2] / root
                root = root.resolve()
                source_path = (root / str(dataset.filepath or "")).resolve()
                if source_path != root and root not in source_path.parents and not source_path.is_relative_to(root):
                    blockers.append("Persistent source dataset path is outside the configured upload storage root.")
                elif not source_path.is_file() or not os.access(source_path, os.R_OK):
                    blockers.append("Persistent source dataset file is missing or unreadable.")
                elif dataset.content_sha256:
                    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
                    if digest != dataset.content_sha256:
                        blockers.append("Persistent source dataset file failed its content hash check.")
        finally:
            db.close()
    for name in ("raw_df", "test_df", "labeled_df"):
        frame = getattr(pipeline_state, name, None)
        if frame is None or frame.empty:
            blockers.append(f"Pipeline {name} observations are unavailable.")
    rec = None
    manifest = None
    if model_id:
        try:
            rec = predict_service.resolve_model(model_id)
            if rec.get("status") != "trained":
                blockers.append("Requested checkpoint is not a verified real-data trained model.")
            manifest = predict_service._load_record_feature_manifest(rec)
            if not rec.get("checkpoint_path") or not os.path.isfile(rec["checkpoint_path"]):
                blockers.append("Trained model checkpoint file is missing.")
            if rec.get("training_dataset") != meta.get("filename"):
                blockers.append("Loaded dataset does not match the dataset used to train this model.")
            expected_fingerprint = manifest.get("source_data_fingerprint")
            actual_fingerprint = data_fingerprint(pipeline_state.raw_df) if pipeline_state.raw_df is not None else None
            if not expected_fingerprint or expected_fingerprint != actual_fingerprint:
                blockers.append("Loaded observations do not match the checkpoint's recorded source-data fingerprint.")
            if manifest.get("preprocessing", {}).get("fit_on") != "train_split_only":
                blockers.append("Checkpoint preprocessing provenance does not verify train-split-only fitting.")
            periods = manifest
            train, validation, test = (periods.get("training_period", {}), periods.get("validation_period", {}),
                                       periods.get("test_period", {}))
            if not all(period.get("start_utc") and period.get("end_utc") for period in (train, validation, test)):
                blockers.append("Checkpoint has no complete chronological train/validation/test boundaries.")
            else:
                train_end = pd.Timestamp(train["end_utc"])
                val_start, val_end = pd.Timestamp(validation["start_utc"]), pd.Timestamp(validation["end_utc"])
                test_start = pd.Timestamp(test["start_utc"])
                if not train_end < val_start <= val_end < test_start:
                    blockers.append("Checkpoint split boundaries are not strictly chronological.")
                actual_test = pipeline_state.test_df
                actual_start = pd.to_datetime(actual_test.timestamp, utc=True).min()
                actual_end = pd.to_datetime(actual_test.timestamp, utc=True).max()
                if actual_start != pd.Timestamp(test["start_utc"]) or actual_end != pd.Timestamp(test["end_utc"]):
                    blockers.append("Loaded test partition does not match the checkpoint's recorded held-out period.")
            if settings.data_mode == "real" and rec.get("status") != "trained":
                blockers.append("Model is not eligible for observational replay.")
        except predict_service.ModelNotFoundError as exc:
            blockers.append(str(exc))
        except predict_service.ModelUnavailableError as exc:
            blockers.append(str(exc))
        except Exception as exc:
            blockers.append(f"Checkpoint or feature contract could not be verified: {exc}")
    return {"ready": not blockers, "blockers": list(dict.fromkeys(blockers)),
            "source_classification": "historical_test_replay", "model_id": model_id,
            "dataset_id": meta.get("dataset_id"),
            "period": manifest.get("test_period") if manifest else None,
            "real_time_feed_available": False}


def _naive_utc(timestamp):
    value = pd.Timestamp(timestamp)
    value = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
    return value.to_pydatetime().replace(tzinfo=None)


def recover_interrupted_replays():
    """Mark volatile in-process replays interrupted; their safe retry policy is full idempotent replay."""
    db = SessionLocal()
    try:
        records = db.query(SimulationReplayRecord).filter(
            SimulationReplayRecord.status.in_(("queued", "running"))
        ).all()
        now = pd.Timestamp.now(tz="UTC").to_pydatetime().replace(tzinfo=None)
        for record in records:
            record.status = "interrupted"
            record.error = "Backend restarted before replay completed; retry restarts from the beginning with idempotent persisted predictions."
            record.finished_at = now
        db.commit()
        return len(records)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _replay_model_fingerprint(record, manifest):
    parts = [json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str)]
    for key in ("checkpoint_path", "transformer_path", "scaler_path"):
        path = record.get(key)
        if path and os.path.isfile(path):
            digest = hashlib.sha256()
            with open(path, "rb") as artifact:
                for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                    digest.update(chunk)
            parts.append(f"{key}:{digest.hexdigest()}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _persist_prediction(result, update=False):
    db = SessionLocal()
    try:
        key = (result.get("replay_id"), result.get("model_id"), _naive_utc(result["forecast_issue_timestamp_utc"]))
        record = db.query(InferenceRecord).filter_by(
            replay_id=key[0], model_id=key[1], forecast_issue_timestamp=key[2]
        ).one_or_none()
        if update:
            if record is None:
                return
            record.outcomes = result.get("outcomes", {})
            record.input_snapshot = result
            db.commit()
            return
        if record is not None:
            # A restart replays deterministically from the beginning. Existing issue
            # rows are reused, and alert processing is retried from its idempotency ledger.
            db.rollback()
        else:
            successful = "error" not in result
            record = InferenceRecord(
                timestamp=_naive_utc(result["replay_processing_timestamp_utc"]), model_id=result.get("model_id"),
                nowcast_state=(result.get("nowcast") or {}).get("state", "unknown"),
                nowcast_confidence=(result.get("nowcast") or {}).get("confidence"),
                forecast=result.get("forecast", {}), risk=result.get("risk", "UNKNOWN"),
                input_snapshot=result, source_classification="historical_test_replay", replay_id=result.get("replay_id"),
                observation_timestamp=_naive_utc(result["observation_timestamp_utc"]),
                forecast_issue_timestamp=key[2], input_window_start=_naive_utc(result["input_window_start_utc"]),
                input_window_end=_naive_utc(result["input_window_end_utc"]), data_quality=result.get("data_quality"),
                processing_status="complete" if successful else "failed", outcomes={},
            )
            db.add(record)
            try:
                db.commit()
            except IntegrityError:
                db.rollback()
                record = db.query(InferenceRecord).filter_by(
                    replay_id=key[0], model_id=key[1], forecast_issue_timestamp=key[2]
                ).one_or_none()
                if record is None:
                    raise
        if "error" not in result:
            transitions = alert_service.process_forecast_prediction(
                db, (result.get("nowcast") or {}).get("state", "unknown"), result.get("forecast", {}),
                result.get("model_id"), is_demo=False, model_id=result.get("model_id"),
                model_version=result.get("model_version"), prediction_id=result.get("prediction_id"),
                source_classification="historical_test_replay",
                observation_timestamp=result.get("observation_timestamp_utc"),
                forecast_issue_timestamp=result.get("forecast_issue_timestamp_utc"),
            )
            result["alert_transitions"] = [{"transition": item["transition"],
                                             "alert_id": item["alert"].id if item.get("alert") else None,
                                             "state": item["alert"].state if item.get("alert") else "NORMAL",
                                             "risk_band": item["alert"].severity if item.get("alert") else None}
                                            for item in transitions]
            record.input_snapshot = result
        replay = db.get(SimulationReplayRecord, result.get("replay_id"))
        if replay is not None and "error" not in result:
            replay.last_observation_timestamp = _naive_utc(result["observation_timestamp_utc"])
            replay.last_prediction_id = result.get("prediction_id")
        db.commit()
    finally:
        db.close()


def _update_replay_status(replay_id, status, error=None):
    db = SessionLocal()
    try:
        record = db.get(SimulationReplayRecord, replay_id)
        if record:
            record.status = status
            record.error = error
            if status in {"completed", "failed", "stopped", "interrupted"}:
                record.finished_at = pd.Timestamp.now(tz="UTC").to_pydatetime().replace(tzinfo=None)
            db.commit()
    finally:
        db.close()


def start_replay(model_id, speed=5, dataset_id=None, replay_start_utc=None, replay_end_utc=None, simulator=None):
    from ml.inference.simulator import live_simulator
    simulator = simulator or live_simulator
    if speed not in {1, 5, 10, 50}:
        raise ReplayNotReadyError(["Replay speed must be one of 1, 5, 10, or 50."])
    if not model_id:
        raise ReplayNotReadyError(["An explicit verified model identifier is required."])
    readiness = assess_replay_readiness(model_id)
    if not readiness["ready"]:
        raise ReplayNotReadyError(readiness["blockers"])
    meta = pipeline_state.dataset_meta
    if dataset_id is not None and str(dataset_id) != str(meta.get("dataset_id")):
        raise ReplayNotReadyError(["Requested dataset identifier does not match the loaded, verified dataset."])
    rec = predict_service.resolve_model(model_id)
    test_frame = pipeline_state.test_df
    test_stamps = pd.to_datetime(test_frame.timestamp, utc=True, errors="coerce", format="mixed")
    first = test_stamps.min()
    last = test_stamps.max()
    def utc_timestamp(value, default):
        stamp = pd.Timestamp(value) if value is not None else pd.Timestamp(default)
        return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    start = utc_timestamp(replay_start_utc, first)
    end = utc_timestamp(replay_end_utc, last)
    if start < first or end > last or start > end:
        raise ReplayNotReadyError(["Requested replay interval must be contained entirely within the held-out test period."])
    fingerprint = data_fingerprint(pipeline_state.raw_df)
    feature_manifest = predict_service._load_record_feature_manifest(rec)
    preprocessing_fingerprint = _replay_model_fingerprint(rec, feature_manifest)
    replay_material = (f"{model_id}|{rec.get('version')}|{meta.get('dataset_id')}|{fingerprint}|"
                       f"{preprocessing_fingerprint}|{start.isoformat()}|{end.isoformat()}")
    replay_id = hashlib.sha256(replay_material.encode("utf-8")).hexdigest()[:32]
    db = SessionLocal()
    try:
        existing = db.get(SimulationReplayRecord, replay_id)
        if existing:
            if existing.status not in {"interrupted", "stopped"}:
                return {"replay_id": replay_id, "status": existing.status, "idempotent": True,
                        "source_classification": existing.source_classification}
            if existing.preprocessing_fingerprint != preprocessing_fingerprint:
                raise ReplayNotReadyError(["Replay preprocessing/model contract changed; this replay cannot be resumed."])
            if simulator.get_state().get("running"):
                raise ReplayNotReadyError(["Another historical replay is already running in this process."])
            existing.status = "queued"
            existing.error = None
            existing.finished_at = None
            db.commit()
        elif simulator.get_state().get("running"):
            raise ReplayNotReadyError(["Another historical replay is already running in this process."])
        else:
            start_utc = start.to_pydatetime()
            end_utc = end.to_pydatetime()
            db.add(SimulationReplayRecord(
                id=replay_id, dataset_id=str(meta.get("dataset_id") or fingerprint), model_id=model_id,
                model_version=str(rec.get("version", "unknown")), source_classification="historical_test_replay",
                status="queued", period_start_utc=start_utc, period_end_utc=end_utc,
                preprocessing_fingerprint=preprocessing_fingerprint,
                configuration={"speed": speed, "target_definition": rec.get("params", {}).get("target_definition"),
                               "forecast_horizons": rec.get("params", {}).get("forecast_horizons"),
                               "thresholds_from_validation": rec.get("params", {}).get("thresholds"),
                               "source_data_fingerprint": fingerprint,
                               "resume_policy": "restart_from_beginning_idempotent"},
            ))
            db.commit()
    except IntegrityError:
        db.rollback()
        return {"replay_id": replay_id, "status": "queued", "idempotent": True,
                "source_classification": "historical_test_replay"}
    finally:
        db.close()

    def predict_fn(prefix):
        rows = prefix[RAW_COLUMNS].copy()
        rows["timestamp"] = rows.timestamp.map(lambda value: value.isoformat())
        return predict_service.predict_from_raw_window(rows.to_dict(orient="records"), model_id)

    def persist_fn(result, update=False):
        result.update({"replay_id": replay_id, "model_id": model_id, "model_version": rec.get("version")})
        _persist_prediction(result, update)

    metadata = {"replay_id": replay_id, "dataset_id": meta.get("dataset_id"),
                "dataset_name": meta.get("filename"), "model_id": model_id,
                "model_name": rec.get("name"), "model_version": rec.get("version"),
                "model_verification_status": "verified_trained_checkpoint",
                "source_classification": "historical_test_replay",
                "period_start_utc": start.isoformat(), "period_end_utc": end.isoformat(),
                "preprocessing_fingerprint": preprocessing_fingerprint,
                "eligible_observations": int(((test_stamps >= start) & (test_stamps <= end)).sum()),
                "data_fingerprint": fingerprint}
    outcomes = pipeline_state.labeled_df[["timestamp"] + [f"forecast_{h}" for h in ("1h", "3h", "6h", "12h", "24h")
                                                       if f"forecast_{h}" in pipeline_state.labeled_df]].copy()
    try:
        simulator.configure(pipeline_state.raw_df[RAW_COLUMNS], start, end, predict_fn,
                            outcome_df=outcomes, metadata=metadata, persist_fn=persist_fn,
                            complete_fn=lambda status, error: _update_replay_status(replay_id, status, error),
                            expected_cadence_seconds=pipeline_state.cadence_seconds)
        simulator.set_speed(speed)
        _update_replay_status(replay_id, "running")
        simulator.start()
    except Exception as exc:
        _update_replay_status(replay_id, "failed", str(exc))
        raise
    return {"replay_id": replay_id, "status": "running", "idempotent": False,
            "source_classification": "historical_test_replay", "provenance": metadata}


def replay_record(db, replay_id):
    return db.get(SimulationReplayRecord, replay_id)


def replay_predictions(db, replay_id, limit=500):
    return (db.query(InferenceRecord).filter_by(replay_id=replay_id)
            .order_by(InferenceRecord.forecast_issue_timestamp.asc()).limit(limit).all())
