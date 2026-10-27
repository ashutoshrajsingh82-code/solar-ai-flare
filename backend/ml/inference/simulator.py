"""Chronological replay controller for an explicitly held-out test period.

The predictor receives only measurement columns through the current issue time.
Reference outcomes are kept separately and become visible only after their
forecast window has elapsed in replay time.
"""
import hashlib
import logging
import threading
import time
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

SPEED_MULTIPLIERS = {1: 1, 5: 5, 10: 10, 50: 50}
BASE_TICK_SECONDS = 2.0
RAW_COLUMNS = ["timestamp", "solexs_flux", "helios_flux"]
HORIZON_MINUTES = {"1h": 60, "3h": 180, "6h": 360, "12h": 720, "24h": 1440}
logger = logging.getLogger("solar-flare-ai")


class LiveSimulator:
    """In-process replay worker; predictions and replay status are persisted by callbacks."""
    def __init__(self):
        self._lock = threading.RLock()
        self._df: Optional[pd.DataFrame] = None
        self._issue_indices = []
        self._cursor = 0
        self._speed = 5
        self._status = "idle"
        self._thread = None
        self._predict_fn = None
        self._outcomes = None
        self._outcome_map = {}
        self._metadata = {}
        self._persist_fn = None
        self._complete_fn = None
        self._latest_result = None
        self._history = []
        self._max_history = 500
        self._expected_cadence_seconds = 60
        self._last_observation = None

    def configure(self, df: pd.DataFrame, issue_start, issue_end, predict_fn,
                  outcome_df=None, metadata=None, persist_fn=None, complete_fn=None,
                  expected_cadence_seconds=60):
        if not isinstance(df, pd.DataFrame) or not set(RAW_COLUMNS).issubset(df.columns):
            raise ValueError("Replay observations must contain timestamp and both supported flux channels.")
        frame = df[RAW_COLUMNS].copy()
        stamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce", format="mixed")
        if stamps.isna().any():
            raise ValueError("Replay observations contain invalid timestamps.")
        frame["timestamp"] = stamps
        if frame["timestamp"].duplicated().any():
            raise ValueError("Replay observations contain duplicate timestamps.")
        if not frame["timestamp"].is_monotonic_increasing:
            raise ValueError("Replay observations must be chronologically ordered; sorting is not implicit.")
        start = pd.Timestamp(issue_start)
        end = pd.Timestamp(issue_end)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        if start > end:
            raise ValueError("Replay period start must not be after its end.")
        indexes = np.flatnonzero(((frame.timestamp >= start) & (frame.timestamp <= end)).to_numpy()).tolist()
        if not indexes:
            raise ValueError("Replay period contains no eligible observations.")
        outcomes = outcome_df.copy() if outcome_df is not None else pd.DataFrame()
        if not outcomes.empty:
            outcomes["timestamp"] = pd.to_datetime(outcomes["timestamp"], utc=True, errors="coerce", format="mixed")
            if outcomes.timestamp.isna().any() or outcomes.timestamp.duplicated().any():
                raise ValueError("Replay reference outcomes contain invalid or duplicate timestamps.")
            self._outcome_map = {row.timestamp: row for row in outcomes.itertuples(index=False)}
        else:
            self._outcome_map = {}
        with self._lock:
            self._df = frame
            self._issue_indices = indexes
            self._cursor = 0
            self._speed = 5
            self._status = "queued"
            self._predict_fn = predict_fn
            self._outcomes = outcomes
            self._metadata = dict(metadata or {})
            self._persist_fn = persist_fn
            self._complete_fn = complete_fn
            self._latest_result = None
            self._history = []
            self._expected_cadence_seconds = max(1, int(expected_cadence_seconds))
            self._last_observation = None

    def set_speed(self, speed: int):
        if speed not in SPEED_MULTIPLIERS:
            raise ValueError(f"Unsupported speed {speed}. Choose from {list(SPEED_MULTIPLIERS)}")
        with self._lock:
            self._speed = speed

    def start(self):
        with self._lock:
            if self._df is None or self._predict_fn is None:
                raise RuntimeError("No eligible verified test replay is configured.")
            if self._status == "running":
                return
            if self._status == "completed":
                raise RuntimeError("This replay has completed; use its persisted results instead of replaying it twice.")
            self._status = "running"
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def pause(self):
        with self._lock:
            if self._status == "running":
                self._status = "paused"

    def resume(self):
        with self._lock:
            if self._df is None:
                raise RuntimeError("No eligible replay is configured.")
            if self._status == "running":
                return
            if self._status == "completed":
                raise RuntimeError("A completed replay cannot be resumed.")
            self._status = "running"
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self):
        with self._lock:
            if self._status in {"running", "paused", "queued"}:
                self._status = "stopped"
            if self._complete_fn:
                self._complete_fn("stopped", None)

    def reset(self):
        self.stop()
        with self._lock:
            self._df = None
            self._issue_indices = []
            self._cursor = 0
            self._status = "idle"
            self._predict_fn = None
            self._outcomes = None
            self._outcome_map = {}
            self._metadata = {}
            self._persist_fn = None
            self._complete_fn = None
            self._history = []
            self._latest_result = None

    def _run_loop(self):
        pending = []
        while True:
            with self._lock:
                if self._status != "running" or self._df is None:
                    return
                if self._cursor >= len(self._issue_indices):
                    self._status = "completed"
                    complete = self._complete_fn
                    break
                index = self._issue_indices[self._cursor]
                issue_row = self._df.iloc[index]
                # Prediction receives only measurements whose timestamp is <= issue time.
                prefix = self._df.iloc[:index + 1][RAW_COLUMNS].copy()
                current = issue_row.timestamp
                previous = self._last_observation
                gap_seconds = (current - previous).total_seconds() if previous is not None else None
                self._last_observation = current
                speed = self._speed
                self._cursor += 1

            warnings = []
            if gap_seconds is not None and gap_seconds > self._expected_cadence_seconds * 1.5:
                warnings.append(f"Observation gap of {gap_seconds:g} seconds; expected {self._expected_cadence_seconds} seconds.")
            observation = {
                "solexs_flux": _finite_or_none(issue_row.solexs_flux),
                "helios_flux": _finite_or_none(issue_row.helios_flux),
            }
            if not any(value is not None for value in observation.values()):
                prediction = {"error": "Observation has no valid instrument measurement."}
            else:
                try:
                    prediction = self._predict_fn(prefix)
                except Exception:
                    prediction = {"error": str(exc)}
            result = {
                **prediction,
                **{key: self._metadata[key] for key in (
                    "model_id", "model_name", "model_version", "model_verification_status",
                    "dataset_id", "dataset_name", "data_fingerprint",
                ) if key in self._metadata},
                "replay_id": self._metadata.get("replay_id"),
                "prediction_id": hashlib.sha256(
                    f"{self._metadata.get('replay_id')}|{current.isoformat()}".encode("utf-8")
                ).hexdigest()[:32],
                "source_classification": "historical_test_replay",
                "data_source_label": "Historical held-out test replay",
                "observation_timestamp_utc": current.isoformat(),
                "forecast_issue_timestamp_utc": current.isoformat(),
                "replay_processing_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "input_window_start_utc": prefix.timestamp.iloc[0].isoformat(),
                "input_window_end_utc": current.isoformat(),
                "input": observation,
                "timestamp": current.isoformat(),
                "solexs_flux": observation["solexs_flux"],
                "helios_flux": observation["helios_flux"],
                "hard_soft_ratio": (observation["helios_flux"] / observation["solexs_flux"]
                                     if observation["helios_flux"] is not None and observation["solexs_flux"] not in (None, 0)
                                     else None),
                "state": (prediction.get("nowcast") or {}).get("state", "unknown"),
                "data_quality": {"warnings": warnings, "cadence_seconds_expected": self._expected_cadence_seconds,
                                 "gap_seconds_from_previous": gap_seconds,
                                 "valid_measurement_count": int(sum(value is not None for value in observation.values()))},
                "processing_status": "failed" if "error" in prediction else "complete",
                "outcomes": {},
            }
            result["forecast_valid_windows_utc"] = {
                horizon: {"start_exclusive_utc": current.isoformat(),
                          "end_inclusive_utc": (current + pd.Timedelta(minutes=minutes)).isoformat()}
                for horizon, minutes in HORIZON_MINUTES.items()
            }
            with self._lock:
                self._latest_result = result
                self._history.append(result)
                if len(self._history) > self._max_history:
                    self._history.pop(0)
                if "error" not in prediction:
                    pending.append(result)
            if self._persist_fn:
                try:
                    self._persist_fn(result, False)
                except Exception as exc:
                    logger.exception("Could not persist replay prediction at %s", current.isoformat())
                    with self._lock:
                        self._status = "failed"
                        self._metadata["error"] = "Replay result could not be committed."
                        complete = self._complete_fn
                    if complete:
                        complete("failed", "Replay result could not be committed; see server logs.")
                    return

            # Outcomes are consulted only after their window's end is observed.
            for prior in list(pending):
                origin = pd.Timestamp(prior["forecast_issue_timestamp_utc"])
                matured = {}
                for horizon, minutes in HORIZON_MINUTES.items():
                    end = origin + pd.Timedelta(minutes=minutes)
                    if end > current:
                        continue
                    row = self._outcome_map.get(origin)
                    label = getattr(row, f"forecast_{horizon}", None) if row is not None else None
                    try:
                        numeric = float(label)
                    except (TypeError, ValueError):
                        numeric = np.nan
                    matured[horizon] = {"status": "observed" if numeric in (0.0, 1.0) else "unknown_insufficient_reference_coverage",
                                        "label": int(numeric) if numeric in (0.0, 1.0) else None,
                                        "outcome_observed_at_utc": current.isoformat()}
                if matured:
                    prior["outcomes"].update(matured)
                    if self._persist_fn:
                        self._persist_fn(prior, True)
                if all((pd.Timestamp(prior["forecast_issue_timestamp_utc"]) + pd.Timedelta(minutes=minutes)) <= current
                       for minutes in HORIZON_MINUTES.values()):
                    pending.remove(prior)

            time.sleep(max(0.01, BASE_TICK_SECONDS / SPEED_MULTIPLIERS.get(speed, 1)))
        if complete:
            complete("completed", None)

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self._status == "running", "paused": self._status == "paused",
                "status": self._status, "speed": self._speed, "cursor": self._cursor,
                "current_index": self._cursor, "total": len(self._issue_indices),
                "total_rows": len(self._issue_indices), "latest": self._latest_result,
                "history": list(self._history[-100:]), "recent_history": list(self._history[-100:]),
                "configured": self._df is not None,
                "source_classification": "historical_test_replay" if self._metadata else "unavailable",
                "source_label": "Historical held-out test replay" if self._metadata else "Real-time satellite feed unavailable",
                "provenance": dict(self._metadata), "error": self._metadata.get("error"),
            }


def _finite_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


live_simulator = LiveSimulator()
