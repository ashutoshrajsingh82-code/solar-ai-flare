"""
Live inference simulator.

Replays the (already preprocessed + featured) demo dataset sequentially,
"as if" it were arriving live from Aditya-L1, and produces a rolling
prediction stream. State is kept in-process (single-process demo server);
for a multi-worker production deployment this would move to Redis/DB.
"""
import threading
import time
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

SPEED_MULTIPLIERS = {1: 1, 5: 5, 10: 10, 50: 50}
BASE_TICK_SECONDS = 2.0  # wall-clock seconds between ticks at 1x


class LiveSimulator:
    def __init__(self):
        self._lock = threading.Lock()
        self._df: Optional[pd.DataFrame] = None
        self._cursor = 0
        self._sequence_length = 60
        self._speed = 5
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._predict_fn = None
        self._latest_result: Optional[Dict[str, Any]] = None
        self._history = []
        self._max_history = 500

    def configure(self, df: pd.DataFrame, sequence_length: int, predict_fn):
        with self._lock:
            self._df = df.reset_index(drop=True)
            self._sequence_length = sequence_length
            self._predict_fn = predict_fn
            self._cursor = sequence_length
            self._history = []
            self._latest_result = None

    def set_speed(self, speed: int):
        if speed not in SPEED_MULTIPLIERS:
            raise ValueError(f"Unsupported speed {speed}. Choose from {list(SPEED_MULTIPLIERS)}")
        self._speed = speed

    def start(self):
        if self._df is None or self._predict_fn is None:
            raise RuntimeError("Simulator not configured. Call configure() first.")
        with self._lock:
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def pause(self):
        with self._lock:
            self._running = False

    def resume(self):
        if self._df is None:
            raise RuntimeError("Simulator not configured.")
        with self._lock:
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def reset(self):
        with self._lock:
            self._running = False
            self._cursor = self._sequence_length
            self._history = []
            self._latest_result = None

    def _run_loop(self):
        while True:
            with self._lock:
                if not self._running or self._df is None:
                    return
                if self._cursor >= len(self._df):
                    self._running = False
                    return
                window = self._df.iloc[self._cursor - self._sequence_length: self._cursor]
                current_row = self._df.iloc[self._cursor - 1]
                try:
                    pred = self._predict_fn(window)
                except Exception as e:
                    pred = {"error": str(e)}

                result = {
                    "timestamp": str(current_row["timestamp"]),
                    "solexs_flux": float(current_row.get("solexs_flux", np.nan)),
                    "helios_flux": float(current_row.get("helios_flux", np.nan)),
                    "hard_soft_ratio": float(current_row.get("hard_soft_ratio", np.nan)) if "hard_soft_ratio" in current_row else None,
                    "cursor": int(self._cursor),
                    "total": int(len(self._df)),
                    **pred,
                }
                self._latest_result = result
                self._history.append(result)
                if len(self._history) > self._max_history:
                    self._history.pop(0)
                self._cursor += 1
                speed = self._speed

            sleep_time = max(0.05, BASE_TICK_SECONDS / SPEED_MULTIPLIERS.get(speed, 1))
            time.sleep(sleep_time)

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "speed": self._speed,
                "cursor": self._cursor,
                "total": len(self._df) if self._df is not None else 0,
                "latest": self._latest_result,
                "history": list(self._history[-100:]),
                "configured": self._df is not None,
            }


live_simulator = LiveSimulator()
