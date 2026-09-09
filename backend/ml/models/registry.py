"""
Lightweight model registry.

Tracks metadata for every trained model (classical + deep) so the dashboard
can list, compare, and activate models. Backed by a JSON index file plus
joblib/torch checkpoint files under ml/saved_models/.
"""
import os
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

REGISTRY_INDEX_FILENAME = "registry_index.json"


@dataclass
class ModelRecord:
    model_id: str
    name: str
    architecture: str          # "logistic_regression" | "random_forest" | "svm" | "cnn_1d" | "lstm" | "cnn_lstm_fusion" | "cnn_transformer"
    version: str
    created_at: str
    training_dataset: str
    input_configuration: str   # "solexs_only" | "helios_only" | "fusion"
    metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoint_path: str = ""
    scaler_path: str = ""
    active: bool = False
    model_kind: str = "classical"  # "classical" | "deep"
    params: Dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    def __init__(self, saved_models_dir: str):
        self.dir = saved_models_dir
        os.makedirs(self.dir, exist_ok=True)
        self.index_path = os.path.join(self.dir, REGISTRY_INDEX_FILENAME)
        if not os.path.exists(self.index_path):
            self._write_index({})

    def _read_index(self) -> Dict[str, dict]:
        with open(self.index_path, "r") as f:
            return json.load(f)

    def _write_index(self, index: Dict[str, dict]):
        with open(self.index_path, "w") as f:
            json.dump(index, f, indent=2, default=str)

    def register(self, record: ModelRecord):
        index = self._read_index()
        index[record.model_id] = asdict(record)
        self._write_index(index)

    def get(self, model_id: str) -> Optional[dict]:
        return self._read_index().get(model_id)

    def list_all(self):
        return list(self._read_index().values())

    def set_active(self, model_id: str):
        index = self._read_index()
        if model_id not in index:
            raise KeyError(f"Model {model_id} not found in registry.")
        for mid, rec in index.items():
            rec["active"] = (mid == model_id)
        self._write_index(index)

    def get_active(self) -> Optional[dict]:
        for rec in self._read_index().values():
            if rec.get("active"):
                return rec
        return None

    def delete(self, model_id: str):
        index = self._read_index()
        index.pop(model_id, None)
        self._write_index(index)


def new_model_id(architecture: str) -> str:
    return f"{architecture}_{int(time.time())}"
