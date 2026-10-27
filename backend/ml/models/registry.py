"""
Lightweight model registry.

Tracks metadata for every trained model (classical + deep) so the dashboard
can list, compare, and activate models. Backed by a JSON index file plus
joblib/torch checkpoint files under ml/saved_models/.
"""
import os
import json
import time
import tempfile
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

REGISTRY_INDEX_FILENAME = "registry_index.json"
SYNTHETIC_ONLY_MARKER = "SYNTHETIC_NOT_FOR_PRODUCTION"


@dataclass
class ModelRecord:
    model_id: str
    name: str
    architecture: str          # baselines (climatology, flux_rule, hard_soft_ratio_rule, two_feature_logreg, logistic_regression, random_forest, svm, gradient_boosting) | "cnn_1d" | "lstm" | "cnn_lstm_fusion" | "cnn_transformer"
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
    status: str = "untrained"
    transformer_required: bool = True
    transformer_path: str = ""


class ModelRegistry:
    def __init__(self, saved_models_dir: str):
        self.dir = saved_models_dir
        os.makedirs(self.dir, exist_ok=True)
        self.index_path = os.path.join(self.dir, REGISTRY_INDEX_FILENAME)
        if not os.path.exists(self.index_path):
            self._write_index({})

    def _read_index(self) -> Dict[str, dict]:
        with open(self.index_path, "r") as f:
            index = json.load(f)
        # Legacy records without explicit lifecycle metadata are uncertain.
        for rec in index.values():
            rec.setdefault("status", "untrained")
            rec.setdefault("transformer_required", True)
            rec.setdefault("transformer_path", rec.get("scaler_path", ""))
            rec.setdefault("scaler_path", rec.get("transformer_path", ""))
            if rec["status"] not in {"untrained", "demo_only", "trained"}:
                rec["status"] = "untrained"
            if rec["status"] == "untrained":
                rec["active"] = False
        return index

    def _write_index(self, index: Dict[str, dict]):
        fd, tmp = tempfile.mkstemp(prefix="registry-", suffix=".tmp", dir=self.dir)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(index, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.index_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def register(self, record: ModelRecord):
        if (record.training_dataset == SYNTHETIC_ONLY_MARKER or
                record.params.get("dataset_status") == SYNTHETIC_ONLY_MARKER or
                record.params.get("model_status") == SYNTHETIC_ONLY_MARKER or
                record.params.get("production_registry_eligible") is False):
            raise ValueError("Synthetic smoke models cannot be registered in the production model registry.")
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
        rec = index[model_id]
        self.validate_artifacts(model_id, rec)
        for mid, item in index.items():
            item["active"] = (mid == model_id)
        self._write_index(index)

    def validate_artifacts(self, model_id: str, rec: Optional[dict] = None):
        rec = rec or self.get(model_id)
        if not rec:
            raise KeyError(f"Model {model_id} not found in registry.")
        params = rec.get("params", {})
        if (rec.get("training_dataset") == SYNTHETIC_ONLY_MARKER or
                params.get("dataset_status") == SYNTHETIC_ONLY_MARKER or
                params.get("model_status") == SYNTHETIC_ONLY_MARKER or
                params.get("production_registry_eligible") is False):
            raise ValueError("Synthetic smoke models cannot be activated in the production model registry.")
        if rec.get("status") == "untrained":
            raise ValueError(f"Model {model_id} is untrained and cannot be activated.")
        if rec.get("status") == "trained":
            params = rec.get("params", {})
            if not params.get("target_definition") or not params.get("forecast_horizons"):
                raise ValueError(f"Model {model_id} has no compatible target and forecast-horizon contract.")
            if not params.get("feature_schema_version"):
                raise ValueError(f"Model {model_id} has no compatible feature-schema version.")
        checkpoint = rec.get("checkpoint_path")
        if not checkpoint or not os.path.isfile(checkpoint):
            raise ValueError(f"Model {model_id} checkpoint is missing.")
        if rec.get("transformer_required", True):
            transformer = rec.get("transformer_path") or rec.get("scaler_path")
            if not transformer or not os.path.isfile(transformer):
                raise ValueError(f"Model {model_id} preprocessing transformer is missing.")
        try:
            if rec.get("model_kind") == "classical":
                import joblib
                artifact = joblib.load(checkpoint)
                if not isinstance(artifact, dict) or "model" not in artifact or not artifact.get("feature_cols"):
                    raise ValueError("Classical checkpoint metadata is incomplete.")
                expected = rec.get("params", {}).get("n_features")
                if expected is not None and len(artifact["feature_cols"]) != expected:
                    raise ValueError("Classical feature metadata does not match the registry.")
            else:
                import torch
                artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
                architecture = rec.get("architecture")
                if artifact.get("model_type") != architecture or "state_dict" not in artifact:
                    raise ValueError("Deep checkpoint metadata does not match the registry.")
                if architecture in {"cnn_transformer", "dlstm"} and artifact.get("architecture_version") != 2:
                    raise ValueError(f"Legacy {architecture} checkpoint requires retraining for this architecture version.")
                from ml.models.deep_models import DEEP_MODEL_BUILDERS
                builder = DEEP_MODEL_BUILDERS.get(architecture)
                if builder is None:
                    raise ValueError("Deep model architecture is unsupported.")
                model_config = artifact.get("model_config", {})
                builder_kwargs = {key: model_config[key]
                                  for key in ("hidden_size", "num_layers", "max_sequence_length")
                                  if model_config.get(key) is not None}
                model = builder(n_channels=artifact.get("n_channels", rec.get("params", {}).get("n_channels", 2)),
                                **builder_kwargs)
                model.load_state_dict(artifact["state_dict"])
            if rec.get("transformer_required", True):
                import joblib
                transformer = joblib.load(rec.get("transformer_path") or rec.get("scaler_path"))
                if not getattr(transformer, "fitted", False) or not callable(getattr(transformer, "transform", None)):
                    raise ValueError("Preprocessing transformer is not fitted or compatible.")
        except Exception as exc:
            if isinstance(exc, ValueError) and "requires retraining" in str(exc):
                raise
            raise ValueError(f"Model {model_id} artifacts are unreadable or incompatible.") from exc

    def get_active(self) -> Optional[dict]:
        for rec in self._read_index().values():
            if rec.get("active") and rec.get("status") != "untrained":
                return rec
        return None

    def delete(self, model_id: str):
        index = self._read_index()
        index.pop(model_id, None)
        self._write_index(index)


def new_model_id(architecture: str) -> str:
    return f"{architecture}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
