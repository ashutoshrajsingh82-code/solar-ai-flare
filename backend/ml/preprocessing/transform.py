"""
Transformation stage: log10(x + eps) then scaling.

CRITICAL: scalers must be fit ONLY on the training split, then reused
(never re-fit) on validation/test/live data. Fitted objects are saved
alongside the trained model via joblib for reproducible inference.
"""
import numpy as np
import pandas as pd
import joblib
import os
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

EPSILON = 1e-12

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


@dataclass
class TransformConfig:
    scaler_type: str = "standard"
    log_transform: bool = True


class FluxTransformer:
    """Stateful transformer: log10 + configurable scaler, fit on train only."""

    def __init__(self, config: TransformConfig = None):
        self.config = config or TransformConfig()
        self.scaler = SCALERS[self.config.scaler_type]()
        self.fitted = False
        self.columns = ["solexs_flux", "helios_flux"]

    def _log(self, arr: np.ndarray) -> np.ndarray:
        return np.log10(arr + EPSILON) if self.config.log_transform else arr

    def fit(self, df: pd.DataFrame):
        x = self._log(df[self.columns].values)
        self.scaler.fit(x)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("FluxTransformer must be fit on training data before transform().")
        x = self._log(df[self.columns].values)
        x_scaled = self.scaler.transform(x)
        out = df.copy()
        out["solexs_flux_scaled"] = x_scaled[:, 0]
        out["helios_flux_scaled"] = x_scaled[:, 1]
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FluxTransformer":
        return joblib.load(path)
