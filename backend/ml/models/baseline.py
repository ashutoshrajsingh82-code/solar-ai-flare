"""
Classical ML baseline models operating on engineered statistical features
(not raw sequences). Used as interpretable baselines against the deep
sequence models.

Each model predicts BOTH tasks:
  - Nowcast: 4-class classification (Quiet / Pre-Flare / Flare / Decay)
  - Forecast: 5 independent binary targets (1h/3h/6h/12h/24h), via MultiOutputClassifier
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier


def build_logistic_regression(class_weight="balanced") -> LogisticRegression:
    return LogisticRegression(max_iter=1000, class_weight=class_weight)


def build_random_forest(class_weight="balanced", n_estimators=200) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_leaf=3,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=42,
    )


def build_svm(class_weight="balanced") -> SVC:
    return SVC(kernel="rbf", probability=True, class_weight=class_weight, random_state=42)


class BaselineMultiTaskModel:
    """
    Wraps a sklearn classifier to jointly handle:
      - single nowcast_model (4-class)
      - forecast_model: MultiOutputClassifier wrapping 5 independent binary classifiers
    """

    def __init__(self, base_estimator_fn, name: str):
        self.name = name
        self.base_estimator_fn = base_estimator_fn
        self.nowcast_model = base_estimator_fn()
        self.forecast_model = MultiOutputClassifier(base_estimator_fn())
        self.feature_names_ = None

    def fit(self, X: np.ndarray, y_nowcast: np.ndarray, y_forecast: np.ndarray, feature_names=None):
        self.feature_names_ = feature_names
        self.nowcast_model.fit(X, y_nowcast)
        self.forecast_model.fit(X, y_forecast)
        return self

    def predict_nowcast_proba(self, X: np.ndarray) -> np.ndarray:
        return self.nowcast_model.predict_proba(X)

    def predict_forecast_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns shape (n_samples, n_horizons) of P(class=1) for each horizon."""
        proba_list = self.forecast_model.predict_proba(X)  # list of (n_samples, n_classes_h) arrays
        out = np.zeros((X.shape[0], len(proba_list)))
        for i, p in enumerate(proba_list):
            if p.shape[1] == 2:
                out[:, i] = p[:, 1]
            else:
                only_class = self.forecast_model.estimators_[i].classes_[0]
                out[:, i] = 1.0 if only_class == 1 else 0.0
        return out

    def feature_importance(self):
        if hasattr(self.nowcast_model, "feature_importances_"):
            return self.nowcast_model.feature_importances_.tolist()
        if hasattr(self.nowcast_model, "coef_"):
            return np.abs(self.nowcast_model.coef_).mean(axis=0).tolist()
        return None


MODEL_BUILDERS = {
    "logistic_regression": lambda: BaselineMultiTaskModel(build_logistic_regression, "logistic_regression"),
    "random_forest": lambda: BaselineMultiTaskModel(build_random_forest, "random_forest"),
    "svm": lambda: BaselineMultiTaskModel(build_svm, "svm"),
}
