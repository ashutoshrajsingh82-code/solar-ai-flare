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
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

from ml.models.baseline_rules import RULE_BUILDERS


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


def build_gradient_boosting(class_weight="balanced") -> HistGradientBoostingClassifier:
    """Strongest tabular baseline; the bar a deep model should clear."""
    return HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.1, max_iter=200,
        class_weight=class_weight, random_state=42,
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
        # Forecast estimators are fitted independently because each horizon can
        # have a different set of known labels.
        self.forecast_model = None  # retained as a compatibility attribute
        self.forecast_models_ = []
        self.feature_names_ = None

    def fit(self, X: np.ndarray, y_nowcast: np.ndarray, y_forecast: np.ndarray, feature_names=None):
        self.feature_names_ = feature_names
        X = np.asarray(X)
        y_forecast = np.asarray(y_forecast, dtype=float)
        self.nowcast_model.fit(X, y_nowcast)
        self.forecast_models_ = []
        for h in range(y_forecast.shape[1]):
            known = np.isfinite(y_forecast[:, h])
            targets = y_forecast[known, h]
            if len(targets) == 0:
                raise ValueError(f"Forecast horizon {h} has no known training labels.")
            classes = np.unique(targets)
            if len(classes) == 1:
                self.forecast_models_.append(float(classes[0]))
            else:
                estimator = self.base_estimator_fn()
                estimator.fit(X[known], targets.astype(int))
                self.forecast_models_.append(estimator)
        return self

    def predict_nowcast_proba(self, X: np.ndarray) -> np.ndarray:
        return self.nowcast_model.predict_proba(X)

    def predict_forecast_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns shape (n_samples, n_horizons) of P(class=1) for each horizon."""
        X = np.asarray(X)
        out = np.zeros((X.shape[0], len(self.forecast_models_)))
        for i, estimator in enumerate(self.forecast_models_):
            if isinstance(estimator, float):
                out[:, i] = estimator
                continue
            probabilities = estimator.predict_proba(X)
            classes = list(estimator.classes_)
            out[:, i] = probabilities[:, classes.index(1)] if 1 in classes else 0.0
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
    "gradient_boosting": lambda: BaselineMultiTaskModel(build_gradient_boosting, "gradient_boosting"),
}

# No-skill + physics-rule baselines (climatology, flux_rule, hard_soft_ratio_rule, two_feature_logreg)
MODEL_BUILDERS.update(RULE_BUILDERS)