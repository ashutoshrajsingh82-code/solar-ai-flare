"""
No-skill and physics-rule baselines.

These sit BELOW the sklearn models in baseline.py. They answer the question
"how much of the skill is already available from trivial or one-physical-
feature rules?" so that the deep models have to earn their complexity.

All classes expose the same API as BaselineMultiTaskModel
(fit / predict_nowcast_proba / predict_forecast_proba / feature_importance)
so they plug into model_service, the registry and predict_service unchanged.

  climatology            predicts the TRAIN-split class priors / flare rates
  flux_rule              depth-limited tree on SoLEXS (or HEL1OS) level + slope
  hard_soft_ratio_rule   depth-limited tree on log(hard/soft ratio) + its slope
                         (tests the Neupert-effect claim; fusion input only)
  two_feature_logreg     logistic regression on flux level + slope
                         (close to operational "current flux + trend" forecasting)

Every threshold / coefficient is fit on the training split only.
Models must stay at module level so joblib can pickle them.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

N_NOWCAST_CLASSES = 4
EPS = 1e-12


def _expand_proba(proba: np.ndarray, classes, n_classes: int) -> np.ndarray:
    """Place predict_proba columns into a fixed (n, n_classes) array.

    sklearn only returns columns for classes seen in training; the app
    expects all 4 nowcast states, so missing classes get probability 0.
    """
    out = np.zeros((proba.shape[0], n_classes))
    for j, c in enumerate(classes):
        out[:, int(c)] = proba[:, j]
    return out


class _BaselineBase:
    name = "baseline"
    feature_names_ = None

    def fit(self, X, y_nowcast, y_forecast, feature_names=None):
        raise NotImplementedError

    def predict_nowcast_proba(self, X):
        raise NotImplementedError

    def predict_forecast_proba(self, X):
        raise NotImplementedError

    def feature_importance(self):
        return None


class ClimatologyBaseline(_BaselineBase):
    """Ignores the input entirely: constant train-split priors."""

    def __init__(self, name="climatology"):
        self.name = name

    def fit(self, X, y_nowcast, y_forecast, feature_names=None):
        self.feature_names_ = feature_names
        y_now = np.asarray(y_nowcast).astype(int)
        counts = np.bincount(y_now, minlength=N_NOWCAST_CLASSES).astype(float)
        self.nowcast_prior_ = counts / max(1.0, counts.sum())
        self.forecast_rate_ = np.asarray(y_forecast, dtype=float).mean(axis=0)
        return self

    def predict_nowcast_proba(self, X):
        return np.tile(self.nowcast_prior_, (len(X), 1))

    def predict_forecast_proba(self, X):
        return np.tile(self.forecast_rate_, (len(X), 1))


class _FeatureSubsetBaseline(_BaselineBase):
    """
    Shared machinery: pick a few named features out of the full matrix,
    transform + standardise them (fit on train), then fit a small model.

    feature_specs: list of (role, candidate_columns, transform)
      - the FIRST candidate present in feature_names is used
      - transform: "identity" | "log10" | "slog" (sign-preserving log for
        heavy-tailed slopes; scale = median |x| on train)
    """

    nowcast_kind = "tree"      # "tree" | "logreg"
    forecast_kind = "tree"     # "tree" | "logreg"
    feature_specs: list = []

    def __init__(self, name):
        self.name = name

    # ---- feature handling -------------------------------------------------
    def _resolve_columns(self, feature_names):
        if feature_names is None:
            raise ValueError(f"{self.name} needs feature_names to locate its input columns.")
        names = list(feature_names)
        idx, used = [], []
        for role, candidates, _ in self.feature_specs:
            hit = next((c for c in candidates if c in names), None)
            if hit is None:
                raise ValueError(
                    f"{self.name} needs one of {candidates} for '{role}', but the selected "
                    f"input configuration does not provide it (hard_soft_ratio_rule requires "
                    f"the 'fusion' input)."
                )
            idx.append(names.index(hit))
            used.append(hit)
        return idx, used

    def _transform_fit(self, Z):
        cols = []
        self.slog_scales_ = {}
        for j, (_, _, kind) in enumerate(self.feature_specs):
            x = Z[:, j].astype(float)
            if kind == "log10":
                x = np.log10(np.clip(x, 0.0, None) + EPS)
            elif kind == "slog":
                nz = np.abs(x[x != 0])
                scale = float(np.median(nz)) if nz.size else 1.0
                self.slog_scales_[j] = max(scale, EPS)
                x = np.sign(x) * np.log10(1.0 + np.abs(x) / self.slog_scales_[j])
            cols.append(x)
        return np.column_stack(cols)

    def _transform_apply(self, Z):
        cols = []
        for j, (_, _, kind) in enumerate(self.feature_specs):
            x = Z[:, j].astype(float)
            if kind == "log10":
                x = np.log10(np.clip(x, 0.0, None) + EPS)
            elif kind == "slog":
                x = np.sign(x) * np.log10(1.0 + np.abs(x) / self.slog_scales_[j])
            cols.append(x)
        return np.column_stack(cols)

    def _prep(self, X):
        Z = np.asarray(X, dtype=float)[:, self.col_idx_]
        Z = self._transform_apply(Z)
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler_.transform(Z)

    # ---- estimators -------------------------------------------------------
    def _make(self, kind, multiclass):
        if kind == "tree":
            depth, leaf = (3, 20) if multiclass else (2, 50)
            return DecisionTreeClassifier(
                max_depth=depth, min_samples_leaf=leaf,
                class_weight="balanced", random_state=42,
            )
        return LogisticRegression(max_iter=1000, class_weight="balanced")

    def fit(self, X, y_nowcast, y_forecast, feature_names=None):
        self.feature_names_ = feature_names
        self.col_idx_, self.used_columns_ = self._resolve_columns(feature_names)
        self.n_total_features_ = np.asarray(X).shape[1]

        Z = np.asarray(X, dtype=float)[:, self.col_idx_]
        Z = self._transform_fit(Z)
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler_ = StandardScaler().fit(Z)
        Zs = self.scaler_.transform(Z)

        y_now = np.asarray(y_nowcast).astype(int)
        self.nowcast_model_ = self._make(self.nowcast_kind, multiclass=True).fit(Zs, y_now)

        y_fc = np.asarray(y_forecast).astype(int)
        self.forecast_models_ = []
        for h in range(y_fc.shape[1]):
            col = y_fc[:, h]
            if len(np.unique(col)) < 2:
                self.forecast_models_.append(float(col[0]) if len(col) else 0.0)  # constant
            else:
                self.forecast_models_.append(self._make(self.forecast_kind, multiclass=False).fit(Zs, col))
        return self

    def predict_nowcast_proba(self, X):
        p = self.nowcast_model_.predict_proba(self._prep(X))
        return _expand_proba(p, self.nowcast_model_.classes_, N_NOWCAST_CLASSES)

    def predict_forecast_proba(self, X):
        Zs = self._prep(X)
        out = np.zeros((Zs.shape[0], len(self.forecast_models_)))
        for h, m in enumerate(self.forecast_models_):
            if isinstance(m, float):
                out[:, h] = m
            else:
                out[:, h] = m.predict_proba(Zs)[:, list(m.classes_).index(1)]
        return out

    def feature_importance(self):
        """Full-length vector (zeros for unused columns) so it lines up with feature_names_."""
        m = self.nowcast_model_
        if hasattr(m, "feature_importances_"):
            used = m.feature_importances_
        elif hasattr(m, "coef_"):
            used = np.abs(m.coef_).mean(axis=0)
        else:
            return None
        full = np.zeros(self.n_total_features_)
        for pos, col in enumerate(self.col_idx_):
            full[col] = used[pos]
        return full.tolist()


_LEVEL = ("level", ["solexs_flux_scaled", "helios_flux_scaled"], "identity")
_SLOPE = ("slope", ["solexs_slope_5m", "helios_slope_5m"], "slog")


class FluxRuleBaseline(_FeatureSubsetBaseline):
    """Thresholds on flux level + short-term slope (tree depth <= 3, fit on train)."""
    nowcast_kind, forecast_kind = "tree", "tree"
    feature_specs = [_LEVEL, _SLOPE]


class HardSoftRatioRuleBaseline(_FeatureSubsetBaseline):
    """Thresholds on log10(HEL1OS/SoLEXS) and its slope: the Neupert-effect rule."""
    nowcast_kind, forecast_kind = "tree", "tree"
    feature_specs = [
        ("log_hard_soft_ratio", ["hard_soft_ratio"], "log10"),
        ("hard_soft_ratio_slope", ["hard_soft_ratio_slope_5m"], "slog"),
    ]


class TwoFeatureLogRegBaseline(_FeatureSubsetBaseline):
    """Logistic regression on flux level + slope: the 'current flux + trend' forecaster."""
    nowcast_kind, forecast_kind = "logreg", "logreg"
    feature_specs = [_LEVEL, _SLOPE]


RULE_BUILDERS = {
    "climatology": lambda: ClimatologyBaseline("climatology"),
    "flux_rule": lambda: FluxRuleBaseline("flux_rule"),
    "hard_soft_ratio_rule": lambda: HardSoftRatioRuleBaseline("hard_soft_ratio_rule"),
    "two_feature_logreg": lambda: TwoFeatureLogRegBaseline("two_feature_logreg"),
}