import numpy as np
import pytest

from ml.models.baseline import MODEL_BUILDERS
from ml.evaluation.metrics import brier_skill_score

NEW = ["climatology", "flux_rule", "hard_soft_ratio_rule", "two_feature_logreg", "gradient_boosting"]
NAMES = ["solexs_flux_scaled", "helios_flux_scaled", "solexs_slope_5m", "helios_slope_5m",
         "hard_soft_ratio", "hard_soft_ratio_slope_5m", "noise"]


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(NAMES)))
    X[:, 4] = np.abs(X[:, 4]) * 3
    y_now = rng.integers(0, 4, size=n)
    y_fc = (rng.random((n, 5)) < [0.02, 0.05, 0.1, 0.2, 0.4]).astype(int)
    y_fc[0] = 1
    return X, y_now, y_fc


@pytest.mark.parametrize("name", NEW)
def test_baseline_api_contract(name):
    X, y_now, y_fc = _data()
    m = MODEL_BUILDERS[name]().fit(X, y_now, y_fc, feature_names=NAMES)
    now = m.predict_nowcast_proba(X)
    fc = m.predict_forecast_proba(X)
    assert now.shape == (len(X), 4) and fc.shape == (len(X), 5)
    assert np.allclose(now.sum(axis=1), 1.0, atol=1e-6)
    assert ((fc >= 0) & (fc <= 1)).all()
    fi = m.feature_importance()
    assert fi is None or len(fi) == len(NAMES)


def test_climatology_matches_train_rates():
    X, y_now, y_fc = _data()
    m = MODEL_BUILDERS["climatology"]().fit(X, y_now, y_fc, feature_names=NAMES)
    assert np.allclose(m.predict_forecast_proba(X[:3])[0], y_fc.mean(axis=0))


def test_missing_nowcast_class_is_zero_padded():
    X, y_now, y_fc = _data()
    y_now = np.where(y_now == 3, 0, y_now)  # class 3 never seen in training
    m = MODEL_BUILDERS["flux_rule"]().fit(X, y_now, y_fc, feature_names=NAMES)
    p = m.predict_nowcast_proba(X)
    assert p.shape[1] == 4 and np.allclose(p[:, 3], 0.0)


def test_hard_soft_ratio_rule_needs_fusion_input():
    X, y_now, y_fc = _data()
    keep = [i for i, n in enumerate(NAMES) if "hard_soft" not in n]
    with pytest.raises(ValueError, match="fusion"):
        MODEL_BUILDERS["hard_soft_ratio_rule"]().fit(
            X[:, keep], y_now, y_fc, feature_names=[NAMES[i] for i in keep])


def test_single_instrument_falls_back_to_available_channel():
    X, y_now, y_fc = _data()
    keep = [i for i, n in enumerate(NAMES) if not n.startswith("solexs") and "hard_soft" not in n]
    names = [NAMES[i] for i in keep]
    m = MODEL_BUILDERS["flux_rule"]().fit(X[:, keep], y_now, y_fc, feature_names=names)
    assert m.used_columns_ == ["helios_flux_scaled", "helios_slope_5m"]


def test_brier_skill_score_reference_is_zero():
    y = np.array([0, 0, 1, 0, 1])
    assert brier_skill_score(y, np.full(5, y.mean()), y.mean()) == pytest.approx(0.0)