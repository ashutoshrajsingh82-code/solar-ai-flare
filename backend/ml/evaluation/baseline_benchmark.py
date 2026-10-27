"""
Baseline benchmark: runs every baseline through the SAME chronological split and
reports forecast skill per horizon, so deep models are compared against a ladder
of baselines rather than in isolation.

    python -m ml.evaluation.baseline_benchmark                      # longer synthetic set (default)
    python -m ml.evaluation.baseline_benchmark --csv path/to.csv    # your own SoLEXS/HEL1OS CSV
    python -m ml.evaluation.baseline_benchmark --ablation --with-svm

Method (leakage-safe):
  * chronological 70/15/15 split; scaler fit on train only
  * decision threshold per horizon is tuned on VALIDATION (max TSS), never on test
  * headline metrics: TSS, HSS, Brier Skill Score vs train climatology
Horizons with no positive events in the test split are reported as n/a, not 0.
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

from ml.data.synthetic_generator import generate_synthetic_dataset
from ml.evaluation.metrics import (
    binary_confusion_counts, true_skill_statistic, heidke_skill_score, brier_skill_score,
)
from ml.features.engineering import FeatureConfig, add_rolling_features, get_feature_columns
from ml.features.labels import LabelConfig, apply_preflare_window, generate_forecast_labels
from ml.features.windowing import chronological_split
from ml.models.baseline import MODEL_BUILDERS
from ml.preprocessing.alignment import align_and_resample
from ml.preprocessing.cleaning import clean_dataframe
from ml.preprocessing.transform import FluxTransformer

FC = ["forecast_1h", "forecast_3h", "forecast_6h", "forecast_12h", "forecast_24h"]
HORIZONS = ["1h", "3h", "6h", "12h", "24h"]
LADDER = ["climatology", "flux_rule", "hard_soft_ratio_rule", "two_feature_logreg",
          "logistic_regression", "random_forest", "gradient_boosting"]


def prepare(raw: pd.DataFrame):
    cleaned, _ = clean_dataframe(raw)
    aligned, _ = align_and_resample(cleaned)
    tr, va, te = chronological_split(aligned)
    transformer = FluxTransformer().fit(tr)
    tf = pd.concat([transformer.transform(x) for x in (tr, va, te)], ignore_index=True)
    feat = add_rolling_features(tf, FeatureConfig())
    cols = get_feature_columns(feat)
    cfg = LabelConfig()
    labeled = generate_forecast_labels(apply_preflare_window(feat, cfg), cfg)
    parts = [x[["timestamp"]].merge(labeled, on="timestamp", how="inner")
             .dropna(subset=cols + ["state"] + FC) for x in (tr, va, te)]
    return parts, cols


def columns_for(cols, config):
    if config == "solexs_only":
        return [c for c in cols if not c.startswith("helios") and "hard_soft_ratio" not in c]
    if config == "helios_only":
        return [c for c in cols if not c.startswith("solexs") and "hard_soft_ratio" not in c]
    return cols


def best_threshold(y_val, p_val):
    """Threshold maximising TSS on validation; 0.5 if validation has no positives."""
    if y_val.sum() == 0:
        return 0.5, False
    best_t, best = 0.5, -2.0
    for t in np.unique(np.round(np.quantile(p_val, np.linspace(0.01, 0.99, 99)), 6)):
        c = binary_confusion_counts(y_val, (p_val >= t).astype(int))
        s = true_skill_statistic(c["tp"], c["fp"], c["tn"], c["fn"])
        if s > best:
            best, best_t = s, float(t)
    return best_t, True


def run(parts, cols, models, configs):
    tr, va, te = parts
    rows = []
    for config in configs:
        use = columns_for(cols, config)
        clim = None
        for name in models:
            try:
                m = MODEL_BUILDERS[name]().fit(
                    tr[use].values, tr["state"].values.astype(int), tr[FC].values, feature_names=use)
            except ValueError as e:      # e.g. hard_soft_ratio_rule on single-instrument input
                rows.append(dict(model=name, config=config, note=str(e)[:60]))
                continue
            p_val, p_te = m.predict_forecast_proba(va[use].values), m.predict_forecast_proba(te[use].values)
            now_pred = m.predict_nowcast_proba(te[use].values).argmax(axis=1)
            from sklearn.metrics import f1_score
            row = dict(model=name, config=config,
                       nowcast_f1_macro=round(float(f1_score(te["state"].values, now_pred, average="macro", zero_division=0)), 3))
            for i, h in enumerate(HORIZONS):
                y_va, y_te = va[FC[i]].values.astype(int), te[FC[i]].values.astype(int)
                if y_te.sum() == 0:
                    row[f"TSS_{h}"] = row[f"HSS_{h}"] = row[f"BSS_{h}"] = np.nan
                    continue
                t, tuned = best_threshold(y_va, p_val[:, i])
                c = binary_confusion_counts(y_te, (p_te[:, i] >= t).astype(int))
                row[f"TSS_{h}"] = round(true_skill_statistic(**c), 3)
                row[f"HSS_{h}"] = round(heidke_skill_score(**c), 3)
                row[f"BSS_{h}"] = round(brier_skill_score(y_te, p_te[:, i], tr[FC[i]].values.mean()), 3)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV with timestamp, solexs_flux, helios_flux, state, flare_class")
    ap.add_argument("--days", type=float, default=15.0)
    ap.add_argument("--flares", type=int, default=30)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--ablation", action="store_true", help="also run solexs_only / helios_only")
    ap.add_argument("--with-svm", action="store_true")
    ap.add_argument("--out", default="ml/saved_models/baseline_benchmark.csv")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    if args.csv:
        raw = pd.read_csv(args.csv)
    else:
        raw, _ = generate_synthetic_dataset(n_days=args.days, n_flares=args.flares, seed=args.seed)
    parts, cols = prepare(raw)
    tr, va, te = parts
    print("rows train/val/test:", len(tr), len(va), len(te))
    print("positive rate per horizon  train:", [round(float(tr[c].mean()), 3) for c in FC])
    print("                           test :", [round(float(te[c].mean()), 3) for c in FC])
    if te[FC].values.sum() == 0:
        print("WARNING: test split has no positive forecast labels; forecast skill is not measurable.")

    models = LADDER + (["svm"] if args.with_svm else [])
    configs = ["fusion"] + (["solexs_only", "helios_only"] if args.ablation else [])
    df = run(parts, cols, models, configs)
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(df.to_string(index=False))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()