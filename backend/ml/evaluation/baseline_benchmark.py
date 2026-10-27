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
from ml.features.engineering import FeatureConfig, add_rolling_features, get_feature_columns, select_feature_columns
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
    # Label each forecast-origin partition independently. An event in a later
    # partition must never create a positive label in an earlier one.
    cadence = cfg.cadence_seconds
    parts = []
    for split in (tr, va, te):
        split_times = set(split.timestamp)
        split_features = feat.loc[feat.timestamp.isin(split_times)].copy().reset_index(drop=True)
        split_labels = generate_forecast_labels(apply_preflare_window(split_features, cfg), cfg)
        for horizon, column in zip(cfg.horizons_hours, FC):
            split_labels[column] = pd.to_numeric(split_labels[column], errors="coerce").astype(float)
            purge = min(int(horizon * 3600 / cadence), len(split_labels))
            if purge:
                split_labels.loc[len(split_labels) - purge:, column] = np.nan
        parts.append(split_labels.dropna(subset=["state"]))
    return parts, cols


def columns_for(frame, config):
    """Use the same provenance-checked registry selection as model training."""
    return select_feature_columns(frame, config)


def best_threshold(y_val, p_val):
    """Threshold maximising TSS on validation; 0.5 if validation has no positives."""
    if len(y_val) == 0 or np.unique(y_val).size < 2:
        return 0.5, False
    best_t, best = 0.5, -2.0
    for t in np.unique(np.round(np.quantile(p_val, np.linspace(0.01, 0.99, 99)), 6)):
        c = binary_confusion_counts(y_val, (p_val >= t).astype(int))
        s = true_skill_statistic(c["tp"], c["fp"], c["tn"], c["fn"])
        if s > best:
            best, best_t = s, float(t)
    return best_t, True


def run(parts, cols, models, configs):
    rows = []
    for config in configs:
        tr_all, va_all, te_all = parts
        use = columns_for(tr_all, config)
        tr = tr_all.dropna(subset=use + ["state"])
        va = va_all.dropna(subset=use + ["state"])
        te = te_all.dropna(subset=use + ["state"])
        if tr.empty or va.empty or te.empty:
            rows.append(dict(model="all", config=config, note="insufficient complete inputs in a split"))
            continue
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
                y_va = pd.to_numeric(va[FC[i]], errors="coerce").to_numpy(dtype=float)
                y_te = pd.to_numeric(te[FC[i]], errors="coerce").to_numpy(dtype=float)
                known_va, known_te = np.isfinite(y_va), np.isfinite(y_te)
                if not known_te.any() or np.unique(y_te[known_te]).size < 2:
                    row[f"TSS_{h}"] = row[f"HSS_{h}"] = row[f"BSS_{h}"] = np.nan
                    continue
                t, tuned = best_threshold(y_va[known_va], p_val[known_va, i]) if known_va.any() else (0.5, False)
                c = binary_confusion_counts(y_te[known_te].astype(int), (p_te[known_te, i] >= t).astype(int))
                row[f"TSS_{h}"] = round(true_skill_statistic(**c), 3)
                row[f"HSS_{h}"] = round(heidke_skill_score(**c), 3)
                train_target = pd.to_numeric(tr[FC[i]], errors="coerce").dropna()
                if train_target.empty:
                    row[f"BSS_{h}"] = np.nan
                else:
                    row[f"BSS_{h}"] = round(brier_skill_score(y_te[known_te], p_te[known_te, i], train_target.mean()), 3)
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
