#!/usr/bin/env python3
"""Build version 3 chronological splits grounded in verified instrument observation days.

Produces:
  - data/labels/split_v3_solexs.json (candidate days: solexs_ok == True)
  - data/labels/split_v3_fusion.json (candidate days: both == True)

Rules:
  - Strict whole-day chronological ordering (~70/15/15) with >= 24h purge gap.
  - Hard requirement validation: >= 20 M/X onsets, >= 5 distinct active regions,
    no single-region burst dominance.
  - Baselines (Climatology, Persistence) evaluated on Val and Test partitions.
  - Explicit warning if a split has fewer than 15 days (bootstrap instability).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

cur_path = Path(__file__).resolve()
if cur_path.parent.name == "scripts" and cur_path.parent.parent.name == "backend":
    REPO_ROOT = cur_path.parent.parent.parent
else:
    REPO_ROOT = cur_path.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from ml.labels.goes_labels import split_by_instrument_days
from scripts.generate_goes_v2_report import (
    compute_binary_metrics,
    block_bootstrap_metrics,
)

LABEL_DIR = REPO_ROOT / "data" / "labels"
DATA_DIR = REPO_ROOT / "data"
OVERLAP_PATH = DATA_DIR / "overlap_days.csv"
PARQUET_PATH = LABEL_DIR / "goes_labels_v2.parquet"
FLARES_PATH = DATA_DIR / "goes" / "2024-05" / "goes_flares.csv"


def load_all_flares() -> pd.DataFrame:
    flares_list = []
    for md in sorted((DATA_DIR / "goes").iterdir()):
        f_p = md / "goes_flares.csv"
        if f_p.exists():
            flares_list.append(pd.read_csv(f_p))
    df = pd.concat(flares_list).drop_duplicates(subset=["start", "goes_class"]).reset_index(drop=True)
    df["start"] = pd.to_datetime(df["start"], utc=True)
    if "is_MX" not in df.columns:
        df["is_MX"] = df["goes_class"].astype(str).str[0].isin(["M", "X"])
    return df


def evaluate_baselines_for_splits(
    splits: list[tuple[str, pd.DataFrame, dict]],
    flares_df: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 6, 12, 24),
) -> dict:
    splits_by_part = {st["partition"]: (df, st) for _, df, st in splits}
    train_df, train_stats = splits_by_part.get("train", (pd.DataFrame(), {}))

    train_base_rates = {}
    for h in horizons:
        col = f"y_{h}h"
        if "forecast_valid" in train_df.columns and col in train_df.columns:
            v = train_df.loc[train_df["forecast_valid"], col].dropna()
            train_base_rates[h] = float(v.mean()) if len(v) else 0.0
        else:
            train_base_rates[h] = 0.0

    mx_starts = flares_df.loc[flares_df["is_MX"], "start"].sort_values().values
    eval_res = {}

    for part_name in ("val", "test"):
        if part_name not in splits_by_part:
            continue
        s_df, s_stats = splits_by_part[part_name]
        n_days = s_stats.get("days_count", 0)

        valid_mask = s_df.get("forecast_valid", pd.Series(True, index=s_df.index)).values
        s_df_valid = s_df[valid_mask].copy()
        eval_timestamps = s_df_valid.index.values
        eval_dates = s_df_valid.index.date

        part_eval = {"days_count": n_days, "bootstrap_reliable": bool(n_days >= 15), "horizons": {}}

        for h in horizons:
            col = f"y_{h}h"
            if col not in s_df_valid.columns:
                continue
            y_true = s_df_valid[col].dropna().values.astype(int)
            cur_dates = eval_dates[:len(y_true)]
            cur_ts = eval_timestamps[:len(y_true)]

            # Climatology
            p_clim = train_base_rates.get(h, 0.0)
            y_prob_c = np.full_like(y_true, fill_value=p_clim, dtype=float)
            y_pred_c = (y_prob_c >= 0.5).astype(int)
            m_clim = compute_binary_metrics(y_true, y_pred_c, y_prob_c)
            ci_clim = block_bootstrap_metrics(cur_dates, y_true, y_pred_c, y_prob_c) if n_days >= 2 else {}

            # Persistence
            h_ns = np.timedelta64(h, "h")
            y_pred_p = np.zeros_like(y_true, dtype=int)
            for idx_t, t_val in enumerate(cur_ts):
                t_low = t_val - h_ns
                if np.any((mx_starts >= t_low) & (mx_starts < t_val)):
                    y_pred_p[idx_t] = 1

            m_pers = compute_binary_metrics(y_true, y_pred_p)
            ci_pers = block_bootstrap_metrics(cur_dates, y_true, y_pred_p) if n_days >= 2 else {}

            part_eval["horizons"][f"{h}h"] = {
                "n_pos": int(np.sum(y_true == 1)),
                "n_neg": int(np.sum(y_true == 0)),
                "climatology": {"metrics": m_clim, "ci": ci_clim},
                "persistence": {"metrics": m_pers, "ci": ci_pers},
            }

        eval_res[part_name] = part_eval

    return eval_res


def main() -> None:
    try:
        from scripts.scan_instrument_overlap import scan_pradan_downloads
        overlap_df = scan_pradan_downloads()
    except Exception:
        overlap_df = pd.read_csv(OVERLAP_PATH)
    labels_df = pd.read_parquet(PARQUET_PATH)
    labels_df.index = pd.to_datetime(labels_df.index, utc=True)
    flares_df = load_all_flares()

    # 1. SoLEXS Variant (candidate days: solexs_ok == True)
    solexs_cand_days = overlap_df.loc[overlap_df["solexs_ok"], "date"].tolist()
    print(f"\nBuilding split_v3_solexs with {len(solexs_cand_days)} candidate days ...")
    solexs_splits = split_by_instrument_days(
        df=labels_df,
        candidate_days=solexs_cand_days,
        flares_df=flares_df,
        ratios=(0.70, 0.15, 0.15),
        purge_gap_days=1,
        split_names=("train", "val", "test"),
        min_mx_onsets=20,
        min_active_regions=5,
    )

    solexs_plan = []
    print("\n[SoLEXS SPLIT GATES]")
    for s_name, sub, stats in solexs_splits:
        solexs_plan.append(stats)
        print(f"  {stats['partition'].upper()} [{stats['start_date']} .. {stats['end_date']}]: "
              f"Days={stats['days_count']}, M/X onsets={stats['mx_onsets']}, ARs={stats['distinct_active_regions']} "
              f"-> {stats['requirement_status']} ({stats['diagnosis']})")

    solexs_baselines = evaluate_baselines_for_splits(solexs_splits, flares_df)

    solexs_out = {
        "variant": "solexs_v3",
        "candidate_rule": "solexs_ok >= 80% day coverage",
        "total_candidate_days": len(solexs_cand_days),
        "purge_gap_days": 1,
        "splits": solexs_plan,
        "baseline_evaluation": solexs_baselines,
    }
    solexs_json_path = LABEL_DIR / "split_v3_solexs.json"
    solexs_json_path.write_text(json.dumps(solexs_out, indent=2), encoding="utf-8")
    print(f"Saved: {solexs_json_path}")

    # 2. Fusion Variant (candidate days: both == True)
    fusion_cand_days = overlap_df.loc[overlap_df["both"], "date"].tolist()
    print(f"\nBuilding split_v3_fusion with {len(fusion_cand_days)} candidate days ...")
    fusion_splits = split_by_instrument_days(
        df=labels_df,
        candidate_days=fusion_cand_days,
        flares_df=flares_df,
        ratios=(0.70, 0.15, 0.15),
        purge_gap_days=1,
        split_names=("train", "val", "test"),
        min_mx_onsets=20,
        min_active_regions=5,
    )

    fusion_plan = []
    print("\n[FUSION SPLIT GATES]")
    for s_name, sub, stats in fusion_splits:
        fusion_plan.append(stats)
        print(f"  {stats['partition'].upper()} [{stats['start_date']} .. {stats['end_date']}]: "
              f"Days={stats['days_count']}, M/X onsets={stats['mx_onsets']}, ARs={stats['distinct_active_regions']} "
              f"-> {stats['requirement_status']} ({stats['diagnosis']})")

    fusion_baselines = evaluate_baselines_for_splits(fusion_splits, flares_df)

    fusion_out = {
        "variant": "fusion_v3",
        "candidate_rule": "both solexs_ok and helios_ok >= 80% day coverage",
        "total_candidate_days": len(fusion_cand_days),
        "purge_gap_days": 1,
        "splits": fusion_plan,
        "baseline_evaluation": fusion_baselines,
    }
    fusion_json_path = LABEL_DIR / "split_v3_fusion.json"
    fusion_json_path.write_text(json.dumps(fusion_out, indent=2), encoding="utf-8")
    print(f"Saved: {fusion_json_path}")


if __name__ == "__main__":
    main()
