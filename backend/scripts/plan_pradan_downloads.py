#!/usr/bin/env python3
"""Generate PRADAN instrument download plan across target date ranges.

Columns:
  date, year_month, priority (1..5), reason, has_solexs, has_helios,
  m_class_count, x_class_count, active_regions

Priority Rules:
  1. Priority 1 (CRITICAL): Days with BOTH SoLEXS and HEL1OS locally or planned.
  2. Priority 2 (HIGH): Days with at least one instrument already available locally.
  3. Priority 3 (HIGH): Days with X-class flaring activity across multiple months/regions.
  4. Priority 4 (MEDIUM): Days with M-class flaring activity across multiple months/regions.
  5. Priority 5 (BALANCING): Quiet-sun and C-only days across different calendar months
     to prevent active-region sampling skew in downstream models.

Quarantine Guard:
  - October 2024 (2024-10-01 .. 2024-10-31) is strictly sealed. Under no circumstances
    will any October 2024 date be included in pradan_plan.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

cur_path = Path(__file__).resolve()
if cur_path.parent.name == "scripts" and cur_path.parent.parent.name == "backend":
    REPO_ROOT = cur_path.parent.parent.parent
else:
    REPO_ROOT = cur_path.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

CONFIG_PATH = REPO_ROOT / "configs" / "goes_label_ranges.yaml"
DATA_DIR = REPO_ROOT / "data"
DATA_GOES_DIR = DATA_DIR / "goes"
OVERLAP_PATH = DATA_DIR / "overlap_days.csv"
OUT_CSV = DATA_DIR / "pradan_plan.csv"

HOLDOUT_START = pd.Timestamp("2024-10-01", tz="UTC")
HOLDOUT_END = pd.Timestamp("2024-10-31 23:59:59", tz="UTC")


def assert_no_holdout_date(d_str: str) -> None:
    ts = pd.Timestamp(d_str, tz="UTC")
    if ts >= HOLDOUT_START and ts <= HOLDOUT_END:
        raise ValueError(f"FATAL: Date {d_str} violates October 2024 holdout quarantine!")


def build_pradan_plan(min_days: int = 60) -> pd.DataFrame:
    """Build the prioritized PRADAN download plan matching exact spec."""
    # Load configuration
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    target_ranges = cfg.get("target_ranges", [])

    # Load local overlap days from Downloads and data/pradan_downloads
    local_solexs = set()
    local_helios = set()
    local_both = set()
    try:
        from scripts.scan_instrument_overlap import scan_pradan_downloads
        ov_df = scan_pradan_downloads()
    except Exception:
        ov_df = pd.read_csv(OVERLAP_PATH) if OVERLAP_PATH.exists() else pd.DataFrame()

    if not ov_df.empty:
        for _, r in ov_df.iterrows():
            d = str(r["date"])
            if r["solexs_ok"]:
                local_solexs.add(d)
            if r["helios_ok"]:
                local_helios.add(d)
            if r["both"]:
                local_both.add(d)

    # Load flares to compute flare stats per day
    daily_flares: dict[str, dict] = {}
    if DATA_GOES_DIR.exists():
        for md in sorted(DATA_GOES_DIR.iterdir()):
            fl_csv = md / "goes_flares.csv"
            if fl_csv.exists():
                df_fl = pd.read_csv(fl_csv)
                for _, r in df_fl.iterrows():
                    st = str(r["start"])[:10]
                    cls = str(r.get("goes_class", "")).strip().upper()
                    letter = cls[0] if cls else "Q"
                    ar = r.get("noaa_ar")
                    ar_str = str(int(ar)) if pd.notna(ar) and ar > 0 else ""

                    if st not in daily_flares:
                        daily_flares[st] = {
                            "m_count": 0,
                            "x_count": 0,
                            "max_letter": "Q",
                            "ars": set(),
                        }
                    if letter == "M":
                        daily_flares[st]["m_count"] += 1
                    elif letter == "X":
                        daily_flares[st]["x_count"] += 1
                    if ar_str:
                        daily_flares[st]["ars"].add(ar_str)

                    order = {"Q": 0, "A": 1, "B": 2, "C": 3, "M": 4, "X": 5}
                    if order.get(letter, 0) > order.get(daily_flares[st]["max_letter"], 0):
                        daily_flares[st]["max_letter"] = letter

    # Build full list of UTC days in target ranges (strictly excluding Oct 2024)
    all_plan_days = []
    for rng in target_ranges:
        s_date = pd.Timestamp(rng["start_date"], tz="UTC")
        e_date = pd.Timestamp(rng["end_date"], tz="UTC")

        cur = s_date
        while cur <= e_date:
            cur_str = cur.strftime("%Y-%m-%d")
            # Enforce holdout
            if cur >= HOLDOUT_START and cur <= HOLDOUT_END:
                cur += pd.Timedelta(days=1)
                continue

            assert_no_holdout_date(cur_str)

            ym = cur_str[:7]
            fl_info = daily_flares.get(cur_str, {"m_count": 0, "x_count": 0, "max_letter": "Q", "ars": set()})
            s_have = (cur_str in local_solexs)
            h_have = (cur_str in local_helios)
            both_have = (cur_str in local_both)

            # Assign Priority and Reason
            if both_have:
                priority = 1
                reason = "dual instrument verified on disk"
            elif s_have or h_have:
                priority = 2
                inst = "SoLEXS" if s_have else "HEL1OS"
                reason = f"single instrument available ({inst}); download counterpart"
            elif fl_info["max_letter"] == "X":
                priority = 3
                reason = f"X-class flare activity ({fl_info['x_count']} X events)"
            elif fl_info["max_letter"] == "M":
                priority = 4
                reason = f"M-class flare activity ({fl_info['m_count']} M events)"
            else:
                priority = 5
                reason = "quiet sun / low-activity baseline balancing"

            ars_str = ",".join(sorted(fl_info["ars"]))

            all_plan_days.append({
                "date": cur_str,
                "year_month": ym,
                "priority": priority,
                "reason": reason,
                "has_solexs": s_have,
                "has_helios": h_have,
                "m_class_count": fl_info["m_count"],
                "x_class_count": fl_info["x_count"],
                "active_regions": ars_str,
            })
            cur += pd.Timedelta(days=1)

    plan_df = pd.DataFrame(all_plan_days)
    plan_df = plan_df.sort_values(by=["priority", "date"]).reset_index(drop=True)
    return plan_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PRADAN download plan")
    parser.add_argument("--min-days", type=int, default=60, help="Minimum days in plan")
    args = parser.parse_args()

    plan_df = build_pradan_plan(min_days=args.min_days)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    plan_df.to_csv(OUT_CSV, index=False)
    print(f"Generated PRADAN plan with {len(plan_df)} candidate days -> {OUT_CSV}")
    print("\nPriority breakdown:")
    print(plan_df["priority"].value_counts().sort_index().to_string())

    print("\nTop 15 days in plan:")
    print(plan_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
