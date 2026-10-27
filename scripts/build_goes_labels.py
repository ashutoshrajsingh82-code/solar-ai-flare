#!/usr/bin/env python3
"""Build multi-month GOES labels from monthly chunks and date-range config.

Produces:
  - data/labels/goes_labels_v2.parquet + .json sidecar
  - data/labels/split_v2.json (chronological ~70/15/15 with >= 24h purge gap)
  - reports/goes_labels_v2_split_plan.json

Rules:
  - October 2024 is quarantined and strictly excluded.
  - v1 labels remain untouched.
  - No interpolation or imputation across gaps.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np
import pandas as pd
import yaml

from ml.labels.goes_labels import (
    GoesLabelConfig,
    load_goes_events,
    make_labels,
    split_by_day_blocks,
    split_by_blocks,
    save_labels,
    generate_label_report,
)

HOLDOUT_START = pd.Timestamp("2024-10-01", tz="UTC")
HOLDOUT_END = pd.Timestamp("2024-10-31 23:59:59", tz="UTC")

GOES_DATA_ROOT = REPO_ROOT / "data" / "goes"
DOWNLOADS_GOES_ROOT = Path("C:/Users/ASHUTOSH/Downloads/data/goes")
CONFIG_PATH = REPO_ROOT / "configs" / "goes_label_ranges.yaml"
LABEL_OUT_DIR = REPO_ROOT / "data" / "labels"
REPORT_DIR = REPO_ROOT / "reports"

logger = logging.getLogger("build_goes_labels")


def assert_no_holdout_timestamps(index: pd.DatetimeIndex) -> None:
    """Raise fatal ValueError if ANY timestamp falls inside sealed October 2024 holdout."""
    idx = pd.to_datetime(index, utc=True)
    in_holdout = (idx >= HOLDOUT_START) & (idx <= HOLDOUT_END)
    if in_holdout.any():
        bad_count = int(in_holdout.sum())
        raise ValueError(
            f"HOLDOUT CONTAMINATION: Found {bad_count} timestamps inside sealed "
            f"October 2024 holdout [{HOLDOUT_START.date()}..{HOLDOUT_END.date()}]. Aborting."
        )


def load_all_monthly_chunks(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], list[str]]:
    """Scan data/goes for monthly chunks YYYY-MM and combine flux and events."""
    all_flux = []
    all_events = []
    satellites_by_month = {}
    source_files = []

    # Check coverage.csv if present
    cov_path = data_dir / "coverage.csv"
    if cov_path.exists():
        cov_df = pd.read_csv(cov_path)
        for _, r in cov_df.iterrows():
            satellites_by_month[str(r["month"])] = str(r["satellite"])

    # Find month directories YYYY-MM
    month_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and len(d.name) == 7 and d.name[4] == "-"])

    for md in month_dirs:
        m_str = md.name
        # Verify holdout protection
        if m_str == "2024-10":
            raise ValueError("HOLDOUT VIOLATION: October 2024 directory encountered in data/goes! Sealed holdout must never be touched.")

        flux_f = md / "goes_xrs_1min.csv"
        flares_f = md / "goes_flares.csv"

        if flux_f.exists() and flares_f.exists():
            df_flx = pd.read_csv(flux_f, parse_dates=["timestamp"], index_col="timestamp")
            df_flx.index = pd.to_datetime(df_flx.index, utc=True)
            if not df_flx.empty:
                all_flux.append(df_flx)
                source_files.append(str(flux_f))

            df_ev = pd.read_csv(flares_f)
            if not df_ev.empty:
                all_events.append(df_ev)
                source_files.append(str(flares_f))

    # If no monthly directories found, check DOWNLOADS_GOES_ROOT
    if not all_flux and DOWNLOADS_GOES_ROOT.exists():
        flx_p = DOWNLOADS_GOES_ROOT / "goes_xrs_1min.csv"
        ev_p = DOWNLOADS_GOES_ROOT / "goes_flares.csv"
        if flx_p.exists() and ev_p.exists():
            df_flx = pd.read_csv(flx_p, parse_dates=["timestamp"], index_col="timestamp")
            df_flx.index = pd.to_datetime(df_flx.index, utc=True)
            all_flux.append(df_flx)
            all_events.append(pd.read_csv(ev_p))
            source_files.extend([str(flx_p), str(ev_p)])

    if not all_flux or not all_events:
        raise RuntimeError("No GOES flux or event data files found to process.")

    combined_flux = pd.concat(all_flux).sort_index()
    combined_flux = combined_flux[~combined_flux.index.duplicated(keep="first")]

    combined_events = pd.concat(all_events).drop_duplicates(subset=["start", "goes_class"]).sort_values("start").reset_index(drop=True)

    return combined_flux, combined_events, satellites_by_month, source_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GOES labels")
    parser.add_argument("--version", default="v2", help="Label version tag (v2)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    parser.add_argument("--purge-gap-days", type=int, default=1, help="Purge gap in calendar days (default: 1 day >= 24h)")
    args = parser.parse_args()

    # Load configuration
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        cfg_yaml = yaml.safe_load(fh)

    label_params = cfg_yaml.get("label_generation_params", {})
    quarantined = cfg_yaml.get("quarantined_holdout", {})

    print("=" * 70)
    print(f"Build GOES Labels  version={args.version}")
    print(f"Quarantined holdout: {quarantined.get('start_date')} .. {quarantined.get('end_date')}")
    print(f"Purge gap between splits: {args.purge_gap_days} day(s) (>= 24h)")
    print("=" * 70)

    # Construct label config
    lbl_cfg = GoesLabelConfig(
        preflare_minutes=label_params.get("preflare_minutes", 30),
        min_class_preflare=label_params.get("min_class_preflare", "C"),
        target_min_class=label_params.get("target_min_class", "M"),
        mask_ongoing_min_class=label_params.get("mask_ongoing_min_class", "C"),
        horizons_hours=tuple(label_params.get("horizons_hours", [1, 3, 6, 12, 24])),
        max_gap_minutes=float(label_params.get("max_gap_minutes", 10.0)),
        cadence=label_params.get("cadence", "1min"),
    )

    LABEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load combined data across all monthly chunks
    flux_df, raw_events_df, sats_by_month, source_files = load_all_monthly_chunks(GOES_DATA_ROOT)

    # Re-verify no holdout timestamps in flux or events
    assert_no_holdout_timestamps(flux_df.index)
    if "start" in raw_events_df.columns:
        assert_no_holdout_timestamps(raw_events_df["start"])

    print(f"Loaded {len(flux_df):,} flux rows ({flux_df.index.min()} to {flux_df.index.max()})")
    print(f"Loaded {len(raw_events_df)} flare events")
    print(f"Satellites by month: {sats_by_month}")

    # Standardize events
    events_df = raw_events_df.copy()
    for col in ("start", "peak", "end"):
        if col in events_df.columns:
            events_df[col] = pd.to_datetime(events_df[col], utc=True, format="ISO8601")

    if "goes_class" in events_df.columns and "class_letter" not in events_df.columns:
        events_df["class_letter"] = events_df["goes_class"].astype(str).str[0].str.upper()

    if "flare_class" not in events_df.columns:
        if "goes_class" in events_df.columns:
            events_df["flare_class"] = events_df["goes_class"]
        elif "class" in events_df.columns:
            events_df["flare_class"] = events_df["class"]

    if args.dry_run:
        print("[DRY-RUN] Complete. No output written.")
        return

    # Generate labels on the flux timestamp grid
    timestamps = flux_df.index
    coverage_end = flux_df.index.max()

    print(f"\nGenerating {args.version} labels ...")
    labels_df = make_labels(
        timestamps=timestamps,
        events_df=events_df,
        config=lbl_cfg,
        goes_flux_df=flux_df,
        coverage_end=coverage_end,
    )

    # Attach satellite column from flux if present
    if "satellite" in flux_df.columns:
        labels_df["satellite"] = flux_df["satellite"]

    # Generate 70/15/15 chronological split with purge gap
    print("\nComputing chronological splits with purge gap ...")
    splits = split_by_day_blocks(
        labels_df,
        ratios=(0.70, 0.15, 0.15),
        purge_gap_days=args.purge_gap_days,
        split_names=("train", "val", "test"),
    )

    split_plan_list = []
    for split_id, sub_df, stats in splits:
        split_plan_list.append(stats)
        mx_1h = stats.get("y_1h_pos_count", 0)
        rate_1h = stats.get("y_1h_pos_rate", 0.0)
        warning_tag = " [UNSTABLE < 20 M/X]" if mx_1h < 20 else ""
        print(f"  [{stats['partition'].upper()}] {stats['start_date']} .. {stats['end_date']} | "
              f"Days: {stats['days_count']} | Rows: {stats['total_rows']:,} | "
              f"M/X (1h): {mx_1h} ({rate_1h:.1%}){warning_tag}")

    # Save data/labels/split_v2.json
    split_v2_path = LABEL_OUT_DIR / f"split_{args.version}.json"
    split_payload = {
        "version": args.version,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "holdout_quarantined": f"{HOLDOUT_START.date()}..{HOLDOUT_END.date()}",
        "purge_gap_days": args.purge_gap_days,
        "splits": split_plan_list,
    }
    split_v2_path.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
    print(f"\nSaved split plan to {split_v2_path}")

    # Save Parquet + JSON sidecar
    out_parquet = LABEL_OUT_DIR / f"goes_labels_{args.version}.parquet"
    metadata = {
        "version": args.version,
        "satellites_by_month": sats_by_month,
        "source_files": source_files,
        "date_ranges": {
            "start": str(flux_df.index.min()),
            "end": str(flux_df.index.max()),
        },
        "purge_gap_days": args.purge_gap_days,
        "split_summary": split_plan_list,
    }
    saved_parquet, saved_sidecar = save_labels(labels_df, out_parquet, metadata=metadata)
    print(f"Saved label parquet: {saved_parquet}")
    print(f"Saved sidecar json:   {saved_sidecar}")

    print("\n[Done] Label build complete.")


if __name__ == "__main__":
    main()
