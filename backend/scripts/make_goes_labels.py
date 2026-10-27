#!/usr/bin/env python3
"""CLI script to generate GOES-based nowcast and forecast labels.

Usage example:
    python scripts/make_goes_labels.py \
        --events C:/Users/ASHUTOSH/Downloads/data/goes/goes_flares.csv \
        --flux C:/Users/ASHUTOSH/Downloads/data/goes/goes_xrs_1min.csv \
        --start 2026-01-31 \
        --end 2026-02-05 \
        --cadence 1min \
        --pre-flare-min 30 \
        --min-class M \
        --out data/labels/goes_labels_v1.parquet \
        --report backend/reports/goes_labels_report.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# Ensure backend is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
REPO_ROOT = BACKEND_DIR.parent if (BACKEND_DIR / "ml").exists() else BACKEND_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from ml.labels.goes_labels import (
    GoesLabelConfig,
    load_goes_events,
    make_labels,
    split_by_blocks,
    save_labels,
    generate_label_report,
)

logger = logging.getLogger("make_goes_labels")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description="Generate GOES-based solar flare labels")
    p.add_argument(
        "--events",
        default="C:/Users/ASHUTOSH/Downloads/data/goes/goes_flares.csv",
        help="Path to GOES flare event list CSV",
    )
    p.add_argument(
        "--flux",
        default="C:/Users/ASHUTOSH/Downloads/data/goes/goes_xrs_1min.csv",
        help="Path to GOES XRS 1-minute flux CSV (optional, for gap masking)",
    )
    p.add_argument("--start", default="2026-01-31", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default="2026-02-05", help="End date (YYYY-MM-DD)")
    p.add_argument(
        "--cadence",
        default="1min",
        help="Cadence for generated timestamps (e.g. '1s', '10s', '1min'). If 'flux', uses flux timestamps.",
    )
    p.add_argument(
        "--pre-flare-min",
        type=int,
        default=30,
        help="Duration of pre-flare state window in minutes (default 30)",
    )
    p.add_argument(
        "--min-class",
        default="M",
        help="Minimum flare class for binary forecast target y_H (default 'M')",
    )
    p.add_argument(
        "--out",
        default="data/labels/goes_labels_v1.parquet",
        help="Output path for label parquet file",
    )
    p.add_argument(
        "--report",
        default="backend/reports/goes_labels_report.md",
        help="Output path for markdown report",
    )
    p.add_argument(
        "--splits",
        type=int,
        default=3,
        help="Number of chronological day-block splits to audit (default 3)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Resolve paths
    events_path = Path(args.events)
    if not events_path.exists():
        # Try relative to repo
        alt_path = REPO_ROOT / args.events
        if alt_path.exists():
            events_path = alt_path
        else:
            logger.error(f"Events file not found: {events_path}")
            return 1

    flux_df = None
    flux_path = Path(args.flux)
    if flux_path.exists():
        logger.info(f"Loading GOES flux from {flux_path}")
        flux_df = pd.read_csv(flux_path)
    else:
        logger.warning(f"GOES flux file not found at {flux_path}; gap masking from flux will be skipped")

    # 2. Load events
    start_str = f"{args.start} 00:00:00"
    end_str = f"{args.end} 23:59:59"
    events_df = load_goes_events(events_path, start_time=start_str, end_time=end_str)
    logger.info(f"Loaded {len(events_df)} filtered events between {start_str} and {end_str}")

    # 3. Build timestamp grid
    if args.cadence.lower() == "flux" and flux_df is not None and "timestamp" in flux_df.columns:
        ts_grid = pd.to_datetime(flux_df["timestamp"], utc=True)
        cadence_desc = "flux_native"
    else:
        cadence_map = {"1s": "1s", "10s": "10s", "1min": "1min", "1m": "1min", "60s": "1min"}
        freq = cadence_map.get(args.cadence.lower(), args.cadence)
        ts_grid = pd.date_range(start=start_str, end=end_str, freq=freq, tz="UTC")
        cadence_desc = freq

    logger.info(f"Generated {len(ts_grid):,} timestamps at cadence '{cadence_desc}'")

    # 4. Generate labels
    config = GoesLabelConfig(
        preflare_minutes=args.pre_flare_min,
        min_class_preflare="C",
        target_min_class=args.min_class,
        mask_ongoing_min_class="C",
        horizons_hours=(1, 3, 6, 12, 24),
        cadence=cadence_desc,
        extra_metadata={
            "source_events_file": str(events_path.resolve()),
            "source_flux_file": str(flux_path.resolve()) if flux_df is not None else None,
            "cli_args": vars(args),
        },
    )

    labels_df = make_labels(
        timestamps=ts_grid,
        events_df=events_df,
        config=config,
        goes_flux_df=flux_df,
        coverage_end=pd.to_datetime(end_str, utc=True),
    )

    # 5. Chronological splitting audit
    logger.info(f"Auditing chronological splits (n={args.splits})...")
    splits = split_by_blocks(labels_df, n_splits=args.splits)
    for s_name, _, s_stats in splits:
        logger.info(f"  Split: {s_name} | Rows: {s_stats['total_rows']} | "
                    f"Valid: {s_stats['forecast_valid_rows']} | M/X Count: {s_stats['m_x_indicator_count']}")

    # 6. Save outputs
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    saved_data, saved_sidecar = save_labels(labels_df, out_path, metadata=config.extra_metadata)

    # Also mirror into backend/data/labels if saved in data/labels
    if "backend" not in out_path.parts:
        backend_mirror = BACKEND_DIR / "data" / "labels" / out_path.name
        save_labels(labels_df, backend_mirror, metadata=config.extra_metadata)

    # 7. Generate markdown report
    report_content = generate_label_report(labels_df, events_df, config)
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding="utf-8")
    logger.info(f"Saved markdown report to {report_path}")

    # Print report to stdout
    print("\n" + "=" * 80)
    print(report_content)
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
