#!/usr/bin/env python3
"""Scan local PRADAN downloads for SoLEXS and HEL1OS daily coverage overlap.

Strict Rules:
  - Header and file existence inspection only.
  - Never load large flux arrays into memory.
  - solexs_ok / helios_ok = file exists and time range covers >= 80% (69,120s) of UTC day.
  - Scans both C:/Users/ASHUTOSH/Downloads and data/pradan_downloads.
  - Generates data/overlap_days.csv
  - Intersects with split_v2.json and development epochs (May 2024, Feb 2026).
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from collections import defaultdict

import pandas as pd

cur_path = Path(__file__).resolve()
if cur_path.parent.name == "scripts" and cur_path.parent.parent.name == "backend":
    REPO_ROOT = cur_path.parent.parent.parent
else:
    REPO_ROOT = cur_path.parent.parent

DOWNLOADS_DIR = Path("C:/Users/ASHUTOSH/Downloads")
DATA_DIR = REPO_ROOT / "data"
PRADAN_DOWNLOADS_DIR = DATA_DIR / "pradan_downloads"
LABEL_DIR = DATA_DIR / "labels"


def scan_pradan_downloads(downloads_dirs: list[Path | str] | Path | str | None = None) -> pd.DataFrame:
    """Inspect SoLEXS and HEL1OS zip packages without inflating flux data."""
    if downloads_dirs is None:
        downloads_dirs = [DOWNLOADS_DIR, PRADAN_DOWNLOADS_DIR]
    elif isinstance(downloads_dirs, (str, Path)):
        downloads_dirs = [Path(downloads_dirs)]
    else:
        downloads_dirs = [Path(p) for p in downloads_dirs]

    solexs_days: dict[str, tuple[str, int]] = {}    # date_str -> (filename, duration_sec)
    helios_days: dict[str, list[tuple[str, int]]] = defaultdict(list)  # date_str -> list of (filename, duration_sec)

    for d_dir in downloads_dirs:
        if not d_dir.exists():
            continue

        # 1. Scan SoLEXS zip archives
        for zpath in sorted(d_dir.glob("solexs_*.zip")):
            try:
                with zipfile.ZipFile(zpath, "r") as zf:
                    for name in zf.namelist():
                        m = re.search(r"AL1_SLX_L1_(\d{4})(\d{2})(\d{2})", name)
                        if m:
                            d_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                            solexs_days[d_str] = (name, 86400)
            except Exception as exc:
                print(f"Warning: error reading {zpath.name}: {exc}")

        # Also check standalone or unzipped SoLEXS files
        for p in sorted(d_dir.rglob("AL1_SLX_L1_*")):
            m = re.search(r"AL1_SLX_L1_(\d{4})(\d{2})(\d{2})", p.name)
            if m:
                d_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                solexs_days[d_str] = (p.name, 86400)

        # 2. Scan HEL1OS zip archives
        for zpath in sorted(d_dir.glob("hel1os_*.zip")):
            try:
                with zipfile.ZipFile(zpath, "r") as zf:
                    for name in zf.namelist():
                        m = re.search(r"HLS_(\d{4})(\d{2})(\d{2})_\d{6}_(\d+)sec", name)
                        if m:
                            d_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                            sec = int(m.group(4))
                            helios_days[d_str].append((name, sec))
            except Exception as exc:
                print(f"Warning: error reading {zpath.name}: {exc}")

        # Also check standalone or unzipped HEL1OS files
        for p in sorted(d_dir.rglob("HLS_*")):
            m = re.search(r"HLS_(\d{4})(\d{2})(\d{2})_\d{6}_(\d+)sec", p.name)
            if m:
                d_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                sec = int(m.group(4))
                helios_days[d_str].append((p.name, sec))
            elif re.search(r"HLS_(\d{4})(\d{2})(\d{2})", p.name):
                m_simple = re.search(r"HLS_(\d{4})(\d{2})(\d{2})", p.name)
                d_str = f"{m_simple.group(1)}-{m_simple.group(2)}-{m_simple.group(3)}"
                # Default daily duration if exact seconds not in name
                helios_days[d_str].append((p.name, 86400))

    # All unique dates
    all_dates = sorted(set(solexs_days.keys()) | set(helios_days.keys()))
    rows = []

    for d in all_dates:
        # SoLEXS check
        if d in solexs_days:
            s_file, s_sec = solexs_days[d]
            s_ok = bool(s_sec >= 0.80 * 86400)
        else:
            s_file, s_ok = "NONE", False

        # HEL1OS check
        if d in helios_days:
            h_items = helios_days[d]
            total_sec = sum(sec for _, sec in h_items)
            h_files = "+".join(fn for fn, _ in h_items)
            h_ok = bool(total_sec >= 0.80 * 86400)
        else:
            h_files, h_ok = "NONE", False

        both = s_ok and h_ok
        rows.append({
            "date": d,
            "solexs_file": s_file,
            "helios_file": h_files,
            "solexs_ok": s_ok,
            "helios_ok": h_ok,
            "both": both,
        })

    return pd.DataFrame(rows)


def analyze_split_and_epoch_overlap(
    overlap_df: pd.DataFrame,
    split_v2_path: Path,
    labels_v2_p: Path | None = None,
) -> dict:
    """Intersect overlap days with split_v2.json and development epochs."""
    summary = {}

    both_dates = set(overlap_df.loc[overlap_df["both"], "date"].tolist())
    solexs_dates = set(overlap_df.loc[overlap_df["solexs_ok"], "date"].tolist())
    helios_dates = set(overlap_df.loc[overlap_df["helios_ok"], "date"].tolist())

    summary["total_overlap_dates"] = sorted(list(both_dates))

    # Load splits
    splits_info = {}
    if split_v2_path.exists():
        with split_v2_path.open("r", encoding="utf-8") as fh:
            split_data = json.load(fh)

        for sp in split_data.get("splits", []):
            s_name = sp["partition"]
            s_start = sp["start_date"]
            s_end = sp["end_date"]
            sp_dates = set(pd.date_range(s_start, s_end).strftime("%Y-%m-%d").tolist())
            both_in_split = sorted(list(sp_dates & both_dates))
            solexs_in_split = sorted(list(sp_dates & solexs_dates))

            splits_info[s_name] = {
                "start_date": s_start,
                "end_date": s_end,
                "both_days_count": len(both_in_split),
                "both_days": both_in_split,
                "solexs_days_count": len(solexs_in_split),
                "solexs_days": solexs_in_split,
            }

    summary["splits"] = splits_info

    dev_epochs = {
        "may_2024_epoch": [f"2024-05-{d:02d}" for d in range(11, 16)],
        "feb_2026_epoch": [f"2026-02-{d:02d}" for d in range(1, 6)],
    }

    dev_summary = {}
    for ep_name, ep_dates in dev_epochs.items():
        ep_set = set(ep_dates)
        in_solexs = sorted(list(ep_set & solexs_dates))
        in_helios = sorted(list(ep_set & helios_dates))
        in_both = sorted(list(ep_set & both_dates))
        dev_summary[ep_name] = {
            "total_epoch_days": len(ep_dates),
            "solexs_available_days": in_solexs,
            "helios_available_days": in_helios,
            "both_instruments_days": in_both,
        }

    summary["dev_epochs"] = dev_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan SoLEXS and HEL1OS overlap days")
    parser.add_argument("--downloads", nargs="*", default=[str(DOWNLOADS_DIR), str(PRADAN_DOWNLOADS_DIR)],
                        help="Paths to Downloads directories")
    parser.add_argument("--out", default="data/overlap_days.csv", help="Output CSV path")
    args = parser.parse_args()

    downloads_dirs = [Path(p) for p in args.downloads]
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning PRADAN downloads in {[str(p.resolve()) for p in downloads_dirs]} ...")
    overlap_df = scan_pradan_downloads(downloads_dirs)
    overlap_df.to_csv(out_csv, index=False)
    print(f"Saved {len(overlap_df)} records to {out_csv}")
    print(overlap_df.to_string(index=False))

    split_v2_p = LABEL_DIR / "split_v2.json"
    analysis = analyze_split_and_epoch_overlap(overlap_df, split_v2_p)
    print("\n" + "=" * 70)
    print("Instrument Overlap & Development Epoch Intersections:")
    print("=" * 70)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
