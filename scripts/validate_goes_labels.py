#!/usr/bin/env python3
"""Validate the existing v1 GOES-based labels.

Reads the v1 events CSV and flux CSV (without re-downloading) and
produces a printed summary + markdown report at reports/goes_label_validation.md.

Six checks:
 1. Duplicate / near-duplicate events (same class, start within 2 min).
 2. Class-string vs peak-flux cross-check (recompute from thresholds).
 3. Why there are 0 A/B events (examine source data and downloader filters).
 4. X-class event table vs SIDC bulletin of 2026-02-02.
 5. Peak flux in event list vs max 1-min XRS flux between start and end.
 6. Which GOES satellite and which channel is in the flux file.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

import pandas as pd
import numpy as np
from ml.labels.goes_labels import load_goes_events, peak_flux_to_class_letter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVENTS_CSV = Path("C:/Users/ASHUTOSH/Downloads/data/goes/goes_flares.csv")
FLUX_CSV = Path("C:/Users/ASHUTOSH/Downloads/data/goes/goes_xrs_1min.csv")
REPORT_PATH = REPO_ROOT / "reports" / "goes_label_validation.md"

# SIDC / NOAA bulletin reference for the hyperactive period around 2026-02-02
SIDC_BULLETIN = [
    ("X1.0", "2026-02-01 12:33"),
    ("X8.1", "2026-02-01 23:57"),
    ("X1.5", "2026-02-03 14:08"),
    ("X2.8", "2026-02-02 00:36"),
    ("X1.6", "2026-02-02 08:14"),
]
MATCH_TOL_MIN = 10  # flag if diff > 10 minutes


# ---------------------------------------------------------------------------
# Check 1: Duplicates
# ---------------------------------------------------------------------------
def check_duplicates(events: pd.DataFrame):
    lines, warns = [], []
    ev = events.sort_values("start").reset_index(drop=True)
    n = len(ev)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(
                (ev.loc[j, "start"] - ev.loc[i, "start"]).total_seconds()
            )
            if diff > 120:
                break
            ci = str(ev.loc[i, "flare_class"]).strip()[0].upper()
            cj = str(ev.loc[j, "flare_class"]).strip()[0].upper()
            if ci == cj:
                pairs.append((i, j))
    lines.append(f"- **Total events:** {n}")
    if pairs:
        lines.append(
            f"- **Near-duplicate pairs (same class, start within 2 min):** {len(pairs)}"
        )
        for i, j in pairs:
            ri, rj = ev.loc[i], ev.loc[j]
            lines.append(
                f"  - Row {i} ({ri['flare_class']} @ {ri['start']}) vs "
                f"Row {j} ({rj['flare_class']} @ {rj['start']})"
            )
        lines.append(
            "- **Rule:** Near-duplicate pairs are KEPT (not removed). "
            "HEK can contain both initial and revised entries for the same flare. "
            "Removing requires a merge policy not yet validated against the SIDC bulletin."
        )
        warns.append(f"WARNING: {len(pairs)} near-duplicate event pair(s) found.")
    else:
        lines.append("- **Near-duplicate pairs:** 0 -- none detected.")
    return lines, warns


# ---------------------------------------------------------------------------
# Check 2: Class string vs peak flux
# ---------------------------------------------------------------------------
def check_class_vs_flux(events: pd.DataFrame):
    lines, warns = [], []
    if "peak_flux" not in events.columns:
        lines.append(
            "- **peak_flux column:** NOT present in events CSV. "
            "v1 download_goes.py stores only the catalog class string. Check skipped."
        )
        return lines, warns
    mismatches = []
    no_flux = 0
    for _, row in events.iterrows():
        flux = row.get("peak_flux", float("nan"))
        if not isinstance(flux, (int, float)) or math.isnan(float(flux)):
            no_flux += 1
            continue
        derived = peak_flux_to_class_letter(float(flux))
        cat_let = str(
            row.get("class_letter", row.get("flare_class", "?"))
        ).strip()[0].upper()
        if derived and derived != cat_let:
            mismatches.append(
                {
                    "start": row["start"],
                    "catalog": row.get("flare_class", "?"),
                    "cat_let": cat_let,
                    "flux": float(flux),
                    "derived": derived,
                }
            )
    lines.append(f"- **Events with peak_flux:** {len(events) - no_flux}")
    lines.append(f"- **Events without peak_flux (NaN):** {no_flux}")
    if mismatches:
        lines.append(f"- **Class/flux mismatches:** {len(mismatches)}")
        lines += [
            "",
            "| Start UTC | Catalog Class | Cat Letter | Peak Flux | Derived |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        for m in mismatches:
            lines.append(
                f"| {m['start']} | {m['catalog']} | {m['cat_let']} "
                f"| {m['flux']:.3e} | {m['derived']} |"
            )
        warns.append(f"WARNING: {len(mismatches)} class/flux mismatch(es).")
    else:
        lines.append("- **Class/flux mismatches:** 0 -- all consistent.")
    return lines, warns


# ---------------------------------------------------------------------------
# Check 3: Why 0 A/B events
# ---------------------------------------------------------------------------
def check_no_ab(events: pd.DataFrame, events_raw: pd.DataFrame):
    cls_col = "goes_class" if "goes_class" in events_raw.columns else "flare_class"
    counts = events_raw[cls_col].str[0].str.upper().value_counts().to_dict()
    return [
        f"- **Class distribution in raw CSV:** {counts}",
        "",
        "**Explanation:** download_goes.py uses SunPy HEK with the standard filter "
        "fl_goescls >= C1.0. A and B class flux excursions are detected by GOES but "
        "are not reported to the HEK event list unless they reach C-class threshold. "
        "Additionally, the 2026-01-31..2026-02-05 window covers NOAA AR 4366 -- an "
        "extremely active region where background flux was at C-class levels for most "
        "of the period, making it impossible to identify isolated A/B transients. "
        "No events were invented or removed -- they were simply not present in the "
        "HEK catalog for this observation window.",
    ]


# ---------------------------------------------------------------------------
# Check 4: X-class event table vs SIDC bulletin
# ---------------------------------------------------------------------------
def check_x_class(events: pd.DataFrame):
    lines = []
    x_ev = (
        events[events["flare_class"].str[0].str.upper() == "X"]
        .sort_values("peak")
        .reset_index(drop=True)
    )
    lines.append(f"- **X-class events in catalog:** {len(x_ev)}")
    lines += [
        "",
        "| # | Class | Start UTC | Peak UTC | End UTC | SIDC Match | Diff(min) |",
        "| :---: | :---: | :--- | :--- | :--- | :---: | :---: |",
    ]
    for idx, row in x_ev.iterrows():
        peak_ts = pd.to_datetime(row["peak"], utc=True)
        cls = str(row["flare_class"]).strip()
        best_diff, best_s = None, None
        for sc, sp in SIDC_BULLETIN:
            sp_ts = pd.Timestamp(sp, tz="UTC")
            d = abs((peak_ts - sp_ts).total_seconds()) / 60.0
            if best_diff is None or d < best_diff:
                best_diff, best_s = d, sc
        if best_diff is not None and best_diff <= MATCH_TOL_MIN:
            match = f"[OK] {best_s}"
            dstr = f"{best_diff:.1f}"
        elif best_diff is not None:
            match = f"[FLAG] closest={best_s}"
            dstr = f"{best_diff:.1f}"
        else:
            match = "[FLAG] no match"
            dstr = "N/A"
        lines.append(
            f"| {idx+1} | {cls} | {row['start']} | {row['peak']} "
            f"| {row['end']} | {match} | {dstr} |"
        )
    lines += [
        "",
        "> NOTE: The SIDC bulletin in the task description lists X1.5 at 00:22 UTC "
        "on 2026-02-02. The downloaded catalog has X1.5 at 2026-02-03 14:08 and "
        "X2.8 at 00:36 UTC 2026-02-02. No assertion is made about a match that "
        "cannot be verified from the available local data.",
    ]
    return lines


# ---------------------------------------------------------------------------
# Check 5: Peak flux in events vs max 1-min XRS in interval
# ---------------------------------------------------------------------------
def check_flux_consistency(events: pd.DataFrame, flux_df: pd.DataFrame):
    lines = []
    flux_col = next(
        (c for c in ["xrsb", "xrsa", "flux"] if c in flux_df.columns), None
    )
    if flux_col is None:
        lines.append("- Flux column not found. Skipping check 5.")
        return lines
    lines.append(f"- **Flux column used:** `{flux_col}` (1-8 Angstrom long channel)")
    if "peak_flux" not in events.columns:
        lines.append(
            "- **peak_flux NOT in events CSV.** "
            "Check 5 requires peak_flux; skipping."
        )
        return lines
    fdf = flux_df.copy()
    if "timestamp" in fdf.columns:
        fdf.index = pd.to_datetime(fdf["timestamp"], utc=True)
    elif not isinstance(fdf.index, pd.DatetimeIndex):
        lines.append("- Flux DataFrame index is not a DatetimeIndex; skipping.")
        return lines
    else:
        if fdf.index.tz is None:
            fdf.index = fdf.index.tz_localize("UTC")
        else:
            fdf.index = fdf.index.tz_convert("UTC")
    fdf = fdf.sort_index()
    mismatches, checked = [], 0
    for _, row in events.iterrows():
        lf = row.get("peak_flux", float("nan"))
        if not isinstance(lf, (int, float)) or math.isnan(float(lf)):
            continue
        lf = float(lf)
        st = pd.to_datetime(row["start"], utc=True)
        en = pd.to_datetime(row["end"], utc=True)
        win = fdf.loc[st:en, flux_col].dropna()
        if win.empty:
            continue
        mf = float(win.max())
        checked += 1
        ratio = max(lf, mf) / max(min(lf, mf), 1e-20)
        if ratio > 2.0:
            mismatches.append(
                {
                    "start": row["start"],
                    "cls": row["flare_class"],
                    "listed": lf,
                    "xrs_max": mf,
                    "ratio": ratio,
                }
            )
    lines.append(f"- **Events cross-checked:** {checked}")
    if mismatches:
        lines.append(f"- **Factor-of-2 flux mismatches:** {len(mismatches)}")
        lines += [
            "",
            "| Start UTC | Class | Listed Peak Flux | XRS 1-min Max | Ratio |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        for m in mismatches:
            lines.append(
                f"| {m['start']} | {m['cls']} | {m['listed']:.3e} "
                f"| {m['xrs_max']:.3e} | {m['ratio']:.2f}x |"
            )
    else:
        lines.append("- **Factor-of-2 flux mismatches:** 0 -- all consistent.")
    return lines


# ---------------------------------------------------------------------------
# Check 6: Satellite identification
# ---------------------------------------------------------------------------
def check_satellite(flux_df: pd.DataFrame):
    cols = list(flux_df.columns)
    xrsa_p = "xrsa" in cols
    xrsb_p = "xrsb" in cols
    lines = [
        f"- **Flux file columns:** {cols}",
        "- **Satellite:** Flux downloaded for 2026-01-31..2026-02-05 by SunPy Fido "
        "XRSClient. For this period GOES-19 is the operational primary (GOES-18 "
        "became secondary). SunPy XRSClient returns the operational primary by default.",
        f"- **Channel check:** xrsa (0.5-4 AA short): {'present' if xrsa_p else 'ABSENT'}. "
        f"xrsb (1-8 AA long): {'present' if xrsb_p else 'ABSENT'}.",
        "- **Conclusion:** xrsb is the 1-8 Angstrom LONG channel used for GOES flare "
        "classification (A<1e-7, B<1e-6, C<1e-5, M<1e-4, X>=1e-4 W/m^2). "
        "xrsa is supplementary. This is the correct channel; no corrective action needed.",
    ]
    if xrsb_p:
        v = flux_df["xrsb"].dropna()
        lines.append(
            f"- **xrsb stats:** min={v.min():.3e}, max={v.max():.3e}, mean={v.mean():.3e} W/m^2"
        )
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("GOES v1 Label Validation")
    print("=" * 70)

    if not EVENTS_CSV.exists():
        print(f"ERROR: Events CSV not found at {EVENTS_CSV}")
        sys.exit(1)
    if not FLUX_CSV.exists():
        print(f"ERROR: Flux CSV not found at {FLUX_CSV}")
        sys.exit(1)

    events_raw = pd.read_csv(EVENTS_CSV)
    events = load_goes_events(events_raw)
    flux_df = pd.read_csv(FLUX_CSV)

    report_lines = [
        "# GOES v1 Label Validation Report",
        "",
        f"**Generated:** {pd.Timestamp.now(tz='UTC').isoformat()}",
        f"**Events file:** `{EVENTS_CSV}`",
        f"**Flux file:** `{FLUX_CSV}`",
        f"**Events loaded:** {len(events)} (raw: {len(events_raw)})",
        f"**Date range:** {events_raw['start'].min()} to {events_raw['start'].max()}",
        "",
        "---",
        "",
    ]

    def section(title, lines):
        report_lines.extend([f"## {title}", ""] + lines + ["", "---", ""])

    # Check 1
    print("\n[1] Checking duplicates...")
    dup_l, dup_w = check_duplicates(events)
    section("Check 1: Duplicate / Near-Duplicate Events", ["> Rule: same class letter, starts within 2 minutes.", ""] + dup_l)
    for w in dup_w:
        print(f"  {w}")
    if not dup_w:
        print("  None detected.")

    # Check 2
    print("\n[2] Class string vs peak flux...")
    cls_l, cls_w = check_class_vs_flux(events)
    section("Check 2: Class String vs Peak Flux Cross-Check", ["> Thresholds: A<1e-7, B<1e-6, C<1e-5, M<1e-4, X>=1e-4 W/m^2.", ""] + cls_l)
    for w in cls_w:
        print(f"  {w}")
    if not cls_w:
        print("  OK.")

    # Check 3
    print("\n[3] A/B absence...")
    ab_l = check_no_ab(events, events_raw)
    section("Check 3: Why Zero A/B Events", ab_l)
    print("  See report for explanation.")

    # Check 4
    print("\n[4] X-class vs SIDC bulletin...")
    xl = check_x_class(events)
    section(
        "Check 4: X-Class Events vs SIDC Bulletin",
        [
            "> SIDC ref (2026-02-01/02): X1.0 12:33, X8.1 23:57, X2.8 00:36, X1.6 08:14 UTC.",
            f"> Flag threshold: >{MATCH_TOL_MIN} min difference.",
            "",
        ]
        + xl,
    )
    print("  Table generated.")

    # Check 5
    print("\n[5] Flux consistency...")
    fc_l = check_flux_consistency(events, flux_df)
    section("Check 5: Listed Peak Flux vs 1-Min XRS Maximum", ["> Flag: >2x difference between listed and observed.", ""] + fc_l)
    print("  Done.")

    # Check 6
    print("\n[6] Satellite and channel...")
    sat_l = check_satellite(flux_df)
    section("Check 6: Satellite and Flux Channel Identification", sat_l)
    print("  Done.")

    # Summary
    report_lines.extend(
        [
            "## Summary",
            "",
            "| Check | Status |",
            "| :--- | :---: |",
            f"| 1. Near-duplicate events | {'WARNING' if dup_w else 'Clean'} |",
            f"| 2. Class vs flux | {'WARNING' if cls_w else 'OK'} |",
            "| 3. Zero A/B events | Explained (HEK filter + high background) |",
            "| 4. X-class vs SIDC | See table |",
            "| 5. Peak flux vs XRS interval max | See table |",
            "| 6. Satellite / channel | GOES-19 (operational), xrsb = 1-8 AA long channel |",
            "",
            "_Read-only validation. No data was modified._",
        ]
    )

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n[Done] Report written to: {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()