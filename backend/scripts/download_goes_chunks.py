#!/usr/bin/env python3
"""Download GOES XRS 1-minute flux and HEK flare event lists in monthly chunks.

Strict Rules:
  - October 2024 (2024-10-01 .. 2024-10-31) is quarantined. Attempting to query
    or process this range raises a fatal ValueError.
  - Satellite identity is extracted directly from the NetCDF `platform` attribute.
  - Never fabricate or interpolate flux values across missing data gaps.
  - Reuses files already on disk.
  - Generates data/goes/coverage.csv and audits satellite transitions.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from ml.labels.goes_labels import peak_flux_to_class_letter

HOLDOUT_START = pd.Timestamp("2024-10-01", tz="UTC")
HOLDOUT_END = pd.Timestamp("2024-10-31 23:59:59", tz="UTC")


def assert_no_holdout(start: pd.Timestamp, end: pd.Timestamp) -> None:
    s = pd.to_datetime(start, utc=True)
    e = pd.to_datetime(end, utc=True)
    if s <= HOLDOUT_END and e >= HOLDOUT_START:
        raise ValueError(
            f"FATAL HOLDOUT GUARD: Range [{s.date()} .. {e.date()}] overlaps "
            f"quarantined October 2024 holdout [{HOLDOUT_START.date()} .. {HOLDOUT_END.date()}]."
        )


def get_netcdf_satellite(nc_path: Path) -> str:
    """Read satellite platform ID from NetCDF file attributes."""
    try:
        ds = netCDF4.Dataset(nc_path)
        plat = getattr(ds, "platform", None)
        if plat is None:
            plat = getattr(ds, "platform_ID", None)
        if plat is None:
            plat = getattr(ds, "satellite_id", None)
        ds.close()
        if plat:
            plat_str = str(plat).strip()
            if plat_str.lower().startswith("g"):
                return f"GOES-{plat_str[1:]}"
            return plat_str
    except Exception as exc:
        print(f"  Warning: could not read NetCDF attribute from {nc_path.name}: {exc}")
    return "UNKNOWN"


def fetch_month_flux(start_str: str, end_str: str, raw_dir: Path) -> tuple[pd.DataFrame, str]:
    """Fetch 1-min XRS flux files for month and return DataFrame + satellite platform."""
    from sunpy.net import Fido, attrs as a
    from sunpy.timeseries import TimeSeries
    from sunpy.util.datatype_factory_base import NoMatchError

    raw_dir.mkdir(parents=True, exist_ok=True)
    sats = (16, 18, 19)

    for sat in sats:
        print(f"  [flux] Searching GOES-{sat} for {start_str} .. {end_str} ...")
        try:
            res = Fido.search(
                a.Time(start_str, end_str),
                a.Instrument("XRS"),
                a.goes.SatelliteNumber(sat),
                a.Resolution("avg1m"),
            )
        except (NoMatchError, Exception) as exc:
            print(f"  [flux] GOES-{sat} search skipped: {exc}")
            continue

        if res.file_num == 0:
            print(f"  [flux] No files for GOES-{sat}")
            continue

        print(f"  [flux] Found {res.file_num} files for GOES-{sat}. Fetching ...")
        fetched = Fido.fetch(res, path=str(raw_dir / "{file}"))
        if not len(fetched):
            continue

        # Extract satellite platform directly from the first valid NetCDF
        sat_id = f"GOES-{sat}"
        for f in sorted(raw_dir.glob("*.nc")):
            p_id = get_netcdf_satellite(f)
            if p_id != "UNKNOWN":
                sat_id = p_id
                break

        print(f"  [flux] Ingesting {len(fetched)} NetCDF files into TimeSeries ...")
        nc_files = sorted(list(raw_dir.glob("*.nc")))
        ts = TimeSeries(nc_files, concatenate=True)
        df = ts.to_dataframe()
        df.index.name = "timestamp"
        keep = [c for c in ("xrsa", "xrsb") if c in df.columns]
        df = df[keep].sort_index()
        df = df[~df.index.duplicated(keep="first")]
        df["satellite"] = sat_id
        return df, sat_id

    raise RuntimeError(f"No GOES flux data could be fetched for {start_str} .. {end_str}")


def fetch_month_flares(start_str: str, end_str: str) -> pd.DataFrame:
    """Query HEK flare catalog for events in range."""
    from sunpy.net import Fido, attrs as a

    print(f"  [flares] Querying HEK for {start_str} .. {end_str} ...")
    res = Fido.search(
        a.Time(start_str, end_str),
        a.hek.EventType("FL"),
        a.hek.FL.GOESCls > "B1.0",
        a.hek.OBS.Observatory == "GOES",
    )
    if "hek" not in res.keys() or len(res["hek"]) == 0:
        print(f"  [flares] No HEK events found.")
        return pd.DataFrame(columns=["start", "peak", "end", "goes_class", "noaa_ar", "is_MX"])

    tbl = res["hek"]
    wanted = {
        "event_starttime": "start",
        "event_peaktime": "peak",
        "event_endtime": "end",
        "fl_goescls": "goes_class",
        "ar_noaanum": "noaa_ar",
    }
    present = [c for c in wanted if c in tbl.colnames]
    df = tbl[present].to_pandas().rename(columns=wanted)

    for c in ("start", "peak", "end"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True)

    df = df.drop_duplicates(subset=["start", "goes_class"]).sort_values("start")
    df["is_MX"] = df["goes_class"].astype(str).str[0].isin(["M", "X"])
    print(f"  [flares] {len(df)} events ({int(df['is_MX'].sum())} M/X)")
    return df.reset_index(drop=True)


def audit_class_vs_peak_flux(flux_df: pd.DataFrame, flares_df: pd.DataFrame) -> dict:
    """Cross-check reported flare class letter against peak 1-8A flux."""
    if flux_df.empty or flares_df.empty or "xrsb" not in flux_df.columns:
        return {"total_checked": 0, "matches": 0, "mismatches": 0, "mismatch_details": []}

    matches = 0
    mismatches = 0
    mismatch_details = []

    # Ensure index is UTC DatetimeIndex
    flux_idx = pd.to_datetime(flux_df.index, utc=True)
    flux_copy = flux_df.copy()
    flux_copy.index = flux_idx

    for _, row in flares_df.iterrows():
        t_start = pd.to_datetime(row["start"], utc=True)
        t_end = pd.to_datetime(row["end"], utc=True)
        reported_cls = str(row.get("goes_class", "")).strip().upper()
        if not reported_cls:
            continue
        reported_letter = reported_cls[0]

        mask = (flux_copy.index >= t_start) & (flux_copy.index <= t_end)
        window_flux = flux_copy.loc[mask, "xrsb"].dropna()
        if window_flux.empty:
            continue

        peak_val = float(window_flux.max())
        derived_letter = peak_flux_to_class_letter(peak_val)

        if derived_letter == reported_letter:
            matches += 1
        else:
            mismatches += 1
            mismatch_details.append({
                "start": str(t_start),
                "reported_class": reported_cls,
                "peak_flux_xrsb": peak_val,
                "derived_class": derived_letter,
            })

    return {
        "total_checked": matches + mismatches,
        "matches": matches,
        "mismatches": mismatches,
        "mismatch_details": mismatch_details[:10],
    }


def process_months(
    months: list[str],
    out_dir: Path,
    existing_data_dir: Path | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Process each month, building coverage.csv and tracking satellite changes."""
    coverage_rows = []
    satellite_changes = []
    class_checks_by_month = {}

    prev_sat = None
    prev_sat_month = None

    for month_str in months:
        print(f"\n================ Processing Month {month_str} ================")
        m_start = pd.Timestamp(f"{month_str}-01 00:00:00", tz="UTC")
        y, m = int(month_str.split("-")[0]), int(month_str.split("-")[1])
        if m == 12:
            next_m = pd.Timestamp(f"{y+1}-01-01 00:00:00", tz="UTC")
        else:
            next_m = pd.Timestamp(f"{y}-{m+1:02d}-01 00:00:00", tz="UTC")
        m_end = next_m - pd.Timedelta(seconds=1)

        # Enforce Holdout Quarantine
        assert_no_holdout(m_start, m_end)

        month_dir = out_dir / month_str
        raw_dir = month_dir / "raw"
        month_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        flux_csv = month_dir / "goes_xrs_1min.csv"
        flares_csv = month_dir / "goes_flares.csv"

        # Check existing cached data in existing_data_dir (e.g. 2026-01 / 2026-02)
        if (not flux_csv.exists() or not flares_csv.exists()) and existing_data_dir and existing_data_dir.exists():
            ex_flux = existing_data_dir / "goes_xrs_1min.csv"
            ex_flares = existing_data_dir / "goes_flares.csv"
            ex_raw = existing_data_dir / "raw"
            if ex_flux.exists() and ex_flares.exists():
                df_ex = pd.read_csv(ex_flux, parse_dates=["timestamp"], index_col="timestamp")
                df_ex.index = pd.to_datetime(df_ex.index, utc=True)
                month_slice = df_ex[(df_ex.index >= m_start) & (df_ex.index <= m_end)]
                if len(month_slice) > 0:
                    print(f"  Reusing existing slice for {month_str} ({len(month_slice)} rows)...")
                    # Copy matching raw NetCDF files
                    if ex_raw.exists():
                        for nc in ex_raw.glob("*.nc"):
                            d_str = month_str.replace("-", "")
                            if d_str in nc.name:
                                dest_nc = raw_dir / nc.name
                                if not dest_nc.exists():
                                    dest_nc.write_bytes(nc.read_bytes())

                    # Identify satellite from NetCDFs
                    sat_id = "UNKNOWN"
                    for nc in sorted(raw_dir.glob("*.nc")):
                        s = get_netcdf_satellite(nc)
                        if s != "UNKNOWN":
                            sat_id = s
                            break
                    if sat_id == "UNKNOWN":
                        # fallback check original
                        for nc in sorted(ex_raw.glob("*.nc")):
                            s = get_netcdf_satellite(nc)
                            if s != "UNKNOWN":
                                sat_id = s
                                break

                    month_slice = month_slice.copy()
                    month_slice["satellite"] = sat_id
                    month_slice.to_csv(flux_csv)

                    flares_ex = pd.read_csv(ex_flares)
                    flares_ex["start"] = pd.to_datetime(flares_ex["start"], utc=True)
                    m_flares = flares_ex[(flares_ex["start"] >= m_start) & (flares_ex["start"] <= m_end)]
                    m_flares.to_csv(flares_csv, index=False)

        # Download if not on disk
        if not flux_csv.exists() or not flares_csv.exists():
            s_query = m_start.strftime("%Y-%m-%d %H:%M:%S")
            e_query = m_end.strftime("%Y-%m-%d %H:%M:%S")
            try:
                flux_df, sat_id = fetch_month_flux(s_query, e_query, raw_dir)
                flux_df.to_csv(flux_csv)
            except Exception as exc:
                print(f"  ERROR fetching flux for {month_str}: {exc}")
                flux_df = pd.DataFrame()
                sat_id = "FAILED"

            try:
                flares_df = fetch_month_flares(s_query, e_query)
                flares_df.to_csv(flares_csv, index=False)
            except Exception as exc:
                print(f"  ERROR fetching flares for {month_str}: {exc}")
                flares_df = pd.DataFrame()
        else:
            flux_df = pd.read_csv(flux_csv, parse_dates=["timestamp"], index_col="timestamp")
            flares_df = pd.read_csv(flares_csv)
            # Read satellite from raw NetCDF
            sat_id = "UNKNOWN"
            for nc in sorted(raw_dir.glob("*.nc")):
                s = get_netcdf_satellite(nc)
                if s != "UNKNOWN":
                    sat_id = s
                    break
            if sat_id == "UNKNOWN" and "satellite" in flux_df.columns:
                sat_id = str(flux_df["satellite"].dropna().iloc[0])

        # Ensure satellite column is populated
        if not flux_df.empty and ("satellite" not in flux_df.columns or flux_df["satellite"].isna().any()):
            flux_df["satellite"] = sat_id
            flux_df.to_csv(flux_csv)

        # Compute coverage stats
        n_rows = len(flux_df)
        if n_rows > 0:
            flux_df.index = pd.to_datetime(flux_df.index, utc=True)
            diffs = flux_df.index.to_series().diff().dropna().dt.total_seconds() / 60.0
            longest_gap = float(diffs.max()) if len(diffs) else 0.0
            total_expected_minutes = int((m_end - m_start).total_seconds() / 60.0) + 1
            pct_missing = round(max(0.0, (1.0 - (n_rows / total_expected_minutes)) * 100.0), 2)
        else:
            longest_gap = 0.0
            pct_missing = 100.0

        n_events = len(flares_df)
        n_mx = int(flares_df["is_MX"].sum()) if ("is_MX" in flares_df.columns and not flares_df.empty) else 0

        cov_item = {
            "month": month_str,
            "satellite": sat_id,
            "rows": n_rows,
            "pct_missing": pct_missing,
            "longest_gap_minutes": round(longest_gap, 1),
            "n_events": n_events,
            "n_mx_events": n_mx,
        }
        coverage_rows.append(cov_item)

        # Re-run class vs peak flux check
        check_res = audit_class_vs_peak_flux(flux_df, flares_df)
        class_checks_by_month[month_str] = check_res

        # Check for satellite changes across consecutive months
        if prev_sat is not None and sat_id not in ("UNKNOWN", "FAILED") and sat_id != prev_sat:
            satellite_changes.append({
                "prev_month": prev_sat_month,
                "prev_satellite": prev_sat,
                "curr_month": month_str,
                "curr_satellite": sat_id,
            })
            print(f"  [SATELLITE CHANGE] {prev_sat} ({prev_sat_month}) -> {sat_id} ({month_str})")

        if sat_id not in ("UNKNOWN", "FAILED"):
            prev_sat = sat_id
            prev_sat_month = month_str

    cov_df = pd.DataFrame(coverage_rows)
    cov_path = out_dir / "coverage.csv"
    cov_df.to_csv(cov_path, index=False)
    print(f"\nSaved coverage table to {cov_path}")
    print(cov_df.to_string(index=False))

    return cov_df, satellite_changes


def audit_satellite_switches(satellite_changes: list[dict], out_dir: Path) -> list[dict]:
    """If satellite changes inside the range, compare flux medians 7 days before vs after."""
    results = []
    for sc in satellite_changes:
        m_prev = sc["prev_month"]
        m_curr = sc["curr_month"]
        flux_prev_p = out_dir / m_prev / "goes_xrs_1min.csv"
        flux_curr_p = out_dir / m_curr / "goes_xrs_1min.csv"

        if not flux_prev_p.exists() or not flux_curr_p.exists():
            continue

        df_prev = pd.read_csv(flux_prev_p, parse_dates=["timestamp"], index_col="timestamp")
        df_curr = pd.read_csv(flux_curr_p, parse_dates=["timestamp"], index_col="timestamp")
        df_prev.index = pd.to_datetime(df_prev.index, utc=True)
        df_curr.index = pd.to_datetime(df_curr.index, utc=True)

        # 7 days before boundary
        t_switch = df_curr.index.min()
        t_7d_before = t_switch - pd.Timedelta(days=7)
        t_7d_after = t_switch + pd.Timedelta(days=7)

        before_slice = df_prev[df_prev.index >= t_7d_before]
        after_slice = df_curr[df_curr.index <= t_7d_after]

        row = dict(sc)
        row["switch_date"] = str(t_switch.date())

        for ch in ("xrsa", "xrsb"):
            if ch in before_slice.columns and ch in after_slice.columns:
                med_before = float(before_slice[ch].dropna().median())
                med_after = float(after_slice[ch].dropna().median())
                jump_pct = abs(med_after - med_before) / med_before if med_before > 0 else 0.0
                row[f"{ch}_med_before"] = med_before
                row[f"{ch}_med_after"] = med_after
                row[f"{ch}_jump_pct"] = round(jump_pct * 100.0, 2)
                row[f"{ch}_flag_gt_20pct"] = bool(jump_pct > 0.20)

        results.append(row)
        print(f"Satellite transition audit for {sc['prev_satellite']} -> {sc['curr_satellite']}:")
        print(f"  Switch date: {t_switch.date()}")
        for ch in ("xrsa", "xrsb"):
            if f"{ch}_jump_pct" in row:
                print(f"  Channel {ch}: before={row[f'{ch}_med_before']:.3e}, after={row[f'{ch}_med_after']:.3e}, jump={row[f'{ch}_jump_pct']}%, flag={row[f'{ch}_flag_gt_20pct']}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GOES data in monthly chunks")
    parser.add_argument("--months", nargs="+", default=["2024-05", "2026-01", "2026-02"], help="Months to process (YYYY-MM)")
    parser.add_argument("--out-dir", default="data/goes", help="Target output directory")
    parser.add_argument("--existing-data", default="C:/Users/ASHUTOSH/Downloads/data/goes", help="Path to existing downloads")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_dir = Path(args.existing_data) if args.existing_data else None

    cov_df, sat_changes = process_months(args.months, out_dir, existing_data_dir=existing_dir)
    switch_audits = audit_satellite_switches(sat_changes, out_dir)


if __name__ == "__main__":
    main()
