import gzip
import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from netCDF4 import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "phase3a_data_audit.py"
SPEC = importlib.util.spec_from_file_location("phase3a_data_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _solexs_gzip(times=(1_700_000_000, 1_700_000_001, 1_700_000_002), counts=(1.0, np.nan, 3.0)):
    rate = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", array=np.asarray(times, dtype=float)),
        fits.Column(name="COUNTS", format="D", array=np.asarray(counts, dtype=float)),
    ], name="RATE")
    output = io.BytesIO()
    fits.HDUList([fits.PrimaryHDU(), rate]).writeto(output)
    return gzip.compress(output.getvalue())


def _hel1os_fits():
    times = pd.date_range("2026-02-05T00:00:00Z", periods=3, freq="s")
    mjd = times.asi8 / 1e9 / 86400 + 40587
    table = fits.BinTableHDU.from_columns([
        fits.Column(name="MJD", format="D", unit="MJD", array=mjd),
        fits.Column(name="ISOT", format="30A", unit="UT", array=np.asarray([t.strftime("%Y-%m-%dT%H:%M:%S") for t in times])),
        fits.Column(name="CTR", format="D", unit="cts/sec", array=[1.0, 2.0, 3.0]),
        fits.Column(name="STAT_ERR", format="D", unit="cts/sec", array=[0.1, 0.2, 0.3]),
    ], name="CZT1_LC_BAND_20.00KEV_TO_40.00KEV")
    output = io.BytesIO()
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(output)
    return output.getvalue()


def _write_goes_nc(path, flags=(0, 4, 1, 2), values=(1e-6, 2e-6, 3e-6, -9999.0)):
    with Dataset(path, "w") as nc:
        nc.createDimension("time", len(flags))
        time = nc.createVariable("time", "f8", ("time",))
        time.units = "seconds since 2000-01-01 12:00:00 UTC"
        time[:] = np.arange(len(flags)) * 60 + 800_000_000
        nc.title = "L2 XRS 1 minute average"
        nc.institution = "DOC/NOAA/NESDIS"
        nc.source = "GOES-R EXIS/XRS L2 xrsf-l2-flx1s"
        for channel in ("xrsa", "xrsb"):
            flux = nc.createVariable(f"{channel}_flux", "f4", ("time",), fill_value=-9999.0)
            flux.units = "W/m2"
            flux[:] = values
            flag = nc.createVariable(f"{channel}_flag", "u1", ("time",), fill_value=255)
            flag.flag_masks = np.asarray([3, 1, 2, 4], dtype=np.uint8)
            flag.flag_values = np.asarray([0, 1, 2, 4], dtype=np.uint8)
            flag.flag_meanings = "good_data eclipse bad_data interpolated"
            flag[:] = flags
            excluded = nc.createVariable(f"{channel}_flag_excluded", "u2", ("time",), fill_value=65535)
            excluded[:] = [0] * len(flags)
        correction = nc.createVariable("electron_correction_flag", "u2", ("time",), fill_value=65535)
        correction[:] = [1] * len(flags)


def test_coverage_splits_internal_gaps_and_is_repeatable():
    times = pd.date_range("2026-01-01", periods=5, freq="min", tz="UTC")
    expected = audit.contiguous_intervals(times, cadence_seconds=60, valid=[1, 1, 0, 1, 1])
    assert expected == audit.contiguous_intervals(times, cadence_seconds=60, valid=[1, 1, 0, 1, 1])
    assert [x["sample_count"] for x in expected] == [2, 2]
    assert expected[0]["end_utc"] < expected[1]["start_utc"]


def test_gti_interval_boundary_policy_is_explicit():
    times = pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="s")
    intervals = [(times[1], times[3])]
    assert audit.gti_membership(times, intervals, closed="both").tolist() == [False, True, True, True, False]
    assert audit.gti_membership(times, intervals, closed="left").tolist() == [False, True, True, False, False]
    assert audit.gti_membership(times, intervals, closed="right").tolist() == [False, False, True, True, False]
    with pytest.raises(ValueError, match="closed"):
        audit.gti_membership(times, intervals, closed="unknown")


def test_joint_candidate_coverage_requires_both_goes_channels_and_deduplicates():
    start, end = "2026-01-01T00:00:00+00:00", "2026-01-01T00:02:00+00:00"
    coverage = [
        {"instrument": "SoLEXS", "start_utc": start, "end_utc": end},
        {"instrument": "HEL1OS", "start_utc": start, "end_utc": end},
        {"instrument": "GOES-18", "source_file": "day.nc:xrsa", "start_utc": start, "end_utc": end},
        {"instrument": "GOES-18", "source_file": "day.nc:xrsb", "start_utc": start, "end_utc": end},
    ]
    events = pd.DataFrame({"start": [pd.Timestamp("2026-01-01T00:01:00Z")]})
    result = audit._candidate_joint_intervals(coverage, events)
    assert len(result) == 1
    assert result[0]["candidate_goes_event_starts"] == 1


def test_solexs_gzip_fits_preserves_raw_missing_count_and_candidate_count():
    result = audit._read_solexs_lc(_solexs_gzip())
    assert len(result["times"]) == 3
    assert result["candidate_valid"].tolist() == [True, False, True]
    assert result["units"]["COUNTS"] is None


def test_goes_fill_and_nonzero_quality_flags_are_excluded(tmp_path):
    path = tmp_path / "sample.nc"
    _write_goes_nc(path)
    result = audit._audit_goes_nc(path)
    assert result["channels"]["xrsa"]["candidate_valid_rows"] == 1
    assert result["channels"]["xrsa"]["sentinel_rows"] == 1
    assert result["channels"]["xrsa"]["excluded_reason_record_counts"]["interpolated"] == 1
    assert result["channels"]["xrsa"]["excluded_reason_record_counts"]["eclipse"] == 1
    assert result["channels"]["xrsa"]["excluded_reason_record_counts"]["bad_data"] == 1
    assert not result["records"][1]["candidate_valid"]


def test_csv_netcdf_values_and_timestamps_are_compared(tmp_path):
    path = tmp_path / "sample.nc"
    _write_goes_nc(path, flags=(0, 0, 0, 0), values=(1e-6, 2e-6, 3e-6, 4e-6))
    daily = audit._audit_goes_nc(path)
    frame = pd.DataFrame({
        "timestamp": daily["timestamps"].astype(str),
        "xrsa": [1e-6, 2e-6, 3e-6, 4e-6],
        "xrsb": [1e-6, 2e-6, 3e-6, 4e-6],
    })
    csv_path = tmp_path / "measurements.csv"
    frame.to_csv(csv_path, index=False)
    result = audit._compare_goes_csv(csv_path, [daily])
    assert result["timestamps_equal_netcdf_exact"] is True
    assert result["channel_comparison"]["xrsa"]["values_equal_nan_aware"] is True


def test_hel1os_mjd_isot_product_fields_parse_without_version_choice():
    result = audit._read_hel1os_lc(_hel1os_fits())
    assert result["rows"] == result["valid_count"] == 3
    assert "20.00KEV_TO_40.00KEV" in result["extension"]
    assert result["columns"][2]["unit"] == "cts/sec"


def test_hel1os_overlapping_versions_are_both_retained(tmp_path):
    outer_path = tmp_path / "outer.zip"
    inner_versions = []
    for version in ("V111", "V211"):
        inner_bytes = io.BytesIO()
        with zipfile.ZipFile(inner_bytes, "w") as inner:
            inner.writestr("2026/02/05/czt/lightcurve_czt1.fits", _hel1os_fits())
        inner_versions.append((f"HLS_20260205_000005_43185sec_lev1_{version}.zip", inner_bytes.getvalue()))
    with zipfile.ZipFile(outer_path, "w") as outer:
        for name, data in inner_versions:
            outer.writestr(name, data)
    result, _ = audit._audit_hel1os(outer_path)
    assert len(result["products"]) == 2
    assert {p["processing_version"] for p in result["products"]} == {"V111", "V211"}
    assert result["version_policy"].startswith("UNRESOLVED")


def test_overlapping_events_and_duplicate_start_class_are_reported(tmp_path):
    event_csv = tmp_path / "events.csv"
    pd.DataFrame([
        ["2026-01-01 00:00:00", "2026-01-01 00:02:00", "2026-01-01 00:04:00", "M1.0", 1, True],
        ["2026-01-01 00:00:00", "2026-01-01 00:02:00", "2026-01-01 00:05:00", "M1.0", 1, True],
        ["2026-01-01 00:03:00", "2026-01-01 00:04:00", "2026-01-01 00:06:00", "X1.0", 1, True],
    ], columns=["start", "peak", "end", "goes_class", "noaa_ar", "is_MX"]).to_csv(event_csv, index=False)
    result = audit._audit_events(event_csv, None)
    assert result["quality_checks"]["duplicate_start_class_rows"] == 2
    assert result["overlapping_event_interval_pairs"] == 3


def test_atomic_report_writer_refuses_to_overwrite_unrecognized_files(tmp_path):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    target = report_dir / "phase3a_data_validation.md"
    target.write_text("user content", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        audit._atomic_write(target, "replacement", "markdown")
    assert target.read_text(encoding="utf-8") == "user content"
