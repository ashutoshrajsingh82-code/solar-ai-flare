"""Automated tests for Phase 4P scalable real observational dataset indexer.

Tests coverage:
- SHA-256 verification for bytes and files.
- Duplicate detection (identical bitstream and counterpart version collisions).
- Overlapping observation interval detection.
- Measurement unit enforcement, detector semantics, and exclusion tagging.
- Contiguous sequence extraction across gaps and missing values.
- Idempotent reproducibility of indexer runs.
"""
from __future__ import annotations

import gzip
import io
import json
import sys
import zipfile
from pathlib import Path

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import numpy as np
import pytest
from astropy.io import fits

from scripts.index_real_observations import (
    compute_sha256_bytes,
    compute_sha256_file,
    RealDatasetIndexer,
)
from ml.experiments.real_self_supervised import extract_contiguous_segments


def _create_mock_fits_bytes(times: np.ndarray, counts: np.ndarray,
                            t_unit: str = "s", c_unit: str = "counts/s") -> bytes:
    """Create in-memory gzipped FITS file bytes."""
    c1 = fits.Column(name="TIME", format="1D", unit=t_unit, array=times)
    c2 = fits.Column(name="COUNTS", format="1E", unit=c_unit, array=counts)
    table_hdu = fits.BinTableHDU.from_columns([c1, c2], name="RATE")
    primary_hdu = fits.PrimaryHDU()
    hdul = fits.HDUList([primary_hdu, table_hdu])
    
    buf = io.BytesIO()
    hdul.writeto(buf)
    return gzip.compress(buf.getvalue())


def test_sha256_computation(tmp_path: Path):
    """Test SHA-256 calculation for in-memory bytes and on-disk files."""
    import hashlib
    data = b"Genuine Aditya-L1 SoLEXS Observational Data"
    expected_sha = hashlib.sha256(data).hexdigest().lower()
    assert compute_sha256_bytes(data) == expected_sha
    assert expected_sha == "ba6deccef74ee09dbada14f9341ca60df625349c4fc0b9ade3644d014ffe93ab"

    test_file = tmp_path / "test_file.bin"
    test_file.write_bytes(data)
    assert compute_sha256_file(test_file) == expected_sha


def test_counterpart_collision_detection():
    """Test detection of HEL1OS V111 vs V211 counterpart version collisions."""
    indexer = RealDatasetIndexer([])
    indexer.inventory = [
        {
            "instrument": "HEL1OS",
            "package_name": "HLS_20260205_000005_43185sec_lev1_V111.zip",
            "product_version": "V111",
            "package_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        },
        {
            "instrument": "HEL1OS",
            "package_name": "HLS_20260205_000005_43185sec_lev1_V211.zip",
            "product_version": "V211",
            "package_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        },
    ]
    indexer._detect_duplicates_and_overlaps()

    assert len(indexer.duplicate_groups) == 1
    collision = indexer.duplicate_groups[0]
    assert collision["type"] == "COUNTERPART_VERSION_COLLISION"
    assert collision["base_identifier"] == "HLS_20260205_000005_43185sec"
    assert "V111" in collision["versions_found"]
    assert "V211" in collision["versions_found"]
    assert collision["blocker_id"] == "B-HL-VERSION-006"


def test_identical_bitstream_duplicate_detection():
    """Test detection of duplicate files with identical bitstreams."""
    indexer = RealDatasetIndexer([])
    shared_sha = "1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff"
    indexer.inventory = [
        {
            "instrument": "SoLEXS",
            "member_path": "path/to/product_A.lc.gz",
            "member_sha256": shared_sha,
        },
        {
            "instrument": "SoLEXS",
            "member_path": "path/to/product_B.lc.gz",
            "member_sha256": shared_sha,
        },
    ]
    indexer._detect_duplicates_and_overlaps()

    assert len(indexer.duplicate_groups) == 1
    dup = indexer.duplicate_groups[0]
    assert dup["type"] == "IDENTICAL_BITSTREAM_DUPLICATE"
    assert dup["sha256"] == shared_sha
    assert len(dup["occurrences"]) == 2


def test_overlapping_interval_detection():
    """Test detection of overlapping time intervals between observation products."""
    indexer = RealDatasetIndexer([])
    indexer.inventory = [
        {
            "eligibility_decision": "ELIGIBLE",
            "member_path": "product_day1.lc.gz",
            "start_time_unix": 1000.0,
            "end_time_unix": 2000.0,
        },
        {
            "eligibility_decision": "ELIGIBLE",
            "member_path": "product_day2_overlap.lc.gz",
            "start_time_unix": 1500.0,
            "end_time_unix": 2500.0,
        },
        {
            "eligibility_decision": "ELIGIBLE",
            "member_path": "product_day3_clean.lc.gz",
            "start_time_unix": 3000.0,
            "end_time_unix": 4000.0,
        },
    ]
    indexer._detect_duplicates_and_overlaps()

    assert len(indexer.overlapping_intervals) == 1
    overlap = indexer.overlapping_intervals[0]
    assert overlap["product_1"] == "product_day1.lc.gz"
    assert overlap["product_2"] == "product_day2_overlap.lc.gz"
    assert overlap["overlap_start_unix"] == 1500.0
    assert overlap["overlap_end_unix"] == 2000.0


def test_non_overlapping_intervals():
    """Test that disjoint time intervals report 0 overlapping conflicts."""
    indexer = RealDatasetIndexer([])
    indexer.inventory = [
        {
            "eligibility_decision": "ELIGIBLE",
            "member_path": "day1.lc.gz",
            "start_time_unix": 1000.0,
            "end_time_unix": 2000.0,
        },
        {
            "eligibility_decision": "ELIGIBLE",
            "member_path": "day2.lc.gz",
            "start_time_unix": 2001.0,
            "end_time_unix": 3000.0,
        },
    ]
    indexer._detect_duplicates_and_overlaps()
    assert len(indexer.overlapping_intervals) == 0


def test_gap_and_missing_value_segmentation():
    """Test that extract_contiguous_segments enforces strictly contiguous spans and min_length."""
    # Span 1: 80 contiguous valid records (enough for 70s segment)
    times1 = np.arange(1000.0, 1080.0, 1.0)
    counts1 = np.full(80, 50.0)

    # Gap of 10 seconds: jump from 1080 to 1090
    # Span 2: 30 contiguous valid records (too short: < 70)
    times2 = np.arange(1090.0, 1120.0, 1.0)
    counts2 = np.full(30, 60.0)

    # Missing values: 5 NaNs
    times_nan = np.arange(1120.0, 1125.0, 1.0)
    counts_nan = np.full(5, np.nan)

    # Span 3: 75 contiguous valid records
    times3 = np.arange(1125.0, 1200.0, 1.0)
    counts3 = np.full(75, 70.0)

    all_times = np.concatenate([times1, times2, times_nan, times3])
    all_counts = np.concatenate([counts1, counts2, counts_nan, counts3])

    segments = extract_contiguous_segments(all_times, all_counts, min_segment_length=70)

    # Should have exactly 2 valid segments (Span 1 and Span 3); Span 2 was too short (<70)
    assert len(segments) == 2
    seg1_t, seg1_c = segments[0]
    assert len(seg1_t) == 80
    assert seg1_t[0] == 1000.0
    assert seg1_t[-1] == 1079.0

    seg2_t, seg2_c = segments[1]
    assert len(seg2_t) == 75
    assert seg2_t[0] == 1125.0
    assert seg2_t[-1] == 1199.0


def test_indexer_mock_solexs_archive(tmp_path: Path):
    """Test complete indexer audit on a structured synthetic SoLEXS archive."""
    times = np.arange(100.0, 200.0, 1.0)
    counts = np.full(100, 42.0)
    fits_gz = _create_mock_fits_bytes(times, counts)

    # Build mock daily zip
    inner_zip_buf = io.BytesIO()
    with zipfile.ZipFile(inner_zip_buf, "w") as inner_zip:
        inner_zip.writestr("level1/al1_solexs_20260201_SDD2_v1.0.lc.gz", fits_gz)
        inner_zip.writestr("level1/al1_solexs_20260201_SDD1_v1.0.lc.gz", fits_gz)
        inner_zip.writestr("level1/al1_solexs_20260201_SDD2_v1.0.pi.gz", b"MOCK_PI_SPECTRA")
    inner_zip_bytes = inner_zip_buf.getvalue()

    outer_zip_path = tmp_path / "solexs_mock_archive.zip"
    with zipfile.ZipFile(outer_zip_path, "w") as outer_zip:
        outer_zip.writestr("al1_solexs_lev1_20260201_daily.zip", inner_zip_bytes)

    indexer = RealDatasetIndexer([outer_zip_path])
    result = indexer.index_all()

    assert result["total_products_indexed"] == 3
    assert result["eligible_for_forecasting"] == 1
    assert result["excluded_products"] == 2

    # Verify SDD2 LC properties
    sdd2_item = next(p for p in result["inventory"] if p.get("detector") == "SDD2" and p.get("product_type") == "LIGHTCURVE")
    assert sdd2_item["eligibility_decision"] == "ELIGIBLE"
    assert sdd2_item["timestamp_units"] == "s"
    assert sdd2_item["measurement_units"] == "counts/s"
    assert sdd2_item["total_records"] == 100
    assert sdd2_item["duty_cycle_pct"] == 100.0
    assert sdd2_item["num_contiguous_segments"] == 1

    # Verify SDD1 exclusion
    sdd1_item = next(p for p in result["inventory"] if p.get("detector") == "SDD1")
    assert sdd1_item["eligibility_decision"] == "EXCLUDED"
    assert sdd1_item["exclusion_code"] == "EXCL-SOLEXS-SDD1-INACTIVE"

    # Verify PI exclusion
    pi_item = next(p for p in result["inventory"] if p.get("product_type") == "SPECTRUM_PI")
    assert pi_item["eligibility_decision"] == "EXCLUDED"
    assert pi_item["exclusion_code"] == "EXCL-SOLEXS-PI-SPECTRA"


def test_indexer_idempotence(tmp_path: Path):
    """Verify that consecutive indexing runs on the same input yield identical outputs."""
    times = np.arange(100.0, 180.0, 1.0)
    counts = np.full(80, 25.0)
    fits_gz = _create_mock_fits_bytes(times, counts)

    inner_buf = io.BytesIO()
    with zipfile.ZipFile(inner_buf, "w") as inner:
        inner.writestr("level1/al1_solexs_20260202_SDD2_v1.0.lc.gz", fits_gz)
    
    outer_path = tmp_path / "solexs_idempotent_test.zip"
    with zipfile.ZipFile(outer_path, "w") as outer:
        outer.writestr("al1_solexs_lev1_20260202_daily.zip", inner_buf.getvalue())

    indexer1 = RealDatasetIndexer([outer_path])
    res1 = indexer1.index_all()

    indexer2 = RealDatasetIndexer([outer_path])
    res2 = indexer2.index_all()

    # Discard non-deterministic timestamp
    del res1["indexed_at_utc"]
    del res2["indexed_at_utc"]

    assert json.dumps(res1, sort_keys=True) == json.dumps(res2, sort_keys=True)
