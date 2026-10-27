"""Unit tests for the Data Foundation overhaul (Part A, B, C, D) and Downloader Fixes.

Verifies:
  1. Holdout quarantine enforcement across planner, downloader, and splitter.
  2. PRADAN download planner invariants (priority levels, no Oct 2024 dates, uniqueness).
  3. Batch downloader safety (polite rate-limiting, <= 15 days/batch, URL secret redaction).
  4. Authentic PRADAN patterns, 404 NOT_AVAILABLE handling, 401/403 auth halt, redirect host checking.
  5. PRADAN_TOKEN quote stripping and zero token leakage in log or console.
  6. Scanning of both Downloads and data/pradan_downloads by overlap scan and planner.
  7. Instrument-day split purge gap (>= 24h) and PASS/FAIL gate logic.
  8. Refreshed GOES coverage column semantics.
  9. Absence of hardcoded secrets across the codebase.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ml.labels.goes_labels import split_by_instrument_days
from scripts.plan_pradan_downloads import (
    HOLDOUT_START,
    HOLDOUT_END,
    build_pradan_plan,
)
from scripts.run_pradan_batch_download import (
    assert_safe_from_holdout,
    redact_url_secrets,
    is_pradan_host,
    get_auth_cookie,
    download_single_file_safe,
    run_batch,
    parse_pradan_script_catalog,
    get_download_items_for_date,
)
from scripts.scan_instrument_overlap import scan_pradan_downloads


class TestHoldoutQuarantine:
    """Ensure October 2024 is strictly forbidden in all new workflows."""

    def test_planner_never_emits_october_2024(self):
        plan_df = build_pradan_plan()
        dates = pd.to_datetime(plan_df["date"]).dt.date
        h_start = HOLDOUT_START.date()
        h_end = HOLDOUT_END.date()
        in_holdout = dates.apply(lambda d: h_start <= d <= h_end)
        assert not in_holdout.any(), f"Planner emitted {in_holdout.sum()} holdout dates!"

    def test_downloader_rejects_october_2024_dates(self):
        with pytest.raises(ValueError, match="FATAL HOLDOUT GUARD"):
            assert_safe_from_holdout(["2024-10-15"])

        with pytest.raises(ValueError, match="FATAL HOLDOUT GUARD"):
            assert_safe_from_holdout(["2024-05-10", "2024-10-01"])

        # Valid dates should pass without exception
        assert_safe_from_holdout(["2024-05-10", "2026-02-01"])

    def test_instrument_splitter_rejects_october_2024(self):
        idx = pd.date_range("2024-10-01", "2024-10-05", freq="1min", tz="UTC")
        df = pd.DataFrame({"y_1h": np.zeros(len(idx))}, index=idx)
        with pytest.raises(ValueError, match="FATAL HOLDOUT GUARD"):
            split_by_instrument_days(df, ["2024-10-02"])


class TestPradanPlannerAndDownloader:
    """Verify planner ordering, priorities, and downloader URL redaction."""

    def test_pradan_plan_structure_and_priorities(self):
        plan_df = build_pradan_plan()
        required_cols = [
            "date", "year_month", "priority", "reason",
            "has_solexs", "has_helios", "m_class_count", "x_class_count", "active_regions",
        ]
        for col in required_cols:
            assert col in plan_df.columns

        assert plan_df["date"].is_unique

        # Priority 1 must be for dual-instrument days
        p1 = plan_df[plan_df["priority"] == 1]
        assert len(p1) > 0
        assert (p1["has_solexs"] & p1["has_helios"]).all()

        # Priority 2 must be for existing single instrument days
        p2 = plan_df[plan_df["priority"] == 2]
        assert len(p2) > 0
        assert (p2["has_solexs"] ^ p2["has_helios"]).all()

    def test_downloader_redacts_url_secrets(self):
        url_with_token = "https://pradan1.issdc.gov.in/download?token=SECRET_TOKEN_XYZ&user=Ashutosh"
        redacted = redact_url_secrets(url_with_token)
        assert "SECRET_TOKEN_XYZ" not in redacted
        assert "[REDACTED]" in redacted or "%5BREDACTED%5D" in redacted

        url_with_pass = "https://example.com/api?password=SUPER_SECRET_123"
        redacted_pass = redact_url_secrets(url_with_pass)
        assert "SUPER_SECRET_123" not in redacted_pass
        assert "[REDACTED]" in redacted_pass or "%5BREDACTED%5D" in redacted_pass


class TestDownloaderSecurityAndRobustness:
    """Verify quote stripping, redirect checks, 404/auth status handling, and token privacy."""

    def test_quote_stripping_from_token(self, monkeypatch):
        # Unquoted
        monkeypatch.setenv("PRADAN_TOKEN", "plain_session_123")
        assert get_auth_cookie() == "plain_session_123"

        # Single quotes
        monkeypatch.setenv("PRADAN_TOKEN", "'single_quoted_cookie'")
        assert get_auth_cookie() == "single_quoted_cookie"

        # Double quotes
        monkeypatch.setenv("PRADAN_TOKEN", '"double_quoted_cookie"')
        assert get_auth_cookie() == "double_quoted_cookie"

        # Whitespace and quotes
        monkeypatch.setenv("PRADAN_TOKEN", "  'padded_cookie'  ")
        assert get_auth_cookie() == "padded_cookie"

        # Nested quotes
        monkeypatch.setenv("PRADAN_TOKEN", " '\"nested_cookie\"' ")
        assert get_auth_cookie() == "nested_cookie"

    def test_is_pradan_host_domain_validation(self):
        assert is_pradan_host("https://pradan1.issdc.gov.in/al1/protected/download") is True
        assert is_pradan_host("https://issdc.gov.in/portal") is True
        assert is_pradan_host("https://sub.issdc.gov.in/file") is True
        assert is_pradan_host("https://evil.attacker.com/steal") is False
        assert is_pradan_host("https://pradan1.issdc.gov.in.evil.com/phish") is False
        assert is_pradan_host("") is False

    def test_redirect_blocked_for_external_host(self, tmp_path):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 302
        mock_resp.headers = {"Location": "https://attacker.com/exfiltrate"}
        mock_session.get.return_value = mock_resp

        target_file = tmp_path / "test.zip"
        status, msg = download_single_file_safe(
            "https://pradan1.issdc.gov.in/file.zip",
            target_file,
            "secret_token_123",
            mock_session,
        )
        assert status == 400
        assert "REDIRECT_TO_NON_PRADAN_HOST_BLOCKED" in msg
        # Ensure only 1 request was made and cookies were never sent to attacker.com
        assert mock_session.get.call_count == 1
        args, kwargs = mock_session.get.call_args
        assert "attacker.com" not in args[0]

    def test_http_404_treated_as_not_available_no_retry(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PRADAN_TOKEN", "test_cookie")
        with patch("scripts.run_pradan_batch_download.download_single_file_safe") as mock_dl:
            mock_dl.return_value = (404, "NOT_FOUND")

            stats = run_batch(["2024-05-10"], instrument="solexs", max_days=1)
            # Exactly 1 attempt per file (no retries on 404!)
            assert mock_dl.call_count == 1
            assert stats["not_available"] == 1
            assert stats["failed"] == 0

    def test_http_401_403_halts_batch_immediately(self, monkeypatch):
        monkeypatch.setenv("PRADAN_TOKEN", "test_cookie")
        with patch("scripts.run_pradan_batch_download.download_single_file_safe") as mock_dl:
            mock_dl.return_value = (401, "UNAUTHORIZED")

            stats = run_batch(["2024-05-10", "2024-05-11", "2024-05-12"], instrument="solexs", max_days=3)
            # Must halt immediately on first date
            assert mock_dl.call_count == 1
            assert stats["auth_expired"] is True

    def test_zero_token_leakage_in_log_or_console(self, monkeypatch, capsys):
        secret_token = "MY_SUPER_SECRET_SESSION_TOKEN_9999"
        monkeypatch.setenv("PRADAN_TOKEN", secret_token)

        with patch("scripts.run_pradan_batch_download.download_single_file_safe") as mock_dl:
            mock_dl.return_value = (404, "NOT_FOUND")
            run_batch(["2024-05-10"], instrument="solexs", max_days=1)

        captured = capsys.readouterr()
        assert secret_token not in captured.out
        assert secret_token not in captured.err

        log_path = REPO_ROOT / "data" / "pradan_download_log.csv"
        if log_path.exists():
            log_content = log_path.read_text(encoding="utf-8")
            assert secret_token not in log_content

    def test_pradan_downloads_directory_read_by_overlap_scan(self, tmp_path):
        # Create mock pradan_downloads directory with a SoLEXS and a HEL1OS file
        mock_pradan_dir = tmp_path / "pradan_downloads"
        mock_pradan_dir.mkdir()

        # Create valid zip for SoLEXS
        s_zip = mock_pradan_dir / "AL1_SLX_L1_20250701_v1.0.zip"
        with zipfile.ZipFile(s_zip, "w") as zf:
            zf.writestr("test.fits", "dummy content")

        # Create valid zip for HEL1OS
        h_zip = mock_pradan_dir / "HLS_20250701_000000_86400sec_lev1_V111.zip"
        with zipfile.ZipFile(h_zip, "w") as zf:
            zf.writestr("test.fits", "dummy content")

        ov_df = scan_pradan_downloads([mock_pradan_dir])
        assert not ov_df.empty
        dates = ov_df["date"].tolist()
        assert "2025-07-01" in dates
        row = ov_df[ov_df["date"] == "2025-07-01"].iloc[0]
        assert bool(row["solexs_ok"]) is True
        assert bool(row["helios_ok"]) is True
        assert bool(row["both"]) is True


class TestInstrumentSplitGateLogic:
    """Verify purge-gap enforcement and PASS/FAIL gate evaluation."""

    def test_instrument_split_purge_gap(self):
        dates = [f"2024-05-{d:02d}" for d in range(1, 11)]
        idx = pd.date_range("2024-05-01", "2024-05-10 23:59:00", freq="1h", tz="UTC")
        df = pd.DataFrame({"y_1h": np.zeros(len(idx))}, index=idx)

        splits = split_by_instrument_days(
            df=df,
            candidate_days=dates,
            ratios=(0.60, 0.20, 0.20),
            purge_gap_days=1,
            split_names=("train", "val", "test"),
        )
        splits_dict = {stats["partition"]: stats for _, _, stats in splits}

        if "train" in splits_dict and "val" in splits_dict:
            train_end = pd.Timestamp(splits_dict["train"]["end_date"])
            val_start = pd.Timestamp(splits_dict["val"]["start_date"])
            assert (val_start - train_end).days > 1

    def test_gate_pass_fail_diagnostics(self):
        idx = pd.date_range("2024-05-01", "2024-05-08 23:59:00", freq="1h", tz="UTC")
        df = pd.DataFrame({"y_1h": np.zeros(len(idx))}, index=idx)
        cand_days = [f"2024-05-{d:02d}" for d in range(1, 9)]

        flares_few = pd.DataFrame({
            "start": ["2024-05-01 10:00:00", "2024-05-01 14:00:00"],
            "goes_class": ["M1.0", "M2.0"],
            "is_MX": [True, True],
            "noaa_ar": [13664, 13664],
        })

        splits = split_by_instrument_days(
            df=df,
            candidate_days=cand_days,
            flares_df=flares_few,
            ratios=(0.60, 0.20, 0.20),
            purge_gap_days=1,
            min_mx_onsets=20,
            min_active_regions=5,
        )
        train_stats = splits[0][2]
        assert train_stats["requirement_status"] == "FAIL"
        assert "need" in train_stats["diagnosis"]
        assert "more M/X onsets" in train_stats["diagnosis"]
        assert "distinct active regions" in train_stats["diagnosis"]

        records = []
        for i in range(25):
            records.append({
                "start": f"2024-05-0{(i % 4) + 1} 12:00:00",
                "goes_class": "M5.0",
                "is_MX": True,
                "noaa_ar": 13660 + (i % 6),
            })
        flares_rich = pd.DataFrame(records)

        splits_rich = split_by_instrument_days(
            df=df,
            candidate_days=cand_days,
            flares_df=flares_rich,
            ratios=(0.60, 0.20, 0.20),
            purge_gap_days=1,
            min_mx_onsets=20,
            min_active_regions=5,
        )
        train_stats_rich = splits_rich[0][2]
        assert train_stats_rich["requirement_status"] == "PASS"
        assert train_stats_rich["diagnosis"] == "PASS: all requirements met"


class TestCoverageColumnSemantics:
    """Verify coverage report contains required decomposed columns."""

    def test_coverage_csv_columns(self):
        cov_path = REPO_ROOT / "data" / "goes" / "coverage.csv"
        assert cov_path.exists()
        cov_df = pd.read_csv(cov_path)
        required_cols = [
            "month",
            "satellite",
            "satellite_changed_from_prev",
            "days_in_month",
            "days_downloaded",
            "rows_expected_for_downloaded_days",
            "rows_present",
            "pct_gap_within_downloaded_days",
            "not_downloaded_days",
            "class_flag",
        ]
        for col in required_cols:
            assert col in cov_df.columns

        assert (cov_df["pct_gap_within_downloaded_days"] >= 0.0).all()
        assert (cov_df["pct_gap_within_downloaded_days"] <= 100.0).all()


class TestSecurityAndZeroCredentialLeakage:
    """Ensure credentials, passwords, and sensitive tokens are strictly absent."""

    def test_no_hardcoded_passwords_in_codebase(self):
        forbidden_patterns = [
            "PRADAN" + "_PASS=",
            "XCPSK" + "v9",
        ]
        scanned_exts = {".py", ".json", ".yaml", ".yml", ".md"}

        for p in REPO_ROOT.rglob("*"):
            if not p.is_file():
                continue
            if p.name == "test_data_foundation.py":
                continue
            parts = p.parts
            if any(x in parts for x in (".git", "venv", ".pytest_cache", "__pycache__", ".gemini", "scratch")):
                continue
            if p.suffix not in scanned_exts:
                continue

            content = p.read_text(encoding="utf-8", errors="ignore")
            for fp in forbidden_patterns:
                assert fp not in content, f"SECURITY VIOLATION: Found forbidden credential pattern in {p}!"
