import importlib.util
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "phase3b_authoritative_audit.py"
SPEC = importlib.util.spec_from_file_location("phase3b_authoritative_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _local(starts, classes=None):
    classes = classes or ["M1.0"] * len(starts)
    return pd.DataFrame({
        "start": starts,
        "peak": [pd.Timestamp(x) + pd.Timedelta(minutes=2) for x in starts],
        "end": [pd.Timestamp(x) + pd.Timedelta(minutes=4) for x in starts],
        "goes_class": classes, "noaa_ar": [1] * len(starts), "is_MX": [True] * len(starts),
    })


def _official(starts, classes=None, ids=None):
    classes = classes or ["M1.0"] * len(starts)
    ids = ids or [f"2026010100{i:02d}" for i in range(len(starts))]
    return pd.DataFrame({
        "start_time": starts,
        "time": [pd.Timestamp(x) + pd.Timedelta(minutes=2) for x in starts],
        "end_time": [pd.Timestamp(x) + pd.Timedelta(minutes=4) for x in starts],
        "flare_class": classes, "flare_id": ids,
        "xrsb_irrad_source": ["GOES-18"] * len(starts), "active_region": [1] * len(starts),
    })


def test_utc_parser_converts_offsets_and_documents_naive_as_utc():
    converted = audit.utc_series(["2026-01-01T01:00:00+01:00", "2026-01-01 00:00:00"])
    assert str(converted.tz) == "UTC"
    assert converted[0] == converted[1]


def test_class_normalization_preserves_magnitude_and_original_value():
    result = audit.normalize_class(" m 1.00 ")
    assert result["original"] == " m 1.00 "
    assert result["normalized"] == "M1.00"
    assert result["family"] == "M" and result["is_mx"] is True
    assert audit.normalize_class("unknown")["family"] is None


def test_unique_timestamp_match_keeps_class_magnitude_revision_visible():
    local = _local(["2026-01-01 00:00:00"], ["M1.0"])
    official = _official(["2026-01-01 00:01:00"], ["M1.1"])
    rows, summary, _, _ = audit.reconcile_events(
        local, official, "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")
    assert summary["unique_mutual_start_matches"] == 1
    assert summary["same_family_magnitude_mismatch_rows"] == 1
    assert rows.iloc[0]["official_class_raw"] == "M1.1"
    assert rows.iloc[0]["local_class_raw"] == "M1.0"
    assert bool(rows.iloc[0]["local_event_identity_uniquely_verified"]) is True


def test_ambiguous_candidate_matches_are_reported_without_closest_selection():
    local = _local(["2026-01-01 00:00:30"])
    official = _official(["2026-01-01 00:00:00", "2026-01-01 00:01:00"], ids=["a", "b"])
    rows, summary, _, _ = audit.reconcile_events(
        local, official, "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")
    assert summary["ambiguous_local_rows"] == 1
    assert summary["unique_mutual_start_matches"] == 0
    assert rows.iloc[0]["reconciliation_status"] == "ambiguous_start_match_no_selection"
    assert rows.iloc[0]["official_flare_id"] is None or pd.isna(rows.iloc[0]["official_flare_id"])


def test_duplicates_are_counted_and_near_duplicates_are_not_merged():
    local = _local(["2026-01-01 00:00:00", "2026-01-01 00:00:30"])
    official = _official(["2026-01-01 00:00:00", "2026-01-01 00:00:30"], ids=["a", "b"])
    _, summary, _, _ = audit.reconcile_events(
        local, official, "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")
    assert len(summary["near_duplicate_start_candidates_within_tolerance_not_merged"]) == 2
    assert all(x["resolution"] == "CANDIDATE_ONLY_NOT_MERGED" for x in summary["near_duplicate_start_candidates_within_tolerance_not_merged"])


def test_unobserved_or_incomplete_horizon_is_never_negative():
    origin = pd.Timestamp("2026-01-01T00:00:00Z")
    covered = [(origin, origin + pd.Timedelta(hours=2))]
    incomplete_catalog = audit.classify_horizon(origin, 3600, False, [], covered)
    assert incomplete_catalog["label"] is None
    assert incomplete_catalog["reason"] == "event_catalog_completeness_unverified"
    missing_future = audit.classify_horizon(
        origin, 3600, True, [], [(origin, origin + pd.Timedelta(minutes=30))])
    assert missing_future["label"] is None
    assert missing_future["reason"] == "missing_or_insufficient_horizon_coverage"


def test_verified_event_inside_open_closed_horizon_is_positive():
    origin = pd.Timestamp("2026-01-01T00:00:00Z")
    event = origin + pd.Timedelta(hours=1)
    result = audit.classify_horizon(origin, 3600, False, [event], [])
    assert result["label"] == 1
    assert result["reason"] == "verified_event_start_in_open_closed_horizon"

