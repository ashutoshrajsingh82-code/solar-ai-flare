import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fits_semantics_audit.py"
SPEC = importlib.util.spec_from_file_location("fits_semantics_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _fits(path, *, gti="valid", science_names=("LIGHTCURVE",), quality="present", time_keys=True):
    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "TEST"
    if time_keys:
        primary.header["TIMESYS"] = "UTC"
        primary.header["TIMEUNIT"] = "s"
    hdus = [primary]
    for name in science_names:
        cols = [fits.Column(name="TIME", format="D", array=[1., 2.])]
        if quality == "present":
            cols.append(fits.Column(name="QUALITY", format="J", array=[0, 1]))
        elif quality == "string":
            cols.append(fits.Column(name="STATUS", format="8A", array=["ok", "bad"]))
        hdus.append(fits.BinTableHDU.from_columns(cols, name=name))
    if gti != "missing":
        if gti == "no_start":
            cols = [fits.Column(name="STOP", format="D", array=[2.])]
        elif gti == "no_stop":
            cols = [fits.Column(name="START", format="D", array=[1.])]
        elif gti == "empty":
            cols = [fits.Column(name="START", format="D", array=[]), fits.Column(name="STOP", format="D", array=[])]
        else:
            values = {"valid": ([1., 3.], [2., 4.]), "bad": ([3., 2., 1., 1., np.nan], [2., 4., 3., 3., 5.])}[gti]
            cols = [fits.Column(name="START", format="D", array=values[0]), fits.Column(name="STOP", format="D", array=values[1])]
        hdus.append(fits.BinTableHDU.from_columns(cols, name="GTI"))
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def test_missing_gti_is_reported_not_treated_as_valid(tmp_path):
    result = audit.process_aditya_l1_level1(_fits(tmp_path / "no-gti.fits", gti="missing"))
    assert result["mode"] == "audit"
    assert not result["gti_extensions"]
    assert "absence does not imply all samples are valid" in result["unresolved_conditions"][0]
    assert result["rejections"]["final_retained"] is None
    assert result["rejections"]["accounting_status"] == "NOT_COMPUTED_AUDIT_MODE_MASKS_NOT_APPLIED"


def test_empty_gti_and_missing_columns_are_structurally_distinct(tmp_path):
    empty = audit.process_aditya_l1_level1(_fits(tmp_path / "empty.fits", gti="empty"))
    missing_start = audit.process_aditya_l1_level1(_fits(tmp_path / "start.fits", gti="no_start"))
    missing_stop = audit.process_aditya_l1_level1(_fits(tmp_path / "stop.fits", gti="no_stop"))
    assert empty["gti_extensions"][0]["interval_count"] == 0
    assert missing_start["gti_extensions"][0]["missing_columns"] == ["START/TSTART"]
    assert missing_stop["gti_extensions"][0]["missing_columns"] == ["STOP/TSTOP"]


def test_gti_bad_interval_counts_nonfinite_reverse_duplicate_overlap_and_sort(tmp_path):
    result = audit.process_aditya_l1_level1(_fits(tmp_path / "gti.fits", gti="bad"))
    gti = result["gti_extensions"][0]
    assert gti["nonfinite_rows"] == 1
    assert gti["reversed_rows"] == 1
    assert gti["duplicate_intervals"] == 1
    assert gti["overlapping_pairs"] >= 1
    assert gti["sorted_by_start"] is False
    assert "no endpoint convention" in gti["semantics"]


def test_tstart_tstop_gti_columns_are_validated_structurally_without_semantic_filtering(tmp_path):
    primary=fits.PrimaryHDU()
    science=fits.BinTableHDU.from_columns([fits.Column(name="TIME",format="D",array=[1.])],name="LIGHTCURVE")
    gti=fits.BinTableHDU.from_columns([fits.Column(name="TSTART",format="D",array=[1.]),fits.Column(name="TSTOP",format="D",array=[2.])],name="GTI_CDTE1")
    path=tmp_path/"tstart-gti.fits";fits.HDUList([primary,science,gti]).writeto(path)
    result=audit.process_aditya_l1_level1(path)
    assert result["gti_extensions"][0]["missing_columns"]==[]
    assert result["gti_extensions"][0]["interval_count"]==1
    assert result["gti_extensions"][0]["malformed_rows"]==0
    assert result["candidate_masks"]["gti"]=="NOT_APPLIED_UNVERIFIED_ENDPOINTS_OR_POLICY"


def test_incomplete_time_reference_and_missing_quality_are_unresolved(tmp_path):
    result = audit.process_aditya_l1_level1(_fits(tmp_path / "metadata.fits", gti="missing", quality="missing", time_keys=False))
    assert any("No recognized time-reference" in x for x in result["unresolved_conditions"])
    assert any("No QUALITY/STATUS" in x for x in result["unresolved_conditions"])


def test_unexpected_status_type_and_multiple_science_extensions_are_reported(tmp_path):
    result = audit.process_aditya_l1_level1(_fits(tmp_path / "multi.fits", science_names=("LIGHTCURVE", "PHA_SPECTRA"), quality="string"))
    assert len(result["science_extensions"]) == 2
    status = [x for h in result["hdus"] for x in h["candidate_columns"] if x["name"] == "STATUS"][0]
    assert status["dtype"].startswith("<U") or status["dtype"].startswith("|S")
    assert status["interpretation"].startswith("CANDIDATE_ONLY")


def test_actual_hel1os_extension_patterns_are_counted_as_science_and_hk_flags_are_candidates(tmp_path):
    primary=fits.PrimaryHDU(); primary.header["INSTRUME"]="HEL1OS"
    lc=fits.BinTableHDU.from_columns([
        fits.Column(name="MJD",format="D",unit="MJD",array=[60000.,60000.]),
        fits.Column(name="CTR",format="D",unit="cts/sec",array=[2.,3.]),
        fits.Column(name="STAT_ERR",format="D",unit="cts/sec",array=[.1,.2]),
    ],name="CDTE1_LC_BAND_5.00KEV_TO_20.00KEV")
    hk=fits.BinTableHDU.from_columns([fits.Column(name="CZT1SATCTR1",format="K",array=[0,1])],name="HLSHK")
    path=tmp_path/"hel1os.fits"; fits.HDUList([primary,lc,hk]).writeto(path)
    result=audit.process_aditya_l1_level1(path)
    assert result["raw_record_count"] == 2
    assert result["science_extensions"][0]["extension_name"].startswith("CDTE1_LC_BAND_")
    saturation=[c for c in result["hdus"][2]["candidate_columns"] if c["name"]=="CZT1SATCTR1"]
    assert saturation[0]["candidate_kind"]=="quality_status"


def test_corrupt_file_and_batch_continues_after_failure(tmp_path):
    (tmp_path / "broken.fits").write_bytes(b"not fits")
    _fits(tmp_path / "valid.fits")
    result = audit.audit_root(tmp_path)
    assert len(result["file_inventory"]) == 2
    assert sum(x["status"] == "INSPECTION_FAILED" for x in result["file_inventory"]) == 1
    assert sum(x["status"] == "AUDITED_STRUCTURALLY_NOT_SCIENCE_READY" for x in result["file_inventory"]) == 1


def test_filtering_refuses_unverified_rules_and_diagnostic_override(tmp_path):
    path = _fits(tmp_path / "filter.fits")
    with pytest.raises(RuntimeError, match="Filtering refused: verified instrument/product rules are absent"):
        audit.process_aditya_l1_level1(path, mode="filtering")
    with pytest.raises(RuntimeError, match="Diagnostic filtering is intentionally not implemented"):
        audit.process_aditya_l1_level1(path, mode="filtering", diagnostic_override="UNVERIFIED_DIAGNOSTIC_ONLY")
    with pytest.raises(RuntimeError, match="audit-only"):
        audit.process_aditya_l1_level1(path, mode="filtering", verified_rules={"gti": "verified"})


def test_hel1os_level1_gti_rules_are_not_inferred_from_filename_or_table(tmp_path):
    path = _fits(tmp_path / "hel1os_lev1.fits", gti="valid")
    result = audit.process_aditya_l1_level1(path)
    assert result["gti_extensions"][0]["semantics"].startswith("STILL_UNVERIFIED")
    assert result["candidate_masks"]["gti"] == "NOT_APPLIED_UNVERIFIED_ENDPOINTS_OR_POLICY"


def test_report_generation_writes_all_machine_readable_outputs(tmp_path):
    _fits(tmp_path / "one.fits")
    report_dir = tmp_path / "reports"
    outputs = audit.write_reports(tmp_path, report_dir)
    assert set(outputs) == {"file_inventory", "hdu_inventory", "gti_validation", "quality_distribution", "summary", "document_evidence", "report"}
    assert all(path.exists() for path in outputs.values())


def test_zip_member_is_audited_without_extraction(tmp_path):
    source = _fits(tmp_path / "inner.fits")
    archive = tmp_path / "products.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(source, "nested/inner.fits")
    source.unlink()
    result = audit.audit_root(tmp_path)
    assert len(result["file_inventory"]) == 1
    assert result["file_inventory"][0]["archive_member"] == "nested/inner.fits"
