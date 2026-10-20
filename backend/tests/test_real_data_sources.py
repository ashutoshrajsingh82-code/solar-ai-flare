import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.services.data_sources.solexs import read_solexs_l1_light_curve
from app.services.data_sources.alignment import align_observations
from app.services.data_sources.hel1os import read_hel1os_product
from app.services.data_sources.goes_labels import read_goes_flare_report_csv, label_observations_from_goes
from app.services.preprocess_service import generate_split_scoped_labels
from app.services.pipeline_state import PipelineState
from app.database.db import Base
from app.database.models import Dataset
from app.services import data_service
from app.services import preprocess_service
from app.schemas.schemas import PreprocessRequest
from ml.features.labels import LabelConfig


def write_solexs_lc(path, times=(1_700_000_000, 1_700_000_001, 1_700_000_002), counts=(2.0, 3.0, 4.0)):
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "ADITYA-L1"
    primary.header["INSTRUME"] = "SoLEXS"
    rate = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", unit="s", array=np.asarray(times, dtype=float)),
        fits.Column(name="COUNTS", format="D", unit="counts/s", array=np.asarray(counts, dtype=float)),
    ], name="RATE")
    fits.HDUList([primary, rate]).writeto(path)


def test_solexs_l1_light_curve_valid_and_gap_report(tmp_path):
    path = tmp_path / "sample.fits"
    write_solexs_lc(path, times=(1_700_000_000, 1_700_000_001, 1_700_000_004), counts=(2, 3, 4))

    result = read_solexs_l1_light_curve(path)

    assert list(result.data.columns) == ["timestamp", "original_time_unix_seconds", "solexs_counts_per_second"]
    assert str(result.data.timestamp.dt.tz) == "UTC"
    assert result.report["source"] == "solexs"
    assert result.report["accepted_rows"] == 3
    assert result.report["missing_intervals"][0]["missing_intervals_estimate"] == 2


@pytest.mark.parametrize("times,counts,message", [
    ((1_700_000_000, 1_700_000_000), (1, 2), "duplicate"),
    ((1_700_000_001, 1_700_000_000), (1, 2), "out of order"),
])
def test_solexs_rejects_duplicate_and_out_of_order_timestamps(tmp_path, times, counts, message):
    path = tmp_path / "bad.fits"
    write_solexs_lc(path, times=times, counts=counts)
    with pytest.raises(ValueError, match=message):
        read_solexs_l1_light_curve(path)


def test_solexs_rejects_missing_required_field(tmp_path):
    path = tmp_path / "missing.fits"
    rate = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", unit="s", array=[1_700_000_000.0]),
    ], name="RATE")
    fits.HDUList([fits.PrimaryHDU(), rate]).writeto(path)
    with pytest.raises(ValueError, match="TIME and COUNTS"):
        read_solexs_l1_light_curve(path)


def test_real_reader_missing_or_truncated_files_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_solexs_l1_light_curve(tmp_path / "missing.fits")
    truncated = tmp_path / "truncated.fits"
    truncated.write_bytes(b"not a FITS product")
    with pytest.raises(ValueError, match="malformed or truncated"):
        read_solexs_l1_light_curve(truncated)


def test_solexs_reports_rejected_measurements_without_interpolation(tmp_path):
    path = tmp_path / "measurements.fits"
    write_solexs_lc(path, counts=(2.0, np.nan, -1.0))
    result = read_solexs_l1_light_curve(path)
    assert result.report["rejected_rows"] == 2
    assert result.report["accepted_rows"] == 1
    assert result.data["solexs_counts_per_second"].tolist() == [2.0]


def test_split_scoped_labels_do_not_see_future_split_event():
    timestamps = pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "solexs_flux": np.ones(12),
        "helios_flux": np.ones(12),
        "state": [0] * 11 + [2],
        "flare_class": ["-"] * 11 + ["X"],
    })
    train = df.iloc[:8].copy()
    validation = df.iloc[8:10].copy()
    test = df.iloc[10:].copy()
    config = LabelConfig(cadence_seconds=3600, preflare_minutes=0, horizons_hours=[1])

    labeled, purged = generate_split_scoped_labels(
        df, {"train": train, "validation": validation, "test": test}, config,
    )

    # The test-period flare cannot create a positive target in training, and
    # the unobserved final forecast hour of every split is excluded.
    assert labeled.loc[:7, "forecast_1h"].dropna().eq(0).all()
    assert pd.isna(labeled.loc[7, "forecast_1h"])
    assert labeled.loc[10, "forecast_1h"] == 1
    assert purged["train"]["forecast_1h"] == 1


def test_alignment_preserves_gaps_and_never_interpolates():
    solexs = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01T00:00:01Z", "2026-01-01T00:00:03Z"]),
        "solexs_counts_per_second": [2.0, 4.0],
    })
    hel1os = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01T00:00:02Z", "2026-01-01T00:00:03Z"]),
        "hel1os_counts_per_second": [8.0, 9.0],
    })

    aligned, report = align_observations(solexs, hel1os, cadence_seconds=1, aggregation="mean")

    middle = aligned.loc[aligned.timestamp == pd.Timestamp("2026-01-01T00:00:02Z")].iloc[0]
    assert pd.isna(middle["solexs_solexs_counts_per_second"])
    assert middle["hel1os_hel1os_counts_per_second"] == 8.0
    assert report["missing_bins"]["solexs"] >= 1
    assert report["interpolation"] == "none"


def test_hel1os_reader_is_fail_closed():
    with pytest.raises(NotImplementedError, match="No demo fallback"):
        read_hel1os_product("not-used.fits")


def test_real_ingestion_persists_provenance_only_after_validation(tmp_path, monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    state = PipelineState(tmp_path / "artifacts", restore=False)
    monkeypatch.setattr(data_service, "pipeline_state", state)
    source_file = tmp_path / "official-name.fits"
    write_solexs_lc(source_file)

    try:
        meta = data_service.load_real_product(
            session, str(source_file), "solexs", "lc", filename="official-name.fits",
        )
        stored = session.query(Dataset).one()
        assert meta["source"] == "solexs"
        assert meta["dataset_id"] == stored.id
        assert stored.source == "solexs_real"
        assert stored.source_metadata["file_sha256"] == meta["file_sha256"]
        assert state.raw_df["solexs_counts_per_second"].tolist() == [2.0, 3.0, 4.0]
        assert "solexs_flux" not in state.raw_df
    finally:
        session.close()
        engine.dispose()


def test_real_native_counts_are_blocked_from_flux_pipeline(monkeypatch, tmp_path):
    state = PipelineState(tmp_path / "unused", restore=False)
    state.raw_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=2, freq="s", tz="UTC"),
        "solexs_counts_per_second": [1.0, 2.0],
    })
    state.dataset_meta = {"source": "solexs"}
    monkeypatch.setattr(preprocess_service, "pipeline_state", state)

    with pytest.raises(ValueError, match="do not match the existing dual-flux pipeline contract"):
        preprocess_service.run_full_pipeline(PreprocessRequest())


def test_unknown_uploaded_labels_are_not_treated_as_quiet(monkeypatch, tmp_path):
    state = PipelineState(tmp_path / "unused", restore=False)
    state.raw_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=2, freq="min", tz="UTC"),
        "solexs_flux": [1.0, 2.0], "helios_flux": [3.0, 4.0], "state": [np.nan, np.nan],
    })
    state.dataset_meta = {"source": "upload"}
    monkeypatch.setattr(preprocess_service, "pipeline_state", state)

    with pytest.raises(ValueError, match="labels are missing or unknown"):
        preprocess_service.run_full_pipeline(PreprocessRequest())


def write_goes_report(path, rows=None):
    rows = rows or [{
        "flare_id": "event-1", "start_time": "2026-01-01 00:05:00",
        "time": "2026-01-01 00:06:00", "end_time": "2026-01-01 00:08:00",
        "flare_class": "M1.2", "xrsb_irrad": 1.2e-5,
    }]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_goes_event_csv_and_coverage_aware_labels(tmp_path):
    path = tmp_path / "sci_xrsf-l2-flrpt_geo_y2026_v1-0-1.csv"
    write_goes_report(path)
    parsed = read_goes_flare_report_csv(path, "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")
    observations = pd.DataFrame({"timestamp": pd.to_datetime([
        "2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z", "2026-01-01T00:06:00Z",
        "2026-01-01T23:30:00Z",
    ])})

    labels, report = label_observations_from_goes(
        observations, parsed.data, "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", horizons_hours=(1,),
    )

    assert parsed.report["product_version"] == "v1-0-1"
    assert labels["forecast_1h"].iloc[0] == 1
    assert labels["forecast_1h"].iloc[1] == 0  # target interval is (t, t+h]
    assert labels["forecast_1h"].iloc[2] == 0
    assert np.isnan(labels["forecast_1h"].iloc[3])  # full forecast window exceeds known coverage
    assert report["horizon_counts"]["1h"]["unknown_rows"] == 1


def test_goes_reader_rejects_incomplete_schema_and_reports_invalid_rows(tmp_path):
    missing = tmp_path / "missing.csv"
    pd.DataFrame({"time": ["2026-01-01"]}).to_csv(missing, index=False)
    with pytest.raises(ValueError, match="missing required fields"):
        read_goes_flare_report_csv(missing, "2026-01-01", "2026-01-02")

    malformed = tmp_path / "malformed.csv"
    write_goes_report(malformed, rows=[{
        "flare_id": "event-2", "start_time": "bad", "time": "2026-01-01 00:06:00",
        "end_time": "2026-01-01 00:08:00", "flare_class": "M1.2",
    }])
    with pytest.raises(ValueError, match="no valid events"):
        read_goes_flare_report_csv(malformed, "2026-01-01", "2026-01-02")


def test_goes_labels_are_attached_to_dataset_with_independent_provenance(tmp_path, monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    state = PipelineState(tmp_path / "artifacts", restore=False)
    monkeypatch.setattr(data_service, "pipeline_state", state)
    source_file = tmp_path / "solexs.fits"
    write_solexs_lc(source_file)
    try:
        data_service.load_real_product(session, str(source_file), "solexs", "lc")
        event_start = pd.Timestamp(1_700_000_001, unit="s", tz="UTC")
        event = {
            "flare_id": "event-3", "start_time": event_start.strftime("%Y-%m-%d %H:%M:%S"),
            "time": (event_start + pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": (event_start + pd.Timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "flare_class": "X1.0", "xrsb_irrad": 1e-4,
        }
        goes_file = tmp_path / "goes.csv"
        write_goes_report(goes_file, rows=[event])

        result = data_service.attach_goes_reference_labels(
            session, str(goes_file), "goes.csv", "2023-11-14T00:00:00Z", "2023-11-16T00:00:00Z",
        )

        dataset = session.query(Dataset).one()
        assert result["is_demo"] is False
        assert state.raw_df["forecast_1h"].iloc[0] == 1
        assert state.raw_df["forecast_1h"].iloc[1] == 0
        assert dataset.source_metadata["reference_labels"]["source"] == "NOAA/NCEI GOES XRS L2 Flare Report"
    finally:
        session.close()
        engine.dispose()
def test_reference_labels_preserved_and_unknowns_stay_unknown():
    timestamps = pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "solexs_flux": np.ones(12),
        "helios_flux": np.ones(12),
        "state": [2] * 12,
        "forecast_1h": [1, 0, np.nan, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    train = df.iloc[:8].copy()
    validation = df.iloc[8:10].copy()
    test = df.iloc[10:].copy()
    config = LabelConfig(
        cadence_seconds=3600,
        preflare_minutes=0,
        horizons_hours=[1],
    )

    labeled, purged = generate_split_scoped_labels(
        df,
        {"train": train, "validation": validation, "test": test},
        config,
        preserve_reference_labels=True,
    )

    assert labeled.loc[0, "forecast_1h"] == 1
    assert labeled.loc[1, "forecast_1h"] == 0
    assert pd.isna(labeled.loc[2, "forecast_1h"])
    assert labeled.loc[3, "forecast_1h"] == 1
    assert labeled.loc[4, "forecast_1h"] == 0

    assert pd.isna(labeled.loc[7, "forecast_1h"])
    assert pd.isna(labeled.loc[9, "forecast_1h"])
    assert pd.isna(labeled.loc[11, "forecast_1h"])
    assert purged["train"]["forecast_1h"] == 1
    assert purged["validation"]["forecast_1h"] == 1
    assert purged["test"]["forecast_1h"] == 1
def test_reference_labels_preserved_and_unknowns_stay_unknown():
    timestamps = pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "solexs_flux": np.ones(12),
        "helios_flux": np.ones(12),
        "state": [2] * 12,
        "forecast_1h": [1, 0, np.nan, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    train = df.iloc[:8].copy()
    validation = df.iloc[8:10].copy()
    test = df.iloc[10:].copy()
    config = LabelConfig(
        cadence_seconds=3600,
        preflare_minutes=0,
        horizons_hours=[1],
    )

    labeled, purged = generate_split_scoped_labels(
        df,
        {"train": train, "validation": validation, "test": test},
        config,
        preserve_reference_labels=True,
    )

    assert labeled.loc[0, "forecast_1h"] == 1
    assert labeled.loc[1, "forecast_1h"] == 0
    assert pd.isna(labeled.loc[2, "forecast_1h"])
    assert labeled.loc[3, "forecast_1h"] == 1
    assert labeled.loc[4, "forecast_1h"] == 0

    assert pd.isna(labeled.loc[7, "forecast_1h"])
    assert pd.isna(labeled.loc[9, "forecast_1h"])
    assert pd.isna(labeled.loc[11, "forecast_1h"])
    assert purged["train"]["forecast_1h"] == 1
    assert purged["validation"]["forecast_1h"] == 1
    assert purged["test"]["forecast_1h"] == 1
