import json

import pandas as pd
import pytest

from app.services.data_sources.goes_xray import acquire_goes_xray, parse_goes_xray_json


def payload():
    return [
        {"time_tag": "2026-10-21T10:00:00Z", "satellite": 18, "flux": 2.1e-6,
         "energy": "0.1-0.8nm", "observed_flux": 2.0e-6, "electron_correction": False},
        {"time_tag": "2026-10-21T10:00:00Z", "satellite": 18, "flux": 1.2e-7,
         "energy": "0.05-0.4nm", "observed_flux": 1.1e-7, "electron_correction": False},
        {"time_tag": "2026-10-21T10:01:00-00:00", "satellite": 17, "flux": 1e-6,
         "energy": "0.1-0.8nm"},
    ]


def test_parse_goes_xrs_preserves_utc_satellite_channels_and_raw_flux():
    result = parse_goes_xray_json(payload())
    assert len(result.data) == 3
    assert isinstance(result.data.timestamp.dtype, pd.DatetimeTZDtype)
    assert str(result.data.timestamp.dt.tz) == "UTC"
    assert set(result.data.satellite) == {17, 18}
    assert set(result.data.energy) == {"0.1-0.8nm", "0.05-0.4nm"}
    long_channel = result.data.loc[result.data.energy == "0.1-0.8nm"].iloc[0]
    assert long_channel.flux_w_m2 == pytest.approx(2.1e-6)
    assert result.report["measurement"].endswith("raw values preserved")
    assert result.report["satellites"] == [17, 18]


def test_parse_goes_xrs_rejects_unknown_or_malformed_schema():
    with pytest.raises(ValueError, match="must contain"):
        parse_goes_xray_json([{"time_tag": "2026-01-01T00:00:00Z"}])
    with pytest.raises(ValueError, match="no usable records"):
        parse_goes_xray_json([{**payload()[0], "flux": -1}])


def test_acquisition_validates_and_saves_raw_bytes_and_manifest(tmp_path):
    raw = json.dumps(payload()).encode()

    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, limit): return raw

    result = acquire_goes_xray("1-day", tmp_path, opener=lambda *_args, **_kwargs: Response(), sleep=lambda _: None)
    raw_path = tmp_path / result.report["raw_file"]
    manifest_path = raw_path.with_suffix(".manifest.json")
    assert raw_path.read_bytes() == raw
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["raw_file_sha256"] == result.report["raw_file_sha256"]
    assert pd.to_datetime(result.report["coverage_start"], utc=True) == pd.Timestamp("2026-10-21T10:00:00Z")
