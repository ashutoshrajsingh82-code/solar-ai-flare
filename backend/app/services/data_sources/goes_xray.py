"""NOAA SWPC rolling GOES XRS observations (raw irradiance, no resampling)."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from app.services.data_sources.base import IngestionResult

BASE_URL = "https://services.swpc.noaa.gov/json/goes/primary/"
PRODUCTS = {"6-hour", "1-day", "3-day", "7-day"}
REQUIRED_FIELDS = {"time_tag", "satellite", "flux", "energy"}


def parse_goes_xray_json(payload, filename="goes-xrays.json") -> IngestionResult:
    """Validate and normalize SWPC XRS JSON, preserving raw measurement semantics."""
    if isinstance(payload, (str, bytes, bytearray)):
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError) as exc:
            raise ValueError("GOES X-ray response is not valid JSON") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("GOES X-ray JSON must be a non-empty array of records")
    rows = []
    rejected = {"invalid_timestamp": 0, "invalid_measurement": 0, "invalid_identity": 0}
    for item in payload:
        if not isinstance(item, dict) or not REQUIRED_FIELDS.issubset(item):
            raise ValueError(f"GOES X-ray record must contain {sorted(REQUIRED_FIELDS)}")
        timestamp = pd.to_datetime(item["time_tag"], utc=True, errors="coerce")
        try:
            satellite = int(item["satellite"])
        except (ValueError, TypeError, OverflowError):
            satellite = None
        try:
            flux = float(item["flux"])
        except (ValueError, TypeError, OverflowError):
            flux = np.nan
        if pd.isna(timestamp):
            rejected["invalid_timestamp"] += 1
            continue
        if satellite is None or satellite <= 0:
            rejected["invalid_identity"] += 1
            continue
        if not np.isfinite(flux) or flux < 0:
            rejected["invalid_measurement"] += 1
            continue
        rows.append({
            "timestamp": timestamp,
            "satellite": satellite,
            "energy": str(item["energy"]).strip(),
            "flux_w_m2": flux,
            "observed_flux": _optional_number(item.get("observed_flux")),
            "electron_correction": item.get("electron_correction"),
            "raw_record": item,
        })
    if not rows:
        raise ValueError(f"GOES X-ray response contains no usable records: {rejected}")
    data = pd.DataFrame(rows).sort_values(["timestamp", "satellite", "energy"], kind="stable")
    duplicate_count = int(data.duplicated(["timestamp", "satellite", "energy"], keep="first").sum())
    data = data.drop_duplicates(["timestamp", "satellite", "energy"], keep="first").reset_index(drop=True)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    report = {
        "status": "validated_with_rejections" if any(rejected.values()) else "validated",
        "source": "NOAA SWPC GOES XRS primary rolling JSON",
        "filename": filename,
        "content_sha256": digest,
        "time_standard": "UTC; SWPC time_tag parsed as UTC",
        "measurement": "SWPC-reported XRS irradiance in W m-2 (flux); raw values preserved",
        "channels": sorted(data["energy"].unique().tolist()),
        "satellites": sorted(int(v) for v in data["satellite"].unique()),
        "coverage_start": data["timestamp"].min().isoformat(),
        "coverage_end": data["timestamp"].max().isoformat(),
        "input_rows": len(payload), "usable_rows": len(data),
        "duplicate_rows_removed": duplicate_count,
        "rejected_rows": int(sum(rejected.values())), "rejection_reasons": rejected,
        "cadence_by_satellite_channel_seconds": _cadence_summary(data),
        "limitations": [
            "This is a rolling SWPC feed, not a historical archive; retrieve archived products for earlier observation periods.",
            "GOES-R and earlier satellite irradiance calibrations differ; satellite identity is retained and values are not cross-calibrated here.",
            "These observations are stored independently; they are not mapped onto SoLEXS/HEL1OS model features.",
            "The SWPC JSON schema is validated against required identity, time, channel and flux fields; additional fields are retained in raw_record.",
        ],
    }
    return IngestionResult(data, report)


def acquire_goes_xray(product="7-day", destination=None, *, opener=None, sleep=time.sleep):
    """Fetch a public rolling SWPC product with bounded retries and immutable bytes."""
    if product not in PRODUCTS:
        raise ValueError(f"product must be one of {sorted(PRODUCTS)}")
    path = f"{product.replace('-', '-')}.json"
    url = BASE_URL + f"xrays-{path}"
    request = Request(url, headers={"User-Agent": "solar-ai-flare/real-data-ingestor"})
    open_url = opener or urlopen
    error = None
    for attempt in range(3):
        try:
            with open_url(request, timeout=20) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    raise ValueError(f"NOAA returned HTTP {status}")
                raw = response.read(100 * 1024 * 1024 + 1)
            if len(raw) > 100 * 1024 * 1024:
                raise ValueError("NOAA GOES response exceeds 100 MiB limit")
            parsed = parse_goes_xray_json(raw, f"xrays-{product}.json")
            if destination is None:
                raise ValueError("A raw-data destination is required")
            destination = Path(destination)
            destination.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(raw).hexdigest()
            final = destination / f"{digest[:16]}-xrays-{product}.json"
            if not final.exists():
                temp = final.with_suffix(final.suffix + ".tmp")
                temp.write_bytes(raw)
                temp.replace(final)
            report = dict(parsed.report)
            report.update({"source_url": url, "raw_file": final.name, "raw_file_sha256": digest,
                           "raw_file_size_bytes": len(raw), "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
                           "product_window": product, "status": "completed"})
            manifest = final.with_suffix(".manifest.json")
            manifest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            return IngestionResult(parsed.data, report)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            error = exc
            if isinstance(exc, ValueError) and not isinstance(exc, (HTTPError, URLError)):
                # A malformed successful response will not improve by retrying.
                if "NOAA returned HTTP" not in str(exc):
                    break
            if attempt < 2:
                sleep(0.5 * (2 ** attempt))
    raise RuntimeError(f"NOAA GOES X-ray acquisition failed after 3 attempts: {error}") from error


def _optional_number(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _cadence_summary(data):
    result = {}
    for (satellite, energy), group in data.groupby(["satellite", "energy"]):
        delta = group["timestamp"].sort_values().diff().dt.total_seconds().dropna()
        result[f"{satellite}:{energy}"] = float(delta.median()) if not delta.empty else None
    return result
