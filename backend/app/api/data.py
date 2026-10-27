import hashlib
import logging
import os
import uuid
from pathlib import Path
from fastapi import Request
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.services import data_service
from app.services.data_sources.goes_xray import acquire_goes_xray, PRODUCTS as GOES_XRAY_PRODUCTS
from app.services.scientific_readiness import generate_scientific_readiness_report
from app.core.config import settings

router = APIRouter(prefix="/api/data", tags=["data"])
logger = logging.getLogger("solar-flare-ai")
CHUNK_SIZE = 1024 * 1024


def _upload_root():
    root = Path(settings.upload_storage_dir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    root = root.resolve()
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    return root, raw


def _safe_original_name(filename):
    name = str(filename or "").replace("\\", "/").split("/")[-1]
    if not name or name in {".", ".."}:
        raise ValueError("A valid filename is required.")
    return name


async def _store_upload(file: UploadFile, allowed_suffixes, request: Request = None):
    original = _safe_original_name(file.filename)
    lower = original.lower()
    suffix = next((candidate for candidate in sorted(allowed_suffixes, key=len, reverse=True)
                   if lower.endswith(candidate)), None)
    if suffix is None:
        raise ValueError(f"Unsupported file type. Allowed: {', '.join(sorted(allowed_suffixes))}.")
    if request is not None:
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.max_upload_bytes + 1024 * 1024:
            raise ValueError("Upload exceeds the configured maximum file size.")
    root, raw = _upload_root()
    temporary = root / f".upload-{uuid.uuid4().hex}.tmp"
    final = raw / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    digest = hashlib.sha256()
    try:
        with temporary.open("xb") as output:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise ValueError("Upload exceeds the configured maximum file size.")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if size == 0:
            raise ValueError("Uploaded file is empty.")
        if suffix == ".csv":
            with temporary.open("rb") as stream:
                sample = stream.read(min(size, 8192))
            if b"\x00" in sample:
                raise ValueError("CSV upload contains binary data.")
            try:
                sample.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError("CSV must use UTF-8 or UTF-8 with BOM encoding.") from exc
        elif suffix == ".fits":
            with temporary.open("rb") as stream:
                if stream.read(9) != b"SIMPLE  =":
                    raise ValueError("FITS file has an invalid primary-header signature.")
        elif suffix == ".fits.gz":
            with temporary.open("rb") as stream:
                if stream.read(2) != b"\x1f\x8b":
                    raise ValueError("Compressed FITS file has an invalid gzip signature.")
        os.replace(temporary, final)
        return {"path": str(final), "storage_key": f"raw/{final.name}", "original_filename": original,
                "content_sha256": digest.hexdigest(), "size_bytes": size,
                "file_format": suffix.lstrip(".").replace(".gz", "+gzip")}
    except Exception:
        temporary.unlink(missing_ok=True)
        final.unlink(missing_ok=True)
        raise


@router.post("/load-demo")
def load_demo(db: Session = Depends(get_db)):
    try:
        meta = data_service.load_demo_data(db)
        return {**meta, "source_used": "demo", "is_demo": True}
    except Exception as e:
        logger.exception("Demo dataset load failed")
        raise HTTPException(status_code=500, detail="Demo dataset could not be loaded; check server logs for diagnostics.") from e


@router.post("/upload")
async def upload_dataset(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    stored = None
    try:
        stored = await _store_upload(file, {".csv"}, request)
        meta = data_service.load_uploaded_csv(
            db, stored["path"], stored["original_filename"], upload_metadata=stored,
        )
        return {**meta, "source_used": "upload", "is_demo": False}
    except ValueError as e:
        if stored:
            Path(stored["path"]).unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        if stored:
            Path(stored["path"]).unlink(missing_ok=True)
        logger.exception("Dataset upload failed")
        raise HTTPException(status_code=500, detail="Dataset upload could not be processed; check server logs for the diagnostic ID.") from e
    finally:
        await file.close()


@router.post("/upload-real")
async def upload_real_product(
    request: Request,
    instrument: str = Query(..., description="solexs or hel1os"),
    product: str = Query(..., description="Supported product type, e.g. lc"),
    file: UploadFile = File(...), db: Session = Depends(get_db),
):
    """Ingest a local product file. This route never falls back to demo data."""
    stored = None
    try:
        if instrument.lower() != "solexs":
            raise ValueError("Only validated SoLEXS FITS light-curve ingestion is currently supported.")
        stored = await _store_upload(file, {".fits", ".fits.gz"}, request)
        meta = data_service.load_real_product(
            db, stored["path"], instrument, product, filename=stored["original_filename"],
            upload_metadata=stored,
        )
        return {**meta, "source_used": instrument.lower(), "is_demo": False}
    except NotImplementedError as exc:
        if stored:
            Path(stored["path"]).unlink(missing_ok=True)
        raise HTTPException(status_code=501, detail=str(exc))
    except FileNotFoundError as exc:
        if stored:
            Path(stored["path"]).unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Uploaded source file could not be read.") from exc
    except ValueError as exc:
        if stored:
            Path(stored["path"]).unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        if stored:
            Path(stored["path"]).unlink(missing_ok=True)
        logger.exception("Real product ingestion failed")
        raise HTTPException(status_code=500, detail="Real product ingestion failed; check server logs for diagnostics.") from exc
    finally:
        await file.close()


@router.post("/upload-goes-labels")
async def upload_goes_reference_labels(
    request: Request,
    coverage_start: str = Query(..., description="UTC start of complete GOES report coverage"),
    coverage_end: str = Query(..., description="UTC end of complete GOES report coverage"),
    file: UploadFile = File(...), db: Session = Depends(get_db),
):
    """Attach local NOAA/NCEI GOES event labels to the already loaded real dataset."""
    stored = None
    try:
        stored = await _store_upload(file, {".csv"}, request)
        return data_service.attach_goes_reference_labels(
            db, stored["path"], stored["original_filename"], coverage_start, coverage_end,
            upload_metadata=stored,
        )
    except FileNotFoundError as exc:
        if stored:
            Path(stored["path"]).unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Uploaded GOES report could not be read.") from exc
    except ValueError as exc:
        if stored:
            Path(stored["path"]).unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        if stored:
            Path(stored["path"]).unlink(missing_ok=True)
        logger.exception("GOES reference-label ingestion failed")
        raise HTTPException(status_code=500, detail="GOES reference-label ingestion failed; check server logs for diagnostics.") from exc
    finally:
        await file.close()


@router.post("/acquire-goes-xray")
def acquire_public_goes_xray(product: str = Query("7-day", description="NOAA SWPC rolling product: 6-hour, 1-day, 3-day, or 7-day")):
    """Acquire and persist a verified public NOAA SWPC rolling XRS product."""
    if product not in GOES_XRAY_PRODUCTS:
        raise HTTPException(status_code=422, detail=f"product must be one of {sorted(GOES_XRAY_PRODUCTS)}")
    _, raw = _upload_root()
    destination = raw / "goes_xray"
    try:
        result = acquire_goes_xray(product, destination)
        return {**result.report, "rows": len(result.data), "is_demo": False}
    except (RuntimeError, ValueError) as exc:
        logger.warning("GOES X-ray acquisition failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/info")
def data_info():
    from app.services.pipeline_state import pipeline_state
    if pipeline_state.raw_df is None:
        if settings.data_mode == "real":
            raise HTTPException(
                status_code=404,
                detail="No real dataset loaded. Upload a local SoLEXS product with /api/data/upload-real; real mode does not load demo data.",
            )
        raise HTTPException(status_code=404, detail="No dataset loaded yet.")
    meta = pipeline_state.dataset_meta
    return {**meta, "source_used": meta.get("source", "unknown"), "is_demo": meta.get("source") == "demo"}


@router.get("/readiness")
def scientific_data_readiness():
    """Persist and return a machine-readable scientific-training readiness audit."""
    return generate_scientific_readiness_report()


@router.get("/preview")
def data_preview(n: int = 50):
    return data_service.preview_dataset(n)


@router.get("/summary")
def data_summary():
    return data_service.dataset_summary_stats()
