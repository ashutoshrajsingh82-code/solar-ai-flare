import os
import shutil
import tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.services import data_service
from app.core.config import settings

router = APIRouter(prefix="/api/data", tags=["data"])

UPLOAD_DIR = "./uploaded_data"


@router.post("/load-demo")
def load_demo(db: Session = Depends(get_db)):
    try:
        meta = data_service.load_demo_data(db)
        return {**meta, "source_used": "demo", "is_demo": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        meta = data_service.load_uploaded_csv(db, dest_path, file.filename)
        return {**meta, "source_used": "upload", "is_demo": False}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {e}")


@router.post("/upload-real")
async def upload_real_product(
    instrument: str = Query(..., description="solexs or hel1os"),
    product: str = Query(..., description="Supported product type, e.g. lc"),
    file: UploadFile = File(...), db: Session = Depends(get_db),
):
    """Ingest a local product file. This route never falls back to demo data."""
    suffix = ".fits.gz" if (file.filename or "").lower().endswith(".fits.gz") else Path(file.filename or "").suffix.lower()
    if instrument.lower() == "solexs" and suffix not in {".fits", ".fits.gz"}:
        raise HTTPException(status_code=422, detail="SoLEXS L1 light-curve upload must be .fits or .fits.gz")
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(prefix="solar-real-", suffix=suffix, delete=False) as target:
            temp_name = target.name
            shutil.copyfileobj(file.file, target)
        meta = data_service.load_real_product(db, temp_name, instrument, product, filename=file.filename)
        return {**meta, "source_used": instrument.lower(), "is_demo": False}
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Real product ingestion failed: {exc}")
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


@router.post("/upload-goes-labels")
async def upload_goes_reference_labels(
    coverage_start: str = Query(..., description="UTC start of complete GOES report coverage"),
    coverage_end: str = Query(..., description="UTC end of complete GOES report coverage"),
    file: UploadFile = File(...), db: Session = Depends(get_db),
):
    """Attach local NOAA/NCEI GOES event labels to the already loaded real dataset."""
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="GOES flare report must be uploaded as CSV")
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(prefix="solar-goes-", suffix=".csv", delete=False) as target:
            temp_name = target.name
            shutil.copyfileobj(file.file, target)
        return data_service.attach_goes_reference_labels(
            db, temp_name, file.filename, coverage_start, coverage_end,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GOES reference-label ingestion failed: {exc}")
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


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


@router.get("/preview")
def data_preview(n: int = 50):
    return data_service.preview_dataset(n)


@router.get("/summary")
def data_summary():
    return data_service.dataset_summary_stats()
