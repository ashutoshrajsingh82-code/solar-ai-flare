import os
import shutil
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
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
        return meta
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
        return meta
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {e}")


@router.get("/info")
def data_info():
    from app.services.pipeline_state import pipeline_state
    if pipeline_state.raw_df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded yet.")
    return pipeline_state.dataset_meta


@router.get("/preview")
def data_preview(n: int = 50):
    return data_service.preview_dataset(n)


@router.get("/summary")
def data_summary():
    return data_service.dataset_summary_stats()
