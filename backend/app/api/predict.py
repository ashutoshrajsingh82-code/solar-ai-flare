from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.schemas.schemas import PredictRequest
from app.services import predict_service, alert_service
from app.database.db import get_db

router = APIRouter(prefix="/api", tags=["predict"])


@router.post("/predict")
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    try:
        result = predict_service.predict_from_raw_window(req.rows, req.model_id)
        alert_service.maybe_create_alert(
            db, result["nowcast"]["state"], result["forecast"], result["model"], is_demo=True
        )
        return result
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@router.post("/predict/batch")
def predict_batch(requests: list[PredictRequest], db: Session = Depends(get_db)):
    results = []
    for req in requests:
        try:
            results.append(predict_service.predict_from_raw_window(req.rows, req.model_id))
        except Exception as e:
            results.append({"error": str(e)})
    return {"results": results}
