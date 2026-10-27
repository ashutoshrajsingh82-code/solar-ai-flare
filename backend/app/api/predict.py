from fastapi import APIRouter, HTTPException, Depends
import logging
import uuid
import datetime as dt
from sqlalchemy.orm import Session

from app.schemas.schemas import PredictRequest, PredictResponse
from app.services import predict_service, alert_service
from app.database.db import get_db
from app.database.models import InferenceRecord

router = APIRouter(prefix="/api", tags=["predict"])
logger = logging.getLogger("solar-flare-ai")


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    request_id = str(uuid.uuid4())
    try:
        result = predict_service.predict_from_raw_window(req.rows, req.model_id)
        db.add(InferenceRecord(
            timestamp=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
            model_id=result["model_id"], nowcast_state=result["nowcast"]["state"],
            nowcast_confidence=float(result["nowcast"]["confidence"]),
            forecast=result["forecast"], risk=result["risk"],
            source_classification="api_prediction",
            observation_timestamp=dt.datetime.fromisoformat(result["forecast_origin_timestamp_utc"].replace("Z", "+00:00")).astimezone(dt.timezone.utc).replace(tzinfo=None),
            forecast_issue_timestamp=dt.datetime.fromisoformat(result["forecast_origin_timestamp_utc"].replace("Z", "+00:00")).astimezone(dt.timezone.utc).replace(tzinfo=None),
            processing_status="complete",
            # Keep an operational summary, never the raw observation payload.
            input_snapshot={key: value for key, value in result.items() if key != "input"},
        ))
        db.commit()
        alert_service.maybe_create_alert(
            db, result["nowcast"]["state"], result["forecast"], result["model"],
            is_demo=result.get("model_status") == "demo_only",
            forecast_decisions=result.get("forecast_decisions"),
            forecast_thresholds=result.get("forecast_thresholds"),
            model_id=result.get("model_id"), model_version=result.get("model_version"),
            prediction_id=result.get("prediction_id"), source_classification="api_prediction",
            observation_timestamp=result.get("forecast_origin_timestamp_utc"),
            forecast_issue_timestamp=result.get("forecast_origin_timestamp_utc"),
        )
        logger.info("prediction_completed request_id=%s prediction_id=%s model_id=%s latency_ms=%s",
                    request_id, result["prediction_id"], result["model_id"], result["inference_latency_ms"])
        return result
    except predict_service.ModelNotFoundError as e:
        logger.warning("prediction_model_not_found request_id=%s", request_id)
        raise HTTPException(status_code=404, detail=str(e), headers={"X-Request-ID": request_id})
    except predict_service.ModelUnavailableError as e:
        logger.warning("prediction_model_unavailable request_id=%s", request_id)
        raise HTTPException(status_code=503, detail=str(e), headers={"X-Request-ID": request_id})
    except (ValueError, KeyError) as e:
        logger.warning("prediction_invalid_request request_id=%s reason=%s", request_id, str(e))
        raise HTTPException(status_code=422, detail=str(e), headers={"X-Request-ID": request_id})
    except Exception as e:
        logger.exception("Prediction request %s failed", request_id)
        db.rollback()
        raise HTTPException(status_code=500, detail="Prediction failed while loading, running, or persisting the result.",
                            headers={"X-Request-ID": request_id})


@router.get("/predict/{prediction_id}", response_model=PredictResponse)
def get_prediction(prediction_id: str, db: Session = Depends(get_db)):
    record = db.query(InferenceRecord).filter(
        InferenceRecord.input_snapshot["prediction_id"].as_string() == prediction_id
    ).one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction result not found.")
    return {**record.input_snapshot, "input": {}}


@router.post("/predict/batch")
def predict_batch(requests: list[PredictRequest], db: Session = Depends(get_db)):
    results = []
    for req in requests:
        try:
            results.append(predict_service.predict_from_raw_window(req.rows, req.model_id))
        except predict_service.ModelNotFoundError as e:
            results.append({"error": str(e), "status_code": 404})
        except predict_service.ModelUnavailableError as e:
            results.append({"error": str(e), "status_code": 503})
        except (ValueError, KeyError) as e:
            results.append({"error": str(e)})
        except Exception:
            logger.exception("Batch prediction failed")
            results.append({"error": "Prediction failed while loading or running the selected model."})
    return {"results": results}
