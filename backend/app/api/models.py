from fastapi import APIRouter, HTTPException
from app.schemas.schemas import TrainRequest, ModelSelectRequest
from app.services import model_service

router = APIRouter(prefix="/api/models", tags=["models"])


@router.post("/train")
def train(req: TrainRequest):
    try:
        return model_service.train_model(req)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")


@router.get("")
def list_models():
    return model_service.list_models()


@router.post("/select")
def select_model(req: ModelSelectRequest):
    try:
        return model_service.set_active_model(req.model_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{model_id}/metrics")
def model_metrics(model_id: str):
    try:
        return model_service.get_model_metrics(model_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
