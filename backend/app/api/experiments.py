from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.schemas.schemas import TrainRequest
from app.services import model_service
from app.database.db import get_db
from app.database.models import ExperimentRecord

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.get("/fusion")
def fusion_experiment(
    model_type: str = "random_forest",
    epochs: int = 8,
    sequence_length: int = 30,
    db: Session = Depends(get_db),
):
    try:
        base_req = TrainRequest(model_type=model_type, epochs=epochs, sequence_length=sequence_length)
        result = model_service.run_fusion_experiment(base_req)

        for config, metrics in result["results"].items():
            if "error" not in metrics:
                record = ExperimentRecord(
                    experiment_name="fusion_experiment", input_configuration=config,
                    model_architecture=model_type, metrics=metrics, dataset_id=None, is_demo=True,
                )
                db.add(record)
        db.commit()
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fusion experiment failed: {e}")
