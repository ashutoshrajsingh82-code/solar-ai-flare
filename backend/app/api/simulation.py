from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.database.models import InferenceRecord, SimulationReplayRecord
from app.schemas.schemas import SimulationStartRequest
from app.services import simulation_service
from app.services.simulation_service import ReplayNotReadyError
from ml.inference.simulator import live_simulator

router = APIRouter(prefix="/api/simulation", tags=["simulation"])


@router.get("/readiness")
def replay_readiness(model_id: str | None = None):
    return simulation_service.assess_replay_readiness(model_id)


@router.post("/start")
def start_simulation(req: SimulationStartRequest):
    try:
        return simulation_service.start_replay(
            req.model_id, speed=req.speed, dataset_id=req.dataset_id,
            replay_start_utc=req.replay_start_utc, replay_end_utc=req.replay_end_utc,
        )
    except ReplayNotReadyError as exc:
        raise HTTPException(status_code=409, detail={"code": "replay_not_ready", "message": str(exc), "blockers": exc.blockers})
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/pause")
def pause_simulation():
    live_simulator.pause()
    return {"status": live_simulator.get_state()["status"]}


@router.post("/resume")
def resume_simulation():
    try:
        live_simulator.resume()
        return {"status": "running"}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/stop")
def stop_simulation():
    live_simulator.stop()
    return {"status": "stopped"}


@router.post("/reset")
def reset_simulation():
    live_simulator.reset()
    return {"status": "reset"}


@router.post("/speed/{speed}")
def set_speed(speed: int):
    try:
        live_simulator.set_speed(speed)
        return {"status": "ok", "speed": speed}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/state")
def simulation_state():
    return live_simulator.get_state()


@router.get("/replays/{replay_id}")
def get_replay(replay_id: str, db: Session = Depends(get_db)):
    record = simulation_service.replay_record(db, replay_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Replay not found.")
    return {"replay_id": record.id, "status": record.status, "error": record.error,
            "source_classification": record.source_classification, "dataset_id": record.dataset_id,
            "model_id": record.model_id, "model_version": record.model_version,
            "period_start_utc": record.period_start_utc.isoformat(), "period_end_utc": record.period_end_utc.isoformat(),
            "configuration": record.configuration, "created_at": record.created_at.isoformat(),
            "finished_at": record.finished_at.isoformat() if record.finished_at else None}


@router.get("/replays/{replay_id}/predictions")
def get_replay_predictions(replay_id: str, limit: int = Query(default=500, ge=1, le=2000), db: Session = Depends(get_db)):
    if db.get(SimulationReplayRecord, replay_id) is None:
        raise HTTPException(status_code=404, detail="Replay not found.")
    records = simulation_service.replay_predictions(db, replay_id, limit)
    return [record.input_snapshot for record in records]
