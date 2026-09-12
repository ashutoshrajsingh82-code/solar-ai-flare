from fastapi import APIRouter, HTTPException
from app.schemas.schemas import SimulationStartRequest
from app.services.pipeline_state import pipeline_state
from app.services import predict_service
from ml.inference.simulator import live_simulator

router = APIRouter(prefix="/api/simulation", tags=["simulation"])


@router.post("/start")
def start_simulation(req: SimulationStartRequest):
    if pipeline_state.labeled_df is None:
        raise HTTPException(status_code=422, detail="Pipeline not preprocessed yet. Run /api/preprocess first.")
    try:
        predict_fn = predict_service.make_simulator_predict_fn(req.model_id)
        live_simulator.configure(pipeline_state.labeled_df, req.sequence_length, predict_fn)
        live_simulator.set_speed(req.speed)
        live_simulator.start()
        return {"status": "started", "speed": req.speed}
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/pause")
def pause_simulation():
    live_simulator.pause()
    return {"status": "paused"}


@router.post("/resume")
def resume_simulation():
    try:
        live_simulator.resume()
        return {"status": "running"}
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/reset")
def reset_simulation():
    live_simulator.reset()
    return {"status": "reset"}


@router.post("/speed/{speed}")
def set_speed(speed: int):
    try:
        live_simulator.set_speed(speed)
        return {"status": "ok", "speed": speed}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/state")
def simulation_state():
    return live_simulator.get_state()
