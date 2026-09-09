import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database.db import init_db

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    init_db()


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "data_mode" in data


def test_dashboard_summary():
    res = client.get("/api/dashboard/summary")
    assert res.status_code == 200
    data = res.json()
    assert "current_state" in data
    assert "risk_level" in data
    assert "pipeline_health" in data


def test_models_list():
    res = client.get("/api/models")
    assert res.status_code == 200
    models = res.json()
    assert isinstance(models, list)
    assert len(models) > 0


def test_alerts_endpoint():
    res = client.get("/api/alerts")
    assert res.status_code == 200
    alerts = res.json()
    assert isinstance(alerts, list)


def test_simulation_state():
    res = client.get("/api/simulation/state")
    assert res.status_code == 200
    state = res.json()
    assert "running" in state
    assert "speed" in state


def test_predict_single_window():
    # Construct a sample 30-step window
    sample_rows = [
        {"timestamp": f"2026-09-08T12:{i:02d}:00Z", "solexs_flux": 1.2e-5, "helios_flux": 4.5e-6}
        for i in range(30)
    ]
    res = client.post("/api/predict", json={"rows": sample_rows})
    assert res.status_code == 200
    pred = res.json()
    assert "nowcast" in pred
    assert "forecast" in pred
    assert "risk" in pred
    assert "1h" in pred["forecast"]
