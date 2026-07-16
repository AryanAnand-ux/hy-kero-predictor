import sys
from pathlib import Path
from fastapi.testclient import TestClient
import importlib

# Add backend/ to path so we can import main
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from main import app

client = TestClient(app)


API_KEY = "hykero-secret-key"
HEADERS = {"X-API-Key": API_KEY}


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert "message" in response.json()


def test_imports_work_from_project_root():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import backend.main as backend_main
    importlib.reload(backend_main)
    assert hasattr(backend_main, "app")


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "model_loaded" in response.json()


def test_health_endpoint_includes_readiness_and_security_headers():
    response = client.get("/api/health")
    assert response.status_code in [200, 503]
    assert "X-Request-ID" in response.headers
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"

    ready_response = client.get("/api/health/ready")
    assert ready_response.status_code in [200, 503]


def test_unauthenticated_api_endpoints():
    # Protected endpoints should return 401 without API Key
    assert client.get("/api/model-metrics").status_code == 401
    assert client.post("/api/predict", json={}).status_code == 401
    assert client.get("/api/history").status_code == 401


def test_model_metrics_endpoint():
    response = client.get("/api/model-metrics", headers=HEADERS)
    # If training has run, it returns 200. If not, it returns 404. Either is acceptable depending on state.
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        data = response.json()
        assert "best_model" in data
        assert "models" in data
        assert len(data["models"]) > 0


def test_predict_endpoint_validation():
    # Sending empty body with key should trigger validation error (422)
    response = client.post("/api/predict", json={}, headers=HEADERS)
    assert response.status_code == 422


def test_history_endpoint_invalid_date():
    # Sending invalid dates should return 400 Bad Request
    response = client.get("/api/history?start_date=invalid-date", headers=HEADERS)
    assert response.status_code == 400


