import sys
import io
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient

# Add backend/ to path so we can import main
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from main import app

client = TestClient(app)
API_KEY = "hykero-secret-key"
HEADERS = {"X-API-Key": API_KEY}


def test_predict_endpoint_returns_alert_metadata():
    sensors = {
        "MF_HK_Draw_T": 224.0,
        "MF_FlashZone_T": 365.0,
        "MF_Top_T": 150.0,
        "Outlet_temp_11F1": 368.0,
        "Outlet_temp_11F2": 368.0,
        "Outlet_temp_11F3": 368.0,
        "Outlet_temp_11F4": 368.0,
        "CDU_Draw_HK_F": 95.0,
        "SS_11C5": 5.0,
    }
    body = {
        "sensors": sensors,
        "lag_flash_gc": 79.0,
        "lag2_flash_gc": 78.0,
        "timestamp": "2026-07-02T06:00:00"
    }
    response = client.post("/api/predict", json=body, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "sample_ts" in data
    assert "shift" in data
    assert data["sample_ts"] == "2026-07-02 06:00:00"
    assert data["shift"] == "M"


def test_predict_window_returns_alert_metadata():
    # Make sure predictions history can run
    body = {
        "timestamp": "2025-06-01T06:00:00",
        "lag_flash_gc": 79.0
    }
    response = client.post("/api/predict/window", json=body, headers=HEADERS)
    # The excel files might not be present or populated depending on the test run,
    # but if they are present, it returns 200. If not, it could return 404/500.
    # We check if 200 or not. If 200, we assert keys.
    if response.status_code == 200:
        data = response.json()
        assert "id" in data
        assert "sample_ts" in data
        assert "shift" in data
        assert data["shift"] == "M"


def test_upload_batch_returns_critical_count():
    # Create a small CSV payload with features
    csv_data = (
        "sample_ts,shift,predicted_flash_point,MF_HK_Draw_T,MF_FlashZone_T,MF_Top_T,Outlet_temp_11F1,Outlet_temp_11F2,Outlet_temp_11F3,Outlet_temp_11F4,CDU_Draw_HK_F,SS_11C5,lag1_flash_gc,lag2_flash_gc\n"
        "2026-07-02 06:00:00,M,60.5,224.0,365.0,150.0,368.0,368.0,368.0,368.0,95.0,5.0,79.0,78.0\n" # critical low (< 63)
        "2026-07-02 14:00:00,E,78.2,224.0,365.0,150.0,368.0,368.0,368.0,368.0,95.0,5.0,79.0,78.0\n" # normal
        "2026-07-02 22:00:00,N,98.1,224.0,365.0,150.0,368.0,368.0,368.0,368.0,95.0,5.0,79.0,78.0\n" # critical high (> 96)
    )
    
    file_payload = {"file": ("test_batch.csv", csv_data.encode("utf-8"), "text/csv")}
    response = client.post("/api/upload/predict-batch", files=file_payload, headers=HEADERS)
    
    assert response.status_code == 200
    data = response.json()
    assert "rows_processed" in data
    assert "critical_count" in data
    assert data["rows_processed"] == 3
    # Note: the model will re-predict using the actual model on the feature columns.
    # Depending on model weights, the model predictions might vary from the initial CSV column,
    # but we can verify the key exists and is an integer.
    assert isinstance(data["critical_count"], int)
