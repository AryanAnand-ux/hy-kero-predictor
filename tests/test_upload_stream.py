import sys
import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add backend/ and src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from main import app

client = TestClient(app)
API_KEY = "hykero-secret-key"
HEADERS = {"X-API-Key": API_KEY}


def test_upload_streaming_size_limit():
    # Construct a payload that is larger than 10MB to test the 413 error
    # To avoid generating huge files in RAM, we can mock or construct a file of 11MB
    # Let's generate a small mock file of 11MB (zeros or spaces)
    large_data = b"0" * (11 * 1024 * 1024)
    file_payload = {"file": ("large_file.csv", large_data, "text/csv")}
    
    response = client.post("/api/upload/predict-batch", files=file_payload, headers=HEADERS)
    assert response.status_code == 413
    assert "File too large" in response.json()["detail"]


def test_upload_invalid_extension():
    file_payload = {"file": ("unsupported.txt", b"some content", "text/plain")}
    response = client.post("/api/upload/predict-batch", files=file_payload, headers=HEADERS)
    assert response.status_code == 400
    assert "Only .xlsx or .csv files are accepted" in response.json()["detail"]


def test_upload_empty_file():
    file_payload = {"file": ("empty.csv", b"", "text/csv")}
    response = client.post("/api/upload/predict-batch", files=file_payload, headers=HEADERS)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower() or "could not parse file" in response.json()["detail"].lower()
