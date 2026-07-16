import sys
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add backend/ and src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import database
from main import app

client = TestClient(app)
API_KEY = "hykero-secret-key"
HEADERS = {"X-API-Key": API_KEY}


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    # Redirect database to temp path for clean testing
    test_db_path = tmp_path / "test_database.db"
    original_db_path = database.DB_PATH
    original_csv_path = database.CSV_PATH
    
    database.DB_PATH = test_db_path
    database.CSV_PATH = Path("/nonexistent/predictions_history.csv")
    database.init_db()
    
    yield
    
    # Restore paths
    database.DB_PATH = original_db_path
    database.CSV_PATH = original_csv_path


def test_chat_unauthenticated_access():
    # Verify 401 response on protected chat endpoints without X-API-Key
    assert client.get("/api/chat/conversations?session_id=test_session").status_code == 401
    assert client.post("/api/chat/conversations", json={"session_id": "test_session", "title": "New"}).status_code == 401
    assert client.delete("/api/chat/conversations/1").status_code == 401
    assert client.get("/api/chat/conversations/1/messages").status_code == 401
    assert client.post("/api/chat/conversations/1/messages", json={"message": "hello"}).status_code == 401


def test_chat_conversation_lifecycle():
    session_id = "test_session_123"
    
    # 1. List conversations (should be empty initially)
    resp = client.get(f"/api/chat/conversations?session_id={session_id}", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert len(resp.json()["conversations"]) == 0
    
    # 2. Create a conversation
    resp = client.post("/api/chat/conversations", json={
        "session_id": session_id,
        "title": "CDU Steam Question"
    }, headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    conv_id = resp.json()["conversation_id"]
    assert conv_id > 0
    
    # 3. List conversations again (should have 1 item)
    resp = client.get(f"/api/chat/conversations?session_id={session_id}", headers=HEADERS)
    assert resp.status_code == 200
    assert len(resp.json()["conversations"]) == 1
    assert resp.json()["conversations"][0]["title"] == "CDU Steam Question"
    assert resp.json()["conversations"][0]["id"] == conv_id

    # 4. Post non-streaming message
    resp = client.post(f"/api/chat/conversations/{conv_id}/messages", json={
        "message": "What is the normal Flash Point spec range?",
        "stream": False
    }, headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert resp.json()["role"] == "assistant"
    assert "63.0" in resp.json()["content"]  # Mock response checks for keyword "spec"
    
    # 5. Fetch message history
    resp = client.get(f"/api/chat/conversations/{conv_id}/messages", headers=HEADERS)
    assert resp.status_code == 200
    messages = resp.json()["messages"]
    assert len(messages) == 2  # 1 User message + 1 Assistant message
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is the normal Flash Point spec range?"
    assert messages[1]["role"] == "assistant"
    
    # 6. Delete conversation
    resp = client.delete(f"/api/chat/conversations/{conv_id}", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    
    # 7. Verify empty list again
    resp = client.get(f"/api/chat/conversations?session_id={session_id}", headers=HEADERS)
    assert len(resp.json()["conversations"]) == 0


def test_chat_streaming_response():
    session_id = "stream_session"
    
    # Create conversation
    resp = client.post("/api/chat/conversations", json={
        "session_id": session_id,
        "title": "Stream Test"
    }, headers=HEADERS)
    conv_id = resp.json()["conversation_id"]
    
    # Post message with stream=True
    resp = client.post(f"/api/chat/conversations/{conv_id}/messages", json={
        "message": "Explain the model accuracy and metrics",
        "stream": True
    }, headers=HEADERS)
    
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    
    # Read stream chunks
    chunks = []
    for line in resp.iter_lines():
        if line.startswith("data: "):
            data = line[6:]
            import json
            payload = json.loads(data)
            if "chunk" in payload:
                chunks.append(payload["chunk"])
            if "done" in payload:
                break
                
    full_text = "".join(chunks)
    assert "RMSE" in full_text
    assert "Huber" in full_text


def test_delete_empty_conversation():
    session_id = "test_empty_session"
    
    # Create conversation
    resp = client.post("/api/chat/conversations", json={
        "session_id": session_id,
        "title": "Empty Conversation"
    }, headers=HEADERS)
    assert resp.status_code == 200
    conv_id = resp.json()["conversation_id"]
    
    # Delete empty conversation (should work and return 200 instead of 404)
    delete_resp = client.delete(f"/api/chat/conversations/{conv_id}", headers=HEADERS)
    assert delete_resp.status_code == 200
    assert delete_resp.json()["status"] == "success"
    
    # Verify it is deleted
    list_resp = client.get(f"/api/chat/conversations?session_id={session_id}", headers=HEADERS)
    assert len(list_resp.json()["conversations"]) == 0
