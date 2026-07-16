# FastAPI Backend API Documentation

The FastAPI backend runs by default at `http://localhost:8000`. All endpoints are prefixed with `/api`.

---

## 🔮 Prediction Endpoints

### 1. Manual Sensor Prediction
Exposes real-time manual predictions for custom sensor readings.

- **URL:** `/api/predict`
- **Method:** `POST`
- **Headers:** `Content-Type: application/json`
- **Request Body:**
  ```json
  {
    "sensors": {
      "MF_HK_Draw_T": 224.5,
      "MF_FlashZone_T": 365.2,
      "MF_Top_T": 151.0
    },
    "lag_flash_gc": 79.0,
    "lag2_flash_gc": 78.5,
    "lag3_flash_gc": 80.0
  }
  ```
- **Response Model (200 OK):**
  ```json
  {
    "predicted_flash_point": 77.2,
    "confidence_lower": 71.68,
    "confidence_upper": 82.72,
    "model_used": "Lasso",
    "unit": "C",
    "status": "success"
  }
  ```

### 2. Time Window Prediction
Computes predictions by aggregating sensor readings around a specific historical timestamp in a ±45-minute window.

- **URL:** `/api/predict/window`
- **Method:** `POST`
- **Request Body:**
  ```json
  {
    "timestamp": "2025-06-01T06:00:00",
    "lag_flash_gc": 79.0,
    "lag2_flash_gc": 78.5,
    "lag3_flash_gc": 80.0
  }
  ```
- **Response Model (200 OK):**
  ```json
  {
    "predicted_flash_point": 78.41,
    "confidence_lower": 72.89,
    "confidence_upper": 83.93,
    "model_used": "Lasso",
    "unit": "C",
    "window_start": "2025-06-01 05:15:00",
    "window_end": "2025-06-01 06:45:00",
    "readings_used": 5,
    "status": "success"
  }
  ```

---

## 📈 Prediction History Endpoints

### 1. Retrieve Historical Predictions
Returns predictions combined with actual lab values.

- **URL:** `/api/history`
- **Method:** `GET`
- **Query Parameters:**
  - `shift` (optional): `M` (Morning), `E` (Evening), or `N` (Night)
  - `start_date` (optional): Filter records on or after `YYYY-MM-DD`
  - `end_date` (optional): Filter records on or before `YYYY-MM-DD`
  - `limit` (optional, default: 200): Limit returned records
- **Response Model (200 OK):**
  ```json
  {
    "total": 728,
    "count": 2,
    "limit": 200,
    "offset": 0,
    "data": [
      {
        "sample_ts": "2026-03-31 22:00:00",
        "shift": "N",
        "actual": 82.0,
        "predicted": 81.45,
        "residual": 0.55
      },
      {
        "sample_ts": "2026-03-31 14:00:00",
        "shift": "E",
        "actual": 79.0,
        "predicted": 78.80,
        "residual": 0.20
      }
    ]
  }
  ```

### 2. Prediction Statistics
- **URL:** `/api/history/stats`
- **Method:** `GET`
- **Response Model (200 OK):**
  ```json
  {
    "total_predictions": 728,
    "actual_mean": 77.41,
    "actual_std": 4.55,
    "actual_min": 58.0,
    "actual_max": 93.0,
    "predicted_mean": 77.38,
    "residual_mean": 0.03,
    "residual_std": 2.82
  }
  ```

---

## 📂 Batch Prediction Upload

### 1. Batch Upload
Run batch predictions on an uploaded dataset.

- **URL:** `/api/upload/predict-batch`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Payload:** File (CSV or Excel) under `file` key. Max size: 10MB.
- **Response Model (200 OK):**
  ```json
  {
    "rows_processed": 100,
    "model_used": "Lasso",
    "critical_count": 0,
    "data": [
      {
        "Timestamp": "2026-04-01 06:00:00",
        "predicted_flash_point": 77.2,
        "confidence_lower": 71.68,
        "confidence_upper": 82.72
      }
    ],
    "status": "success"
  }
  ```

---

## 💬 Chatbot Endpoints

### 1. List Conversations
Retrieve list of conversation sessions.
- **URL:** `/api/chat/conversations`
- **Method:** `GET`
- **Query Parameters:**
  - `session_id` (required): Unique guest session identifier
- **Response Model (200 OK):**
  ```json
  {
    "status": "success",
    "conversations": [
      {
        "id": 1,
        "session_id": "session-xyz",
        "title": "CDU Stripping Steam Question",
        "created_at": "2026-07-01 10:15:30"
      }
    ]
  }
  ```

### 2. Create Conversation
Start a new chatbot conversation thread.
- **URL:** `/api/chat/conversations`
- **Method:** `POST`
- **Request Body:**
  ```json
  {
    "session_id": "session-xyz",
    "title": "New Discussion"
  }
  ```
- **Response Model (200 OK):**
  ```json
  {
    "status": "success",
    "conversation_id": 2,
    "title": "New Discussion"
  }
  ```

### 3. Delete Conversation
- **URL:** `/api/chat/conversations/{id}`
- **Method:** `DELETE`
- **Response Model (200 OK):**
  ```json
  {
    "status": "success",
    "message": "Conversation deleted successfully."
  }
  ```

### 4. Fetch Message History
- **URL:** `/api/chat/conversations/{id}/messages`
- **Method:** `GET`
- **Response Model (200 OK):**
  ```json
  {
    "status": "success",
    "messages": [
      {"role": "user", "content": "What is the normal operating range?"},
      {"role": "assistant", "content": "The normal operating specification range is 63°C to 96°C."}
    ]
  }
  ```

### 5. Send Message (with optional SSE Stream)
- **URL:** `/api/chat/conversations/{id}/messages`
- **Method:** `POST`
- **Request Body:**
  ```json
  {
    "message": "What is the RMSE of the Huber Tuned model?",
    "stream": true
  }
  ```
- **Response Model (200 OK / SSE Stream chunks):**
  - Standard JSON:
    ```json
    {
      "status": "success",
      "role": "assistant",
      "content": "The Huber Tuned model test RMSE is 2.21°C."
    }
    ```
  - SSE Stream Event:
    ```text
    data: {"chunk": "The"}
    data: {"chunk": " Lasso"}
    data: {"chunk": " model"}
    data: {"done": true}
    ```

---

## 🏥 Health Probes

### 1. Liveness Probe
Lightweight validation that the server application has started.
- **URL:** `/api/health/live`
- **Method:** `GET`
- **Response (200 OK):**
  ```json
  {"status": "alive"}
  ```

### 2. Readiness Probe
Validates that connections to the model storage files and SQLite database are active.
- **URL:** `/api/health/ready`
- **Method:** `GET`
- **Response (200 OK):**
  ```json
  {
    "status": "ready",
    "model_loaded": true,
    "model_name": "Lasso",
    "database_ready": true
  }
  ```

### 3. Integrated Health Probe
Combines status and readiness checks for Docker orchestration.
- **URL:** `/api/health`
- **Method:** `GET`
- **Response (200 OK):**
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "model_name": "Lasso",
    "ready": true
  }
  ```
