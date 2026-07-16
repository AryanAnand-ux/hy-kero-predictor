# Technical Architecture Document

This document outlines the directory structure, component details, data schema, and API endpoint routing of the HY Kero Flash Point Prediction system.

---

## 📂 Project Structure

```bash
HY Kero Flash Point/
├── backend/                  # FastAPI Web Backend
│   ├── routes/               # Route definitions
│   │   ├── history.py        # GET /api/history & /api/history/stats
│   │   ├── models.py         # GET /api/model-metrics & /api/feature-importance
│   │   ├── predict.py        # POST /api/predict & /api/predict/window (±45 min window)
│   │   ├── upload.py         # POST /api/upload/predict-batch
│   │   └── chat.py           # Chatbot message & session management (with SSE stream)
│   ├── Dockerfile            # Backend container configuration
│   ├── main.py               # Application entrypoint & security middleware
│   ├── constants.py          # Unified backend specifications and constants
│   ├── database.py           # Thread-safe SQLite/Postgres connection pooling
│   ├── ai_provider.py        # Abstract LLM provider (Gemini or local fallback)
│   └── requirements.txt      # PyPI dependencies (lean, no optuna)
│
├── frontend/                 # React SPA (Vite + CSS)
│   ├── src/
│   │   ├── components/       # Reusable components (Kpi, Detail, ErrorBoundary, FlashBadge)
│   │   ├── pages/            # View components
│   │   │   ├── DashboardPage.jsx
│   │   │   ├── HistoryPage.jsx
│   │   │   ├── ModelsPage.jsx
│   │   │   ├── PredictPage.jsx
│   │   │   └── UploadPage.jsx
│   │   ├── App.jsx           # Main navigation & toast notifications
│   │   ├── api.js            # Fetch client wrapper
│   │   ├── constants.js      # Unified frontend variables and neobrutalist color tokens
│   │   ├── utils.js          # Shared utility methods (e.g. getFlashPointStatus)
│   │   ├── index.css         # Minimal Neo-brutalist theme CSS
│   │   └── main.jsx          # DOM mounting entrypoint
│   ├── Dockerfile            # Multi-stage production build configuration
│   └── package.json          # Node dependencies
│
├── src/                      # ML Pipeline Code
│   ├── __init__.py           # Package initializer
│   ├── preprocess.py         # Tag mapping & shift alignment (±45 min lookup window)
│   ├── features.py           # Feature engineering & splits (leakage fixed)
│   ├── train.py              # TimeSeries CV & Hyperparameter tuning (compressed models)
│   └── predict.py            # Singleton artifact cache & inference
│
├── data/                     # Raw & processed file storage
│   ├── processed/            # Merged CSVs & JSON metrics reports
│   └── models/               # Scaler and model weights binaries (.pkl)
│
├── docker-compose.yml        # Multi-service local orchestrator
├── README.md                 # Project quickstart guide
└── .gitignore                # Git ignore configuration (xlsx ignored)
```

---

## 📡 API Routing Design

The FastAPI backend exposes the following endpoints (prefixed with `/api`):

### 1. Prediction Router (`routes/predict.py`)
- **`POST /predict`**:
  - Accepts manual sensor name key-values (JSON).
  - Pre-loads scaler and selected model, matches input to features, fills missing values with training medians, runs scaling, computes predictions, and calculates 95% Confidence Intervals.
- **`POST /predict/window`**:
  - Accepts an ISO datetime.
  - Aggregates real-time sensor measurements in a ±45-minute window around that date from the raw Excel spreadsheet, runs `predict_from_raw`, and returns predicted values and metadata.

### 2. History Router (`routes/history.py`)
- **`GET /history`**:
  - Fetches chronologically ordered prediction history from the SQLite database.
  - Supports filtering by shift (`M`, `E`, `N`), `start_date`, and `end_date` (YYYY-MM-DD format).
- **`GET /history/stats`**:
  - Returns overall predictive metrics (residual mean, residual std, actual target range, and mean bias).

### 3. Models Router (`routes/models.py`)
- **`GET /model-metrics`**:
  - Returns cross-validation and testing scores (RMSE, MAE, R², MAPE) and chronological 5-Fold scores for all trained models.
- **`GET /feature-importance`**:
  - Retrieves the top 15 features sorted by linear coefficients or feature importance.

### 4. Upload Router (`routes/upload.py`)
- **`POST /upload/predict-batch`**:
  - Handles batch uploads of CSV/Excel files.
  - Performs column matching, missing value filling, scaling, inference, confidence interval bounding, and returns records in JSON format suitable for client-side tabular rendering and CSV downloads. Limit: 10MB.

### 5. Chat Router (`routes/chat.py`)
- **`GET /chat/conversations`**: List conversations for a session ID.
- **`POST /chat/conversations`**: Create a conversation tab.
- **`DELETE /chat/conversations/{id}`**: Delete conversation and associated messages.
- **`GET /chat/conversations/{id}/messages`**: Retrieve message sequence.
- **`POST /chat/conversations/{id}/messages`**: Post message and stream SSE chunks. Enriches prompt with real-time predictions database context if date/time query is detected.

---

## 🗄️ Database Schema

The SQLite database (`data/database.db`) stores prediction records and chatbot session state.

### `predictions`
Used to log manual, window-based, and batch predictions.
```sql
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    sample_ts TIMESTAMP NOT NULL,
    shift VARCHAR(10) NOT NULL,
    actual REAL,
    predicted REAL NOT NULL,
    residual REAL,
    confidence_lower REAL,
    confidence_upper REAL,
    sensors TEXT NOT NULL,         -- JSON stringified sensor readings dictionary
    lag_flash_gc REAL,
    lag2_flash_gc REAL,
    lag3_flash_gc REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### `conversations`
Maintains chatbot session thread items.
```sql
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    session_id VARCHAR(255) NOT NULL,
    title VARCHAR(255) NOT NULL DEFAULT 'New Conversation',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### `messages`
Stores conversation trees.
```sql
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    conversation_id INTEGER NOT NULL,
    role VARCHAR(50) NOT NULL,    -- 'user' or 'assistant'
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
```
