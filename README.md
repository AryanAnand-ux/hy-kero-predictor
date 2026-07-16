# HY Kero Flash Point Prediction — Full Stack ML Project

> Real-time Flash Point prediction for Heavy Kerosene using Crude Distillation Unit (CDU) process sensor data.

This project is a production-grade full-stack machine learning application designed for **IOCL (Indian Oil Corporation Ltd)**. It predicts the **Flash Point** of Heavy Kerosene (HY Kero) using 41 DCS sensor readings, providing process operators with continuous feedback and reducing reliance on slow laboratory Gas Chromatography analysis.

---

## ✨ Features Added in Overhaul & Hardening

- **Robust ML Pipeline**: Implemented chronological `TimeSeriesSplit` cross-validation, feature correlation filtering, and target variable leakage fixes (`roll3_flash_gc` target leakage fixed).
- **Unified Feature Transforms**: Extracted row-level feature calculation logic into a single shared module `src/feature_transforms.py` for inference `predict.py` calls, resolving architectural duplication risk.
- **FastAPI Backend Hardening**: Pre-loads ML models at startup, implements lifespan handlers, supports dynamic CORS settings, and includes request body size checks (1MB payload cap) for JSON endpoints.
- **95% Confidence Intervals**: Computes 95% Confidence Intervals (using model test RMSE bounds) for all manual, window-based, and batch predictions.
- **Interactive React Dashboard**: High-fidelity dashboard visualizing historical predictions, real-time manually entered predictions, model validation metrics, and feature importances.
- **AI Chatbot Assistant**: Specialized heavy kerosene process chatbot with domain-specific knowledge and real-time database prediction lookup context, secured with prompt injection filtering. Falls back gracefully to a smart offline mock assistant if no Gemini API key is configured.
- **Optimized SQL Statistics**: Rewrote `get_prediction_stats()` DB function to compute population standard deviation directly via SQL aggregate variance, reducing memory overhead by avoiding pulling thousands of rows into python runtime memory.
- **Advanced Filtering & CSV Export**: Allows filtering prediction history by shift and date ranges with client-side CSV downloads.
- **Batch Upload & Processing**: Allows drag-and-drop file uploads (up to 10MB) for batch predictions with annotated output CSV file generation.
- **Docker Orchestration**: Complete containerization using multi-stage Node/Nginx builds and Python slim Dockerfiles with root `docker-compose.yml`.

---

## 📈 ML Methodology & Overfitting Fix

During data discovery, a **severe distribution shift** (+3.74°C target mean change) was detected between the training period (Apr 2025 – Jan 2026) and testing period (Jan 2026 – Mar 2026). Overly complex tree-based models (XGBoost, Random Forest) overfit the training domain, yielding negative R² values on the test set. 

This was resolved by:
1. Moving from complex tree structures to L1-regularized **Lasso** and **ElasticNet** linear models, which generalize stable trends outside of the training domain.
2. Dropping highly correlated ($r > 0.92$) and near-constant features (17 redundant and 7 low-variance features removed).
3. Fixing target variable leakage in `roll3_flash_gc` rolling averages.

---

## 🚀 Environment Variables

### Backend (`.env`)
| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | The port the FastAPI backend runs on | `8000` |
| `APP_ENV` | Application environment mode (`development` or `production`) | `development` |
| `API_KEY` | Secret token used to authenticate API requests | `hykero-secret-key` |
| `GEMINI_API_KEY` | Google Gemini API key used for the chatbot assistant | *(empty/fallback to Mock)* |
| `CORS_ORIGINS` | Comma-separated list of allowed CORS origins | `http://localhost,http://localhost:80,http://localhost:5173,http://127.0.0.1:5173` |
| `DATABASE_URL` | SQLite database file connection path | `sqlite:///data/database.db` |

### Frontend (`frontend/.env` or `package.json`)
| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Base URL endpoint path for backend API | `http://localhost:8000/api` |
| `VITE_API_KEY` | Client-side API key sent in `X-API-Key` headers | `hykero-secret-key` |

---

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)
Launch the entire stack (FastAPI backend + React frontend) instantly without installing local dependencies:
```bash
docker-compose up --build
```
- **React Dashboard:** `http://localhost` (Port 80)
- **FastAPI Docs:** `http://localhost:8000/docs`
- **Health check:** `http://localhost/api/health`

For production-style deployments, the frontend now proxies API requests through Nginx to the backend so the UI can call `/api/...` from a single origin.

### Option 2: Local Development Setup (With Virtual Environment)
1. **Set up Virtual Environment & Install Dependencies:**
   Create a localized environment and install packages (configured with relaxed constraints for Python 3.14):
   ```bash
   # Create virtual environment
   python -m venv venv --prompt "hykero"
   
   # Activate environment
   # Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   # Linux/macOS:
   source venv/bin/activate
   
   # Install requirements (forced binary prebuilt wheels)
   pip install --only-binary :all: -r backend/requirements.txt pytest optuna
   ```
2. **Run Pipeline (Generates Model Binaries):**
   ```bash
   python run_pipeline.py
   ```
3. **Start FastAPI Backend:**
   ```bash
   cd backend
   uvicorn main:app --reload --port 8000
   ```
4. **Start React Frontend:**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   Open `http://localhost:5173`.

5. **Run Tests:**
   ```bash
   # Run all database, API, chat and unit tests
   pytest tests/test_database.py tests/test_inference.py tests/test_api.py tests/test_chat.py tests/test_alerts.py tests/test_upload_stream.py tests/test_preprocess.py tests/test_training.py -v
   ```

---

## 📂 Project Structure

```bash
HY Kero Flash Point/
├── src/                      # ML pipeline modules
│   ├── preprocess.py         # Data alignment (±45 min windows)
│   ├── features.py           # Feature engineering & splits (leakage fixed)
│   ├── feature_transforms.py # Shared feature computation logic (single source of truth)
│   ├── train.py              # TimeSeries CV & Hyperparameter tuning (compressed models)
│   └── predict.py            # Singleton artifact cache & inference (imports from feature_transforms)
├── backend/                  # FastAPI REST API
│   ├── main.py               # Entrypoint, secure auth, 1MB size checks & lifespan
│   ├── constants.py          # Unified backend specifications and constants
│   ├── database.py           # Dual SQLite/Postgres adapter (optimized SQL stats)
│   ├── ai_provider.py        # Abstract LLM provider (Gemini or local fallback)
│   └── routes/
│       ├── predict.py        # POST /api/predict (Manual + Window ±45 min lookup)
│       ├── history.py        # GET /api/history (With date/shift filters)
│       ├── models.py         # GET /api/model-metrics & CV scores
│       ├── upload.py         # POST /api/upload/predict-batch
│       └── chat.py           # Secured chatbot with prompt injection filtering
├── frontend/                 # React + Vite dashboard
│   ├── src/
│   │   ├── components/       # Extracted subcomponents (Kpi, Detail, ErrorBoundary, FlashBadge)
│   │   ├── pages/            # DashboardPage, PredictPage, HistoryPage, ModelsPage, UploadPage
│   │   ├── App.jsx           # Main layout frame & toast notifications
│   │   ├── api.js            # API fetch client wrapper (boundary boundary FormData fix)
│   │   ├── constants.js      # Unified frontend variables and neobrutalist color tokens
│   │   └── utils.js          # Shared utility methods (e.g. getFlashPointStatus)
├── data/
│   ├── processed/            # Merged datasets and quality reports
│   └── models/               # Scaler and model binaries (.pkl)
├── run_pipeline.py           # One-click pipeline orchestrator
├── docker-compose.yml        # Multi-service local orchestrator
├── README.md
```

---

## 🔌 API Summary

| Method | Endpoint | Description |
|--------|---------|-------------|
| **POST** | `/api/predict` | Predict from manual sensor inputs with 95% CI |
| **POST** | `/api/predict/window` | Predict from a timestamp (reads raw file in ±45 min window) |
| **GET** | `/api/history` | Prediction history with shift and date range filters |
| **GET** | `/api/history/stats` | Summary statistics of prediction residuals (SQL aggregated) |
| **GET** | `/api/model-metrics` | All model comparison metrics (including CV folds) |
| **GET** | `/api/feature-importance` | Top 15 feature coefficients |
| **POST** | `/api/upload/predict-batch` | Batch prediction from uploaded CSV/XLSX |
| **GET** | `/api/chat/conversations` | List conversations for the active session |
| **POST** | `/api/chat/conversations` | Create a new chatbot conversation |
| **DELETE** | `/api/chat/conversations/{id}` | Delete a conversation |
| **GET** | `/api/chat/conversations/{id}/messages` | Get message history |
| **POST** | `/api/chat/conversations/{id}/messages` | Post a chat message (streams response chunks, secured) |
| **GET** | `/api/health` | Liveness/readiness health check |
