# HY Kero Flash Point Prediction — React Frontend Dashboard

This directory contains the **React + Vite** single-page application (SPA) frontend dashboard for the Heavy Kerosene (HY Kero) Flash Point Predictor system. 

It provides process operators with real-time prediction entry tools, validation metric charts, historical data search/filters with CSV downloads, and a process troubleshooting chatbot.

---

## 🎨 Theme & Styling

The frontend is styled using **Vanilla CSS** with a modern, high-contrast **Neo-brutalist** design theme:
- Curated color tokens (sleek dark mode backgrounds, vibrant brand borders, flat shadows).
- Custom typography, micro-animations, and responsive layouts.
- Component layouts separated into clean files with high accessibility support.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
npm install
```

### 2. Configure Environment Variables
Create a `.env` or use environment variables during running. By default, the application calls `/api` endpoints:
- `VITE_API_URL`: The URL to the FastAPI backend API (defaults to `http://localhost:8000/api` during dev).
- `VITE_API_KEY`: The authorization token matched with backend (defaults to `hykero-secret-key` in dev).

### 3. Run Development Server
```bash
npm run dev
```
Open `http://localhost:5173` in your browser.

### 4. Build for Production
Generates minified static assets in the `dist/` directory:
```bash
npm run build
```

---

## 📂 Codebase Overview

- **`src/App.jsx`**: Main layout frame, page routing table, global Toast notification provider, and global error boundaries.
- **`src/constants.js`**: Centralized neobrutalist style tokens (`BRUT_COLORS_OBJ`, `BRUT_COLORS_ARR`) and physical process variables constants.
- **`src/utils.js`**: Shared helpers such as `getFlashPointStatus(value)` to classify predicted values into Normal, Alert, or Danger alerts.
- **`src/components/`**:
  - `ErrorBoundary.jsx`: Secure layout fallback screen.
  - `FlashBadge.jsx`: Uniform colored status badges.
  - `Detail.jsx`: Side-drawer details layout.
  - `Kpi.jsx`: Standardized metric KPI cards.
- **`src/pages/`**:
  - `DashboardPage.jsx`: Main telemetry overview and KPI indicators.
  - `PredictPage.jsx`: Manual process variable inference calculator.
  - `UploadPage.jsx`: Drag-and-drop batch upload processing.
  - `HistoryPage.jsx`: SQLite database prediction logs search and CSV exports.
  - `ModelsPage.jsx`: Recharts comparison curves and feature importance tables.
- **`src/api.js`**: Unified fetch client handling credentials headers, FormData boundaries, and JSON body generation.
