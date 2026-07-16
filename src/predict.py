"""
predict.py
──────────
Inference module: given a dict of raw sensor readings for a time window,
returns a predicted Flash Point.

Uses lazy-singleton pattern to cache model artifacts in memory.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE   = Path(__file__).resolve().parent.parent
MODELS = BASE / "data" / "models"
PROC   = BASE / "data" / "processed"

import sys
sys.path.insert(0, str(BASE))
try:
    from backend.constants import DEFAULT_RMSE, CI_Z_SCORE_95
except ImportError:
    DEFAULT_RMSE = 2.8
    CI_Z_SCORE_95 = 1.96

try:
    from feature_transforms import compute_derived_features, compute_time_features
except ImportError:
    from src.feature_transforms import compute_derived_features, compute_time_features

import threading

# ── Lazy-loaded singleton cache ──────────────────────────────────────────────
_cache = {}
_cache_lock = threading.Lock()


def load_artifacts():
    """Load and cache model artifacts. Returns (model, scaler, feature_cols, model_name, rmse)."""
    with _cache_lock:
        if "model" not in _cache:
            _cache["model"]        = joblib.load(MODELS / "best_model.pkl")
            _cache["scaler"]       = joblib.load(MODELS / "scaler.pkl")
            _cache["feature_cols"] = joblib.load(MODELS / "feature_cols.pkl")
            _cache["model_name"]   = joblib.load(MODELS / "best_model_name.pkl")
            
            # Load medians
            try:
                _cache["train_medians"] = joblib.load(MODELS / "train_medians.pkl")
            except Exception:
                _cache["train_medians"] = {fc: 0.0 for fc in _cache["feature_cols"]}
            
            # Load metrics to get RMSE / residuals std
            try:
                import json
                with open(PROC / "model_metrics.json", "r") as f:
                    metrics = json.load(f)
                model_name = _cache["model_name"]
                model_test_metrics = metrics.get(model_name, {}).get("test", {})
                rmse = model_test_metrics.get("residuals_std", model_test_metrics.get("rmse", DEFAULT_RMSE))
            except Exception:
                rmse = DEFAULT_RMSE
            _cache["rmse"] = rmse
            
        return _cache["model"], _cache["scaler"], _cache["feature_cols"], _cache["model_name"], _cache["rmse"]


def clear_cache():
    """Clear the artifact cache (e.g., after retraining)."""
    with _cache_lock:
        _cache.clear()



def predict_from_raw(sensor_readings: dict,
                     lag_flash_gc: float | None = None,
                     lag2_flash_gc: float | None = None,
                     lag3_flash_gc: float | None = None,
                     timestamp: pd.Timestamp | None = None) -> dict:
    """
    Given a dict of {sensor_name: value} for a 1.5-hour (90-min) window mean,
    return the predicted Flash Point.

    sensor_readings: {friendly_name: float}  — mean over the ±45 min window
    lag_flash_gc:    float | None  — previous shift's Flash GC (if available)
    lag2_flash_gc:   float | None  — 2 shifts ago Flash GC
    lag3_flash_gc:   float | None  — 3 shifts ago Flash GC
    timestamp:       pd.Timestamp | None — prediction timestamp for time features
    """
    model, scaler, feature_cols, model_name, rmse = load_artifacts()

    # Build a single-row DataFrame matching training feature columns
    # Initialize using training-set medians as defaults
    train_medians = _cache.get("train_medians", {})
    row = {fc: train_medians.get(fc, 0.0) for fc in feature_cols}

    # Fill in sensor mean columns (std features removed — model trains on means only)
    for sensor, val in sensor_readings.items():
        mean_key = f"{sensor}_mean"
        if mean_key in row:
            row[mean_key] = float(val)

    # Lag flash GC values
    if lag_flash_gc is not None and "lag1_flash_gc" in row:
        row["lag1_flash_gc"] = float(lag_flash_gc)
    if lag2_flash_gc is not None and "lag2_flash_gc" in row:
        row["lag2_flash_gc"] = float(lag2_flash_gc)
    if lag3_flash_gc is not None and "lag3_flash_gc" in row:
        row["lag3_flash_gc"] = float(lag3_flash_gc)

    # Flash GC momentum
    if lag_flash_gc is not None and lag2_flash_gc is not None and "flash_gc_delta" in row:
        row["flash_gc_delta"] = float(lag_flash_gc) - float(lag2_flash_gc)

    # ── Compute derived physics features (mirrors features.py) ──
    row = compute_derived_features(row)

    # ── Compute cyclical time encodings (mirrors features.py) ──
    row = compute_time_features(row, timestamp)

    # Only keep feature columns the model expects
    df = pd.DataFrame([{fc: row.get(fc, train_medians.get(fc, 0.0)) for fc in feature_cols}])[feature_cols]
    df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_cols)

    pred = float(model.predict(df_scaled)[0])
    
    # 95% Confidence Interval: +/- 1.96 * RMSE / residuals std
    ci_half = CI_Z_SCORE_95 * rmse
    
    return {
        "predicted_flash_point": round(pred, 2),
        "confidence_lower": round(pred - ci_half, 2),
        "confidence_upper": round(pred + ci_half, 2),
        "model_used": model_name,
        "unit": "C"
    }


if __name__ == "__main__":
    # Quick smoke test
    result = predict_from_raw({"MF_HK_Draw_T": 224.0, "MF_FlashZone_T": 365.0},
                               lag_flash_gc=79.0)
    print(result)
