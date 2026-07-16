
import sys
import os
import threading
import math
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
from fastapi import APIRouter, HTTPException

BACKEND_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = BACKEND_ROOT.parent / "src"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from predict import predict_from_raw, load_artifacts

from pydantic import BaseModel

try:
    from ..database import insert_prediction
    from ..constants import (
        FLASH_POINT_SPEC_MIN,
        FLASH_POINT_SPEC_MAX,
        PREDICTION_WINDOW_MINUTES,
        FLASH_POINT_VALUE_MIN,
        FLASH_POINT_VALUE_MAX,
    )
except ImportError:
    from database import insert_prediction
    from constants import (
        FLASH_POINT_SPEC_MIN,
        FLASH_POINT_SPEC_MAX,
        PREDICTION_WINDOW_MINUTES,
        FLASH_POINT_VALUE_MIN,
        FLASH_POINT_VALUE_MAX,
    )

logger = logging.getLogger("hykero.routes.predict")
router = APIRouter()
BASE = Path(__file__).resolve().parent.parent.parent


def preload_model():
    """Pre-load model artifacts at startup."""
    load_artifacts()


def _get_cached_model():
    """Get the cached model artifacts for health checks."""
    model, scaler, feature_cols, model_name, _ = load_artifacts()
    return model, scaler, feature_cols, model_name


class SensorInput(BaseModel):
    sensors: dict[str, float]          # {sensor_friendly_name: value}
    timestamp: Optional[str] = None    # ISO format, defaults to current time
    lag_flash_gc: Optional[float] = None
    lag2_flash_gc: Optional[float] = None
    lag3_flash_gc: Optional[float] = None


class WindowInput(BaseModel):
    timestamp: str                     # ISO format e.g. "2025-06-01T06:00:00"
    lag_flash_gc: Optional[float] = None
    lag2_flash_gc: Optional[float] = None
    lag3_flash_gc: Optional[float] = None


# ── In-Memory Excel Data Cache (Thread-Safe) ──────────────────────────────────
_excel_cache = {}
_excel_cache_lock = threading.Lock()


def load_cached_sensor_data(sensor_file: Path, tag_map_file: Path):
    """Loads and caches Excel sensor data, checking modification times for freshness."""
    with _excel_cache_lock:
        s_mtime = os.path.getmtime(sensor_file)
        t_mtime = os.path.getmtime(tag_map_file)

        if (_excel_cache.get("sensor_mtime") == s_mtime and
                _excel_cache.get("tag_mtime") == t_mtime and
                "sensor_df" in _excel_cache):
            return _excel_cache["sensor_df"]

        logger.info("Excel cache miss or file modified. Loading Excel sheets...")
        sensor_df = pd.read_excel(sensor_file, sheet_name="Data")
        sensor_df["Timestamp"] = pd.to_datetime(sensor_df["Timestamp"])

        tag_df = pd.read_excel(tag_map_file)
        tag_col = tag_df.columns[3]
        name_col = tag_df.columns[4]
        tag_map = dict(zip(tag_df[tag_col].astype(str).str.strip(),
                           tag_df[name_col].astype(str).str.strip()))

        sensor_df = sensor_df.rename(columns={c: tag_map.get(c.strip(), c)
                                              for c in sensor_df.columns})

        _excel_cache["sensor_df"] = sensor_df
        _excel_cache["sensor_mtime"] = s_mtime
        _excel_cache["tag_mtime"] = t_mtime
        logger.info("Successfully loaded and cached Excel data sheets.")
        return sensor_df


# ── Helpers & Input Validation ────────────────────────────────────────────────
SENSOR_BOUNDS = {
    "MF_HK_Draw_T": (100.0, 350.0),
    "MF_FlashZone_T": (200.0, 450.0),
    "MF_Top_T": (50.0, 250.0),
    "Outlet_temp_11F1": (200.0, 450.0),
    "Outlet_temp_11F2": (200.0, 450.0),
    "Outlet_temp_11F3": (200.0, 450.0),
    "Outlet_temp_11F4": (200.0, 450.0),
    "CDU_Draw_HK_F": (0.0, 500.0),
    "Crude_Tput": (100.0, 3000.0),
    "SS_11C1": (0.0, 30.0),
    "SS_11C5": (0.0, 30.0),
}


def validate_input_data(sensors: dict, lag1: float | None = None, lag2: float | None = None, lag3: float | None = None):
    for key, val in sensors.items():
        base_key = key.replace("_mean", "").replace("_std", "")
        if not isinstance(val, (int, float)):
            raise ValueError(f"Sensor '{key}' value must be a number.")
        if not math.isfinite(val):
            raise ValueError(f"Sensor '{key}' value must be a finite number.")
        if base_key in SENSOR_BOUNDS:
            lo, hi = SENSOR_BOUNDS[base_key]
            if not (lo <= val <= hi):
                raise ValueError(f"Sensor '{key}' value {val} is out of bounds [{lo}, {hi}].")

    for lag_name, lag_val in [("lag_flash_gc", lag1), ("lag2_flash_gc", lag2), ("lag3_flash_gc", lag3)]:
        if lag_val is not None:
            if not math.isfinite(lag_val):
                raise ValueError(f"{lag_name} value must be a finite number.")
            if not (FLASH_POINT_VALUE_MIN <= lag_val <= FLASH_POINT_VALUE_MAX):
                raise ValueError(f"{lag_name} value {lag_val} is out of bounds [{FLASH_POINT_VALUE_MIN}, {FLASH_POINT_VALUE_MAX}].")


def determine_shift(hour: int) -> str:
    if 2 <= hour < 10:
        return "M"
    elif 10 <= hour < 18:
        return "E"
    else:
        return "N"


def _log_flash_point_alert(predicted_fp: float, ts_str: str):
    if predicted_fp < FLASH_POINT_SPEC_MIN:
        logger.error(f"🚨 CRITICAL ALERT: Predicted flash point {predicted_fp}°C is BELOW SPEC MINIMUM ({FLASH_POINT_SPEC_MIN}°C) at {ts_str}!")
    elif predicted_fp > FLASH_POINT_SPEC_MAX:
        logger.error(f"🚨 CRITICAL ALERT: Predicted flash point {predicted_fp}°C is ABOVE SPEC MAXIMUM ({FLASH_POINT_SPEC_MAX}°C) at {ts_str}!")


def _save_prediction_to_db(ts_str: str, shift: str, predicted: float, confidence_lower: float,
                           confidence_upper: float, sensors: dict, lag1: float | None,
                           lag2: float | None, lag3: float | None) -> Optional[int]:
    try:
        return insert_prediction(
            sample_ts=ts_str,
            shift=shift,
            predicted=predicted,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            sensors=sensors,
            lag_flash_gc=lag1,
            lag2_flash_gc=lag2,
            lag3_flash_gc=lag3
        )
    except Exception as db_err:
        logger.error(f"Failed to log prediction to database: {db_err}")
        return None


@router.post("/predict")
def predict_from_sensors(body: SensorInput):
    """Predict Flash Point from manually provided sensor values."""
    try:
        validate_input_data(body.sensors, body.lag_flash_gc, body.lag2_flash_gc, body.lag3_flash_gc)
    except ValueError as val_err:
        raise HTTPException(400, f"Validation failed: {val_err}")

    try:
        if body.timestamp:
            ts = pd.Timestamp(body.timestamp)
        else:
            ts = pd.Timestamp.now()
        
        shift = determine_shift(ts.hour)
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")

        res = predict_from_raw(
            body.sensors,
            lag_flash_gc=body.lag_flash_gc,
            lag2_flash_gc=body.lag2_flash_gc,
            lag3_flash_gc=body.lag3_flash_gc
        )
        res["status"] = "success"

        prediction_id = _save_prediction_to_db(
            ts_str=ts_str,
            shift=shift,
            predicted=res["predicted_flash_point"],
            confidence_lower=res["confidence_lower"],
            confidence_upper=res["confidence_upper"],
            sensors=body.sensors,
            lag1=body.lag_flash_gc,
            lag2=body.lag2_flash_gc,
            lag3=body.lag3_flash_gc
        )

        res["id"] = prediction_id
        res["sample_ts"] = ts_str
        res["shift"] = shift

        _log_flash_point_alert(res["predicted_flash_point"], ts_str)

        return res
    except Exception as e:
        logger.error(f"Manual inference failed: {e}", exc_info=True)
        raise HTTPException(500, "Inference failed due to an internal server error.")


@router.post("/predict/window")
def predict_from_window(body: WindowInput):
    """Predict Flash Point using a timeframe window from the raw sensor file."""
    try:
        ts = pd.Timestamp(body.timestamp)
    except Exception:
        raise HTTPException(400, "Invalid timestamp format. Use ISO 8601.")

    try:
        validate_input_data({}, body.lag_flash_gc, body.lag2_flash_gc, body.lag3_flash_gc)
    except ValueError as val_err:
        raise HTTPException(400, f"Validation failed: {val_err}")

    sensor_file = BASE / "HY Kero  AI ML Data 15 Mins.xlsx"
    tag_map_file = BASE / "Tag to names Mapping.xlsx"

    simulated_mode = not sensor_file.exists() or not tag_map_file.exists()

    if simulated_mode:
        logger.warning("Excel files not found. Running in SIMULATED fallback mode.")
        import random
        # Generate simulated sensor readings based on standard physical ranges
        means = {
            "MF_HK_Draw_T": random.uniform(220.0, 230.0),
            "MF_FlashZone_T": random.uniform(360.0, 370.0),
            "CDU_Draw_HK_F": random.uniform(90.0, 100.0),
            "SS_11C5": random.uniform(4.0, 6.0),
            "MF_Top_T": random.uniform(145.0, 155.0),
            "Outlet_temp_11F1": random.uniform(365.0, 370.0),
            "Outlet_temp_11F2": random.uniform(365.0, 370.0),
            "Outlet_temp_11F3": random.uniform(365.0, 370.0),
            "Outlet_temp_11F4": random.uniform(365.0, 370.0),
        }
        res = predict_from_raw(
            means,
            lag_flash_gc=body.lag_flash_gc,
            lag2_flash_gc=body.lag2_flash_gc,
            lag3_flash_gc=body.lag3_flash_gc
        )
        lo = ts - pd.Timedelta(minutes=PREDICTION_WINDOW_MINUTES)
        hi = ts + pd.Timedelta(minutes=PREDICTION_WINDOW_MINUTES)
        res.update({
            "window_start": str(lo),
            "window_end":   str(hi),
            "readings_used": 5,
            "status": "success",
            "is_simulated": True
        })
        shift = determine_shift(ts.hour)
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        
        prediction_id = _save_prediction_to_db(
            ts_str=ts_str,
            shift=shift,
            predicted=res["predicted_flash_point"],
            confidence_lower=res["confidence_lower"],
            confidence_upper=res["confidence_upper"],
            sensors=means,
            lag1=body.lag_flash_gc,
            lag2=body.lag2_flash_gc,
            lag3=body.lag3_flash_gc
        )
        res["id"] = prediction_id
        res["sample_ts"] = ts_str
        res["shift"] = shift
        _log_flash_point_alert(res["predicted_flash_point"], ts_str)
        return res

    try:
        sensor_df = load_cached_sensor_data(sensor_file, tag_map_file)

        lo = ts - pd.Timedelta(minutes=PREDICTION_WINDOW_MINUTES)
        hi = ts + pd.Timedelta(minutes=PREDICTION_WINDOW_MINUTES)
        
        window = sensor_df[(sensor_df["Timestamp"] >= lo) &
                           (sensor_df["Timestamp"] <= hi)]

        if window.empty:
            raise HTTPException(404, f"No sensor readings found in window {lo} – {hi}")

        feat_cols = [c for c in window.columns if c != "Timestamp"]
        means = window[feat_cols].mean().to_dict()

        # Sanitize any NaNs in means
        means = {k: (0.0 if pd.isna(v) else v) for k, v in means.items()}

        res = predict_from_raw(
            means,
            lag_flash_gc=body.lag_flash_gc,
            lag2_flash_gc=body.lag2_flash_gc,
            lag3_flash_gc=body.lag3_flash_gc
        )
        
        res.update({
            "window_start": str(lo),
            "window_end":   str(hi),
            "readings_used": len(window),
            "status": "success"
        })

        shift = determine_shift(ts.hour)
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        
        prediction_id = _save_prediction_to_db(
            ts_str=ts_str,
            shift=shift,
            predicted=res["predicted_flash_point"],
            confidence_lower=res["confidence_lower"],
            confidence_upper=res["confidence_upper"],
            sensors=means,
            lag1=body.lag_flash_gc,
            lag2=body.lag2_flash_gc,
            lag3=body.lag3_flash_gc
        )

        res["id"] = prediction_id
        res["sample_ts"] = ts_str
        res["shift"] = shift

        _log_flash_point_alert(res["predicted_flash_point"], ts_str)

        return res

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Window inference failed: {e}", exc_info=True)
        raise HTTPException(500, "Inference failed due to an internal server error.")
