""

import io
import os
import re
import sys
import logging
from pathlib import Path
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

logger = logging.getLogger("hykero.routes.upload")
router = APIRouter()
BASE = Path(__file__).resolve().parent.parent.parent

# Add src/ to path so we can import predict module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from predict import load_artifacts

try:
    from ..database import insert_prediction
    from ..constants import (
        FLASH_POINT_SPEC_MIN,
        FLASH_POINT_SPEC_MAX,
        CI_Z_SCORE_95,
        MAX_UPLOAD_SIZE_BYTES,
        UPLOAD_CHUNK_SIZE,
        BATCH_DB_LOG_LIMIT,
    )
except ImportError:
    from database import insert_prediction
    from constants import (
        FLASH_POINT_SPEC_MIN,
        FLASH_POINT_SPEC_MAX,
        CI_Z_SCORE_95,
        MAX_UPLOAD_SIZE_BYTES,
        UPLOAD_CHUNK_SIZE,
        BATCH_DB_LOG_LIMIT,
    )


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and shell injection risks."""
    base = os.path.basename(filename)
    return re.sub(r"[^a-zA-Z0-9._-]", "_", base)


@router.post("/upload/predict-batch")
async def upload_and_predict(file: UploadFile = File(...)):
    """
    Upload a new CSV or Excel file containing sensor features.
    Predicts flash point for all rows and returns the results.
    """
    sanitized_name = sanitize_filename(file.filename)
    if not sanitized_name.lower().endswith((".xlsx", ".csv")):
        raise HTTPException(400, "Only .xlsx or .csv files are accepted.")

    # Read content incrementally (streaming size limit check to prevent memory exhaustion)
    try:
        size = 0
        chunks = []
        while True:
            chunk = await file.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(413, f"File too large. Maximum size allowed is {MAX_UPLOAD_SIZE_BYTES / (1024*1024):.1f}MB.")
            chunks.append(chunk)

        content = b"".join(chunks)

        if sanitized_name.lower().endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to parse uploaded batch file: {e}", exc_info=True)
        raise HTTPException(400, "Could not parse file. Verify formatting is correct.")

    if df.empty:
        raise HTTPException(400, "The uploaded file is empty.")

    # Load model artifacts and training medians
    try:
        model, scaler, feature_cols, model_name, rmse = load_artifacts()
        from predict import _cache
        train_medians = _cache.get("train_medians", {})
    except Exception as e:
        logger.error(f"Error loading model artifacts for batch prediction: {e}", exc_info=True)
        raise HTTPException(503, "Model artifacts are not loaded. Run training first.")

    # Build feature matrix in one pass to avoid pandas fragmentation warnings
    feature_columns = []
    for col in feature_cols:
        if col in df.columns:
            feature_columns.append(df[col].rename(col))
        elif col.replace("_mean", "") in df.columns:
            feature_columns.append(df[col.replace("_mean", "")].rename(col))
        else:
            feature_columns.append(pd.Series(train_medians.get(col, 0.0), index=df.index, name=col))

    df_features = pd.concat(feature_columns, axis=1)
    df_features = df_features.fillna({col: train_medians.get(col, 0.0) for col in feature_cols})

    try:
        # Scale and predict
        scaled = pd.DataFrame(scaler.transform(df_features), columns=feature_cols)
        preds = model.predict(scaled).tolist()
    except Exception as e:
        logger.error(f"Batch inference execution failed: {e}", exc_info=True)
        raise HTTPException(400, "Inference execution failed. Verify columns align with process variables.")

    # Build response DataFrame containing predictions
    result_df = df.copy()
    result_df["predicted_flash_point"] = [round(p, 2) for p in preds]
    
    # 95% Confidence interval
    ci_half = CI_Z_SCORE_95 * rmse
    result_df["confidence_lower"] = [round(p - ci_half, 2) for p in preds]
    result_df["confidence_upper"] = [round(p + ci_half, 2) for p in preds]

    # Convert to JSON serializable dictionary
    result_df = result_df.replace([float("inf"), float("-inf")], None)
    result_df = result_df.where(pd.notnull(result_df), None)

    # Log predictions into SQLite database (limit to BATCH_DB_LOG_LIMIT to prevent clogging)
    critical_count = sum(1 for p in preds if p < FLASH_POINT_SPEC_MIN or p > FLASH_POINT_SPEC_MAX)
    if critical_count > 0:
        logger.error(f"🚨 CRITICAL ALERT: {critical_count} critical flash point violations detected in batch upload of {len(preds)} rows!")

    try:
        from database import insert_prediction
        logged_count = 0
        for _, row in result_df.head(BATCH_DB_LOG_LIMIT).iterrows():
            ts_val = None
            if "sample_ts" in row and row["sample_ts"]:
                ts_val = str(row["sample_ts"])
            elif "Timestamp" in row and row["Timestamp"]:
                ts_val = str(row["Timestamp"])
            
            if not ts_val:
                ts_val = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

            shift_val = row.get("shift", row.get("SHIFT", "M"))
            
            row_sensors = {}
            for col in feature_cols:
                tag_name = col.replace("_mean", "").replace("_std", "")
                if tag_name in row:
                    row_sensors[tag_name] = row[tag_name]
                elif col in row:
                    row_sensors[col] = row[col]

            insert_prediction(
                sample_ts=ts_val,
                shift=shift_val,
                predicted=row["predicted_flash_point"],
                confidence_lower=row["confidence_lower"],
                confidence_upper=row["confidence_upper"],
                sensors=row_sensors
            )
            logged_count += 1
        logger.info(f"Successfully logged {logged_count} batch predictions to database.")
    except Exception as db_err:
        logger.error(f"Failed to log batch predictions to database: {db_err}")

    return {
        "rows_processed": len(result_df),
        "model_used": model_name,
        "critical_count": critical_count,
        "data": result_df.to_dict(orient="records"),
        "status": "success"
    }
