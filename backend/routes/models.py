
import json
import pandas as pd
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException
import joblib

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

try:
    from ..constants import FEATURE_IMPORTANCE_SORT_SENTINEL, DEFAULT_FEATURE_IMPORTANCE_TOP_N
except ImportError:
    from constants import FEATURE_IMPORTANCE_SORT_SENTINEL, DEFAULT_FEATURE_IMPORTANCE_TOP_N

router = APIRouter()
BASE = Path(__file__).resolve().parent.parent.parent
PROC   = BASE / "data" / "processed"
MODELS = BASE / "data" / "models"


@router.get("/model-metrics")
def get_model_metrics():
    metrics_file = PROC / "model_metrics.json"
    cv_file      = PROC / "cv_results.json"
    best_file    = MODELS / "best_model_name.pkl"

    if not metrics_file.exists():
        raise HTTPException(404, "Metrics not found. Run training first.")

    with open(metrics_file) as f:
        metrics_data = json.load(f)

    cv_data = {}
    if cv_file.exists():
        try:
            with open(cv_file) as f:
                cv_data = json.load(f)
        except Exception:
            pass

    best_name = None
    if best_file.exists():
        try:
            best_name = joblib.load(best_file)
        except Exception:
            pass
    if not best_name and metrics_data:
        best_name = next(iter(metrics_data))

    # Flatten for easier frontend consumption
    rows = []
    for model_name, m in metrics_data.items():
        cv_info = cv_data.get(model_name, {})
        rows.append({
            "model":       model_name,
            "train_rmse":  round(m["train"]["rmse"], 3) if "train" in m else None,
            "test_rmse":   round(m["test"]["rmse"],  3) if "test" in m else None,
            "train_mae":   round(m["train"]["mae"],  3) if "train" in m else None,
            "test_mae":    round(m["test"]["mae"],   3) if "test" in m else None,
            "train_r2":    round(m["train"]["r2"],   4) if "train" in m else None,
            "test_r2":     round(m["test"]["r2"],    4) if "test" in m else None,
            "test_mape":   round(m["test"]["mape"],  2) if "test" in m else None,
            "cv_rmse":     cv_info.get("mean_rmse"),
            "cv_folds":    cv_info.get("fold_scores", []),
            "is_best":     model_name == best_name,
        })

    # Sort: best (lowest test RMSE) first
    rows.sort(key=lambda r: r["test_rmse"] if r["test_rmse"] is not None else FEATURE_IMPORTANCE_SORT_SENTINEL)
    return {"best_model": best_name, "models": rows}


@router.get("/feature-importance")
def get_feature_importance(top_n: int = DEFAULT_FEATURE_IMPORTANCE_TOP_N):
    fi_file = PROC / "feature_importance.csv"
    if not fi_file.exists():
        raise HTTPException(404, "Feature importance not found. Run training first.")

    try:
        df = pd.read_csv(fi_file).head(top_n)
        return {"features": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(500, f"Error reading feature importance: {e}")
