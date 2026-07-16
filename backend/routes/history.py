

import logging
import sys
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

try:
    from ..database import get_predictions, get_prediction_stats
except ImportError:
    from database import get_predictions, get_prediction_stats

logger = logging.getLogger("hykero.routes.history")
router = APIRouter()


@router.get("/history")
def get_history(
    shift: Optional[str] = Query(None, description="Filter by shift: M, E, or N"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD or ISO)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD or ISO)"),
    limit: int = Query(200, ge=1, le=2000, description="Max rows to return (1-2000)"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    # Validate start_date and end_date formats
    start_ts = None
    if start_date:
        try:
            start_ts = pd.Timestamp(start_date)
        except Exception:
            raise HTTPException(400, "Invalid start_date format. Use YYYY-MM-DD.")

    end_ts = None
    if end_date:
        try:
            end_ts = pd.Timestamp(end_date)
        except Exception:
            raise HTTPException(400, "Invalid end_date format. Use YYYY-MM-DD.")

    if start_ts and end_ts and start_ts > end_ts:
        raise HTTPException(400, "start_date cannot be greater than end_date.")

    try:
        start_str = start_ts.strftime("%Y-%m-%d %H:%M:%S") if start_ts else None
        end_str = end_ts.strftime("%Y-%m-%d %H:%M:%S") if end_ts else None

        total_count, records = get_predictions(
            shift=shift,
            start_date=start_str,
            end_date=end_str,
            limit=limit,
            offset=offset
        )

        return {
            "total": total_count,
            "count": len(records),
            "limit": limit,
            "offset": offset,
            "data": records
        }
    except Exception as e:
        logger.error(f"Error fetching prediction history: {e}", exc_info=True)
        raise HTTPException(500, "Failed to retrieve history.")


@router.get("/history/stats")
def get_history_stats_route():
    try:
        stats = get_prediction_stats()
        return stats
    except Exception as e:
        logger.error(f"Error calculating prediction stats: {e}", exc_info=True)
        raise HTTPException(500, "Failed to retrieve history stats.")

