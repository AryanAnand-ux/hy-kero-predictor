"""
feature_transforms.py
─────────────────────
Shared single source of truth for row-level (single-observation) feature
computation used during inference.

Both functions mirror the DataFrame-based implementations in features.py
(add_derived_features / add_time_features) but operate on a plain dict so
they can be called from predict.py without constructing a DataFrame.

Exported:
    compute_derived_features(row: dict) -> dict
    compute_time_features(row: dict, timestamp=None) -> dict
"""

import numpy as np
import pandas as pd


def compute_derived_features(row: dict) -> dict:
    """
    Calculate physics-inspired derived features for a single observation.
    Mirrors features.py:add_derived_features() -- only the 4 proven useful features.
    """

    def get(name: str) -> float | None:
        key = f"{name}_mean"
        val = row.get(key)
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return None
        return float(val)

    # Furnace outlet temperature spread (max - min)
    furnace_vals = [get(f"Outlet_temp_11F{i}") for i in range(1, 5)]
    furnace_vals = [v for v in furnace_vals if v is not None]
    if len(furnace_vals) >= 2:
        row["feat_furnace_T_spread"] = max(furnace_vals) - min(furnace_vals)

    # HK / LN draw flow ratio
    hk_flow = get("CDU_Draw_HK_F")
    ln_flow = get("CDU_Draw_LN_MF_F")
    if hk_flow is not None and ln_flow is not None and ln_flow != 0:
        row["feat_HK_LN_draw_ratio"] = hk_flow / ln_flow

    # Total stripping steam
    ss_keys = [k for k in row if k.startswith("SS_") and k.endswith("_mean")]
    ss_vals = [row[k] for k in ss_keys if row[k] is not None and isinstance(row[k], (int, float)) and np.isfinite(row[k])]
    if ss_vals:
        row["feat_total_SS"] = sum(ss_vals)

    # Stripping efficiency: HK draw temp minus Flash Zone temp
    hk_draw = get("MF_HK_Draw_T")
    fz_temp = get("MF_FlashZone_T")
    if hk_draw is not None and fz_temp is not None:
        row["feat_HK_strip_eff"] = hk_draw - fz_temp

    return row


def compute_time_features(row: dict, timestamp=None) -> dict:
    """
    Calculate cyclical time encodings and shift ordinal matching features.py:add_time_features().
    """
    if timestamp is None:
        timestamp = pd.Timestamp.now()

    hour = timestamp.hour
    dow = timestamp.dayofweek
    month = timestamp.month

    row["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
    row["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
    row["dow_sin"] = float(np.sin(2 * np.pi * dow / 7))
    row["dow_cos"] = float(np.cos(2 * np.pi * dow / 7))
    row["month_sin"] = float(np.sin(2 * np.pi * month / 12))
    row["month_cos"] = float(np.cos(2 * np.pi * month / 12))

    # Shift ordinal: M=0, E=1, N=2
    if 2 <= hour < 10:
        row["shift_ord"] = 0
    elif 10 <= hour < 18:
        row["shift_ord"] = 1
    else:
        row["shift_ord"] = 2

    return row
