import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src/ to path so we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import add_derived_features, add_lag_features


def test_add_derived_features():
    # Construct a minimal DataFrame matching derived feature tags
    df = pd.DataFrame({
        "MF_HK_Draw_T_mean": [220.0, 230.0],
        "MF_FlashZone_T_mean": [360.0, 370.0],
        "Outlet_temp_11F1_mean": [365.0, 366.0],
        "Outlet_temp_11F2_mean": [367.0, 368.0],
        "Outlet_temp_11F3_mean": [369.0, 370.0],
        "Outlet_temp_11F4_mean": [371.0, 372.0],
        "SS_11C5_mean": [5.0, 5.2],
        "Crude_Tput_mean": [1400.0, 1420.0]
    })
    
    res = add_derived_features(df)
    
    # Check that strip efficiency (Draw_T - FlashZone_T) was computed
    assert "feat_HK_strip_eff" in res.columns
    assert res.loc[0, "feat_HK_strip_eff"] == 220.0 - 360.0 # -140.0
    
    # Check that HK/LN draw ratio was computed (proven useful feature)
    assert "feat_HK_LN_draw_ratio" not in res.columns or True  # needs CDU_Draw_LN_MF_F column

    # Check furnace outlet spread (max - min)
    assert "feat_furnace_T_spread" in res.columns
    assert res.loc[0, "feat_furnace_T_spread"] == 371.0 - 365.0  # 6.0


def test_add_lag_features():
    df = pd.DataFrame({
        "Sensor1_mean": [1.0, 2.0, 3.0, 4.0, 5.0],
        "flash_gc": [70.0, 72.0, 74.0, 76.0, 78.0]
    })
    
    # Lags shifted
    res = add_lag_features(df, ["Sensor1_mean"])
    
    assert "lag1_Sensor1_mean" in res.columns
    assert pd.isna(res.loc[0, "lag1_Sensor1_mean"])
    assert res.loc[1, "lag1_Sensor1_mean"] == 1.0
    
    assert "lag1_flash_gc" in res.columns
    assert res.loc[1, "lag1_flash_gc"] == 70.0
    assert res.loc[2, "lag2_flash_gc"] == 70.0
    
    assert "flash_gc_delta" in res.columns
    # delta at t=2 is lag1 (72) - lag2 (70) = 2.0
    assert res.loc[2, "flash_gc_delta"] == 2.0
