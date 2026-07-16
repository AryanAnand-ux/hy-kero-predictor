import sys
from pathlib import Path
import pytest
import numpy as np

# Add src/ and backend/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from predict import predict_from_raw, load_artifacts


def test_predict_from_raw_smoke():
    # Verify artifact loader returns correct types
    model, scaler, feature_cols, model_name, rmse = load_artifacts()
    assert model is not None
    assert scaler is not None
    assert isinstance(feature_cols, list)
    assert len(feature_cols) > 0
    assert isinstance(model_name, str)
    assert isinstance(rmse, float)


def test_predict_from_raw_inference():
    # Make a dummy prediction with normal inputs
    sensors = {
        "MF_HK_Draw_T": 224.0,
        "MF_FlashZone_T": 365.0,
        "MF_Top_T": 150.0,
        "Outlet_temp_11F1": 368.0,
        "Outlet_temp_11F2": 368.0,
        "Outlet_temp_11F3": 368.0,
        "Outlet_temp_11F4": 368.0,
        "CDU_Draw_HK_F": 95.0,
        "SS_11C5": 5.0,
    }
    
    result = predict_from_raw(sensors, lag_flash_gc=79.0, lag2_flash_gc=78.0)
    
    assert "predicted_flash_point" in result
    assert "confidence_lower" in result
    assert "confidence_upper" in result
    assert "model_used" in result
    assert result["unit"] == "C"
    
    pred = result["predicted_flash_point"]
    lower = result["confidence_lower"]
    upper = result["confidence_upper"]
    
    assert lower <= pred <= upper
    assert 50.0 <= pred <= 110.0  # reasonable temperature bounds for Heavy Kerosene flash point
