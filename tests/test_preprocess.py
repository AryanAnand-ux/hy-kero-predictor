import sys
from pathlib import Path
import pandas as pd

# Add src/ to path so we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from preprocess import compute_sample_timestamp, extract_window_features


def test_default_alignment_window_and_lag():
    import preprocess

    assert preprocess.WINDOW_MINUTES == 45
    assert preprocess.RESIDENCE_TIME_LAG_MINUTES == 75


def test_compute_sample_timestamp():
    # Test morning shift (hour = 6)
    row_m = pd.Series({"DATE": pd.Timestamp("2026-06-24"), "SHIFT": "M"})
    ts_m = compute_sample_timestamp(row_m)
    assert ts_m == pd.Timestamp("2026-06-24 06:00:00")

    # Test evening shift (hour = 14)
    row_e = pd.Series({"DATE": pd.Timestamp("2026-06-24"), "SHIFT": "E"})
    ts_e = compute_sample_timestamp(row_e)
    assert ts_e == pd.Timestamp("2026-06-24 14:00:00")

    # Test night shift (hour = 22)
    row_n = pd.Series({"DATE": pd.Timestamp("2026-06-24"), "SHIFT": "N"})
    ts_n = compute_sample_timestamp(row_n)
    assert ts_n == pd.Timestamp("2026-06-24 22:00:00")


def test_extract_window_features():
    # Construct dummy sensor dataframe
    timestamps = pd.date_range("2026-06-24 05:00:00", "2026-06-24 07:00:00", freq="15min")
    sensor_df = pd.DataFrame({
        "Timestamp": timestamps,
        "Sensor1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "Sensor2": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    })
    
    sample_ts = pd.Timestamp("2026-06-24 06:00:00")
    # Window +/- 30 min contains 05:30, 05:45, 06:00, 06:15, 06:30 (5 values)
    # Sensor1 values: 3.0, 4.0, 5.0, 6.0, 7.0 (mean = 5.0, std = 1.4142)
    # Sensor2 values: 10.0 (mean = 10.0, std = 0.0)
    
    feats = extract_window_features(sensor_df, sample_ts, ["Sensor1", "Sensor2"])
    assert feats is not None
    assert feats["Sensor1_mean"] == 5.0
    assert feats["Sensor2_mean"] == 10.0
    assert "Sensor1_std" not in feats.index
    assert "Sensor2_std" not in feats.index
