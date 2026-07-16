"""
preprocess.py
─────────────
Merges the 15-min DCS sensor data with SAP Flash GC lab results.

Key logic:
  - For each SAP row (date + shift), identify the lab sample time:
      M → 06:00,  E → 14:00,  N → 22:00
  - Extract sensor readings within ±45 min of that sample time
  - Average those ~4-5 readings into one feature row
  - Join with the Flash GC target value
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
RAW  = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

SENSOR_FILE = BASE / "HY Kero  AI ML Data 15 Mins.xlsx"
SAP_FILE    = BASE / "SAP_HK_DATA Final.xlsx"
TAG_MAP_FILE= BASE / "Tag to names Mapping.xlsx"

# ── Shift → Sample time mapping ───────────────────────────────────────────────
SHIFT_SAMPLE_HOUR = {"M": 6, "E": 14, "N": 22}
WINDOW_MINUTES    = 45   # ±45 min around sample time
RESIDENCE_TIME_LAG_MINUTES = 75   # 75-min lag to align with physical kerosene rundown travel time


def load_tag_mapping() -> dict:
    """Returns {raw_tag: friendly_name} from the mapping file."""
    df = pd.read_excel(TAG_MAP_FILE)
    # columns: From, Unnamed:1, Unnamed:2, Unnamed:3 (raw tag), Tag Names
    tag_col  = df.columns[3]   # raw tag column
    name_col = df.columns[4]   # friendly name column
    mapping  = dict(zip(df[tag_col].astype(str).str.strip(),
                        df[name_col].astype(str).str.strip()))
    return mapping


def load_sensor_data() -> pd.DataFrame:
    """Load 15-min DCS data and rename columns to friendly names."""
    print("Loading sensor data (this may take ~30 sec)…")
    df = pd.read_excel(SENSOR_FILE, sheet_name="Data")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Rename columns using tag mapping
    tag_map = load_tag_mapping()
    rename  = {}
    for col in df.columns:
        key = col.strip()
        if key in tag_map:
            rename[col] = tag_map[key]
    df = df.rename(columns=rename)

    print(f"  Sensor data shape: {df.shape}")
    print(f"  Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    return df


def load_sap_data() -> pd.DataFrame:
    """Load SAP Flash GC results and parse dates."""
    df = pd.read_excel(SAP_FILE, sheet_name="Format")
    df["DATE"]  = pd.to_datetime(df["DATE"])
    df["SHIFT"] = df["SHIFT"].astype(str).str.strip().str.upper()
    df = df[df["SHIFT"].isin(["M", "E", "N"])].copy()
    df = df.dropna(subset=["Flash GC"]).reset_index(drop=True)
    print(f"  SAP data shape: {df.shape}")
    print(f"  Date range: {df['DATE'].min()} to {df['DATE'].max()}")
    return df


def compute_sample_timestamp(row: pd.Series) -> pd.Timestamp:
    """Build the exact lab sample datetime for a given SAP row."""
    hour = SHIFT_SAMPLE_HOUR[row["SHIFT"]]
    return row["DATE"] + pd.Timedelta(hours=hour)


def extract_window_features(sensor_df: pd.DataFrame,
                             sample_ts: pd.Timestamp,
                             feature_cols: list) -> pd.Series | None:
    """
    Extract ±45 min of sensor readings around sample_ts,
    return the column-wise mean (one row of features).
    Returns None if no readings found in the window.
    """
    lo = sample_ts - pd.Timedelta(minutes=WINDOW_MINUTES)
    hi = sample_ts + pd.Timedelta(minutes=WINDOW_MINUTES)
    mask = (sensor_df["Timestamp"] >= lo) & (sensor_df["Timestamp"] <= hi)
    window = sensor_df.loc[mask, feature_cols]

    if window.empty:
        return None

    means = window.mean()
    means.index = [f"{c}_mean" for c in means.index]
    return means


def build_merged_dataset(sensor_df: pd.DataFrame,
                          sap_df: pd.DataFrame) -> pd.DataFrame:
    """Main merge: for each SAP row, find sensor window, join features."""
    feature_cols = [c for c in sensor_df.columns if c != "Timestamp"]

    records = []
    missing = 0
    for _, row in sap_df.iterrows():
        sample_ts = compute_sample_timestamp(row)
        # Shift lookup window by process residence time lag to match physical rundown travel time
        aligned_ts = sample_ts - pd.Timedelta(minutes=RESIDENCE_TIME_LAG_MINUTES)
        feats     = extract_window_features(sensor_df, aligned_ts, feature_cols)

        if feats is None:
            missing += 1
            continue

        record = {
            "date":       row["DATE"],
            "shift":      row["SHIFT"],
            "sample_ts":  sample_ts,
            "flash_gc":   row["Flash GC"],
        }
        record.update(feats.to_dict())
        records.append(record)

    print(f"  Matched: {len(records)} rows | Skipped (no sensor data): {missing}")
    merged = pd.DataFrame(records)
    merged = merged.sort_values("sample_ts").reset_index(drop=True)
    return merged


def run():
    print("\n=== Phase 1: Data Preprocessing ===\n")

    sensor_df = load_sensor_data()
    sap_df    = load_sap_data()

    # Filter to overlapping date range
    overlap_start = max(sensor_df["Timestamp"].min(), sap_df["DATE"].min())
    overlap_end   = min(sensor_df["Timestamp"].max(), sap_df["DATE"].max())
    print(f"\n  Overlap period: {overlap_start.date()} to {overlap_end.date()}")

    sap_df = sap_df[(sap_df["DATE"] >= overlap_start) &
                    (sap_df["DATE"] <= overlap_end)].copy()

    merged = build_merged_dataset(sensor_df, sap_df)

    # Save
    out_path = PROC / "merged_dataset.csv"
    merged.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")
    print(f"  Final dataset shape: {merged.shape}")
    print(f"\n  Flash GC stats:\n{merged['flash_gc'].describe().to_string()}")

    # ── Data quality report ──
    import json
    quality = {
        "total_rows": len(merged),
        "total_columns": len(merged.columns),
        "date_range": {
            "start": str(merged["sample_ts"].min()),
            "end":   str(merged["sample_ts"].max()),
        },
        "target_stats": {
            "mean":  round(float(merged["flash_gc"].mean()), 2),
            "std":   round(float(merged["flash_gc"].std()), 2),
            "min":   round(float(merged["flash_gc"].min()), 2),
            "max":   round(float(merged["flash_gc"].max()), 2),
            "median":round(float(merged["flash_gc"].median()), 2),
        },
        "shift_distribution": merged["shift"].value_counts().to_dict(),
        "nan_percentage": round(float(merged.isnull().mean().mean() * 100), 2),
        "sensor_count": len([c for c in merged.columns if c.endswith("_mean")]),
    }
    with open(PROC / "data_quality_report.json", "w") as f:
        json.dump(quality, f, indent=2)
    print(f"  Saved data_quality_report.json")

    return merged


if __name__ == "__main__":
    run()

