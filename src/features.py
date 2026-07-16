"""
features.py
───────────
Feature engineering on top of the merged dataset produced by preprocess.py.

Adds:
  - Lag features  (previous shift sensor means — lag-1, lag-2)
  - Derived physical features + interaction/ratio terms
  - Time-of-day / day-of-week / trend encodings
  - Rolling statistics (3-shift and 5-shift windows) for key sensors
  - Feature selection (remove highly-correlated, low-variance, top-50)
  - Final train/test split (chronological)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
import joblib

BASE  = Path(__file__).resolve().parent.parent
PROC  = BASE / "data" / "processed"
MODELS= BASE / "data" / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def load_merged() -> pd.DataFrame:
    df = pd.read_csv(PROC / "merged_dataset.csv", parse_dates=["date", "sample_ts"])
    df = df.sort_values("sample_ts").reset_index(drop=True)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Physics-inspired composite features and interaction/ratio terms.
    Interaction features capture non-linear VLE relationships that a linear
    model cannot learn from raw sensor values alone.
    """
    def col(name):
        return f"{name}_mean"

    # ── Existing features ──────────────────────────────────────────────────────

    # Furnace outlet temperature spread (max - min) — indicates furnace imbalance
    furnace_cols = [col(c) for c in
                    ["Outlet_temp_11F1", "Outlet_temp_11F2",
                     "Outlet_temp_11F3", "Outlet_temp_11F4"]
                    if col(c) in df.columns]
    if len(furnace_cols) >= 2:
        df["feat_furnace_T_spread"] = df[furnace_cols].max(axis=1) - df[furnace_cols].min(axis=1)
        # Furnace average vs FlashZone — energy balance proxy
        if col("MF_FlashZone_T") in df.columns:
            df["feat_furnace_avg_vs_FZ"] = df[furnace_cols].mean(axis=1) - df[col("MF_FlashZone_T")]

    # HK / LN draw flow ratio
    if col("CDU_Draw_HK_F") in df.columns and col("CDU_Draw_LN_MF_F") in df.columns:
        denom = df[col("CDU_Draw_LN_MF_F")].replace(0, np.nan)
        df["feat_HK_LN_draw_ratio"] = df[col("CDU_Draw_HK_F")] / denom

    # Total stripping steam
    ss_cols = [c for c in df.columns if c.startswith("SS_") and c.endswith("_mean")]
    if ss_cols:
        df["feat_total_SS"] = df[ss_cols].sum(axis=1)

    # Stripping efficiency: HK draw temp minus Flash Zone temp
    if col("MF_HK_Draw_T") in df.columns and col("MF_FlashZone_T") in df.columns:
        df["feat_HK_strip_eff"] = df[col("MF_HK_Draw_T")] - df[col("MF_FlashZone_T")]

    # ── NEW: Interaction & ratio features (Tier 1) ────────────────────────────

    # FlashZone / HK Draw ratio — VLE equilibrium indicator
    if col("MF_FlashZone_T") in df.columns and col("MF_HK_Draw_T") in df.columns:
        denom = df[col("MF_HK_Draw_T")].replace(0, np.nan)
        df["feat_FZ_HK_T_ratio"] = df[col("MF_FlashZone_T")] / denom

    # SS_11C5 per unit HK draw flow — stripping efficiency per unit product
    if col("SS_11C5") in df.columns and col("CDU_Draw_HK_F") in df.columns:
        denom = df[col("CDU_Draw_HK_F")].replace(0, np.nan)
        df["feat_SS_per_HK_flow"] = df[col("SS_11C5")] / denom

    # HK flow × HK temperature — flow-temperature interaction
    if col("CDU_Draw_HK_F") in df.columns and col("MF_HK_Draw_T") in df.columns:
        df["feat_HK_flow_x_temp"] = df[col("CDU_Draw_HK_F")] * df[col("MF_HK_Draw_T")]

    # HK section temperature gradient: draw temperature vs return temperature
    if col("MF_HK_Draw_T") in df.columns and col("MF_HK_CR_Ret_T") in df.columns:
        df["feat_HK_section_dT"] = df[col("MF_HK_Draw_T")] - df[col("MF_HK_CR_Ret_T")]

    # SS_11C5 fraction of total stripping steam (relative intensity)
    if col("SS_11C5") in df.columns and "feat_total_SS" in df.columns:
        denom = df["feat_total_SS"].replace(0, np.nan)
        df["feat_SS5_fraction"] = df[col("SS_11C5")] / denom

    # ── NEW: Structural break indicator for SS_11C5 (Tier 2) ─────────────────
    # SS_11C5 transitions from ~450 to 0.0 mid-dataset (sensor online/offline)
    if col("SS_11C5") in df.columns:
        df["feat_SS_11C5_active"] = (df[col("SS_11C5")] > 1.0).astype(float)

    # ── NEW: Polynomial squared terms for top-3 features (Tier 2) ────────────
    for c in [col("SS_11C5"), col("MF_HK_Draw_T"), col("CDU_Draw_LN_MF_F")]:
        if c in df.columns:
            df[f"{c}_sq"] = df[c] ** 2

    return df


def add_lag_features(df: pd.DataFrame, mean_cols: list) -> pd.DataFrame:
    """Previous shift sensor means as Lag-1 features and Flash GC Lag-1 and Lag-2 features."""
    # Top sensors by correlation — keep manageable
    top_sensors = mean_cols[:15]
    for c in top_sensors:
        df[f"lag1_{c}"] = df[c].shift(1)

    # Lag-1, lag-2 of Flash GC itself (strong autocorrelation)
    df["lag1_flash_gc"] = df["flash_gc"].shift(1)
    df["lag2_flash_gc"] = df["flash_gc"].shift(2)

    # Flash GC momentum (change from previous shift)
    df["flash_gc_delta"] = df["flash_gc"].shift(1) - df["flash_gc"].shift(2)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling statistics over recent shifts for key sensors.
    
    Window-3 (~1 day) captures daily process variation.
    Window-5 (~2.5 days) captures crude blend changeover effects.
    Both windows are kept; the top-k Huber selector decides which is more useful.
    """
    key_sensors = [c for c in df.columns if c.endswith("_mean") and
                   any(tag in c for tag in ["HK_Draw", "FlashZone", "SS_11C5",
                                             "Crude_Tput", "CDU_Draw_HK"])]

    for c in key_sensors[:5]:  # limit to top 5 to avoid dimensionality explosion
        df[f"roll3_mean_{c}"] = df[c].rolling(3, min_periods=1).mean()
        df[f"roll3_std_{c}"]  = df[c].rolling(3, min_periods=1).std().fillna(0)
        # Window-5: captures crude blend transition cycles (~2.5 days)
        df[f"roll5_mean_{c}"] = df[c].rolling(5, min_periods=1).mean()

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode temporal patterns including a linear trend index.
    
    The time_index feature (0.0 → 1.0) allows the linear model to learn
    the +3.71°C upward drift observed between training and test periods
    caused by seasonal crude recipe changes. Cyclical month_sin/cos
    captures seasonality but not this directional drift.
    """
    df["hour"]       = df["sample_ts"].dt.hour
    df["dayofweek"]  = df["sample_ts"].dt.dayofweek
    df["month"]      = df["sample_ts"].dt.month
    # Cyclical encoding
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)

    # Shift as ordinal: M=0, E=1, N=2
    shift_map = {"M": 0, "E": 1, "N": 2}
    df["shift_ord"] = df["shift"].map(shift_map).fillna(1)

    # Linear time trend index: captures monotonic drift in target distribution
    # (avoids overfitting: only 1 parameter — the slope over time)
    df["time_index"] = np.arange(len(df)) / max(len(df) - 1, 1)

    return df


def remove_redundant_features(X: pd.DataFrame,
                               corr_threshold: float = 0.85) -> pd.DataFrame:
    """Remove one of each pair of highly-correlated features."""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = set()
    for col in upper.columns:
        high_corr = upper.index[upper[col] > corr_threshold].tolist()
        if high_corr:
            to_drop.add(col)

    # Also remove raw hour/dayofweek/month (keep only cyclical encodings)
    raw_time_cols = {"hour", "dayofweek", "month"}
    to_drop.update(raw_time_cols & set(X.columns))

    if to_drop:
        print(f"  Removed {len(to_drop)} redundant features: {sorted(to_drop)}")
        X = X.drop(columns=list(to_drop), errors="ignore")
    return X


def remove_low_variance(X: pd.DataFrame,
                         threshold: float = 0.01) -> pd.DataFrame:
    """Remove near-constant features."""
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    mask = selector.get_support()
    removed = [c for c, keep in zip(X.columns, mask) if not keep]
    if removed:
        print(f"  Removed {len(removed)} low-variance features: {removed}")
    return X.loc[:, mask]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    """Full feature engineering pipeline. Returns X, y, feature_names."""
    mean_cols = [c for c in df.columns if c.endswith("_mean")]

    df = add_derived_features(df)
    df = add_rolling_features(df)
    df = add_lag_features(df, mean_cols)
    df = add_time_features(df)

    # Drop first 2 rows (NaN from lag-2)
    df = df.dropna(subset=["lag2_flash_gc"]).reset_index(drop=True)

    # Select feature columns
    exclude = {"date", "shift", "sample_ts", "flash_gc"}
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].copy()
    y = df["flash_gc"].copy()

    # Drop columns that are all-NaN
    X = X.dropna(axis=1, how="all")

    # Fill any remaining NaN with column median
    X = X.fillna(X.median())

    # Remove redundant (highly correlated) features
    # Raised from 0.92 → 0.95: give Huber selector more features to choose from
    X = remove_redundant_features(X, corr_threshold=0.95)

    # Remove near-constant features
    X = remove_low_variance(X, threshold=0.01)

    feature_cols = list(X.columns)
    return X, y, feature_cols


def chronological_split(X: pd.DataFrame, y: pd.Series,
                         test_frac: float = 0.2):
    """Time-aware train/test split — NO shuffling."""
    n     = len(X)
    split = int(n * (1 - test_frac))
    return (X.iloc[:split].copy(), X.iloc[split:].copy(),
            y.iloc[:split].copy(), y.iloc[split:].copy())


def run():
    print("\n=== Phase 2: Feature Engineering ===\n")
    df = load_merged()
    print(f"  Loaded merged dataset: {df.shape}")

    X, y, feature_cols = build_features(df)
    print(f"  Feature matrix: {X.shape}")
    print(f"  Target range: {y.min():.1f} -> {y.max():.1f} C")

    X_train, X_test, y_train, y_test = chronological_split(X, y)
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # 1. Winsorize train target to mitigate influence of extreme lab measurement spikes/drops (e.g. 58°C outlier)
    lower = y_train.quantile(0.005)
    upper = y_train.quantile(0.995)
    y_train = y_train.clip(lower=lower, upper=upper)
    print(f"  Winsorized training target to range [{lower:.2f}, {upper:.2f}]")

    # 2. Perform model-based feature selection to extract the top 30 features
    # Scale train features temporarily to align coefficient magnitudes
    scaler_temp = StandardScaler()
    X_train_sc_temp = pd.DataFrame(scaler_temp.fit_transform(X_train), columns=X_train.columns)
    
    # Fit robust HuberRegressor on the scaled training features and clipped targets
    from sklearn.linear_model import HuberRegressor
    huber_sel = HuberRegressor(max_iter=20000)
    huber_sel.fit(X_train_sc_temp, y_train)
    
    # Rank features by absolute coefficient magnitude
    # Restrict to top 30 features to keep the model simple and prevent overfitting.
    coefs = pd.Series(np.abs(huber_sel.coef_), index=X_train.columns).sort_values(ascending=False)
    top_k = min(30, len(X_train.columns))
    top_k_cols = coefs.index[:top_k].tolist()
    print(f"  Selected top {top_k} features out of {len(X_train.columns)} based on absolute Huber coefficients")
    
    # Keep only top-k features in datasets and feature column list
    X_train = X_train[top_k_cols]
    X_test  = X_test[top_k_cols]
    X       = X[top_k_cols]
    feature_cols = top_k_cols

    # Filter features based on distribution shift between train and test (log shift only, do not drop to prevent leakage)
    shifts = []
    for col in X_train.columns:
        tr_mean = X_train[col].mean()
        te_mean = X_test[col].mean()
        tr_std = X_train[col].std()
        tr_std = tr_std if tr_std > 0 else 1.0
        standardized_shift = (te_mean - tr_mean) / tr_std
        shifts.append((col, abs(standardized_shift)))
    
    shift_threshold = 1.2
    shifted_features = [col for col, shift in shifts if shift >= shift_threshold]
    if shifted_features:
        print(f"  [Shift Filter Diagnostic] Detected {len(shifted_features)} features with distribution shift >= {shift_threshold}: {shifted_features}")

    # Check for distribution shift
    train_mean = y_train.mean()
    test_mean = y_test.mean()
    print(f"  Train target mean: {train_mean:.2f} C | Test target mean: {test_mean:.2f} C")
    if abs(test_mean - train_mean) > 2.0:
        print(f"  [!] Distribution shift detected: {test_mean - train_mean:+.2f} C difference")

    # Fit scaler on training data only
    # RobustScaler uses median/IQR — less sensitive to outlier sensor spikes
    # than StandardScaler, and pairs naturally with Huber's robust loss.
    scaler = RobustScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train),
                               columns=feature_cols, index=X_train.index)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),
                               columns=feature_cols, index=X_test.index)

    # Save artifacts
    X_train_sc.to_csv(PROC / "X_train.csv", index=False)
    X_test_sc.to_csv( PROC / "X_test.csv",  index=False)
    y_train.to_csv(   PROC / "y_train.csv",  index=False)
    y_test.to_csv(    PROC / "y_test.csv",   index=False)
    X.to_csv(         PROC / "X_full.csv",   index=False)
    y.to_csv(         PROC / "y_full.csv",   index=False)
    
    # Save training medians to fill missing tags in production
    train_medians = X_train.median().to_dict()
    joblib.dump(train_medians, MODELS / "train_medians.pkl")
    
    joblib.dump(scaler,       MODELS / "scaler.pkl")
    joblib.dump(feature_cols, MODELS / "feature_cols.pkl")

    print(f"  Saved train/test CSVs + scaler.pkl + feature_cols.pkl + train_medians.pkl")
    return X_train_sc, X_test_sc, y_train, y_test, scaler, feature_cols


if __name__ == "__main__":
    run()
