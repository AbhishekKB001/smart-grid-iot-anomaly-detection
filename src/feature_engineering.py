"""
Smart Grid IoT Feature Engineering Module
=========================================
Performs temporal decomposition, diurnal segmentation, constant-time O(1)
sliding window statistical transformation, and dynamic rolling Z-score computation.
"""

import numpy as np
import pandas as pd


def classify_diurnal_slice(hour):
    """
    Classifies hour of the day into 4 standard operational grid time-slices:
    - 00:00 - 06:00: Night Valley (Base load & off-peak)
    - 06:00 - 12:00: Morning Peak (Commercial ramp-up & residential wake-up)
    - 12:00 - 18:00: Afternoon Solar/Work (Workday plateau & solar PV offset)
    - 18:00 - 24:00: Evening Peak (Maximum residential demand & grid stress)
    """
    if 0 <= hour < 6:
        return "00:00 - 06:00 (Night Valley)"
    elif 6 <= hour < 12:
        return "06:00 - 12:00 (Morning Peak)"
    elif 12 <= hour < 18:
        return "12:00 - 18:00 (Afternoon Solar/Work)"
    else:
        return "18:00 - 24:00 (Evening Peak)"


def add_temporal_features(df):
    """
    Decomposes timestamps into 11 temporal feature dimensions for Power BI
    hierarchies, time-series slicing, and diurnal behavioral analysis.
    """
    df = df.copy()
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
    df["Date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["Time"] = df["timestamp"].dt.strftime("%H:%M:%S")
    df["Year"] = df["timestamp"].dt.year
    df["Quarter"] = "Q" + df["timestamp"].dt.quarter.astype(str)
    df["Month_Name"] = df["timestamp"].dt.strftime("%B")
    df["Month_Num"] = df["timestamp"].dt.month
    df["Day_Of_Month"] = df["timestamp"].dt.day
    df["Day_Of_Week"] = df["timestamp"].dt.strftime("%A")
    df["Day_Num"] = df["timestamp"].dt.dayofweek + 1
    df["Is_Weekend"] = np.where(df["timestamp"].dt.dayofweek >= 5, "Weekend", "Weekday")
    df["Hour"] = df["timestamp"].dt.hour
    df["Minute"] = df["timestamp"].dt.minute
    df["Time_Slice"] = df["Hour"].apply(classify_diurnal_slice)
    
    return df


def compute_rolling_stats_vectorized(series_values, window_size=96):
    """
    Computes rolling mean and standard deviation in constant O(1) time per step
    using prefix cumulative sums (cumsum and cumsum_sq).
    
    Mathematical formulation:
        S_t = sum_{k=1}^t x_k,    S_t^(2) = sum_{k=1}^t x_k^2
        mean_t = (S_t - S_{t-W}) / W
        var_t  = max(0.0, (S_t^(2) - S_{t-W}^(2)) / W - mean_t^2)
        std_t  = sqrt(var_t)
        
    Args:
        series_values (np.ndarray): 1D array of numerical values.
        window_size (int): Moving window size (default 96 steps = 24 hours at 15-min cadence).
        
    Returns:
        tuple[np.ndarray, np.ndarray]: (rolling_mean, rolling_std)
    """
    series_values = np.asarray(series_values, dtype=np.float64)
    n = len(series_values)
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        
    rolling_mean = np.zeros(n, dtype=np.float64)
    rolling_std = np.zeros(n, dtype=np.float64)
    
    cumsum = np.cumsum(np.insert(series_values, 0, 0.0))
    cumsum_sq = np.cumsum(np.insert(series_values**2, 0, 0.0))
    
    for i in range(n):
        start_idx = max(0, i - window_size + 1)
        k = i - start_idx + 1
        
        sum_w = cumsum[i + 1] - cumsum[start_idx]
        sum_sq_w = cumsum_sq[i + 1] - cumsum_sq[start_idx]
        
        mean_val = sum_w / k
        # max with 0.0 prevents floating point precision underflow into negative variance
        var_val = max(0.0, (sum_sq_w / k) - (mean_val ** 2))
        
        rolling_mean[i] = mean_val
        rolling_std[i] = np.sqrt(var_val)
        
    return rolling_mean, rolling_std


def compute_dynamic_zscores(df, window_size=96, z_threshold=3.0, z_warning_threshold=2.0):
    """
    Groups telemetry by zone_id, sorts chronologically, and computes:
    - 24-hour moving average: rolling_mean_kwh
    - 24-hour moving standard deviation: rolling_std_kwh
    - Dynamic rolling standard score: consumption_zscore = (x - mean) / (std + epsilon)
    - Binary anomaly flags for warning (2.0 <= |Z| < 3.0) and critical (|Z| >= 3.0)
    
    Args:
        df (pd.DataFrame): Telemetry DataFrame containing zone_id, timestamp, and energy_consumption_kwh.
        window_size (int): 96 steps for 24-hour diurnal cycle.
        z_threshold (float): Critical decision boundary (+-3.0 standard deviations).
        z_warning_threshold (float): Warning boundary (+-2.0 standard deviations).
        
    Returns:
        pd.DataFrame: Augmented DataFrame with rolling statistics and Z-scores.
    """
    df = df.copy()
    processed_groups = []
    epsilon = 1e-6
    
    for zone_id, group in df.groupby("zone_id"):
        group = group.sort_values("timestamp").copy()
        kwh_vals = group["energy_consumption_kwh"].to_numpy(dtype=np.float64)
        
        r_mean, r_std = compute_rolling_stats_vectorized(kwh_vals, window_size=window_size)
        z_scores = (kwh_vals - r_mean) / (r_std + epsilon)
        
        group["rolling_mean_kwh"] = np.round(r_mean, 4)
        group["rolling_std_kwh"] = np.round(r_std, 4)
        group["consumption_zscore"] = np.round(z_scores, 4)
        
        group["is_zscore_critical"] = (np.abs(z_scores) >= z_threshold).astype(int)
        group["is_zscore_warning"] = (
            (np.abs(z_scores) >= z_warning_threshold) & (np.abs(z_scores) < z_threshold)
        ).astype(int)
        
        processed_groups.append(group)
        
    return pd.concat(processed_groups, ignore_index=True)
