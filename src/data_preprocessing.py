"""
Smart Grid IoT Telemetry Preprocessing & Quality Auditing Module
================================================================
Handles data loading, schema type enforcement, missing value detection,
duplicate detection, timestamp continuity verification, and physical electrical range validation.
"""

import os
import numpy as np
import pandas as pd


def load_raw_data(filepath="data/raw/raw_smart_grid_data.csv"):
    """
    Loads raw telemetry CSV and parses the timestamp column.
    Falls back to alternative relative paths if necessary.
    """
    candidate_paths = [
        filepath,
        os.path.join(os.path.dirname(__file__), "..", filepath),
        os.path.join(os.path.dirname(__file__), "..", "data", "raw", "raw_smart_grid_data.csv"),
        "raw_smart_grid_data.csv"
    ]
    
    target_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            target_path = path
            break
            
    if target_path is None:
        raise FileNotFoundError(f"Raw dataset file not found in candidate paths: {candidate_paths}")
        
    df = pd.read_csv(target_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def audit_raw_data_integrity(df):
    """
    Performs comprehensive data quality audit on raw smart grid telemetry.
    Checks:
    1. Total records and columns
    2. Missing values per column
    3. Duplicate meter readings (timestamp + meter_id)
    4. Monotonic cadence gaps (verifying expected 15-minute sampling interval)
    5. Physical range violations for electrical quantities
    
    Returns:
        dict: A structured audit summary dictionary.
    """
    total_records = len(df)
    missing_by_column = df.isnull().sum().to_dict()
    total_missing = sum(missing_by_column.values())
    duplicate_rows = int(df.duplicated(subset=["timestamp", "meter_id"]).sum())
    
    # Check timestamp monotonicity per zone
    df_sorted = df.sort_values(["zone_id", "timestamp"]).copy()
    df_sorted["time_diff"] = df_sorted.groupby("zone_id")["timestamp"].diff()
    expected_delta = pd.Timedelta(minutes=15)
    cadence_violations = int((df_sorted["time_diff"].dropna() != expected_delta).sum())
    
    # Physical Range Validations
    kwh_neg = int((df["energy_consumption_kwh"] < 0).sum())
    v_out_of_bounds = int(((df["voltage_v"] < 50.0) | (df["voltage_v"] > 320.0)).sum())
    pf_out_of_bounds = int(((df["power_factor"] < 0.0) | (df["power_factor"] > 1.0)).sum())
    freq_out_of_bounds = int(((df["grid_frequency_hz"] < 45.0) | (df["grid_frequency_hz"] > 55.0)).sum())
    
    audit_report = {
        "total_records": total_records,
        "distinct_meters": int(df["meter_id"].nunique()),
        "distinct_zones": int(df["zone_id"].nunique()),
        "start_timestamp": str(df["timestamp"].min()),
        "end_timestamp": str(df["timestamp"].max()),
        "total_missing_cells": total_missing,
        "missing_percentage": (total_missing / (total_records * len(df.columns))) * 100.0,
        "duplicate_rows": duplicate_rows,
        "cadence_violations": cadence_violations,
        "physical_violations": {
            "negative_kwh": kwh_neg,
            "voltage_extrema": v_out_of_bounds,
            "power_factor_bounds": pf_out_of_bounds,
            "frequency_extrema": freq_out_of_bounds
        },
        "missing_by_column": missing_by_column
    }
    
    return audit_report


def validate_and_clean_telemetry(df):
    """
    Validates data types, removes any duplicate timestamps per meter,
    sorts observations chronologically, and returns a verified DataFrame.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Deduplicate if necessary
    initial_len = len(df)
    df = df.drop_duplicates(subset=["timestamp", "meter_id"], keep="first")
    if len(df) < initial_len:
        print(f"[*] Removed {initial_len - len(df)} duplicate records.")
        
    # Sort chronologically per zone
    df = df.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)
    
    # Type enforcements
    float_cols = [
        "energy_consumption_kwh", "reactive_power_kvarh",
        "voltage_v", "current_a", "power_factor",
        "grid_frequency_hz", "ambient_temp_c"
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.float64)
            
    return df
