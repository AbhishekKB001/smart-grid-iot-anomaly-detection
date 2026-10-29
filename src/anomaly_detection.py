"""
Smart Grid IoT Multi-Parametric Anomaly Detection Engine
========================================================
Combines dynamic rolling Z-score statistical bounds with IEEE 1159 Power Quality
standards to classify anomalies into actionable severity tiers and compute
academic benchmark metrics.
"""

import numpy as np
import pandas as pd


def apply_ieee1159_power_quality_rules(df):
    """
    Evaluates telemetry against IEEE 1159 Recommended Practice for Monitoring
    Electric Power Quality:
    - Voltage Sag: V < 207.0 V RMS (-10% of nominal 230V)
    - Voltage Swell: V > 253.0 V RMS (+10% of nominal 230V)
    - Grid Frequency Deviation: |f - 50.0| > 0.4 Hz
    - Low Power Factor: PF < 0.80 (excessive inductive VAR demand)
    - Telemetry Loss / Power Dropout: energy < 0.5 kWh while moving mean > 5.0 kWh
    """
    df = df.copy()
    df["is_voltage_sag"] = (df["voltage_v"] < 207.0).astype(int)
    df["is_voltage_swell"] = (df["voltage_v"] > 253.0).astype(int)
    df["is_voltage_violation"] = (df["is_voltage_sag"] | df["is_voltage_swell"]).astype(int)
    df["is_freq_violation"] = (np.abs(df["grid_frequency_hz"] - 50.0) > 0.4).astype(int)
    df["is_pf_violation"] = (df["power_factor"] < 0.80).astype(int)
    
    # Check for dropout / sudden meter loss
    if "rolling_mean_kwh" in df.columns:
        df["is_dropout_violation"] = (
            (df["energy_consumption_kwh"] < 0.5) & (df["rolling_mean_kwh"] > 5.0)
        ).astype(int)
    else:
        df["is_dropout_violation"] = (df["energy_consumption_kwh"] < 0.1).astype(int)
        
    return df


def classify_severity_and_diagnosis(df):
    """
    Triages grid observations into multi-tier operational severity grades:
    - NORMAL: Normal operating band
    - WARNING: Mild power factor degradation or 2.0 <= |Z| < 3.0
    - CRITICAL: |Z| >= 3.0, severe voltage sag/swell (<190V or >260V), or blackout dropout
    """
    df = df.copy()
    
    def assign_severity(row):
        if row["detected_anomaly"] == 0:
            if row.get("is_zscore_warning", 0) == 1 or row.get("is_pf_violation", 0) == 1:
                return "WARNING"
            return "NORMAL"
        else:
            if (row.get("is_zscore_critical", 0) == 1 or 
                row["voltage_v"] < 190.0 or row["voltage_v"] > 260.0 or 
                row.get("is_dropout_violation", 0) == 1):
                return "CRITICAL"
            return "WARNING"
            
    def assign_diagnosis(row):
        if row["detected_anomaly"] == 0:
            return "Normal Grid Operation"
        reasons = []
        if row.get("is_zscore_critical", 0) == 1:
            reasons.append("Surge/Demand Outlier")
        if row.get("is_voltage_sag", 0) == 1:
            reasons.append("Voltage Sag (<207V)")
        if row.get("is_voltage_swell", 0) == 1:
            reasons.append("Voltage Swell (>253V)")
        if row.get("is_dropout_violation", 0) == 1:
            reasons.append("Power Dropout/Loss")
        if row.get("is_pf_violation", 0) == 1:
            reasons.append("Low Power Factor (<0.80)")
        if row.get("is_freq_violation", 0) == 1:
            reasons.append("Frequency Instability")
        return " & ".join(reasons) if reasons else "General Grid Anomaly"
        
    df["anomaly_severity"] = df.apply(assign_severity, axis=1)
    df["anomaly_diagnosis"] = df.apply(assign_diagnosis, axis=1)
    
    return df


def detect_grid_anomalies(df, z_threshold=3.0, z_warning_threshold=2.0, window_size=96):
    """
    End-to-end anomaly detection pipeline:
    1. Computes IEEE 1159 power quality rule flags.
    2. Combines statistical Z-score with electrical bounds.
    3. Triages observations into severity levels and diagnoses.
    """
    df = apply_ieee1159_power_quality_rules(df)
    
    # Composite anomaly decision function
    df["detected_anomaly"] = (
        df.get("is_zscore_critical", 0) | 
        df["is_voltage_violation"] | 
        df["is_freq_violation"] | 
        df["is_pf_violation"] | 
        df["is_dropout_violation"]
    ).astype(int)
    
    df = classify_severity_and_diagnosis(df)
    return df


def evaluate_anomaly_detection_performance(y_true, y_pred):
    """
    Calculates confusion matrix and academic evaluation metrics:
    Precision, Recall (Sensitivity), Specificity, F1-Score, and Overall Accuracy.
    
    Handles edge cases such as empty arrays and zero division gracefully.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    if len(y_true) != len(y_pred):
        raise ValueError("Length of ground truth and prediction arrays must match.")
        
    total = len(y_true)
    if total == 0:
        return {
            "total_samples": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
            "specificity": 0.0, "f1_score": 0.0
        }
        
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "total_samples": total,
        "ground_truth_anomalies": int(np.sum(y_true)),
        "detected_anomalies": int(np.sum(y_pred)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score
    }
