"""
Smart Grid IoT Energy Consumption & Anomaly Detection Package
============================================================
A modular framework for smart meter telemetry simulation, data preprocessing,
feature engineering, statistical anomaly detection, and scientific visualization.

Course: BIS586 - 5th Semester ISE Minor Project (Section A)
Academic Standards: IEEE 1159 (Power Quality) & ACM SIGENERGY Research Standards
"""

__version__ = "1.0.0"
__author__ = "AbhishekKB001 & Section A Project Team"

from .data_preprocessing import load_raw_data, audit_raw_data_integrity
from .feature_engineering import add_temporal_features, compute_rolling_stats_vectorized, compute_dynamic_zscores
from .anomaly_detection import detect_grid_anomalies, evaluate_anomaly_detection_performance
from .visualization import generate_all_figures

__all__ = [
    "load_raw_data",
    "audit_raw_data_integrity",
    "add_temporal_features",
    "compute_rolling_stats_vectorized",
    "compute_dynamic_zscores",
    "detect_grid_anomalies",
    "evaluate_anomaly_detection_performance",
    "generate_all_figures",
]
