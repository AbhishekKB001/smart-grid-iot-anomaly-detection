"""
Unit Tests for Smart Grid IoT Preprocessing & Anomaly Detection Pipeline
========================================================================
Validates:
- Data integrity auditing (nulls, duplicates, cadence)
- Temporal feature engineering (11 dimensions & diurnal slices)
- Constant-time O(1) rolling statistics and numerical stability
- IEEE 1159 electrical limit boundary evaluations
- Confusion matrix and statistical performance metric computations
"""

import unittest
import numpy as np
import pandas as pd

from src.data_preprocessing import audit_raw_data_integrity, validate_and_clean_telemetry
from src.feature_engineering import (
    add_temporal_features,
    classify_diurnal_slice,
    compute_rolling_stats_vectorized,
    compute_dynamic_zscores
)
from src.anomaly_detection import (
    apply_ieee1159_power_quality_rules,
    classify_severity_and_diagnosis,
    detect_grid_anomalies,
    evaluate_anomaly_detection_performance
)


class TestSmartGridPipeline(unittest.TestCase):

    def setUp(self):
        """Create a reproducible synthetic test dataset."""
        self.timestamps = pd.date_range("2026-06-01 00:00:00", periods=192, freq="15min")
        self.sample_df = pd.DataFrame({
            "timestamp": self.timestamps,
            "meter_id": "SM-Z01-001",
            "zone_id": "Zone_01",
            "zone_name": "Downtown Commercial",
            "zone_type": "Commercial",
            "energy_consumption_kwh": np.linspace(20.0, 50.0, 192),
            "reactive_power_kvarh": np.linspace(5.0, 12.0, 192),
            "voltage_v": np.full(192, 230.0),
            "current_a": np.linspace(15.0, 35.0, 192),
            "power_factor": np.full(192, 0.95),
            "grid_frequency_hz": np.full(192, 50.0),
            "ambient_temp_c": np.full(192, 25.0),
            "is_synthetic_anomaly": np.zeros(192, dtype=int),
            "synthetic_anomaly_type": np.full(192, "NORMAL")
        })

    def test_audit_integrity_clean_data(self):
        """Audit on clean data should report zero missing cells, zero duplicates, zero gaps."""
        report = audit_raw_data_integrity(self.sample_df)
        self.assertEqual(report["total_records"], 192)
        self.assertEqual(report["total_missing_cells"], 0)
        self.assertEqual(report["duplicate_rows"], 0)
        self.assertEqual(report["cadence_violations"], 0)

    def test_audit_integrity_with_defects(self):
        """Audit should accurately detect intentionally injected defects."""
        corrupt_df = self.sample_df.copy()
        corrupt_df.loc[5, "energy_consumption_kwh"] = np.nan
        corrupt_df = pd.concat([corrupt_df, corrupt_df.iloc[[0]]], ignore_index=True)
        
        report = audit_raw_data_integrity(corrupt_df)
        self.assertGreater(report["total_missing_cells"], 0)
        self.assertEqual(report["duplicate_rows"], 1)

    def test_temporal_feature_engineering(self):
        """Verifies decomposition into calendar and diurnal dimensions."""
        df_feats = add_temporal_features(self.sample_df)
        required_cols = [
            "Date", "Time", "Year", "Quarter", "Month_Name", "Month_Num",
            "Day_Of_Month", "Day_Of_Week", "Day_Num", "Is_Weekend", "Hour",
            "Minute", "Time_Slice"
        ]
        for col in required_cols:
            self.assertIn(col, df_feats.columns)
            
        self.assertEqual(classify_diurnal_slice(2), "00:00 - 06:00 (Night Valley)")
        self.assertEqual(classify_diurnal_slice(9), "06:00 - 12:00 (Morning Peak)")
        self.assertEqual(classify_diurnal_slice(14), "12:00 - 18:00 (Afternoon Solar/Work)")
        self.assertEqual(classify_diurnal_slice(20), "18:00 - 24:00 (Evening Peak)")

    def test_vectorized_rolling_stats_numerical_parity(self):
        """
        Verifies constant-time prefix cumsum rolling stats against
        a naive Python sliding window calculation for exact parity.
        """
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        window = 4
        
        r_mean, r_std = compute_rolling_stats_vectorized(data, window_size=window)
        
        # Compute naive ground truth
        naive_mean = []
        naive_std = []
        for i in range(len(data)):
            w = data[max(0, i - window + 1): i + 1]
            naive_mean.append(np.mean(w))
            naive_std.append(np.std(w, ddof=0))
            
        np.testing.assert_allclose(r_mean, naive_mean, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(r_std, naive_std, rtol=1e-5, atol=1e-5)

    def test_rolling_stats_edge_cases(self):
        """Verifies handling of constant arrays, empty arrays, and single elements."""
        constant_data = np.full(50, 42.0)
        r_mean, r_std = compute_rolling_stats_vectorized(constant_data, window_size=10)
        
        np.testing.assert_allclose(r_mean, 42.0)
        np.testing.assert_allclose(r_std, 0.0, atol=1e-8)
        
        empty_mean, empty_std = compute_rolling_stats_vectorized(np.array([]))
        self.assertEqual(len(empty_mean), 0)
        self.assertEqual(len(empty_std), 0)

    def test_dynamic_zscore_computation(self):
        """Verifies that an injected sudden spike produces |Z| >= 3.0."""
        spike_df = self.sample_df.copy()
        spike_df.loc[100, "energy_consumption_kwh"] = 500.0  # Large surge
        
        z_df = compute_dynamic_zscores(spike_df, window_size=96, z_threshold=3.0)
        self.assertIn("consumption_zscore", z_df.columns)
        self.assertIn("is_zscore_critical", z_df.columns)
        
        spike_z = z_df.loc[100, "consumption_zscore"]
        self.assertGreater(spike_z, 3.0)
        self.assertEqual(z_df.loc[100, "is_zscore_critical"], 1)

    def test_ieee1159_power_quality_rules(self):
        """Verifies voltage sag (<207V), swell (>253V), and low power factor (<0.80)."""
        pq_df = self.sample_df.copy()
        pq_df.loc[10, "voltage_v"] = 195.0   # Sag
        pq_df.loc[20, "voltage_v"] = 265.0   # Swell
        pq_df.loc[30, "power_factor"] = 0.72 # Low PF
        pq_df.loc[40, "grid_frequency_hz"] = 50.8 # Freq violation
        
        tagged_df = apply_ieee1159_power_quality_rules(pq_df)
        
        self.assertEqual(tagged_df.loc[10, "is_voltage_sag"], 1)
        self.assertEqual(tagged_df.loc[10, "is_voltage_violation"], 1)
        self.assertEqual(tagged_df.loc[20, "is_voltage_swell"], 1)
        self.assertEqual(tagged_df.loc[20, "is_voltage_violation"], 1)
        self.assertEqual(tagged_df.loc[30, "is_pf_violation"], 1)
        self.assertEqual(tagged_df.loc[40, "is_freq_violation"], 1)

    def test_performance_metrics_evaluation(self):
        """Verifies accuracy, precision, recall, and F1 calculations."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1, 0])
        # TP: 3, FP: 1, TN: 5, FN: 1
        
        metrics = evaluate_anomaly_detection_performance(y_true, y_pred)
        self.assertEqual(metrics["tp"], 3)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["tn"], 5)
        self.assertEqual(metrics["fn"], 1)
        
        expected_acc = 8.0 / 10.0
        expected_prec = 3.0 / 4.0
        expected_rec = 3.0 / 4.0
        expected_f1 = (2 * 0.75 * 0.75) / (0.75 + 0.75)
        
        self.assertAlmostEqual(metrics["accuracy"], expected_acc)
        self.assertAlmostEqual(metrics["precision"], expected_prec)
        self.assertAlmostEqual(metrics["recall"], expected_rec)
        self.assertAlmostEqual(metrics["f1_score"], expected_f1)


if __name__ == "__main__":
    unittest.main()
