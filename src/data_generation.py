"""
Smart Grid IoT Telemetry Synthesis & Anomaly Injection Engine
=============================================================
Synthesizes 15-minute resolution smart grid telemetry across 10 municipal zone
topologies for 90 continuous days (86,400 observations).

Programmatically injects ground-truth IEEE 1159 electrical power quality anomalies:
- Active load surges & dropouts
- Voltage sags (<207V RMS) and swells (>253V RMS)
- Reactive power & low power factor distortions (PF < 0.80)
- Grid frequency instabilities (|f - 50.0| > 0.4 Hz)
"""

import os
import numpy as np
import pandas as pd


def get_zone_profiles():
    """
    Defines baseline load profiles, characteristics, and daily diurnal shapes
    for 10 distinct city zones reflecting urban electrical grid topologies.
    """
    return [
        {
            "zone_id": "Zone_01",
            "zone_name": "Downtown Commercial",
            "zone_type": "Commercial",
            "meter_id": "SM-Z01-001",
            "base_kwh": 35.0,
            "peak_kwh": 85.0,
            "peak_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            "weekend_factor": 0.45,
            "temp_sensitivity": 0.8,
            "nominal_pf": 0.94
        },
        {
            "zone_id": "Zone_02",
            "zone_name": "North Residential",
            "zone_type": "Residential",
            "meter_id": "SM-Z02-001",
            "base_kwh": 12.0,
            "peak_kwh": 42.0,
            "peak_hours": [6, 7, 8, 18, 19, 20, 21, 22],
            "weekend_factor": 1.25,
            "temp_sensitivity": 0.6,
            "nominal_pf": 0.96
        },
        {
            "zone_id": "Zone_03",
            "zone_name": "Industrial Park",
            "zone_type": "Industrial",
            "meter_id": "SM-Z03-001",
            "base_kwh": 90.0,
            "peak_kwh": 140.0,
            "peak_hours": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "weekend_factor": 0.80,
            "temp_sensitivity": 0.3,
            "nominal_pf": 0.88
        },
        {
            "zone_id": "Zone_04",
            "zone_name": "Tech Hub & Data Center",
            "zone_type": "High-Tech Commercial",
            "meter_id": "SM-Z04-001",
            "base_kwh": 65.0,
            "peak_kwh": 95.0,
            "peak_hours": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "weekend_factor": 0.70,
            "temp_sensitivity": 0.5,
            "nominal_pf": 0.97
        },
        {
            "zone_id": "Zone_05",
            "zone_name": "Suburban West",
            "zone_type": "Residential",
            "meter_id": "SM-Z05-001",
            "base_kwh": 10.0,
            "peak_kwh": 38.0,
            "peak_hours": [6, 7, 8, 17, 18, 19, 20, 21, 22, 23],
            "weekend_factor": 1.30,
            "temp_sensitivity": 0.7,
            "nominal_pf": 0.95
        },
        {
            "zone_id": "Zone_06",
            "zone_name": "Healthcare District",
            "zone_type": "Institutional Critical",
            "meter_id": "SM-Z06-001",
            "base_kwh": 55.0,
            "peak_kwh": 80.0,
            "peak_hours": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "weekend_factor": 0.95,
            "temp_sensitivity": 0.4,
            "nominal_pf": 0.98
        },
        {
            "zone_id": "Zone_07",
            "zone_name": "Harbor & Logistics Terminal",
            "zone_type": "Industrial Logistics",
            "meter_id": "SM-Z07-001",
            "base_kwh": 40.0,
            "peak_kwh": 110.0,
            "peak_hours": [4, 5, 6, 7, 14, 15, 16, 20, 21, 22],
            "weekend_factor": 0.60,
            "temp_sensitivity": 0.2,
            "nominal_pf": 0.90
        },
        {
            "zone_id": "Zone_08",
            "zone_name": "University Campus",
            "zone_type": "Educational",
            "meter_id": "SM-Z08-001",
            "base_kwh": 25.0,
            "peak_kwh": 70.0,
            "peak_hours": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            "weekend_factor": 0.35,
            "temp_sensitivity": 0.6,
            "nominal_pf": 0.93
        },
        {
            "zone_id": "Zone_09",
            "zone_name": "South Suburban Solar District",
            "zone_type": "Residential Solar",
            "meter_id": "SM-Z09-001",
            "base_kwh": 11.0,
            "peak_kwh": 36.0,
            "peak_hours": [6, 7, 18, 19, 20, 21, 22],
            "weekend_factor": 1.20,
            "temp_sensitivity": 0.5,
            "nominal_pf": 0.95
        },
        {
            "zone_id": "Zone_10",
            "zone_name": "Agricultural Outskirts",
            "zone_type": "Agricultural",
            "meter_id": "SM-Z10-001",
            "base_kwh": 8.0,
            "peak_kwh": 45.0,
            "peak_hours": [22, 23, 0, 1, 2, 3, 4, 5],
            "weekend_factor": 1.05,
            "temp_sensitivity": 0.9,
            "nominal_pf": 0.89
        }
    ]


def generate_smart_grid_dataset(
    start_date="2026-06-01 00:00:00",
    days=90,
    freq="15min",
    random_seed=42,
    output_path="data/raw/raw_smart_grid_data.csv"
):
    """
    Synthesizes the full 90-day 15-minute telemetry dataset for all 10 zones.
    """
    rng = np.random.default_rng(random_seed)
    timestamps = pd.date_range(start=start_date, periods=days * 96, freq=freq)
    zones = get_zone_profiles()
    all_data = []

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    t_int = np.arange(len(timestamps))
    base_temp = 25.0 + 8.0 * np.sin(2 * np.pi * (t_int % 96) / 96.0 - np.pi / 2)
    daily_seasonal_temp = 3.0 * np.sin(2 * np.pi * t_int / (96.0 * 30.0))
    ambient_temp = base_temp + daily_seasonal_temp + rng.normal(0, 1.2, len(timestamps))

    for zone in zones:
        n_samples = len(timestamps)
        base_load = np.full(n_samples, zone["base_kwh"])
        peak_delta = zone["peak_kwh"] - zone["base_kwh"]
        
        hours = timestamps.hour
        is_peak = np.isin(hours, zone["peak_hours"]).astype(float)
        diurnal_variation = peak_delta * is_peak * (0.8 + 0.4 * np.sin(np.pi * (timestamps.minute / 60.0)))
        
        is_weekend = np.isin(timestamps.dayofweek, [5, 6]).astype(float)
        weekend_mult = 1.0 + is_weekend * (zone["weekend_factor"] - 1.0)
        
        temp_effect = zone["temp_sensitivity"] * np.maximum(0, ambient_temp - 24.0)
        stochastic_noise = rng.normal(0, 0.04 * zone["peak_kwh"], n_samples)
        
        energy = (base_load + diurnal_variation) * weekend_mult + temp_effect + stochastic_noise
        energy = np.clip(energy, a_min=0.5, a_max=None)
        
        voltage = rng.normal(230.0, 3.2, n_samples)
        power_factor = rng.normal(zone["nominal_pf"], 0.015, n_samples)
        power_factor = np.clip(power_factor, 0.70, 0.99)
        frequency = rng.normal(50.0, 0.05, n_samples)
        
        apparent_power_kva = energy / (power_factor * 0.25)
        current = (apparent_power_kva * 1000.0) / (np.sqrt(3) * voltage)
        reactive_power_kvarh = energy * np.tan(np.arccos(power_factor))
        
        is_anomaly = np.zeros(n_samples, dtype=int)
        anomaly_type = np.full(n_samples, "NORMAL", dtype=object)
        
        # Inject realistic anomalies (2.5% rate)
        n_anomalies = int(n_samples * 0.025)
        anomaly_indices = rng.choice(n_samples, size=n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            anomaly_class = rng.choice(["SURGE", "DROPOUT", "VOLTAGE_SAG", "VOLTAGE_SWELL", "LOW_PF", "FREQ_DEV"],
                                       p=[0.30, 0.20, 0.20, 0.15, 0.10, 0.05])
            is_anomaly[idx] = 1
            anomaly_type[idx] = anomaly_class
            
            if anomaly_class == "SURGE":
                energy[idx] *= rng.uniform(2.4, 4.0)
                current[idx] *= rng.uniform(2.0, 3.5)
            elif anomaly_class == "DROPOUT":
                energy[idx] = rng.uniform(0.01, 0.2)
                current[idx] = rng.uniform(0.1, 0.5)
            elif anomaly_class == "VOLTAGE_SAG":
                voltage[idx] = rng.uniform(180.0, 204.0)
                current[idx] *= rng.uniform(1.15, 1.30)
            elif anomaly_class == "VOLTAGE_SWELL":
                voltage[idx] = rng.uniform(255.0, 275.0)
            elif anomaly_class == "LOW_PF":
                power_factor[idx] = rng.uniform(0.60, 0.78)
                reactive_power_kvarh[idx] = energy[idx] * np.tan(np.arccos(power_factor[idx]))
            elif anomaly_class == "FREQ_DEV":
                frequency[idx] += rng.choice([-1, 1]) * rng.uniform(0.5, 1.2)
                
        zone_df = pd.DataFrame({
            "timestamp": timestamps,
            "meter_id": zone["meter_id"],
            "zone_id": zone["zone_id"],
            "zone_name": zone["zone_name"],
            "zone_type": zone["zone_type"],
            "energy_consumption_kwh": np.round(energy, 3),
            "reactive_power_kvarh": np.round(reactive_power_kvarh, 3),
            "voltage_v": np.round(voltage, 2),
            "current_a": np.round(current, 2),
            "power_factor": np.round(power_factor, 3),
            "grid_frequency_hz": np.round(frequency, 3),
            "ambient_temp_c": np.round(ambient_temp, 2),
            "is_synthetic_anomaly": is_anomaly,
            "synthetic_anomaly_type": anomaly_type
        })
        all_data.append(zone_df)

    df_full = pd.concat(all_data, ignore_index=True)
    df_full = df_full.sort_values(["timestamp", "zone_id"]).reset_index(drop=True)
    df_full.to_csv(output_path, index=False)
    print(f"[OK] Generated {len(df_full):,} raw telemetry records saved to '{output_path}'.")
    return df_full


if __name__ == "__main__":
    generate_smart_grid_dataset()
