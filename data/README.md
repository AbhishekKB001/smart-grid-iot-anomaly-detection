# Smart Grid IoT Telemetry Dataset Documentation

## 1. Provenance & Dataset Nature

> ⚠️ **IMPORTANT NOTE ON DATASET PROVENANCE:**  
> This dataset is **100% synthetically generated** for academic research and evaluation in **Course BIS586 (Information Science and Engineering Minor Project, Section A)**.  
> It is **not** real-world utility customer telemetry. It was programmatically synthesized using [`src/data_generation.py`](../src/data_generation.py) to simulate realistic urban electrical distribution topologies and to inject ground-truth electrical power quality anomalies in strict adherence to **IEEE 1159 standards**.

---

## 2. Dataset Overview

* **Temporal Scope:** 90 continuous days (June 1, 2026 – August 29, 2026).
* **Sampling Cadence:** 15-minute periodic intervals ($96\text{ readings/day/meter}$).
* **Total Records:** 86,400 observations across 10 municipal zone topologies ($8,640\text{ records/zone}$).
* **Temporal Continuity:** 100% unbroken chronological timeline (0 cadence gaps, 0 null cells, 0 duplicate timestamps).

---

## 3. Directory Layout

```
data/
├── README.md               # Dataset documentation and schema specifications
├── raw/
│   └── raw_smart_grid_data.csv        # Raw synthesized telemetry (14 columns, 86,400 rows)
└── processed/
    ├── preprocessed_smart_grid_data.csv # Preprocessed dataset with rolling statistics & labels (41 columns)
    └── PowerBI_Ready_GridData.csv      # Star-Schema aligned dataset for Microsoft Power BI
```

---

## 4. Municipal Zone Profiles (10 Urban Topologies)

| Zone ID | Zone Name | Zone Type | Meter ID | Baseline Load | Peak Load | Peak Operating Hours |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| `Zone_01` | Downtown Commercial | Commercial | `SM-Z01-001` | 35.0 kWh | 85.0 kWh | 09:00 – 18:00 (Workday plateau) |
| `Zone_02` | North Residential | Residential | `SM-Z02-001` | 12.0 kWh | 42.0 kWh | 06:00–08:30 & 18:00–22:00 (Bimodal) |
| `Zone_03` | Industrial Park | Industrial | `SM-Z03-001` | 90.0 kWh | 140.0 kWh | 08:00 – 20:00 (High continuous load) |
| `Zone_04` | Tech Hub & Data Center | High-Tech | `SM-Z04-001` | 65.0 kWh | 95.0 kWh | 08:00 – 19:00 (Continuous server HVAC) |
| `Zone_05` | Suburban West | Residential | `SM-Z05-001` | 10.0 kWh | 38.0 kWh | 06:00–08:00 & 17:00–23:00 (Commuter) |
| `Zone_06` | Healthcare District | Institutional | `SM-Z06-001` | 55.0 kWh | 80.0 kWh | 24/7 Flat High (Critical hospital load) |
| `Zone_07` | Harbor & Logistics Terminal | Industrial Logistics | `SM-Z07-001` | 40.0 kWh | 110.0 kWh | 04:00–07:00 & 14:00–22:00 (Freight) |
| `Zone_08` | University Campus | Educational | `SM-Z08-001` | 25.0 kWh | 70.0 kWh | 08:00 – 17:00 (Classroom/Lab cycle) |
| `Zone_09` | South Suburban Solar | Residential Solar | `SM-Z09-001` | 11.0 kWh | 36.0 kWh | 11:00 – 15:00 (Solar PV "Duck Curve") |
| `Zone_10` | Agricultural Outskirts | Agricultural | `SM-Z10-001` | 8.0 kWh | 45.0 kWh | 22:00 – 05:00 (Nocturnal water pumping) |

---

## 5. Schema Data Dictionary

### 5.1 Raw Dataset Schema (`data/raw/raw_smart_grid_data.csv`)

| Column Name | Data Type | Physical Range | Unit / Meaning |
| :--- | :--- | :--- | :--- |
| `timestamp` | `datetime64[ns]` | `2026-06-01` to `2026-08-29` | ISO-8601 periodic timestamp (15-min cadence) |
| `meter_id` | `string` | `SM-Z01-001` to `SM-Z10-001` | Unique smart meter hardware identifier |
| `zone_id` | `string` | `Zone_01` to `Zone_10` | Substation distribution zone identifier |
| `zone_name` | `string` | 10 municipal names | Human-readable municipal zone name |
| `zone_type` | `string` | Commercial / Residential / etc. | Urban consumer classification |
| `energy_consumption_kwh` | `float64` | $0.01$ to $550.0\text{ kWh}$ | Active electrical energy consumed per 15-min interval |
| `reactive_power_kvarh` | `float64` | $0.00$ to $250.0\text{ kvarh}$ | Reactive electrical energy ($Q = P \tan \theta$) |
| `voltage_v` | `float64` | $175.0$ to $285.0\text{ V}$ | RMS line-to-neutral voltage (Nominal 230V) |
| `current_a` | `float64` | $0.1$ to $150.0\text{ A}$ | Calculated line RMS current draw |
| `power_factor` | `float64` | $0.55$ to $0.99$ | Displacement power factor ($\cos \theta$) |
| `grid_frequency_hz` | `float64` | $48.5$ to $51.5\text{ Hz}$ | AC system frequency (Nominal 50.0 Hz) |
| `ambient_temp_c` | `float64` | $16.0$ to $38.0\text{ ^\circ C}$ | Ambient dry-bulb air temperature |
| `is_synthetic_anomaly` | `int64` | `0` (Normal) or `1` (Anomaly) | Programmatic ground-truth label |
| `synthetic_anomaly_type` | `string` | `SURGE`, `DROPOUT`, etc. | Ground-truth fault mechanism |

### 5.2 Preprocessed & Power BI Schema Additions (`data/processed/PowerBI_Ready_GridData.csv`)

The preprocessed dataset expands to 41 features including:
* **Temporal Hierarchies:** `Date`, `Time`, `Year`, `Quarter`, `Month_Name`, `Month_Num`, `Day_Of_Month`, `Day_Of_Week`, `Day_Num`, `Is_Weekend`, `Hour`, `Minute`, `Time_Slice`.
* **Rolling Statistics:** `rolling_mean_kwh`, `rolling_std_kwh` (24-hour moving window $W=96$).
* **Standard Scores:** `consumption_zscore` ($Z_t = \frac{x_t - \mu_t}{\sigma_t + \epsilon}$).
* **IEEE 1159 Boundary Flags:** `is_voltage_sag`, `is_voltage_swell`, `is_voltage_violation`, `is_freq_violation`, `is_pf_violation`, `is_dropout_violation`, `detected_anomaly`.
* **Semantic Triage:** `anomaly_severity` (`NORMAL`, `WARNING`, `CRITICAL`), `anomaly_diagnosis`.

---

## 6. How to Regenerate the Data

To regenerate the raw telemetry from scratch:

```bash
# From repository root:
python src/data_generation.py
```

To run preprocessing, compute rolling Z-scores, and generate Power BI tables:

```bash
python src/data_preprocessing.py
```
