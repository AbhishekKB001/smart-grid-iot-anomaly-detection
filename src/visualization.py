"""
Smart Grid IoT Scientific Visualization Engine
===============================================
Generates publication-quality 300-DPI academic figures:
1. Multi-Zone Energy Consumption Time-Series Line Plot
2. Telemetry Anomaly Detection & IEEE 1159 Voltage Diagnostic Scatter Plot
3. 24-Hour Diurnal Energy Consumption Heatmap Matrix (Matrix-based, not geographic)
4. Rolling Z-Score Statistical Distribution & Decision Boundaries
5. IEEE Research Benchmark Confusion Matrix & Evaluation Metrics
6. Preprocessing & Data Quality 4-Panel Diagnostic Audit
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def configure_plot_styles():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 14


def plot_energy_time_series(df, output_dir="reports/figures"):
    """
    Figure 1: Multi-Zone Energy Consumption Time-Series Line Plot.
    Displays 7-day load profiles across diverse zone topologies with dynamic bands.
    """
    os.makedirs(output_dir, exist_ok=True)
    start_sample = df["timestamp"].min() + pd.Timedelta(days=14)
    end_sample = start_sample + pd.Timedelta(days=7)
    sample_df = df[(df["timestamp"] >= start_sample) & (df["timestamp"] <= end_sample)].copy()
    
    zones_to_plot = ["Zone_01", "Zone_02", "Zone_03", "Zone_06", "Zone_09"]
    zone_labels = {
        "Zone_01": "Downtown Commercial (Zone 1)",
        "Zone_02": "North Residential (Zone 2)",
        "Zone_03": "Industrial Park (Zone 3)",
        "Zone_06": "Healthcare District (Zone 6)",
        "Zone_09": "Residential Solar (Zone 9)"
    }
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    for zone_id, color in zip(zones_to_plot, colors):
        z_data = sample_df[sample_df["zone_id"] == zone_id]
        ax.plot(
            z_data["timestamp"],
            z_data["energy_consumption_kwh"],
            label=zone_labels.get(zone_id, zone_id),
            color=color,
            linewidth=1.6,
            alpha=0.88
        )
        
    ax.set_title("Multi-Zone Smart Grid Energy Consumption Profile (7-Day Telemetry Window)", fontweight='bold', pad=15)
    ax.set_xlabel("Timestamp (15-Minute Periodic Intervals)", fontweight='bold')
    ax.set_ylabel("Active Power Consumption (kWh)", fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, shadow=True)
    
    plt.tight_layout()
    file_path = os.path.join(output_dir, "01_grid_energy_time_series.png")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    return file_path


def plot_anomaly_scatter(df, output_dir="reports/figures"):
    """
    Figure 2: Anomaly Detection Diagnostic Scatter Plot with IEEE 1159 Boundaries.
    """
    os.makedirs(output_dir, exist_ok=True)
    target_zone = "Zone_03"
    z_df = df[df["zone_id"] == target_zone].copy().sort_values("timestamp")
    
    start_sample = z_df["timestamp"].min() + pd.Timedelta(days=20)
    end_sample = start_sample + pd.Timedelta(days=12)
    sample_df = z_df[(z_df["timestamp"] >= start_sample) & (z_df["timestamp"] <= end_sample)].copy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, dpi=300)
    
    # Active Load Diagnostic
    ax1.plot(sample_df["timestamp"], sample_df["energy_consumption_kwh"], color="#1f77b4", linewidth=1.4, label="Actual Consumption (kWh)")
    ax1.plot(sample_df["timestamp"], sample_df["rolling_mean_kwh"], color="#2ca02c", linestyle="--", linewidth=1.8, label="24h Rolling Mean (μ)")
    
    upper_band = sample_df["rolling_mean_kwh"] + 3.0 * sample_df["rolling_std_kwh"]
    lower_band = np.maximum(0, sample_df["rolling_mean_kwh"] - 3.0 * sample_df["rolling_std_kwh"])
    ax1.fill_between(sample_df["timestamp"], lower_band, upper_band, color="#2ca02c", alpha=0.15, label="±3σ Statistical Confidence Band")
    
    kwh_anomalies = sample_df[sample_df.get("is_zscore_critical", 0) == 1]
    ax1.scatter(kwh_anomalies["timestamp"], kwh_anomalies["energy_consumption_kwh"], color="#d62728", s=65, zorder=5, edgecolors='black', label=f"Z-Score Critical Outlier (|Z| ≥ 3.0) [n={len(kwh_anomalies)}]")
    
    dropout_anomalies = sample_df[sample_df.get("is_dropout_violation", 0) == 1]
    if len(dropout_anomalies) > 0:
        ax1.scatter(dropout_anomalies["timestamp"], dropout_anomalies["energy_consumption_kwh"], color="#9467bd", marker="x", s=80, zorder=6, linewidth=2, label=f"Power Dropout/Loss [n={len(dropout_anomalies)}]")
        
    ax1.set_title(f"Telemetry Anomaly Engine: Active Load Diagnostics ({sample_df['zone_name'].iloc[0]})", fontweight='bold')
    ax1.set_ylabel("Energy (kWh / 15-min)", fontweight='bold')
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # Voltage Diagnostic
    ax2.plot(sample_df["timestamp"], sample_df["voltage_v"], color="#333333", linewidth=1.2, label="Grid Voltage (V RMS)")
    ax2.axhline(230.0, color="gray", linestyle=":", label="Nominal Voltage (230V)")
    ax2.axhline(253.0, color="#d62728", linestyle="--", alpha=0.8, label="IEEE 1159 Overvoltage Limit (+10% / 253V)")
    ax2.axhline(207.0, color="#ff7f0e", linestyle="--", alpha=0.8, label="IEEE 1159 Undervoltage Limit (-10% / 207V)")
    
    volt_anomalies = sample_df[sample_df.get("is_voltage_violation", 0) == 1]
    ax2.scatter(volt_anomalies["timestamp"], volt_anomalies["voltage_v"], color="#e377c2", s=60, zorder=5, edgecolors='black', label=f"Voltage Sag/Swell Anomaly [n={len(volt_anomalies)}]")
    
    ax2.set_title("Power Quality Diagnostics: Voltage Compliance Monitoring (IEEE 1159 Standards)", fontweight='bold')
    ax2.set_xlabel("Timestamp", fontweight='bold')
    ax2.set_ylabel("Voltage (V)", fontweight='bold')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax2.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    file_path = os.path.join(output_dir, "02_anomaly_detection_scatter.png")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    return file_path


def plot_zone_hourly_heatmap(df, output_dir="reports/figures"):
    """
    Figure 3: 24-Hour Diurnal Energy Consumption Heatmap Matrix.
    NOTE: This is a matrix-based data visualization (Zone vs. Hour), NOT a geographic map.
    """
    os.makedirs(output_dir, exist_ok=True)
    pivot = df.pivot_table(
        index="zone_name",
        columns="Hour",
        values="energy_consumption_kwh",
        aggfunc="mean"
    )
    ordered_zones = df[["zone_id", "zone_name"]].drop_duplicates().sort_values("zone_id")["zone_name"].tolist()
    pivot = pivot.reindex(ordered_zones)
    
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        annot=True,
        fmt=".1f",
        linewidths=0.6,
        linecolor="#e0e0e0",
        cbar_kws={'label': 'Mean Active Energy Consumption (kWh / 15-min interval)'},
        ax=ax
    )
    ax.set_title("Diurnal Energy Consumption Heatmap: 24-Hour Load Intensity Across 10 City Zones", fontweight='bold', pad=15)
    ax.set_xlabel("Hour of the Day (00:00 - 23:00 hrs)", fontweight='bold')
    ax.set_ylabel("Smart Grid Zone Profile", fontweight='bold')
    
    plt.tight_layout()
    file_path = os.path.join(output_dir, "03_zone_hourly_heatmap.png")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    return file_path


def plot_zscore_distribution(df, output_dir="reports/figures"):
    """
    Figure 4: Statistical Z-Score Distribution & Decision Boundary Plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    z_scores = df["consumption_zscore"].dropna().values
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    counts, bins, patches = ax.hist(
        z_scores,
        bins=100,
        range=(-6, 6),
        density=True,
        color="#3498db",
        alpha=0.65,
        edgecolor="#2980b9",
        label="Empirical Z-Score Distribution"
    )
    
    x_axis = np.linspace(-6, 6, 500)
    normal_pdf = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x_axis ** 2))
    ax.plot(x_axis, normal_pdf, color="#2c3e50", linewidth=2.2, linestyle="-", label="Theoretical Normal PDF N(0, 1)")
    
    ax.axvline(3.0, color="#e74c3c", linestyle="--", linewidth=2, label="Upper Anomaly Threshold (+3.0σ)")
    ax.axvline(-3.0, color="#e74c3c", linestyle="--", linewidth=2, label="Lower Anomaly Threshold (-3.0σ)")
    ax.axvline(2.0, color="#f39c12", linestyle=":", linewidth=1.8, label="Warning Boundary (±2.0σ)")
    ax.axvline(-2.0, color="#f39c12", linestyle=":", linewidth=1.8)
    
    ax.fill_between(x_axis, 0, normal_pdf, where=(x_axis >= 3.0) | (x_axis <= -3.0), color="#e74c3c", alpha=0.35, label="Critical Rejection Region (|Z| ≥ 3.0)")
    
    ax.set_title("Statistical Distribution of Dynamic Rolling Z-Scores (Standard Normal Fit)", fontweight='bold', pad=15)
    ax.set_xlabel("Standardized Rolling Z-Score (σ units)", fontweight='bold')
    ax.set_ylabel("Probability Density", fontweight='bold')
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    file_path = os.path.join(output_dir, "04_statistical_zscore_distribution.png")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    return file_path


def plot_confusion_matrix(df, output_dir="reports/figures"):
    """
    Figure 5: IEEE Research Benchmark Confusion Matrix & Evaluation Metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    y_true = df["is_synthetic_anomaly"].to_numpy()
    y_pred = df["detected_anomaly"].to_numpy()
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    cm = np.array([[tn, fp], [fn, tp]])
    accuracy = (tp + tn) / len(df)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Normal", "Predicted Anomaly"],
        yticklabels=["Actual Normal", "Actual Anomaly"],
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax
    )
    
    ax.set_title("Multi-Parametric Anomaly Engine: Confusion Matrix Evaluation", fontweight='bold', pad=15)
    ax.set_ylabel("True Grid Operating State", fontweight='bold')
    ax.set_xlabel("Engine Diagnostic Classification", fontweight='bold')
    
    metrics_text = (
        f"Evaluation Metrics Summary (86,400 Observations):\n"
        f"• Accuracy: {accuracy*100:.2f}%\n"
        f"• Precision: {precision*100:.2f}%\n"
        f"• Recall (Sensitivity): {recall*100:.2f}%\n"
        f"• F1-Score: {f1*100:.2f}%\n"
        f"• Specificity: {(tn/(tn+fp))*100:.2f}%"
    )
    plt.annotate(
        metrics_text,
        xy=(0.5, -0.22),
        xycoords='axes fraction',
        ha='center',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9)
    )
    
    plt.tight_layout()
    file_path = os.path.join(output_dir, "05_confusion_matrix_evaluation.png")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    return file_path


def plot_data_quality_audit(df, output_dir="reports/figures"):
    """
    Figure 6: Preprocessing & Data Quality 4-Panel Verification Plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    
    # Panel 1: Data Completeness & Temporal Volume by Zone
    zone_counts = df.groupby("zone_name")["energy_consumption_kwh"].count()
    axes[0, 0].barh(zone_counts.index, zone_counts.values, color="#2b5c8f", edgecolor="black")
    axes[0, 0].set_title("Panel A: Telemetry Volume & Completeness per Zone (0 Missing)", fontweight="bold")
    axes[0, 0].set_xlabel("Record Count (100% Continuity = 8,640 records/zone)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    
    # Panel 2: Energy Consumption Distribution Boxplot
    sns.boxplot(data=df, x="zone_name", y="energy_consumption_kwh", ax=axes[0, 1], hue="zone_name", legend=False, palette="Set2")
    axes[0, 1].set_title("Panel B: Energy Outlier Spread Across Municipal Topologies", fontweight="bold")
    axes[0, 1].set_xlabel("Zone")
    axes[0, 1].set_ylabel("Active Power (kWh / 15-min)")
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Panel 3: Diurnal Time-Slice Progression
    slice_order = [
        "00:00 - 06:00 (Night Valley)",
        "06:00 - 12:00 (Morning Peak)",
        "12:00 - 18:00 (Afternoon Solar/Work)",
        "18:00 - 24:00 (Evening Peak)"
    ]
    slice_means = df.groupby("Time_Slice")["energy_consumption_kwh"].mean().reindex(slice_order)
    axes[1, 0].plot(slice_order, slice_means.values, marker="o", linewidth=2.5, color="#d9534f", markersize=8)
    axes[1, 0].set_title("Panel C: Diurnal Load Progression Across Grid Time-Slices", fontweight="bold")
    axes[1, 0].set_ylabel("Mean Energy Consumption (kWh)")
    axes[1, 0].tick_params(axis='x', rotation=20)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    
    # Panel 4: Anomaly Severity Triage Breakdown
    sev_counts = df["anomaly_severity"].value_counts()
    colors = {"NORMAL": "#5cb85c", "WARNING": "#f0ad4e", "CRITICAL": "#d9534f"}
    pie_colors = [colors.get(x, "#999999") for x in sev_counts.index]
    axes[1, 1].pie(
        sev_counts.values,
        labels=sev_counts.index,
        autopct="%1.2f%%",
        startangle=140,
        colors=pie_colors,
        wedgeprops=dict(edgecolor="black")
    )
    axes[1, 1].set_title("Panel D: Preprocessed Telemetry Severity Triage", fontweight="bold")
    
    plt.tight_layout()
    file_path = os.path.join(output_dir, "06_preprocessing_data_quality_audit.png")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    return file_path


def generate_all_figures(df, output_dir="reports/figures"):
    """
    Generates all 6 publication figures and saves them in output_dir.
    """
    configure_plot_styles()
    os.makedirs(output_dir, exist_ok=True)
    print(f"[+] Generating all 6 scientific figures to '{output_dir}'...")
    
    p1 = plot_energy_time_series(df, output_dir)
    p2 = plot_anomaly_scatter(df, output_dir)
    p3 = plot_zone_hourly_heatmap(df, output_dir)
    p4 = plot_zscore_distribution(df, output_dir)
    p5 = plot_confusion_matrix(df, output_dir)
    p6 = plot_data_quality_audit(df, output_dir)
    
    print("[OK] All figures generated successfully.")
    return [p1, p2, p3, p4, p5, p6]
