"""
Smart Grid IoT Energy Consumption & Anomaly Detection Dashboard
==============================================================
Interactive Operations & Monitoring Web Application built with Streamlit and Plotly.

Course: BIS586 - 5th Semester ISE Minor Project (Section A)
Academic Standards: IEEE 1159 (Power Quality Monitoring) & ACM SIGENERGY Guidelines

Launch command:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Smart Grid IoT Anomaly Engine | BIS586",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive cards and UI refinement
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%);
        border-radius: 10px;
        padding: 14px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #111827;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #10B981;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_grid_data():
    """
    Loads preprocessed smart grid telemetry with fallback path resolution.
    """
    candidate_paths = [
        "data/processed/PowerBI_Ready_GridData.csv",
        "data/processed/preprocessed_smart_grid_data.csv",
        os.path.join(os.path.dirname(__file__), "..", "data", "processed", "PowerBI_Ready_GridData.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "processed", "preprocessed_smart_grid_data.csv"),
        "PowerBI_Ready_GridData.csv",
        "preprocessed_smart_grid_data.csv",
    ]
    
    target_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            target_path = path
            break
            
    if target_path is None:
        return None
        
    df = pd.read_csv(target_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        df["Date"] = df["timestamp"].dt.date
    return df


def main():
    df = load_grid_data()
    if df is None:
        st.error(
            "⚠️ Preprocessed dataset `PowerBI_Ready_GridData.csv` was not found!\n\n"
            "Please run the pipeline first:\n"
            "```bash\n"
            "python src/data_generation.py\n"
            "python src/data_preprocessing.py\n"
            "```"
        )
        st.stop()
        
    st.markdown('<div class="main-header">⚡ Smart Grid IoT Energy & Anomaly Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header"><b>Course Code: BIS586</b> | 5th Semester Information Science & Engineering Minor Project (Section A) | <i>IEEE 1159 & ACM Power Quality Standards</i></div>', unsafe_allow_html=True)
    
    # Notice on synthetic data
    st.info("ℹ️ **Dataset Note:** Telemetry displayed is a high-fidelity synthetic benchmark dataset modeled across 10 municipal zone topologies with programmatically injected IEEE 1159 electrical anomalies for academic evaluation.")
    
    # Sidebar Controls
    st.sidebar.title("🎛️ Grid Control Center")
    st.sidebar.markdown("---")
    
    all_zones = sorted(df["zone_name"].unique())
    selected_zones = st.sidebar.multiselect("📍 Select City Zones", options=all_zones, default=all_zones[:4])
    if not selected_zones:
        selected_zones = all_zones
        
    min_date = df["Date"].min()
    if hasattr(min_date, "date"):
        min_date = min_date.date()
    max_date = df["Date"].max()
    if hasattr(max_date, "date"):
        max_date = max_date.date()
        
    default_end = min(min_date + pd.Timedelta(days=14), max_date)
    
    date_range = st.sidebar.date_input(
        "📅 Date Window Filter",
        value=(min_date, default_end),
        min_value=min_date,
        max_value=max_date
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_filter, end_filter = date_range
    else:
        start_filter = min_date
        end_filter = default_end
        
    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 Statistical Calibration")
    z_threshold = st.sidebar.slider(
        "Dynamic Z-Score Threshold (|Z| ≥ σ)",
        min_value=1.5,
        max_value=5.0,
        value=3.0,
        step=0.1
    )
    
    severity_filter = st.sidebar.selectbox(
        "⚠️ Anomaly Severity Filter",
        options=["All Readings", "Anomalies Only (Critical & Warning)", "Critical Only", "Normal Only"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ IEEE 1159 Grid Standards")
    enable_ieee = st.sidebar.checkbox("Enforce Voltage Bounds (207V - 253V)", value=True)
    
    # Filter Data
    date_series = df["Date"].dt.date if hasattr(df["Date"], "dt") else df["Date"]
    filtered_df = df[
        (df["zone_name"].isin(selected_zones)) & 
        (date_series >= start_filter) & 
        (date_series <= end_filter)
    ].copy()
    
    filtered_df["dynamic_zscore_anomaly"] = (np.abs(filtered_df["consumption_zscore"]) >= z_threshold).astype(int)
    
    if enable_ieee:
        filtered_df["dynamic_detected_anomaly"] = (
            filtered_df["dynamic_zscore_anomaly"] |
            filtered_df.get("is_voltage_violation", 0) |
            filtered_df.get("is_freq_violation", 0) |
            filtered_df.get("is_pf_violation", 0) |
            filtered_df.get("is_dropout_violation", 0)
        ).astype(int)
    else:
        filtered_df["dynamic_detected_anomaly"] = filtered_df["dynamic_zscore_anomaly"]
        
    if severity_filter == "Anomalies Only (Critical & Warning)":
        display_df = filtered_df[filtered_df["dynamic_detected_anomaly"] == 1]
    elif severity_filter == "Critical Only":
        display_df = filtered_df[filtered_df["anomaly_severity"] == "CRITICAL"]
    elif severity_filter == "Normal Only":
        display_df = filtered_df[filtered_df["dynamic_detected_anomaly"] == 0]
    else:
        display_df = filtered_df
        
    # KPI Metrics Banner
    total_energy_mwh = filtered_df["energy_consumption_kwh"].sum() / 1000.0
    total_samples = len(filtered_df)
    total_anomalies = int(filtered_df["dynamic_detected_anomaly"].sum())
    anomaly_rate = (total_anomalies / total_samples * 100.0) if total_samples > 0 else 0.0
    voltage_violations = int(filtered_df.get("is_voltage_violation", 0).sum())
    voltage_compliance = 100.0 - ((voltage_violations / total_samples * 100.0) if total_samples > 0 else 0.0)
    mean_pf = filtered_df["power_factor"].mean()
    grid_health_index = max(0.0, 100.0 - (anomaly_rate * 8.5))
    
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    with kpi1:
        st.markdown(f'<div class="metric-card" style="border-left-color: #3B82F6;"><div class="metric-title">Total Active Energy</div><div class="metric-value">{total_energy_mwh:,.1f} <span style="font-size: 0.9rem;">MWh</span></div><div class="metric-sub">{total_samples:,} readings</div></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="metric-card" style="border-left-color: #10B981;"><div class="metric-title">Grid Health Index</div><div class="metric-value">{grid_health_index:.1f}%</div><div class="metric-sub">Power Quality Score</div></div>', unsafe_allow_html=True)
    with kpi3:
        st.markdown(f'<div class="metric-card" style="border-left-color: #EF4444;"><div class="metric-title">Anomalies Detected</div><div class="metric-value">{total_anomalies:,}</div><div class="metric-sub" style="color: #EF4444;">{anomaly_rate:.2f}% anomaly rate</div></div>', unsafe_allow_html=True)
    with kpi4:
        st.markdown(f'<div class="metric-card" style="border-left-color: #F59E0B;"><div class="metric-title">Voltage Compliance</div><div class="metric-value">{voltage_compliance:.1f}%</div><div class="metric-sub">{voltage_violations:,} IEEE 1159 events</div></div>', unsafe_allow_html=True)
    with kpi5:
        st.markdown(f'<div class="metric-card" style="border-left-color: #8B5CF6;"><div class="metric-title">Mean Power Factor</div><div class="metric-value">{mean_pf:.3f}</div><div class="metric-sub">Target: ≥ 0.900</div></div>', unsafe_allow_html=True)
    with kpi6:
        st.markdown(f'<div class="metric-card" style="border-left-color: #6366F1;"><div class="metric-title">Z-Threshold Set</div><div class="metric-value">±{z_threshold:.1f}σ</div><div class="metric-sub">Decision Boundary</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Time-Series & Anomaly Diagnostics",
        "🗺️ 24-Hour Diurnal Heatmap Matrix",
        "⚡ Power Quality & IEEE 1159 Compliance",
        "📋 Incident Log & CSV Export"
    ])
    
    with tab1:
        st.subheader("📊 Dynamic Telemetry & Moving Statistical Confidence Bands")
        inspect_zone = st.selectbox("Select Zone for In-Depth Telemetry Inspection:", options=selected_zones, index=0)
        z_df = filtered_df[filtered_df["zone_name"] == inspect_zone].sort_values("timestamp")
        
        if len(z_df) == 0:
            st.warning("No data available for the selected filters.")
        else:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=(
                    f"Active Energy Consumption (kWh) & Statistical Confidence Band (±{z_threshold}σ)",
                    "Grid RMS Voltage (V) & IEEE 1159 Compliance Boundaries (207V - 253V)"
                )
            )
            upper_band = z_df["rolling_mean_kwh"] + z_threshold * z_df["rolling_std_kwh"]
            lower_band = np.maximum(0, z_df["rolling_mean_kwh"] - z_threshold * z_df["rolling_std_kwh"])
            
            fig.add_trace(go.Scatter(x=z_df["timestamp"], y=upper_band, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
            fig.add_trace(go.Scatter(x=z_df["timestamp"], y=lower_band, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(46, 204, 113, 0.15)', name=f'±{z_threshold}σ Band'), row=1, col=1)
            fig.add_trace(go.Scatter(x=z_df["timestamp"], y=z_df["energy_consumption_kwh"], mode='lines', line=dict(color='#2563EB', width=1.5), name='Actual Consumption (kWh)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=z_df["timestamp"], y=z_df["rolling_mean_kwh"], mode='lines', line=dict(color='#059669', width=2, dash='dash'), name='24h Rolling Mean (μ)'), row=1, col=1)
            
            anom_points = z_df[z_df["dynamic_detected_anomaly"] == 1]
            if len(anom_points) > 0:
                fig.add_trace(go.Scatter(
                    x=anom_points["timestamp"], y=anom_points["energy_consumption_kwh"],
                    mode='markers', marker=dict(color='#DC2626', size=9, symbol='circle', line=dict(color='black', width=1)),
                    name=f'Detected Anomaly (n={len(anom_points)})'
                ), row=1, col=1)
                
            fig.add_trace(go.Scatter(x=z_df["timestamp"], y=z_df["voltage_v"], mode='lines', line=dict(color='#374151', width=1.3), name='Grid Voltage (V RMS)'), row=2, col=1)
            fig.add_hline(y=230.0, line=dict(color="gray", width=1, dash="dot"), row=2, col=1)
            fig.add_hline(y=253.0, line=dict(color="#DC2626", width=1.5, dash="dash"), row=2, col=1)
            fig.add_hline(y=207.0, line=dict(color="#D97706", width=1.5, dash="dash"), row=2, col=1)
            
            fig.update_layout(height=650, template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
    with tab2:
        st.subheader("🗺️ 24-Hour Diurnal Energy Consumption Heatmap Matrix")
        st.caption("ℹ️ *Note: This is a matrix-based visualization of mean active load (kWh) indexed by Zone vs. Hour of Day, illustrating diurnal human demand patterns across urban topologies. It is not a geographic map.*")
        
        pivot_data = filtered_df.pivot_table(index="zone_name", columns="Hour", values="energy_consumption_kwh", aggfunc="mean")
        fig_heat = px.imshow(
            pivot_data,
            labels=dict(x="Hour of Day (00:00 - 23:00)", y="Zone Profile", color="Mean kWh"),
            color_continuous_scale="YlOrRd",
            text_auto=".1f",
            aspect="auto"
        )
        fig_heat.update_layout(height=480)
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with tab3:
        st.subheader("⚡ Power Quality Compliance & Electrical Diagnostics (IEEE 1159)")
        c1, c2 = st.columns(2)
        with c1:
            fig_v = px.histogram(
                filtered_df, x="voltage_v", nbins=80, color="zone_type",
                title="RMS Line Voltage Distribution vs IEEE 1159 Limits"
            )
            fig_v.add_vline(x=207.0, line_dash="dash", line_color="orange", annotation_text="Undervoltage (-10%)")
            fig_v.add_vline(x=253.0, line_dash="dash", line_color="red", annotation_text="Overvoltage (+10%)")
            st.plotly_chart(fig_v, use_container_width=True)
            
        with c2:
            fig_pf = px.histogram(
                filtered_df, x="power_factor", nbins=60, color="zone_type",
                title="Grid Power Factor Distribution (Compliance Target ≥ 0.80)"
            )
            fig_pf.add_vline(x=0.80, line_dash="dash", line_color="red", annotation_text="Minimum PF Threshold")
            st.plotly_chart(fig_pf, use_container_width=True)
            
    with tab4:
        st.subheader("📋 Grid Anomaly Incident Logs & Export")
        st.dataframe(
            display_df[[
                "timestamp", "zone_name", "energy_consumption_kwh",
                "rolling_mean_kwh", "consumption_zscore", "voltage_v",
                "power_factor", "anomaly_severity", "anomaly_diagnosis"
            ]].head(500),
            use_container_width=True
        )
        csv_download = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Telemetry & Incident Records (CSV)",
            data=csv_download,
            file_name="smart_grid_filtered_incidents.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
