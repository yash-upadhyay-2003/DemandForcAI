from __future__ import annotations

import os
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Configuration & Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Daily Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

DEFAULT_API_URL = os.getenv("FORECAST_API_URL", "https://demandforcai-fastapi-backend.onrender.com")
try:
    DEFAULT_API_URL = st.secrets.get("FORECAST_API_URL", DEFAULT_API_URL)
except Exception:
    pass
DEFAULT_API_URL = DEFAULT_API_URL.rstrip("/")

# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Forecast Controls")
    horizon = st.radio(
        "Forecast Horizon",
        options=[30, 60, 90],
        index=2,
        horizontal=True,
        help="Select the recursive forecast horizon in days."
    )
    
    history_window = st.selectbox(
        "Historical Context View",
        options=["Last 90 Days", "Last 180 Days", "Last 1 Year", "Full History (2011–2014)"],
        index=0,
        help="Controls how much historical data is displayed before the forecast start line."
    )
    
    st.divider()
    api_url = st.text_input(
        "Backend API URL",
        value=DEFAULT_API_URL,
        help="Render FastAPI endpoint URL."
    ).rstrip("/")
    st.caption("Forecasts are generated recursively via the deployed Tuned Ridge pipeline.")

# -----------------------------------------------------------------------------
# Data Fetching
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_api_data(base_url: str, horizon_val: int):
    headers = {"Accept": "application/json"}
    dashboard_res = requests.get(f"{base_url}/api/dashboard?horizon={horizon_val}", headers=headers, timeout=35)
    dashboard_res.raise_for_status()
    dashboard = dashboard_res.json()
    
    meta_res = requests.get(f"{base_url}/api/metadata", headers=headers, timeout=15)
    metadata = meta_res.json() if meta_res.status_code == 200 else {}
    
    health_res = requests.get(f"{base_url}/health", headers=headers, timeout=10)
    health = health_res.json() if health_res.status_code == 200 else {}
    
    return dashboard, metadata, health

try:
    dashboard, metadata, health = fetch_api_data(api_url, horizon)
except Exception as exc:
    st.error("⚠️ Unable to connect to the backend forecasting API.")
    st.info("If Render is spinning up from idle, please wait 30 seconds and refresh.")
    st.code(f"Target API: {api_url}")
    st.caption(f"Error details: {exc}")
    st.stop()

# Helper formatter
def fmt_currency(val: float | int) -> str:
    return f"${val:,.2f}"

# -----------------------------------------------------------------------------
# Header & KPI Metrics
# -----------------------------------------------------------------------------
st.title("Daily Sales Forecasting")
st.caption("Wholesale revenue forecasting powered by regularized recursive lag and product assortment lookbacks.")

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Total Forecast Revenue", fmt_currency(dashboard["total_forecast_revenue"]))
kpi2.metric("Average Daily Forecast", fmt_currency(dashboard["average_daily_forecast_revenue"]))
kpi3.metric("Forecast Horizon", f"{dashboard['horizon']} Days")

# -----------------------------------------------------------------------------
# Main Interactive Chart: Historical Sales & Future Forecast
# -----------------------------------------------------------------------------
st.subheader("Historical Sales & Future Forecast")

chart_raw = pd.DataFrame(dashboard["chart_data"])
chart_raw["Date"] = pd.to_datetime(chart_raw["Date"])

hist_df = chart_raw[chart_raw["Series"] == "Historical"].sort_values("Date").reset_index(drop=True)
fc_df = chart_raw[chart_raw["Series"] == "Forecast"].sort_values("Date").reset_index(drop=True)

# Filter historical window for optimal visual clarity
if history_window == "Last 90 Days":
    cutoff_date = hist_df["Date"].max() - pd.Timedelta(days=90)
    hist_filtered = hist_df[hist_df["Date"] >= cutoff_date]
elif history_window == "Last 180 Days":
    cutoff_date = hist_df["Date"].max() - pd.Timedelta(days=180)
    hist_filtered = hist_df[hist_df["Date"] >= cutoff_date]
elif history_window == "Last 1 Year":
    cutoff_date = hist_df["Date"].max() - pd.Timedelta(days=365)
    hist_filtered = hist_df[hist_df["Date"] >= cutoff_date]
else:
    hist_filtered = hist_df

forecast_start_date = fc_df["Date"].min()

# Build Plotly Figure matching reference design
fig = go.Figure()

# 1. Historical Actual Sales trace (sleek dark line)
fig.add_trace(go.Scatter(
    x=hist_filtered["Date"],
    y=hist_filtered["Revenue"],
    mode="lines",
    name="Historical Actual Sales",
    line=dict(color="#2c3e50", width=1.8),
    hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Actual Sales:</b> $%{y:,.2f}<extra></extra>"
))

# 2. Forecasted Daily Sales trace (vibrant blue with markers)
fig.add_trace(go.Scatter(
    x=fc_df["Date"],
    y=fc_df["Revenue"],
    mode="lines+markers",
    name=f"Forecasted Daily Sales ({horizon} Days)",
    line=dict(color="#1f77b4", width=2.5),
    marker=dict(size=4, color="#1f77b4"),
    hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Forecasted Sales:</b> $%{y:,.2f}<extra></extra>"
))

# 3. Vertical dashed divider line labeled "Forecast Starts"
fig.add_vline(
    x=forecast_start_date.timestamp() * 1000, # plotly ms timestamp
    line_width=1.5,
    line_dash="dash",
    line_color="#7f8c8d",
    annotation_text="Forecast Starts",
    annotation_position="top left",
    annotation_font=dict(size=12, color="#1f77b4")
)

# Responsive Layout & Styling
fig.update_layout(
    height=480,
    margin=dict(l=40, r=20, t=20, b=40),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0.01,
        font=dict(size=12)
    ),
    xaxis=dict(
        showgrid=True,
        gridcolor="#f0f2f5",
        linecolor="#dcdde1",
        tickformat="%b %d<br>%Y"
    ),
    yaxis=dict(
        title="Daily Sales ($)",
        showgrid=True,
        gridcolor="#f0f2f5",
        linecolor="#dcdde1",
        tickprefix="$",
        tickformat=","
    )
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Secondary Row: Monthly Forecast & Historical Performance
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Monthly Forecast Revenue")
    monthly_data = pd.DataFrame(dashboard["monthly_forecast"])
    if not monthly_data.empty:
        fig_bar = go.Figure(data=[
            go.Bar(
                x=monthly_data["Month"],
                y=monthly_data["Forecast_Revenue"],
                marker_color="#1f77b4",
                text=[fmt_currency(v) for v in monthly_data["Forecast_Revenue"]],
                textposition="auto",
                hovertemplate="<b>Month:</b> %{x}<br><b>Projected Total:</b> $%{y:,.2f}<extra></extra>"
            )
        ])
        fig_bar.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=10, b=30),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            yaxis=dict(showgrid=True, gridcolor="#f0f2f5", tickprefix="$", tickformat=","),
            xaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No monthly forecast data available.")

with col_right:
    st.subheader("Historical Performance")
    h1, h2 = st.columns(2)
    h1.metric("Total Historical Revenue", fmt_currency(dashboard["total_historical_revenue"]))
    h2.metric("Average Daily Sales", fmt_currency(dashboard["average_daily_historical_revenue"]))
    st.metric(
        "Peak Sales Day",
        dashboard["peak_sales_date"],
        fmt_currency(dashboard["peak_sales_revenue"])
    )

# -----------------------------------------------------------------------------
# Daily Forecast Table & Download
# -----------------------------------------------------------------------------
st.subheader("Daily Forecast")
daily_fc_df = pd.DataFrame(dashboard["daily_forecast"])
if not daily_fc_df.empty:
    display_table = daily_fc_df.copy()
    display_table["Forecast_Revenue"] = display_table["Forecast_Revenue"].apply(fmt_currency)
    display_table.columns = ["Date", "Forecasted Revenue"]
    st.dataframe(display_table, use_container_width=True, hide_index=True)
    
    csv_export = daily_fc_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Forecast as CSV",
        data=csv_export,
        file_name=f"daily_sales_forecast_{horizon}_days.csv",
        mime="text/csv"
    )

# -----------------------------------------------------------------------------
# Model Performance and Metadata (Preserving Our Exact Ridge Architecture)
# -----------------------------------------------------------------------------
with st.expander("Model Performance and Metadata"):
    st.write(f"**Model:** {metadata.get('model', 'Tuned Ridge Regression')}")
    steps = metadata.get('pipeline_steps', ['StandardScaler', 'Ridge(alpha=1000)'])
    st.write("**Pipeline Architecture:** " + " → ".join(steps))
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Test MAE", fmt_currency(metadata.get('test_mae', 2050.81)))
    m2.metric("Test RMSE", fmt_currency(metadata.get('test_rmse', 3656.24)))
    m3.metric("Test R² Score", f"{metadata.get('test_r2', 0.007):.3f}")
    
    features = metadata.get('features', [
        "Lag_1", "Lag_7", "Lag_30", "Rolling_Mean_7", "Rolling_Mean_30", "Product_Lag_1"
    ])
    st.write(f"**Engineered Features ({len(features)}):** " + ", ".join(features))
    st.write(f"**Historical Timeline:** {metadata.get('history_start', '2011-01-21')} to {metadata.get('history_end', '2014-09-29')}")

st.caption(
    f"Historical Span: {dashboard['history_start']} to {dashboard['history_end']} | "
    f"Forecast Window: {dashboard['forecast_start']} to {dashboard['forecast_end']} | "
    f"Status: {health.get('status', 'Online').upper()}"
)
