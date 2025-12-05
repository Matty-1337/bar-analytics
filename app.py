import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import folium
from folium import plugins
from streamlit_folium import st_folium
from datetime import datetime
import os

# =============================
# PAGE CONFIG
# =============================

st.set_page_config(
    page_title="Best Regards Intelligence System",
    layout="wide"
)

# =============================
# UI THEME (GLASS DARK)
# =============================

st.markdown("""
<style>

/* Core background */
body {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e5e7eb;
}
.block-container {
    padding-top: 1.5rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #020617);
    border-right: 1px solid rgba(255,255,255,.08);
}

/* Glass Cards */
.glass {
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    background: rgba(255,255,255,.08);
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,.12);
    padding: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,.4);
}

/* Executive Cards */
.kpi {
    text-align: center;
    padding: 25px;
    min-height: 140px;
}
.kpi h1 {
    font-size: 2.2rem;
    margin: 0;
}
.kpi p {
    opacity: .7;
    margin: 0;
}

/* Titles */
h1, h2, h3 {
    font-weight: 600;
    letter-spacing: .3px;
    color: #f8fafc;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,.06);
    border-radius: 14px;
    padding: 12px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #9333ea);
    color: white;
}

/* Charts */
.js-plotly-plot {
    border-radius: 16px;
    overflow: hidden;
}

/* Buttons */
button {
    border-radius: 12px !important;
}

/* Remove Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)


# =============================
# DATA LOADERS
# =============================

def safe_load(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

data = {
    "forecast": safe_load("forecast_values.csv"),
    "metrics": safe_load("forecast_metrics.csv"),
    "menu": safe_load("menu_forensics.csv"),
    "map": safe_load("map_data.csv"),
    "servers": safe_load("suspicious_servers.csv"),
    "geo": safe_load("geo_pressure.csv"),
    "sentiment": safe_load("sentiment.csv")
}


# =============================
# SANITIZATION
# =============================

def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                .str.replace(",", "")
                .str.replace("$", "")
                .astype(float, errors="ignore")
                .fillna(0)
            )
    return df

data["menu"] = to_num(data["menu"], ["Total_Revenue", "Qty_Sold"])
data["servers"] = to_num(data["servers"], ["Void_Rate", "Void_Z_Score", "Potential_Loss"])
data["sentiment"] = to_num(data["sentiment"], ["CX_Index", "BestRegards_Revenue"])

# =============================
# KPI ENGINE
# =============================

def kpis():
    theft = data["servers"].get("Potential_Loss", pd.Series(dtype=float)).sum()
    high_risk = len(data["servers"].query("Void_Z_Score >= 2")) if "Void_Z_Score" in data["servers"].columns else 0
    sentiment = data["sentiment"]["CX_Index"].iloc[-1] if not data["sentiment"].empty else 0

    stars = len(data["menu"].query("BCG_Matrix == 'Star'")) if "BCG_Matrix" in data["menu"].columns else 0
    dogs = len(data["menu"].query("BCG_Matrix == 'Dog'")) if "BCG_Matrix" in data["menu"].columns else 0

    forecast = data["forecast"]
    if "Revenue" in forecast.columns:
        avg_rev = forecast["Revenue"].head(3).mean()
    else:
        avg_rev = 0

    return avg_rev, theft, high_risk, sentiment, stars, dogs

avg_rev, theft, high_risk, sentiment, stars, dogs = kpis()


# =============================
# SIDEBAR
# =============================

with st.sidebar:
    st.markdown("<h2>Best Regards</h2>", unsafe_allow_html=True)
    st.markdown("<p>Executive Intelligence Portal</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.metric("Average Monthly Revenue", f"${avg_rev:,.0f}")
    st.metric("Estimated Monthly Loss", f"${theft:,.0f}")
    st.metric("High Risk Employees", f"{high_risk}")
    st.metric("Customer Sentiment Index", f"{sentiment:.2f}")

    st.caption(f"Data Timestamp: {datetime.now().strftime('%B %d, %Y')}")

# =============================
# HEADER KPI ROW
# =============================

st.markdown("## Business Intelligence Overview")

c1, c2, c3, c4 = st.columns(4)

def kpi_box(title, value, subtitle):
    return f"""
    <div class='glass kpi'>
        <h1>{value}</h1>
        <p>{title}</p>
        <small>{subtitle}</small>
    </div>
    """

c1.markdown(kpi_box("Revenue Projection", f"${avg_rev:,.0f}", "Monthly Average"), unsafe_allow_html=True)
c2.markdown(kpi_box("Operational Loss", f"${theft:,.0f}", "Estimated Monthly"), unsafe_allow_html=True)
c3.markdown(kpi_box("High Risk Staff", high_risk, "Requiring Review"), unsafe_allow_html=True)
c4.markdown(kpi_box("Customer Sentiment", f"{sentiment:.2f}", "CX Index"), unsafe_allow_html=True)


# =============================
# TABS
# =============================

tabs = st.tabs([
    "Revenue Forecast",
    "Menu Engineering",
    "Competitive Environment",
    "Operational Risk",
    "Sentiment",
    "Executive Summary"
])

# =============================
# FORECAST TAB
# =============================

with tabs[0]:
    st.markdown("### Revenue Projection")
    df = data["forecast"]

    if not df.empty:
        df.columns = ["Month", "Revenue"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Month"],
            y=df["Revenue"],
            mode="lines+markers",
            name="Projected Revenue"
        ))

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Forecast data not available.")


# =============================
# MENU TAB
# =============================

with tabs[1]:
    st.markdown("### Menu Performance Matrix")

    df = data["menu"]

    if not df.empty and "BCG_Matrix" in df.columns:

        fig = go.Figure()

        for cat in ["Star", "Plowhorse", "Puzzle", "Dog"]:
            subset = df[df["BCG_Matrix"] == cat]
            fig.add_trace(go.Scatter(
                x=subset["Qty_Sold"],
                y=subset["Total_Revenue"],
                mode="markers",
                name=cat
            ))

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Menu data unavailable.")


# =============================
# COMPETITIVE TAB
# =============================

with tabs[2]:
    st.markdown("### Competitive Environment")

    df = data["map"]
    if "Latitude" in df.columns and "Longitude" in df.columns:

        center = [df["Latitude"].mean(), df["Longitude"].mean()]
        m = folium.Map(location=center, zoom_start=14, tiles="cartodbdark_matter")

        for _, r in df.iterrows():
            folium.CircleMarker(
                [r["Latitude"], r["Longitude"]],
                radius=6,
                color="#7c3aed",
                fill=True
            ).add_to(m)

        st_folium(m, height=600, width="100%")

    else:
        st.info("Map data unavailable.")


# =============================
# RISK TAB
# =============================

with tabs[3]:
    st.markdown("### Internal Risk & Loss Analysis")

    df = data["servers"]
    if not df.empty:
        st.dataframe(df, use_container_width=True)

        recovery = df.get("Potential_Loss", pd.Series()).sum()
        st.markdown(f"Estimated Monthly Recovery Potential: ${recovery:,.0f}")

    else:
        st.info("No risk data found.")


# =============================
# SENTIMENT TAB
# =============================

with tabs[4]:
    st.markdown("### Customer Sentiment")

    df = data["sentiment"]
    if not df.empty:

        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["Month"], y=df["BestRegards_Revenue"], name="Revenue"))
        fig.add_trace(go.Scatter(x=df["Month"], y=df["CX_Index"], yaxis="y2", name="CX Index"))

        fig.update_layout(
            yaxis2=dict(overlaying="y", side="right"),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sentiment data unavailable.")


# =============================
# SUMMARY TAB
# =============================

with tabs[5]:
    st.markdown("### Board Summary")

    opportunity = (theft + stars * 1500 - dogs * 800) * 12

    st.markdown(f"""
    <div class='glass'>
    <h2>Estimated Annual Performance Improvement</h2>
    <h1>${opportunity:,.0f}</h1>

    <p><strong>Primary Drivers</strong></p>
    <ul>
    <li>Internal loss elimination</li>
    <li>Menu rationalization</li>
    <li>Training improvements</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
