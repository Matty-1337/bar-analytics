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

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #020617);
    border-right: 1px solid rgba(255,255,255,.12);
}

/* Ensure sidebar text is clearly visible */
section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
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

/* Executive KPI cards */
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
    opacity: .8;
    margin: 0;
}
.kpi small {
    opacity: .6;
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
    """Read CSV if it exists, otherwise return empty DataFrame."""
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
    """Convert selected columns to numeric, stripping $ and commas."""
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

data["menu"] = to_num(data["menu"], ["Total_Revenue", "Qty_Sold"])
data["servers"] = to_num(data["servers"], ["Void_Rate", "Void_Z_Score", "Potential_Loss"])
data["sentiment"] = to_num(data["sentiment"], ["CX_Index", "BestRegards_Revenue"])
data["map"] = to_num(data["map"], ["Latitude", "Longitude", "Total_Revenue"])


# =============================
# KPI ENGINE
# =============================

def kpis():
    # Theft and high-risk employees
    theft = data["servers"]["Potential_Loss"].sum() if "Potential_Loss" in data["servers"].columns else 0
    if "Void_Z_Score" in data["servers"].columns:
        high_risk = len(data["servers"].query("Void_Z_Score >= 2"))
    else:
        high_risk = 0

    # Sentiment
    if not data["sentiment"].empty and "CX_Index" in data["sentiment"].columns:
        sentiment_val = data["sentiment"]["CX_Index"].iloc[-1]
    else:
        sentiment_val = 0

    # Menu stars and dogs
    if not data["menu"].empty and "BCG_Matrix" in data["menu"].columns:
        stars = len(data["menu"].query("BCG_Matrix == 'Star'"))
        dogs = len(data["menu"].query("BCG_Matrix == 'Dog'"))
    else:
        stars = 0
        dogs = 0

    # Revenue baseline from forecast
    forecast = data["forecast"].copy()
    if not forecast.empty:
        # Normalise columns: first col Month, second col Revenue
        cols = list(forecast.columns)
        if len(cols) >= 1:
            forecast.rename(columns={cols[0]: "Month"}, inplace=True)
        if len(cols) >= 2:
            forecast.rename(columns={cols[1]: "Revenue"}, inplace=True)

        if "Revenue" in forecast.columns:
            avg_rev = forecast["Revenue"].head(3).mean()
        else:
            avg_rev = 0
    else:
        avg_rev = 0

    return avg_rev, theft, high_risk, sentiment_val, stars, dogs

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
    st.metric("High-Risk Employees", f"{high_risk}")
    st.metric("Customer Sentiment Index", f"{sentiment:.2f}")

    st.caption(f"Data timestamp: {datetime.now().strftime('%B %d, %Y')}")


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

c1.markdown(kpi_box("Revenue Projection", f"${avg_rev:,.0f}", "Monthly average"), unsafe_allow_html=True)
c2.markdown(kpi_box("Operational Loss", f"${theft:,.0f}", "Estimated monthly"), unsafe_allow_html=True)
c3.markdown(kpi_box("High-Risk Staff", high_risk, "Requires review"), unsafe_allow_html=True)
c4.markdown(kpi_box("Customer Sentiment", f"{sentiment:.2f}", "CX index (0–1)"), unsafe_allow_html=True)


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

    df = data["forecast"].copy()
    if not df.empty:
        cols = list(df.columns)
        if len(cols) >= 1:
            df.rename(columns={cols[0]: "Month"}, inplace=True)
        if len(cols) >= 2:
            df.rename(columns={cols[1]: "Revenue"}, inplace=True)

        if "Month" in df.columns:
            # Try to parse Month as datetime if possible
            df["Month"] = pd.to_datetime(df["Month"], errors="ignore")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Month"],
            y=df["Revenue"],
            mode="lines+markers",
            name="Projected revenue"
        ))

        fig.update_layout(
            height=500,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Month",
            yaxis_title="Revenue ($)"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Forecast data not available.")


# =============================
# MENU TAB
# =============================

with tabs[1]:
    st.markdown("### Menu Performance Matrix")

    df = data["menu"].copy()

    if not df.empty and {"BCG_Matrix", "Qty_Sold", "Total_Revenue"}.issubset(df.columns):

        fig = go.Figure()

        categories = ["Star", "Plowhorse", "Puzzle", "Dog"]
        colors = {
            "Star": "#22c55e",
            "Plowhorse": "#0ea5e9",
            "Puzzle": "#eab308",
            "Dog": "#f97316"
        }

        for cat in categories:
            subset = df[df["BCG_Matrix"] == cat]
            if subset.empty:
                continue
            fig.add_trace(go.Scatter(
                x=subset["Qty_Sold"],
                y=subset["Total_Revenue"],
                mode="markers",
                name=cat,
                marker=dict(
                    size=10,
                    color=colors.get(cat, "#a855f7"),
                    opacity=0.9
                ),
                text=subset.get("Menu Item", None)
            ))

        fig.update_layout(
            height=500,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Units sold",
            yaxis_title="Total revenue ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Detailed menu table"):
            st.dataframe(df, use_container_width=True)

    else:
        st.info("Menu data unavailable or missing required columns.")


# =============================
# COMPETITIVE TAB
# =============================

with tabs[2]:
    st.markdown("### Competitive Environment")

    df = data["map"].copy()

    # Require valid numeric coordinates
    if {"Latitude", "Longitude"}.issubset(df.columns):
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

        df_map = df.dropna(subset=["Latitude", "Longitude"]).copy()

        if not df_map.empty:
            center = [df_map["Latitude"].mean(), df_map["Longitude"].mean()]
            m = folium.Map(location=center, zoom_start=14, tiles="cartodbdark_matter")

            for _, r in df_map.iterrows():
                try:
                    lat = float(r["Latitude"])
                    lon = float(r["Longitude"])
                except (TypeError, ValueError):
                    continue

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    color="#7c3aed",
                    fill=True,
                    fill_color="#a855f7",
                    fill_opacity=0.85,
                    weight=1
                ).add_to(m)

            st_folium(m, height=600, width="100%")
        else:
            st.info("No valid competitor coordinates available after cleaning.")
    else:
        st.info("Map data unavailable or missing Latitude/Longitude columns.")


# =============================
# RISK TAB
# =============================

with tabs[3]:
    st.markdown("### Internal Risk and Loss Analysis")

    df = data["servers"].copy()
    if not df.empty:
        st.dataframe(df, use_container_width=True)

        if "Potential_Loss" in df.columns:
            recovery = df["Potential_Loss"].sum()
        else:
            recovery = 0

        st.markdown(f"Estimated monthly recovery potential with proper controls: **${recovery:,.0f}**")
    else:
        st.info("No risk data found.")


# =============================
# SENTIMENT TAB
# =============================

with tabs[4]:
    st.markdown("### Customer Sentiment")

    df = data["sentiment"].copy()
    if not df.empty and {"Month", "BestRegards_Revenue", "CX_Index"}.issubset(df.columns):

        df["Month"] = pd.to_datetime(df["Month"], errors="ignore")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["Month"],
            y=df["BestRegards_Revenue"],
            name="Revenue",
            marker_color="#22c55e",
            opacity=0.7
        ))
        fig.add_trace(go.Scatter(
            x=df["Month"],
            y=df["CX_Index"],
            yaxis="y2",
            name="CX index",
            marker_color="#f97316"
        ))

        fig.update_layout(
            height=500,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Month",
            yaxis_title="Revenue ($)",
            yaxis2=dict(
                title="CX index (0–1)",
                overlaying="y",
                side="right",
                range=[0, 1]
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Sentiment data unavailable or missing required columns.")


# =============================
# SUMMARY TAB
# =============================

with tabs[5]:
    st.markdown("### Executive Summary")

    # Very simple opportunity model using theft and menu signals
    opportunity = (theft + stars * 1500 - dogs * 800) * 12

    st.markdown(f"""
    <div class='glass'>
        <h2>Estimated Annual Performance Improvement</h2>
        <h1>${opportunity:,.0f}</h1>
        <p>
            This estimate combines recoverable internal loss, menu engineering upside,
            and efficiency gains from rationalising underperforming items.
        </p>
        <p>
            Immediate levers:
        </p>
        <ul>
            <li>Investigate high-risk staff and tighten point-of-sale controls.</li>
            <li>Focus menu design around high-performing items and remove low-yield products.</li>
            <li>Align service training with the sentiment and review patterns observed.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
