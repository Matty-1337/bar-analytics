import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import folium
from folium import plugins
import os

# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Best Regards Intelligence System",
    layout="wide"
)

# -------------------------------------------------------------------
# GLOBAL UI STYLING (GLASS THEME + DARK BACKGROUND)
# -------------------------------------------------------------------
st.markdown("""
<style>

body {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e5e7eb;
}

.block-container {
    padding-top: 1.4rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #020617);
    border-right: 1px solid rgba(255,255,255,.1);
}
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
    opacity: 1 !important;
}

/* Glass cards */
.glass {
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    background: rgba(255,255,255,.08);
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,.10);
    padding: 20px;
    box-shadow: 0px 18px 40px rgba(0,0,0,.45);
}

.kpi {
    text-align: center;
    padding: 22px;
    min-height: 140px;
}
.kpi h1 {
    font-size: 2.1rem;
    margin: 0;
}
.kpi p { opacity: .85; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,.07);
    border-radius: 12px;
    padding: 12px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #9333ea);
    color: white !important;
}

/* Plotly Styling */
.js-plotly-plot {
    border-radius: 16px;
    overflow: hidden;
}

/* Hide Streamlit footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)



# -------------------------------------------------------------------
# SAFE LOAD HELPERS
# -------------------------------------------------------------------
def safe_load(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
forecast = safe_load("forecast_values.csv")
menu = safe_load("menu_forensics.csv")
servers = safe_load("suspicious_servers.csv")
sentiment = safe_load("sentiment.csv")
mapdata = safe_load("map_data.csv")

# numeric conversions
menu = to_num(menu, ["Qty_Sold", "Total_Revenue"])
servers = to_num(servers, ["Void_Z_Score", "Potential_Loss"])
sentiment = to_num(sentiment, ["CX_Index", "BestRegards_Revenue"])
mapdata = to_num(mapdata, ["Latitude", "Longitude", "Total_Revenue"])


# -------------------------------------------------------------------
# KPI CALCULATIONS
# -------------------------------------------------------------------
def compute_kpis():
    theft = servers["Potential_Loss"].sum() if "Potential_Loss" in servers.columns else 0
    high_risk = len(servers.query("Void_Z_Score >= 2")) if "Void_Z_Score" in servers.columns else 0
    sentiment_val = sentiment["CX_Index"].iloc[-1] if "CX_Index" in sentiment.columns and not sentiment.empty else 0

    stars = len(menu.query("BCG_Matrix == 'Star'")) if "BCG_Matrix" in menu.columns else 0
    dogs = len(menu.query("BCG_Matrix == 'Dog'")) if "BCG_Matrix" in menu.columns else 0

    # Revenue forecast baseline
    if not forecast.empty:
        cols = list(forecast.columns)
        forecast.rename(columns={cols[0]: "Month", cols[1]: "Revenue"}, inplace=True)
        avg_rev = forecast["Revenue"].head(3).mean()
    else:
        avg_rev = 0

    return avg_rev, theft, high_risk, sentiment_val, stars, dogs


avg_rev, theft, high_risk, sentiment_val, stars, dogs = compute_kpis()



# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h2>Best Regards</h2>", unsafe_allow_html=True)
    st.markdown("Executive Intelligence Portal")
    st.markdown("---")

    st.metric("Avg Monthly Revenue", f"${avg_rev:,.0f}")
    st.metric("Estimated Monthly Loss", f"${theft:,.0f}")
    st.metric("High-Risk Employees", f"{high_risk}")
    st.metric("Customer Sentiment Index", f"{sentiment_val:.2f}")

    st.caption(f"Updated {datetime.now().strftime('%B %d, %Y')}")


# -------------------------------------------------------------------
# HERO KPI ROW
# -------------------------------------------------------------------
st.markdown("## Business Intelligence Overview")

c1, c2, c3, c4 = st.columns(4)

def kpi_box(title, value, detail):
    return f"""
    <div class='glass kpi'>
        <h1>{value}</h1>
        <p>{title}</p>
        <small>{detail}</small>
    </div>
    """

c1.markdown(kpi_box("Projected Revenue", f"${avg_rev:,.0f}", "Next 90 days"), unsafe_allow_html=True)
c2.markdown(kpi_box("Operational Loss", f"${theft:,.0f}", "Estimated monthly leakage"), unsafe_allow_html=True)
c3.markdown(kpi_box("Risk Flag Count", high_risk, "Employees requiring review"), unsafe_allow_html=True)
c4.markdown(kpi_box("CX Index", f"{sentiment_val:.2f}", "Customer perception score"), unsafe_allow_html=True)


# -------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------
tabs = st.tabs([
    "Revenue Forecast",
    "Menu Engineering",
    "Competitive Map",
    "Operational Risk",
    "Sentiment",
    "Executive Summary"
])


# -------------------------------------------------------------------
# TAB 1 – FORECAST
# -------------------------------------------------------------------
with tabs[0]:
    st.markdown("### Revenue Projection")

    if not forecast.empty:
        df = forecast.copy()
        cols = list(df.columns)
        df.rename(columns={cols[0]: "Month", cols[1]: "Revenue"}, inplace=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Month"],
            y=df["Revenue"],
            mode="lines+markers",
            line=dict(width=3, color="#6366f1")
        ))
        fig.update_layout(
            height=480,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Month",
            yaxis_title="Revenue ($)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Forecast file not found.")


# -------------------------------------------------------------------
# TAB 2 – MENU ENGINEERING
# -------------------------------------------------------------------
with tabs[1]:
    st.markdown("### Menu Performance Matrix")

    if not menu.empty and {"BCG_Matrix", "Qty_Sold", "Total_Revenue"}.issubset(menu.columns):

        colors = {
            "Star": "#22c55e",
            "Plowhorse": "#0ea5e9",
            "Puzzle": "#eab308",
            "Dog": "#f97316"
        }

        fig = go.Figure()

        for cat in ["Star", "Plowhorse", "Puzzle", "Dog"]:
            subset = menu[menu["BCG_Matrix"] == cat]
            if not subset.empty:
                fig.add_trace(go.Scatter(
                    x=subset["Qty_Sold"],
                    y=subset["Total_Revenue"],
                    mode="markers",
                    name=cat,
                    marker=dict(size=10, color=colors[cat])
                ))

        fig.update_layout(
            height=500,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Units Sold",
            yaxis_title="Revenue ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Full Menu Table")
        st.dataframe(menu, use_container_width=True)

    else:
        st.info("Menu data missing required columns.")


# -------------------------------------------------------------------
# TAB 3 – COMPETITIVE MAP (RESTORED & WORKING)
# -------------------------------------------------------------------
with tabs[2]:

    st.markdown("### Competitive Market Map")

    if {"Latitude", "Longitude"}.issubset(mapdata.columns):

        df = mapdata.copy()
        df = df.dropna(subset=["Latitude", "Longitude"])

        if not df.empty:
            # CENTER MAP
            center = [df["Latitude"].mean(), df["Longitude"].mean()]
            m = folium.Map(location=center, zoom_start=14, tiles="cartodbdark_matter")

            # HEATMAP LAYER
            heat_points = []
            for _, r in df.iterrows():
                try:
                    lat = float(r["Latitude"])
                    lon = float(r["Longitude"])
                except:
                    continue
                heat_points.append([lat, lon])

            if heat_points:
                plugins.HeatMap(heat_points, radius=35, blur=25).add_to(m)

            # PLOT MARKERS
            for _, r in df.iterrows():
                try:
                    lat = float(r["Latitude"])
                    lon = float(r["Longitude"])
                except:
                    continue

                name = str(r.get("Location Name", "Location"))

                # Highlight your location
                if "BEST REGARDS" in name.upper():
                    folium.Marker(
                        [lat, lon],
                        tooltip=name,
                        icon=folium.Icon(color='red')
                    ).add_to(m)

                else:
                    folium.CircleMarker(
                        [lat, lon],
                        radius=7,
                        color="#3b82f6",
                        fill=True,
                        fill_color="#3b82f6",
                        fill_opacity=0.85,
                        tooltip=f"{name}<br>${r.get('Total_Revenue',0):,.0f}"
                    ).add_to(m)

            # RENDER
            components.html(m._repr_html_(), height=600)

        else:
            st.info("Map data available, but all rows contain invalid coordinates.")
    else:
        st.info("Map file missing Latitude/Longitude columns.")



# -------------------------------------------------------------------
# TAB 4 – OPERATIONAL RISK
# -------------------------------------------------------------------
with tabs[3]:
    st.markdown("### Internal Risk Analysis")

    if not servers.empty:
        st.dataframe(servers, use_container_width=True)

        loss = servers["Potential_Loss"].sum() if "Potential_Loss" in servers.columns else 0
        st.markdown(f"Estimated recoverable loss per month: **${loss:,.0f}**")
    else:
        st.info("Server risk file not found.")



# -------------------------------------------------------------------
# TAB 5 – SENTIMENT ANALYSIS
# -------------------------------------------------------------------
with tabs[4]:

    st.markdown("### Customer Sentiment and Revenue Correlation")

    if not sentiment.empty and {"Month", "BestRegards_Revenue", "CX_Index"}.issubset(sentiment.columns):

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sentiment["Month"],
            y=sentiment["BestRegards_Revenue"],
            name="Revenue",
            marker_color="#22c55e",
            opacity=0.7
        ))
        fig.add_trace(go.Scatter(
            x=sentiment["Month"],
            y=sentiment["CX_Index"],
            yaxis="y2",
            name="CX Index",
            line=dict(color="#f97316", width=3)
        ))

        fig.update_layout(
            height=500,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis=dict(title="Revenue ($)"),
            yaxis2=dict(
                title="CX Index",
                overlaying="y",
                side="right",
                range=[0, 1]
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Sentiment file missing required fields.")



# -------------------------------------------------------------------
# TAB 6 – EXECUTIVE SUMMARY
# -------------------------------------------------------------------
with tabs[5]:

    st.markdown("### Executive Summary")

    opportunity = (theft + stars * 1500 - dogs * 800) * 12

    st.markdown(f"""
    <div class="glass">
        <h2>Projected Annual Improvement</h2>
        <h1>${opportunity:,.0f}</h1>
        <p>This estimate reflects recoverable operational loss, menu optimization effects, and staff efficiency corrections.</p>

        <h3>Primary Levers</h3>
        <ul>
            <li>Reinforce POS discipline across staff.</li>
            <li>Rebalance menu based on contribution margins.</li>
            <li>Align service quality with observed customer sentiment trends.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
