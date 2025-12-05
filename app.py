
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium

# --- PAGE SETUP ---
st.set_page_config(page_title="Best Regards Analytics", layout="wide")

# --- SPEED BOOSTER (CACHING) ---
# This tells Streamlit: "Run this function ONCE and save the result in memory."
# This makes clicking tabs instant.
@st.cache_data
def load_all_data():
    # 1. Generate/Load Forecast Data
    dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    df_forecast = pd.DataFrame({
        'Date': dates,
        'Revenue': np.random.normal(150000, 10000, 12).cumsum()
    })
    
    # 2. Generate/Load Menu Data
    df_menu = pd.DataFrame({
        'Item': ['Burger', 'Fries', 'Steak', 'Salad', 'Beer', 'Wine'],
        'Qty Sold': [500, 450, 100, 300, 800, 200],
        'Profit': [5000, 2000, 4000, 1500, 6000, 2500],
        'Category': ['Star', 'Plowhorse', 'Puzzle', 'Dog', 'Star', 'Puzzle']
    })
    
    return df_forecast, df_menu

# Load the data into memory
df_forecast, df_menu = load_all_data()

# --- HIDE MENUS ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🍸 Best Regards")
    st.success("⚡ High-Speed Mode Active")
    st.write("---")

# --- MAIN CONTENT ---
st.title("📊 Executive Business Intelligence")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📉 Forecast Engine", "🍔 Menu Matrix", "🗺️ Competitor Map"])

with tab1:
    st.header("Revenue Forecast")
    st.line_chart(df_forecast.set_index('Date'))
    st.caption("Data cached for instant retrieval.")

with tab2:
    st.header("Menu Engineering (BCG Matrix)")
    fig = px.scatter(df_menu, x='Qty Sold', y='Profit', color='Category', 
                     size='Profit', hover_name='Item', title="Profit vs Popularity")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Geospatial Competitor Analysis")
    m = folium.Map(location=[29.7604, -95.3698], zoom_start=13)
    folium.Marker([29.7604, -95.3698], popup="Best Regards", icon=folium.Icon(color='red')).add_to(m)
    folium.Marker([29.7654, -95.3748], popup="Competitor A", icon=folium.Icon(color='blue')).add_to(m)
    st_folium(m, width=800, height=400)
