import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Best Regards Analytics",
    page_icon="🍸",
    layout="wide"
)

# --- 2. DATA LOADING ENGINE ---
@st.cache_data
def load_all_data():
    data = {}

    def load_csv(filename):
        try:
            return pd.read_csv(filename)
        except FileNotFoundError:
            return pd.DataFrame()

    # --- A. LOAD MASTER (Parquet or CSV) ---
    try:
        data['df_master'] = pd.read_parquet('master_data.parquet')
    except:
        try:
            data['df_master'] = pd.read_parquet('master_data_recent.parquet')
        except:
            data['df_master'] = load_csv('master_data.csv')

    # Fix Master Dates
    if not data['df_master'].empty:
        cols = data['df_master'].columns
        # Try to find a date column
        if 'Date' in cols:
            data['df_master']['Date'] = pd.to_datetime(data['df_master']['Date'], errors='coerce')
            data['df_master']['Month'] = data['df_master']['Date'].dt.to_period('M').astype(str)

    # --- B. LOAD ANALYTICS FILES ---
    data['df_menu'] = load_csv('menu_forensics.csv')
    data['df_sentiment'] = load_csv('sentiment.csv')
    data['df_forecast'] = load_csv('forecast_values.csv')
    data['df_geo'] = load_csv('geo_pressure.csv')
    data['df_map'] = load_csv('map_data.csv')
    data['df_servers'] = load_csv('suspicious_servers.csv')
    data['df_voids_h'] = load_csv('hourly_voids.csv')
    data['df_voids_d'] = load_csv('daily_voids.csv')
    data['df_combo'] = load_csv('suspicious_combinations.csv')

    # --- C. PREPARE MONTHLY REVENUE (HISTORICAL) ---
    if not data['df_master'].empty:
        clean_df = data['df_master']
        # Filter Voids if column exists
        if 'is_void' in clean_df.columns:
            clean_df = clean_df[~clean_df['is_void']]
        
        # Group by Month
        if 'Month' in clean_df.columns:
            monthly_data = clean_df.groupby('Month')['Net Price'].sum().reset_index()
            monthly_data.columns = ['Month', 'Revenue']
            monthly_data['Type'] = 'Historical'
            data['monthly_revenue'] = monthly_data
        else:
            data['monthly_revenue'] = pd.DataFrame()
    else:
        data['monthly_revenue'] = pd.DataFrame()

    return data

data = load_all_data()

# --- 3. SIDEBAR ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 1rem;}
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title("🍸 Best Regards")
    st.write("### Executive Portal")
    if not data['df_master'].empty:
        st.success("✅ Master Data: Active")
    else:
        st.error("❌ Master Data: Missing")

# --- 4. MAIN TABS ---
st.title("📊 Business Intelligence Dashboard")

# DEFINING TABS AS A LIST (This fixes the NameError)
tabs = st.tabs([
    "📉 Forecast", 
    "🍔 Menu Matrix", 
    "🗺️ Competitor Map", 
    "🚨 Theft Detection", 
    "❤️ Sentiment"
])

# --- TAB 0: FORECAST ---
with tabs[0]:
    st.header("Revenue Forecast")
    
    if not data['df_forecast'].empty:
        fc_data = data['df_forecast'].copy()
        
        # Emergency Column Renaming (Fixes KeyError)
        if len(fc_data.columns) >= 1:
            fc_data.rename(columns={fc_data.columns[0]: 'Month'}, inplace=True)
        if len(fc_data.columns) >= 2:
            fc_data.rename(columns={fc_data.columns[1]: 'Revenue'}, inplace=True)
            
        fc_data['Type'] = 'Forecast'
        
        # Convert to Datetime safely
        try:
            fc_data['Month'] = pd.to_datetime(fc_data['Month'])
        except:
            pass # Keep as string if conversion fails

        # Combine with Historical Data if available
        if not data['monthly_revenue'].empty:
            try:
                data['monthly_revenue']['Month'] = pd.to_datetime(data['monthly_revenue']['Month'])
                combined_df = pd.concat([data['monthly_revenue'], fc_data], ignore_index=True)
            except:
                combined_df = fc_data
        else:
            combined_df = fc_data

        # Plot
        fig = px.line(combined_df, x='Month', y='Revenue', color='Type', 
                      title="Projected Revenue Trajectory", 
                      color_discrete_map={'Historical': 'gray', 'Forecast': '#FF4B4B'})
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("⚠️ Forecast data is empty or missing.")

# --- TAB 1: MENU ---
with tabs[1]:
    st.header("Menu Engineering (BCG Matrix)")
    if not data['df_menu'].empty:
        fig = px.scatter(
            data['df_menu'], 
            x="Qty_Sold", 
            y="Total_Revenue", 
            color="BCG_Matrix",
            size="Total_Revenue", 
            hover_name="Menu Item", 
            title="Profitability vs Popularity"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Menu data missing.")

# --- TAB 2: MAP ---
with tabs[2]:
    st.header("Geospatial Competitor Analysis")
    if not data['df_map'].empty:
        # Clean Data
        df_m = data['df_map'].copy()
        df_m['Latitude'] = pd.to_numeric(df_m['Latitude'], errors='coerce')
        df_m['Longitude'] = pd.to_numeric(df_m['Longitude'], errors='coerce')
        df_m = df_m.dropna(subset=['Latitude', 'Longitude'])
        
        if not df_m.empty:
            m = folium.Map(location=[df_m['Latitude'].mean(), df_m['Longitude'].mean()], zoom_start=13)
            for i, row in df_m.iterrows():
                # Color 'Best Regards' Red, others Blue
                color = 'red' if 'BEST REGARDS' in str(row.get('Location Name','')).upper() else 'blue'
                folium.CircleMarker(
                    [row['Latitude'], row['Longitude']], 
                    radius=10, 
                    color=color, 
                    fill=True, 
                    fill_color=color,
                    tooltip=str(row.get('Location Name', 'Competitor'))
                ).add_to(m)
            st_folium(m, width=800, height=500)
    else:
        st.warning("⚠️ Map data missing.")

# --- TAB 3: VOIDS ---
with tabs[3]:
    st.header("Operational Risk & Void Detection")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Suspicious Servers")
        if not data['df_servers'].empty:
            st.dataframe(data['df_servers'], hide_index=True)
        else:
            st.info("No server alerts.")
            
    with col2:
        st.subheader("High Risk Hours")
        if not data['df_voids_h'].empty:
            st.bar_chart(data['df_voids_h'].set_index('Hour_of_Day')['Void_Rate'])
        else:
            st.info("No hourly data.")

# --- TAB 4: SENTIMENT ---
with tabs[4]:
    st.header("Sentiment Analysis")
    if not data['df_sentiment'].empty:
        # Auto-detect sentiment column
        cols = data['df_sentiment'].columns
        if len(cols) > 1:
            st.line_chart(data['df_sentiment'].set_index(cols[0])[cols[1]])
        else:
            st.dataframe(data['df_sentiment'])
    else:
        st.warning("⚠️ Sentiment data missing.")
