import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Best Regards Analytics", layout="wide", page_icon="🍸")

# --- 2. DIAGNOSTIC MODE (Top of App) ---
# This box will tell us EXACTLY what files GitHub sees
with st.expander("🛠️ DEBUG MENU (Check this first!)", expanded=True):
    st.write("### 1. Files found in GitHub Repo:")
    files = os.listdir('.')
    st.write(files)
    
    st.write("### 2. Critical File Check:")
    if 'master_data.parquet' in files:
        st.success("✅ master_data.parquet found (PERFECT)")
    elif 'master_data.csv' in files:
        st.warning("⚠️ master_data.csv found (Parquet is better, but this works)")
    else:
        st.error("❌ MASTER DATA MISSING! Please upload 'master_data.parquet' or 'master_data.csv'")

    if 'forecast_values.csv' in files:
        st.success("✅ forecast_values.csv found")
    else:
        st.error("❌ forecast_values.csv MISSING")

# --- 3. ROBUST DATA LOADER ---
@st.cache_data
def load_all_data():
    data = {}

    # Helper to load CSV safely
    def load_safe(filename):
        if os.path.exists(filename):
            try:
                return pd.read_csv(filename)
            except:
                return pd.DataFrame()
        return pd.DataFrame()

    # --- A. LOAD MASTER (Try Parquet First) ---
    try:
        data['df_master'] = pd.read_parquet('master_data.parquet')
    except:
        try:
            data['df_master'] = pd.read_csv('master_data.csv', low_memory=False)
        except:
            data['df_master'] = pd.DataFrame()

    # Normalize Master Dates (Crucial for Forecast Line)
    if not data['df_master'].empty:
        # Find the date column automatically
        potential_date_cols = [c for c in data['df_master'].columns if 'Date' in c or 'Time' in c]
        if potential_date_cols:
            date_col = potential_date_cols[0]
            data['df_master']['Date'] = pd.to_datetime(data['df_master'][date_col], errors='coerce')
            data['df_master']['Month'] = data['df_master']['Date'].dt.to_period('M').astype(str)

    # --- B. LOAD OTHER FILES ---
    # We use 'load_safe' so the app doesn't crash if one is missing
    data['df_forecast'] = load_safe('forecast_values.csv')
    data['df_menu'] = load_safe('menu_forensics.csv')
    data['df_map'] = load_safe('map_data.csv')
    data['df_servers'] = load_safe('suspicious_servers.csv')
    data['df_voids_h'] = load_safe('hourly_voids.csv')
    data['df_voids_d'] = load_safe('daily_voids.csv')
    data['df_geo'] = load_safe('geo_pressure.csv')
    data['df_sentiment'] = load_safe('sentiment.csv')

    return data

data = load_all_data()

# --- 4. SIDEBAR STATUS ---
with st.sidebar:
    st.title("🍸 Best Regards")
    st.write("---")
    # Debug Status Lights
    if not data['df_master'].empty:
        st.success("Master Data: Connected")
    else:
        st.error("Master Data: Disconnected")
    
    if not data['df_forecast'].empty:
        st.success("Forecast: Loaded")
    else:
        st.warning("Forecast: Missing")

# --- 5. MAIN DASHBOARD ---
st.title("📊 Business Intelligence Dashboard")
tab1, tab2, tab3, tab4 = st.tabs(["📉 Forecast", "🗺️ Map", "🍔 Menu", "🚨 Voids"])

# --- TAB 1: FORECAST (Fixed Flat Line) ---
with tab1:
    st.header("Revenue Forecast")
    
    # 1. Prepare Historical Data (From Master)
    if not data['df_master'].empty and 'Month' in data['df_master'].columns:
        # Filter valid sales (exclude voids)
        valid_sales = data['df_master']
        if 'is_void' in valid_sales.columns:
            valid_sales = valid_sales[~valid_sales['is_void']]
        
        hist_df = valid_sales.groupby('Month')['Net Price'].sum().reset_index()
        hist_df.columns = ['Date', 'Revenue']
        hist_df['Type'] = 'Historical'
    else:
        hist_df = pd.DataFrame()

    # 2. Prepare Forecast Data (From CSV)
    if not data['df_forecast'].empty:
        fc_df = data['df_forecast'].copy()
        # Force column names to be standard
        if len(fc_df.columns) >= 2:
            fc_df = fc_df.iloc[:, :2] # Take first two cols
            fc_df.columns = ['Date', 'Revenue']
            fc_df['Type'] = 'Forecast'
    else:
        fc_df = pd.DataFrame()

    # 3. Combine and Plot
    if not hist_df.empty or not fc_df.empty:
        # Convert dates to datetime objects for proper plotting
        if not hist_df.empty: hist_df['Date'] = pd.to_datetime(hist_df['Date'], errors='coerce')
        if not fc_df.empty: fc_df['Date'] = pd.to_datetime(fc_df['Date'], errors='coerce')
        
        plot_df = pd.concat([hist_df, fc_df], ignore_index=True)
        plot_df = plot_df.sort_values('Date') # Sort chronologically
        
        fig = px.line(plot_df, x='Date', y='Revenue', color='Type', 
                      title="Revenue Trajectory (Historical + Forecast)",
                      color_discrete_map={'Historical': 'gray', 'Forecast': 'red'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No data available for Forecast. Check 'Debug Menu' above.")

# --- TAB 2: MAP (Fixed Columns) ---
with tab2:
    st.header("Competitor Map")
    if not data['df_map'].empty:
        # Debug: Show columns if map is broken
        # st.write("Map Columns Detected:", data['df_map'].columns.tolist())
        
        map_df = data['df_map'].copy()
        
        # Smart Column Rename: Look for variations of Lat/Lon
        cols = map_df.columns.str.lower()
        if 'latitude' in cols: 
            map_df.rename(columns={map_df.columns[list(cols).index('latitude')]: 'latitude'}, inplace=True)
        if 'longitude' in cols: 
            map_df.rename(columns={map_df.columns[list(cols).index('longitude')]: 'longitude'}, inplace=True)
            
        # Ensure numeric
        map_df['latitude'] = pd.to_numeric(map_df['latitude'], errors='coerce')
        map_df['longitude'] = pd.to_numeric(map_df['longitude'], errors='coerce')
        map_df = map_df.dropna(subset=['latitude', 'longitude'])
        
        if not map_df.empty:
            # Create Map
            m = folium.Map(location=[map_df['latitude'].mean(), map_df['longitude'].mean()], zoom_start=13)
            
            for idx, row in map_df.iterrows():
                # Try to find a name column
                name_col = [c for c in map_df.columns if 'name' in c.lower()]
                name = str(row[name_col[0]]) if name_col else "Location"
                
                color = 'red' if 'BEST REGARDS' in name.upper() else 'blue'
                
                folium.Marker(
                    [row['latitude'], row['longitude']],
                    popup=name,
                    icon=folium.Icon(color=color)
                ).add_to(m)
            st_folium(m, width=800, height=500)
        else:
            st.warning("Map data loaded but Latitude/Longitude columns are empty.")
    else:
        st.warning("Map CSV not found.")

# --- TAB 3: MENU ---
with tab3:
    st.header("Menu Analysis")
    if not data['df_menu'].empty:
        fig = px.scatter(data['df_menu'], x="Qty_Sold", y="Total_Revenue", color="BCG_Matrix",
                         size="Total_Revenue", hover_name="Menu Item", title="Menu Matrix")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Menu CSV not found.")

# --- TAB 4: VOIDS ---
with tab4:
    st.header("Operational Risks")
    if not data['df_servers'].empty:
        st.dataframe(data['df_servers'])
    else:
        st.info("No server alerts.")
