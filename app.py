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
    """
    Loads all data files from the local repository.
    Handles Parquet for large files and CSV for standard files.
    """
    data = {}

    # Helper for safe CSV loading
    def load_csv(filename):
        try:
            return pd.read_csv(filename)
        except FileNotFoundError:
            return pd.DataFrame()

    # --- A. LOAD MASTER DATA (Parquet Support) ---
    # This handles the 119MB file issue by reading the compressed version
    try:
        data['df_master'] = pd.read_parquet('master_data.parquet')
    except FileNotFoundError:
        try:
            # Fallback if you uploaded the "recent" version instead
            data['df_master'] = pd.read_parquet('master_data_recent.parquet')
        except FileNotFoundError:
            # Final fallback to CSV if you managed to upload it
            data['df_master'] = load_csv('master_data.csv')

    # Ensure Date format if master loaded
    if not data['df_master'].empty:
        if 'Date' in data['df_master'].columns:
            data['df_master']['Date'] = pd.to_datetime(data['df_master']['Date'], errors='coerce')
        if 'Month' not in data['df_master'].columns:
             data['df_master']['Month'] = data['df_master']['Date'].dt.to_period('M').astype(str)

    # --- B. LOAD ANALYTICS DATA ---
    data['df_menu'] = load_csv('menu_forensics.csv')
    data['df_sentiment'] = load_csv('sentiment.csv')
    data['df_forecast'] = load_csv('forecast_values.csv')
    data['df_geo'] = load_csv('geo_pressure.csv')
    data['df_map'] = load_csv('map_data.csv')
    data['df_servers'] = load_csv('suspicious_servers.csv')
    data['df_voids_h'] = load_csv('hourly_voids.csv')
    data['df_voids_d'] = load_csv('daily_voids.csv')
    data['df_combo'] = load_csv('suspicious_combinations.csv')

    # --- C. PREPARE HISTORICAL REVENUE ---
    # We derive monthly revenue from the Master file to graph alongside the Forecast
    if not data['df_master'].empty:
        # Filter out voids if 'is_void' exists, otherwise assume all valid
        clean_df = data['df_master']
        if 'is_void' in clean_df.columns:
            clean_df = clean_df[~clean_df['is_void']]
        
        # Group by Month
        monthly_data = clean_df.groupby('Month')['Net Price'].sum().reset_index()
        monthly_data.columns = ['Month', 'Revenue']
        monthly_data['Type'] = 'Historical'
        data['monthly_revenue'] = monthly_data
    else:
        data['monthly_revenue'] = pd.DataFrame()

    return data

# Load the data
data = load_all_data()

# --- 3. STYLING & SIDEBAR ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 1rem;}
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title("🍸 Best Regards")
    st.write("### Executive Portal")
    st.write("---")
    
    # Status Indicators
    if not data['df_master'].empty:
        st.success("✅ Master Data: Connected")
    else:
        st.error("❌ Master Data: Missing")
        
    if not data['df_forecast'].empty:
        st.success("✅ Forecast AI: Active")
    else:
        st.warning("⚠️ Forecast: Missing")

# --- 4. MAIN DASHBOARD ---
st.title("📊 Business Intelligence Dashboard")
st.markdown("Real-time analysis of Sales, Menu Performance, Competition, and Operational Risks.")

# Create Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 Revenue Forecast", 
    "🍔 Menu Engineering", 
    "🗺️ Competitor Map", 
    "🚨 Void Analysis",
    "❤️ Sentiment"
])

# --- TAB 1: FORECAST ---
with tab1:
    st.header("Revenue Forecast & Trajectory")
    
    if not data['df_forecast'].empty:
        # Prepare Forecast Data
        fc_data = data['df_forecast'].copy()
        # Rename columns for consistency if needed
        if 'Forecasted_Revenue' in fc_data.columns:
            fc_data = fc_data.rename(columns={'Forecasted_Revenue': 'Revenue'})
            fc_data.index.name = 'Month'
            fc_data = fc_data.reset_index()
        
        fc_data['Type'] = 'Forecast'
        
        # Combine with Historical if available
        if not data['monthly_revenue'].empty:
            # Ensure date formats match
            fc_data['Month'] = pd.to_datetime(fc_data['Month'])
            data['monthly_revenue']['Month'] = pd.to_datetime(data['monthly_revenue']['Month'])
            
            combined_df = pd.concat([data['monthly_revenue'], fc_data], ignore_index=True)
        else:
            combined_df = fc_data

        # Plot
        fig_forecast = px.line(
            combined_df, 
            x='Month', 
            y='Revenue', 
            color='Type',
            title="Revenue Trajectory (Historical vs. Projected)",
            color_discrete_map={'Historical': 'gray', 'Forecast': '#FF4B4B'}
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Metrics
        proj_avg = fc_data['Revenue'].mean()
        st.metric("Projected Monthly Average", f"${proj_avg:,.0f}")
        
    else:
        st.warning("⚠️ Forecast data not found. Please check 'forecast_values.csv' in GitHub.")

# --- TAB 2: MENU ENGINEERING ---
with tabs[2]:
    st.header("Menu Matrix (BCG Analysis)")
    st.caption("Identify Stars (Keep), Dogs (Remove), Plowhorses (Reprice), and Puzzles (Promote).")
    
    if not data['df_menu'].empty:
        # Scatter Plot
        fig_bcg = px.scatter(
            data['df_menu'], 
            x="Qty_Sold", 
            y="Total_Revenue", 
            color="BCG_Matrix", 
            size="Total_Revenue", 
            hover_name="Menu Item",
            title="Profitability vs Popularity",
            color_discrete_map={
                'Star': '#00CC96',      # Green
                'Dog': '#EF553B',       # Red
                'Plowhorse': '#AB63FA', # Purple
                'Puzzle': '#FFA15A'     # Orange
            }
        )
        st.plotly_chart(fig_bcg, use_container_width=True)
        
        # Data Table
        with st.expander("View Detailed Menu Data"):
            st.dataframe(data['df_menu'])
    else:
        st.warning("⚠️ Menu data not found. Please check 'menu_forensics.csv' in GitHub.")

# --- TAB 3: COMPETITOR MAP ---
with tabs[3]:
    st.header("Geospatial Competitor Analysis")
    
    if not data['df_map'].empty:
        # Clean Data
        valid_map = data['df_map'].copy()
        valid_map['Latitude'] = pd.to_numeric(valid_map['Latitude'], errors='coerce')
        valid_map['Longitude'] = pd.to_numeric(valid_map['Longitude'], errors='coerce')
        valid_map = valid_map.dropna(subset=['Latitude', 'Longitude'])

        if not valid_map.empty:
            # Center Map
            center_lat = valid_map['Latitude'].mean()
            center_lon = valid_map['Longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
            
            for idx, row in valid_map.iterrows():
                # Logic to color 'Best Regards' differently
                name = str(row.get('Location Name', '')).upper()
                is_us = 'BEST REGARDS' in name
                
                folium.Marker(
                    [row['Latitude'], row['Longitude']],
                    popup=row.get('Location Name', 'Unknown'),
                    icon=folium.Icon(color='red' if is_us else 'blue', icon='star' if is_us else 'info-sign')
                ).add_to(m)
            
            st_folium(m, width=800, height=500)
            
            # Agglomeration Chart
            if not data['df_geo'].empty:
                st.subheader("Agglomeration Effect")
                fig_geo = px.scatter(
                    data['df_geo'], x="GeoPressure_Total", y="BestRegards_Revenue",
                    title="Impact of Competitor Density on Revenue"
                )
                st.plotly_chart(fig_geo, use_container_width=True)
        else:
            st.warning("Map data exists but contains invalid Latitude/Longitude values.")
    else:
        st.warning("⚠️ Map data not found. Please check 'map_data.csv' in GitHub.")

# --- TAB 4: VOID ANALYSIS ---
with tabs[4]:
    st.header("Operational Risk & Void Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Suspicious Servers")
        if not data['df_servers'].empty:
            st.dataframe(data['df_servers'], hide_index=True)
        else:
            st.info("No suspicious server activity detected.")
            
    with col2:
        st.subheader("Suspicious Combinations")
        st.caption("Servers frequently voiding specific Tab Names")
        if not data['df_combo'].empty:
            st.dataframe(data['df_combo'], hide_index=True)
        else:
            st.info("No suspicious combinations detected.")
            
    st.markdown("---")
    st.subheader("Void Timing Analysis")
    
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        if not data['df_voids_h'].empty:
            st.write("**By Hour of Day**")
            st.bar_chart(data['df_voids_h'].set_index('Hour_of_Day')['Void_Rate'])
            
    with v_col2:
        if not data['df_voids_d'].empty:
            st.write("**By Day of Week**")
            st.bar_chart(data['df_voids_d'].set_index('Day_of_Week_Name')['Void_Rate'])

# --- TAB 5: SENTIMENT ---
with tabs[5]: # Note: Streamlit tabs are 0-indexed, but list was length 5
    st.header("Customer Sentiment")
    if not data['df_sentiment'].empty:
        # Assuming columns CX_Index and BestRegards_Revenue
        st.line_chart(data['df_sentiment'].set_index('Month')[['CX_Index', 'BestRegards_Revenue']])
        st.caption("Correlation between Customer Experience Index and Monthly Revenue.")
    else:
        st.warning("⚠️ Sentiment data not found. Please check 'sentiment.csv' in GitHub.")
