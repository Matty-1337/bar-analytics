import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
import os

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
    errors = []

    # Helper to load CSV safely
    def load_safe(filename):
        if os.path.exists(filename):
            try:
                return pd.read_csv(filename)
            except:
                return pd.DataFrame()
        return pd.DataFrame()

    # --- A. LOAD MASTER (Try Parquet First, then CSV) ---
    data['df_master'] = pd.DataFrame()
    try:
        # Primary method: Parquet
        data['df_master'] = pd.read_parquet('master_data.parquet')
    except Exception as e_parquet:
        # Fallback method: CSV (if they uploaded csv instead or pyarrow is missing)
        try:
            data['df_master'] = pd.read_csv('master_data.csv', low_memory=False)
        except Exception as e_csv:
            # Record error for sidebar display
            errors.append(f"Master Data Load Failed: {str(e_parquet)} | {str(e_csv)}")

    # Fix Master Dates
    if not data['df_master'].empty:
        cols = data['df_master'].columns
        if 'Date' in cols:
            data['df_master']['Date'] = pd.to_datetime(data['df_master']['Date'], errors='coerce')
            data['df_master']['Month'] = data['df_master']['Date'].dt.to_period('M').astype(str)

    # --- B. LOAD ANALYTICS FILES ---
    data['df_forecast'] = load_safe('forecast_values.csv')
    data['df_metrics'] = load_safe('forecast_metrics.csv')
    data['df_menu'] = load_safe('menu_forensics.csv')
    data['df_map'] = load_safe('map_data.csv')
    data['df_servers'] = load_safe('suspicious_servers.csv')
    data['df_voids_h'] = load_safe('hourly_voids.csv')
    data['df_voids_d'] = load_safe('daily_voids.csv')
    data['df_combo'] = load_safe('suspicious_combinations.csv')
    data['df_geo'] = load_safe('geo_pressure.csv')
    data['df_sentiment'] = load_safe('sentiment.csv')

    # --- C. PREPARE MONTHLY REVENUE (HISTORICAL) ---
    if not data['df_master'].empty:
        clean_df = data['df_master']
        if 'is_void' in clean_df.columns:
            clean_df = clean_df[~clean_df['is_void']]
        
        if 'Month' in clean_df.columns:
            monthly_data = clean_df.groupby('Month')['Net Price'].sum().reset_index()
            monthly_data.columns = ['Month', 'Revenue']
            monthly_data['Type'] = 'Historical'
            data['monthly_revenue'] = monthly_data
        else:
            data['monthly_revenue'] = pd.DataFrame()
    else:
        data['monthly_revenue'] = pd.DataFrame()

    return data, errors

data, load_errors = load_all_data()

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
    
    # Status Indicators
    if not data['df_master'].empty:
        st.success("✅ Master Data: Active")
    else:
        st.error("❌ Master Data: Missing")
        if load_errors:
            with st.expander("See Error Details"):
                for e in load_errors:
                    st.write(e)
    
    if not data['df_forecast'].empty:
        st.success("✅ Forecast Model: Active")
    else:
        st.warning("⚠️ Forecast: Missing")

# --- 4. MAIN TABS ---
st.title("📊 Business Intelligence Dashboard")

tabs = st.tabs([
    "📉 Forecast", 
    "🍔 Menu Matrix", 
    "🗺️ Competitor Map", 
    "🚨 Theft Detection", 
    "❤️ Sentiment"
])

# --- TAB 0: FORECAST ---
with tabs[0]:
    st.header("Revenue Forecast (24-Month Horizon)")
    
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        if not data['df_forecast'].empty:
            fc_data = data['df_forecast'].copy()
            
            # Column Cleanup
            if len(fc_data.columns) >= 1: fc_data.rename(columns={fc_data.columns[0]: 'Month'}, inplace=True)
            if len(fc_data.columns) >= 2: fc_data.rename(columns={fc_data.columns[1]: 'Revenue'}, inplace=True)
                
            fc_data['Type'] = 'Projection'
            
            # Combine Historical + Forecast for full timeline
            combined_df = fc_data
            if not data['monthly_revenue'].empty:
                try:
                    # Align columns
                    hist = data['monthly_revenue'].copy()
                    hist['Month'] = hist['Month'].astype(str)
                    fc_data['Month'] = fc_data['Month'].astype(str)
                    combined_df = pd.concat([hist, fc_data], ignore_index=True)
                    
                    # Convert back to datetime for sorting/plotting
                    combined_df['Month'] = pd.to_datetime(combined_df['Month'])
                    combined_df = combined_df.sort_values('Month')
                except:
                    combined_df = fc_data

            fig = px.line(combined_df, x='Month', y='Revenue', color='Type', 
                          title="Revenue Trajectory: Historical Performance vs. Projection", 
                          color_discrete_map={'Historical': 'gray', 'Projection': '#FF4B4B'},
                          markers=True)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("⚠️ Forecast data is empty or missing.")

    with col_b:
        st.info("💡 **Analyst Insight:**\nThe red line represents the projected revenue trajectory based on historical patterns. Identifying divergences here allows us to intervene before revenue dips occur.")
        
        if not data['df_metrics'].empty:
            st.write("### Model Accuracy")
            st.write("Comparing predictive models (ARIMA vs Prophet):")
            st.dataframe(data['df_metrics'], hide_index=True)
        else:
            st.write("*(Model metrics unavailable)*")

# --- TAB 1: MENU ---
with tabs[1]:
    st.header("Menu Engineering (BCG Matrix)")
    
    if not data['df_menu'].empty:
        col1, col2 = st.columns([3, 1])
        with col1:
            fig = px.scatter(
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
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.info("""
            **Analyst Insight:**
            - ⭐ **Stars:** High Profit, High Volume. (Keep & Promote)
            - 🐴 **Plowhorses:** Low Profit, High Volume. (Increase Price)
            - 🧩 **Puzzles:** High Profit, Low Volume. (Marketing Push)
            - 🐕 **Dogs:** Low Profit, Low Volume. (Remove from Menu)
            """)
            
        with st.expander("View Full Menu Data Table"):
            st.dataframe(data['df_menu'])
    else:
        st.warning("⚠️ Menu data missing.")

# --- TAB 2: MAP ---
with tabs[2]:
    st.header("Geospatial Competitor Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if not data['df_map'].empty:
            df_m = data['df_map'].copy()
            # Robust Rename
            cols = df_m.columns.str.lower()
            if 'latitude' in cols: df_m.rename(columns={df_m.columns[list(cols).index('latitude')]: 'Latitude'}, inplace=True)
            if 'longitude' in cols: df_m.rename(columns={df_m.columns[list(cols).index('longitude')]: 'Longitude'}, inplace=True)
            
            df_m['Latitude'] = pd.to_numeric(df_m['Latitude'], errors='coerce')
            df_m['Longitude'] = pd.to_numeric(df_m['Longitude'], errors='coerce')
            df_m = df_m.dropna(subset=['Latitude', 'Longitude'])
            
            if not df_m.empty:
                m = folium.Map(location=[df_m['Latitude'].mean(), df_m['Longitude'].mean()], zoom_start=14)
                for i, row in df_m.iterrows():
                    name_col = [c for c in df_m.columns if 'name' in c.lower()]
                    loc_name = str(row[name_col[0]]) if name_col else "Location"
                    
                    is_us = 'BEST REGARDS' in loc_name.upper()
                    color = 'red' if is_us else 'blue'
                    icon = 'star' if is_us else 'info-sign'
                    
                    folium.Marker(
                        [row['Latitude'], row['Longitude']], 
                        popup=loc_name, 
                        tooltip=loc_name,
                        icon=folium.Icon(color=color, icon=icon)
                    ).add_to(m)
                st_folium(m, width=800, height=500)
            else:
                st.warning("Map coordinates invalid.")
        else:
            st.warning("⚠️ Map data missing.")

    with col2:
        st.info("💡 **Analyst Insight:**\nThis map visualizes the 'Agglomeration Effect'. Blue markers represent competitors. High density of blue markers near Best Regards (Red) often correlates with higher foot traffic, rather than lost sales.")
        if not data['df_map'].empty:
            st.write("### Competitor List")
            st.dataframe(data['df_map'])

# --- TAB 3: VOIDS ---
with tabs[3]:
    st.header("Operational Risk & Void Detection")
    st.markdown("💡 **Analyst Insight:** *High void rates, specifically on Friday nights, indicate potential operational slippage or theft. The servers listed below have Z-Scores > 2.0, meaning they are statistically anomalous compared to the staff average.*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Suspicious Servers (Z-Score > 2)")
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
            
    st.subheader("Suspicious Combinations (Server + Tab)")
    if not data['df_combo'].empty:
        st.dataframe(data['df_combo'], hide_index=True)

# --- TAB 4: SENTIMENT ---
with tabs[4]:
    st.header("Sentiment Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if not data['df_sentiment'].empty:
            cols = data['df_sentiment'].columns
            if len(cols) > 1:
                # Plot Correlation
                st.line_chart(data['df_sentiment'].set_index(cols[0])[cols[1]])
            else:
                st.dataframe(data['df_sentiment'])
        else:
            st.warning("⚠️ Sentiment data missing.")
            
    with col2:
        st.info("💡 **Analyst Insight:**\nWe correlate the Customer Experience (CX) Index with actual Revenue. Note the 'Lag Effect'—negative reviews often impact revenue 2-4 weeks after they are posted.")
