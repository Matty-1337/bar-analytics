import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium import plugins
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

    def load_safe(filename):
        if os.path.exists(filename):
            try:
                return pd.read_csv(filename)
            except:
                return pd.DataFrame()
        return pd.DataFrame()

    def clean_numeric(df, cols):
        """Forces columns to be numeric, replacing errors with 0"""
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        return df

    # --- A. LOAD MASTER ---
    try:
        data['df_master'] = pd.read_parquet('master_data.parquet')
    except:
        try:
            data['df_master'] = pd.read_csv('master_data.csv', low_memory=False)
        except Exception as e:
            data['df_master'] = pd.DataFrame()
            errors.append(f"Master Load Error: {e}")

    # Fix Master Data Types
    if not data['df_master'].empty:
        # Fix Dates
        cols = data['df_master'].columns
        if 'Date' in cols:
            data['df_master']['Date'] = pd.to_datetime(data['df_master']['Date'], errors='coerce')
            data['df_master']['Month'] = data['df_master']['Date'].dt.to_period('M').astype(str)
        # Fix Numbers
        data['df_master'] = clean_numeric(data['df_master'], ['Net Price', 'Qty'])

    # --- B. LOAD ANALYTICS FILES ---
    data['df_forecast'] = clean_numeric(load_safe('forecast_values.csv'), ['Forecasted_Revenue'])
    data['df_metrics'] = load_safe('forecast_metrics.csv')
    
    # Menu: Filter out Zeros immediately
    menu_raw = load_safe('menu_forensics.csv')
    menu_raw = clean_numeric(menu_raw, ['Qty_Sold', 'Total_Revenue', 'Item_Void_Rate'])
    data['df_menu'] = menu_raw[(menu_raw['Total_Revenue'] > 0) & (menu_raw['Qty_Sold'] > 0)]

    data['df_map'] = load_safe('map_data.csv')
    
    data['df_servers'] = clean_numeric(load_safe('suspicious_servers.csv'), ['Void_Rate', 'Void_Z_Score', 'Potential_Loss'])
    data['df_voids_h'] = clean_numeric(load_safe('hourly_voids.csv'), ['Void_Rate', 'Hour_of_Day'])
    data['df_voids_d'] = clean_numeric(load_safe('daily_voids.csv'), ['Void_Rate'])
    
    data['df_combo'] = load_safe('suspicious_combinations.csv')
    data['df_geo'] = clean_numeric(load_safe('geo_pressure.csv'), ['GeoPressure_Total'])
    data['df_sentiment'] = clean_numeric(load_safe('sentiment.csv'), ['CX_Index', 'BestRegards_Revenue'])

    # --- C. PREPARE MONTHLY REVENUE ---
    if not data['df_master'].empty and 'Month' in data['df_master'].columns:
        clean_df = data['df_master']
        if 'is_void' in clean_df.columns:
            clean_df = clean_df[~clean_df['is_void']]
        
        monthly_data = clean_df.groupby('Month')['Net Price'].sum().reset_index()
        monthly_data.columns = ['Month', 'Revenue']
        monthly_data['Type'] = 'Historical'
        data['monthly_revenue'] = monthly_data
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
    if not data['df_master'].empty:
        st.success("✅ Master Data: Active")
    else:
        st.error("❌ Master Data: Missing")
    
    if not data['df_forecast'].empty:
        st.success("✅ Forecast Model: Active")

# --- 4. MAIN TABS ---
st.title("📊 Business Intelligence Dashboard")

tabs = st.tabs([
    "📉 Forecast", 
    "🍔 Menu Matrix", 
    "🔥 Market Heatmap", 
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
            # Emergency Column Rename
            if len(fc_data.columns) >= 1: fc_data.rename(columns={fc_data.columns[0]: 'Month'}, inplace=True)
            if len(fc_data.columns) >= 2: fc_data.rename(columns={fc_data.columns[1]: 'Revenue'}, inplace=True)
            
            fc_data['Type'] = 'Projection'
            
            # Combine
            combined_df = fc_data
            if not data['monthly_revenue'].empty:
                try:
                    hist = data['monthly_revenue'].copy()
                    hist['Month'] = hist['Month'].astype(str)
                    fc_data['Month'] = fc_data['Month'].astype(str)
                    combined_df = pd.concat([hist, fc_data], ignore_index=True)
                    combined_df['Month'] = pd.to_datetime(combined_df['Month'])
                    combined_df = combined_df.sort_values('Month')
                except:
                    pass

            fig = px.line(combined_df, x='Month', y='Revenue', color='Type', 
                          title="Revenue Trajectory: Historical vs. Projection", 
                          color_discrete_map={'Historical': 'gray', 'Projection': '#FF4B4B'},
                          markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Forecast data is empty or missing.")

    with col_b:
        st.info("💡 **Analyst Insight:**\nThe red line represents the projected revenue trajectory. We utilize this to identify potential revenue dips before they happen.")
        if not data['df_metrics'].empty:
            st.write("### Statistical Accuracy")
            st.dataframe(
                data['df_metrics'], 
                hide_index=True,
                column_config={
                    "RMSE": st.column_config.NumberColumn("RMSE", format="$%.2f"),
                    "MAE": st.column_config.NumberColumn("MAE", format="$%.2f")
                }
            )

# --- TAB 1: MENU ---
with tabs[1]:
    st.header("Menu Engineering (BCG Matrix)")
    if not data['df_menu'].empty:
        col1, col2 = st.columns([3, 1])
        with col1:
            fig = px.scatter(
                data['df_menu'], 
                x="Qty_Sold", y="Total_Revenue", 
                color="BCG_Matrix", size="Total_Revenue", 
                hover_name="Menu Item", 
                color_discrete_map={'Star': '#00CC96', 'Dog': '#EF553B', 'Plowhorse': '#AB63FA', 'Puzzle': '#FFA15A'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.info("**Strategy:** Promote Stars, Re-price Plowhorses, Re-invent Puzzles, Remove Dogs.")
            
        with st.expander("View Active Menu Data"):
            st.dataframe(
                data['df_menu'],
                column_config={
                    "Total_Revenue": st.column_config.NumberColumn("Revenue", format="$%.2f"),
                    "Item_Void_Rate": st.column_config.NumberColumn("Void Rate", format="%.2f%%")
                }
            )
    else:
        st.warning("⚠️ Menu data missing.")

# --- TAB 2: MARKET HEATMAP ---
with tabs[2]:
    st.header("Geospatial Market Pressure")
    
    if not data['df_map'].empty:
        # 1. Clean Map Data (Crash Prevention)
        df_m = data['df_map'].copy()
        
        # Robust Rename
        cols = df_m.columns.str.lower()
        if 'latitude' in cols: df_m.rename(columns={df_m.columns[list(cols).index('latitude')]: 'Latitude'}, inplace=True)
        if 'longitude' in cols: df_m.rename(columns={df_m.columns[list(cols).index('longitude')]: 'Longitude'}, inplace=True)
        
        # Force Numeric & Drop NaNs
        df_m['Latitude'] = pd.to_numeric(df_m['Latitude'], errors='coerce')
        df_m['Longitude'] = pd.to_numeric(df_m['Longitude'], errors='coerce')
        df_m = df_m.dropna(subset=['Latitude', 'Longitude'])
        
        if not df_m.empty:
            # 2. Setup Time Data (Only if valid)
            heat_data = []
            time_index = []
            
            # Check Geo Data
            if not data['df_geo'].empty and 'Month' in data['df_geo'].columns and 'GeoPressure_Total' in data['df_geo'].columns:
                try:
                    df_g = data['df_geo'].copy()
                    df_g['Month'] = pd.to_datetime(df_g['Month'])
                    df_g = df_g.sort_values('Month')
                    
                    max_p = df_g['GeoPressure_Total'].max()
                    df_g['Intensity'] = df_g['GeoPressure_Total'] / max_p if max_p > 0 else 0.5
                    
                    for _, row in df_g.iterrows():
                        monthly_points = []
                        intensity = row['Intensity']
                        # Add intensity to all competitor locations
                        for _, loc in df_m.iterrows():
                            monthly_points.append([loc['Latitude'], loc['Longitude'], intensity])
                        
                        if monthly_points:
                            heat_data.append(monthly_points)
                            time_index.append(row['Month'].strftime('%Y-%m'))
                except:
                    st.warning("Time-Series data corrupt. Showing static map.")

            # 3. Render Map
            # Center on average of valid points
            m = folium.Map(location=[df_m['Latitude'].mean(), df_m['Longitude'].mean()], zoom_start=13)
            
            # Static Markers
            for _, row in df_m.iterrows():
                name_col = [c for c in df_m.columns if 'name' in c.lower()]
                loc_name = str(row[name_col[0]]) if name_col else "Location"
                color = 'red' if 'BEST REGARDS' in loc_name.upper() else 'blue'
                folium.CircleMarker(
                    [row['Latitude'], row['Longitude']], 
                    radius=5, color=color, fill=True, fill_color=color, fill_opacity=0.8,
                    tooltip=loc_name
                ).add_to(m)

            # Time Player (Only if valid data exists)
            if heat_data and len(heat_data) > 0:
                plugins.HeatMapWithTime(
                    heat_data,
                    index=time_index,
                    auto_play=True,
                    radius=40,
                    max_opacity=0.6
                ).add_to(m)
            
            st_folium(m, width=800, height=500)
            
        else:
            st.error("Map Data exists but contains no valid Latitude/Longitude.")
    else:
        st.warning("⚠️ Map data missing.")

# --- TAB 3: VOIDS ---
with tabs[3]:
    st.header("Operational Risk & Void Detection")
    st.markdown("💡 **Analyst Insight:** *High void rates indicate potential operational slippage. Servers with Z-Scores > 2.0 are statistically anomalous.*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Suspicious Servers")
        if not data['df_servers'].empty:
            st.dataframe(
                data['df_servers'], 
                hide_index=True,
                column_config={
                    "Void_Rate": st.column_config.NumberColumn("Void Rate", format="%.2f%%"),
                    "Void_Z_Score": st.column_config.NumberColumn("Z-Score", format="%.2f"),
                    "Potential_Loss": st.column_config.NumberColumn("Est. Loss", format="$%.2f")
                }
            )
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
        cols = data['df_sentiment'].columns
        if len(cols) > 1:
            st.line_chart(data['df_sentiment'].set_index(cols[0])[cols[1]])
        else:
            st.dataframe(data['df_sentiment'])
    else:
        st.warning("⚠️ Sentiment data missing.")
