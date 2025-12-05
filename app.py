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

    # A. LOAD MASTER (Parquet or CSV)
    try:
        data['df_master'] = pd.read_parquet('master_data.parquet')
    except:
        try:
            data['df_master'] = pd.read_parquet('master_data_recent.parquet')
        except:
            data['df_master'] = load_csv('master_data.csv')

    # Fix Master Dates
    if not data['df_master'].empty:
        # Find the date column (it might be 'Date' or something else)
        cols = data['df_master'].columns
        if 'Date' in cols:
            data['df_master']['Date'] = pd.to_datetime(data['df_master']['Date'], errors='coerce')
            data['df_master']['Month'] = data['df_master']['Date'].dt.to_period('M').astype(str)

    # B. LOAD OTHER FILES
    data['df_menu'] = load_csv('menu_forensics.csv')
    data['df_sentiment'] = load_csv('sentiment.csv')
    data['df_forecast'] = load_csv('forecast_values.csv')
    data['df_geo'] = load_csv('geo_pressure.csv')
    data['df_map'] = load_csv('map_data.csv')
    data['df_servers'] = load_csv('suspicious_servers.csv')
    data['df_voids_h'] = load_csv('hourly_voids.csv')
    data['df_voids_d'] = load_csv('daily_voids.csv')
    data['df_combo'] = load_csv('suspicious_combinations.csv')

    # C. PREPARE MONTHLY REVENUE (HISTORICAL)
    if not data['df_master'].empty:
        clean_df = data['df_master']
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 Forecast", "🍔 Menu", "🗺️ Map", "🚨 Voids", "❤️ Sentiment"
])

# --- TAB 1: FORECAST (FIXED) ---
with tab1:
    st.header("Revenue Forecast")
    
    if not data['df_forecast'].empty:
        fc_data = data['df_forecast'].copy()
        
        # --- EMERGENCY COLUMN FIX ---
        # If columns are weird, force rename the first two columns
        if len(fc_data.columns) >= 1:
            # Assume 1st column is Date/Month
            fc_data.rename(columns={fc_data.columns[0]: 'Month'}, inplace=True)
        if len(fc_data.columns) >= 2:
            # Assume 2nd column is Revenue
            fc_data.rename(columns={fc_data.columns[1]: 'Revenue'}, inplace=True)
            
        fc_data['Type'] = 'Forecast'
        
        # Convert Month to Datetime
        try:
            fc_data['Month'] = pd.to_datetime(fc_data['Month'])
        except Exception as e:
            st.error(f"Date Error: {e}")

        # Combine with Historical
        if not data['monthly_revenue'].empty:
            data['monthly_revenue']['Month'] = pd.to_datetime(data['monthly_revenue']['Month'])
            combined_df = pd.concat([data['monthly_revenue'], fc_data], ignore_index=True)
        else:
            combined_df = fc_data

        # Plot
        fig = px.line(combined_df, x='Month', y='Revenue', color='Type', 
                      title="Projected Revenue", color_discrete_map={'Historical': 'gray', 'Forecast': 'red'})
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("Forecast CSV is empty or missing.")

# --- TAB 2: MENU ---
with tab2:
    st.header("Menu Engineering")
    if not data['df_menu'].empty:
        fig = px.scatter(data['df_menu'], x="Qty_Sold", y="Total_Revenue", color="BCG_Matrix",
                         size="Total_Revenue", hover_name="Menu Item", title="Menu Matrix")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Menu CSV missing.")

# --- TAB 3: MAP ---
with tab3:
    st.header("Competitor Map")
    if not data['df_map'].empty:
        # Clean Data
        df_m = data['df_map'].copy()
        df_m['Latitude'] = pd.to_numeric(df_m['Latitude'], errors='coerce')
        df_m['Longitude'] = pd.to_numeric(df_m['Longitude'], errors='coerce')
        df_m = df_m.dropna(subset=['Latitude', 'Longitude'])
        
        if not df_m.empty:
            m = folium.Map(location=[df_m['Latitude'].mean(), df_m['Longitude'].mean()], zoom_start=13)
            for i, row in df_m.iterrows():
                color = 'red' if 'BEST REGARDS' in str(row.get('Location Name','')).upper() else 'blue'
                folium.CircleMarker([row['Latitude'], row['Longitude']], radius=10, color=color, fill=True, fill_color=color).add_to(m)
            st_folium(m, width=800, height=500)
    else:
        st.warning("Map CSV missing.")

# --- TAB 4: VOIDS ---
with tabs[4]:
    st.header("Theft Detection")
    col1, col2 = st.columns(2)
    with col1:
        if not data['df_servers'].empty:
            st.write("Suspicious Servers")
            st.dataframe(data['df_servers'])
    with col2:
        if not data['df_voids_h'].empty:
            st.write("High Risk Hours")
            st.bar_chart(data['df_voids_h'].set_index('Hour_of_Day')['Void_Rate'])

# --- TAB 5: SENTIMENT ---
with tabs[5]:
    st.header("Sentiment Analysis")
    if not data['df_sentiment'].empty:
        st.line_chart(data['df_sentiment'].set_index('Month')[['CX_Index']])
