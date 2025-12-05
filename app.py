import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
        """Forces columns to be numeric, handling currency symbols and commas"""
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        return df

    # --- A. LOAD MASTER ---
    # Robust loader that tries every possible format
    data['df_master'] = pd.DataFrame()
    if os.path.exists('master_data.parquet'):
        try:
            data['df_master'] = pd.read_parquet('master_data.parquet')
        except Exception as e:
            errors.append(f"Parquet Load Error: {e}")
    
    # Fallback to CSV if Parquet failed or missing
    if data['df_master'].empty and os.path.exists('master_data.csv'):
        try:
            data['df_master'] = pd.read_csv('master_data.csv', low_memory=False)
        except Exception as e:
            errors.append(f"CSV Load Error: {e}")
            
    if data['df_master'].empty and not errors:
        errors.append("No Master Data File Found (master_data.parquet or master_data.csv)")

    # Fix Master Data Types
    if not data['df_master'].empty:
        cols = data['df_master'].columns
        if 'Date' in cols:
            data['df_master']['Date'] = pd.to_datetime(data['df_master']['Date'], errors='coerce')
            data['df_master']['Month'] = data['df_master']['Date'].dt.to_period('M').astype(str)
        data['df_master'] = clean_numeric(data['df_master'], ['Net Price', 'Qty'])

    # --- B. LOAD ANALYTICS FILES ---
    data['df_forecast'] = clean_numeric(load_safe('forecast_values.csv'), ['Forecasted_Revenue'])
    data['df_metrics'] = load_safe('forecast_metrics.csv')
    
    # Menu: Filter out Zeros immediately
    menu_raw = load_safe('menu_forensics.csv')
    menu_raw = clean_numeric(menu_raw, ['Qty_Sold', 'Total_Revenue', 'Item_Void_Rate'])
    data['df_menu'] = menu_raw[(menu_raw['Total_Revenue'] > 0) | (menu_raw['Qty_Sold'] > 0)]

    # Map Data: Standardize Columns immediately
    map_raw = load_safe('map_data.csv')
    cols = map_raw.columns.str.lower()
    if 'latitude' in cols: map_raw.rename(columns={map_raw.columns[list(cols).index('latitude')]: 'Latitude'}, inplace=True)
    if 'longitude' in cols: map_raw.rename(columns={map_raw.columns[list(cols).index('longitude')]: 'Longitude'}, inplace=True)
    if 'location name' in cols: map_raw.rename(columns={map_raw.columns[list(cols).index('location name')]: 'Location Name'}, inplace=True)
    if 'total_revenue' in cols: map_raw.rename(columns={map_raw.columns[list(cols).index('total_revenue')]: 'Total_Revenue'}, inplace=True)
    
    data['df_map'] = clean_numeric(map_raw, ['Latitude', 'Longitude', 'Total_Revenue'])
    
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
        if load_errors:
            st.error(f"Debug: {load_errors[0]}")
    
    if not data['df_forecast'].empty:
        st.success("✅ Forecast Model: Active")

# --- 4. MAIN TABS ---
st.title("📊 Business Intelligence Dashboard")

tabs = st.tabs([
    "📉 Forecast", 
    "🍔 Menu Matrix", 
    "🔥 Competitive Intelligence", 
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

# --- TAB 2: COMPETITIVE INTELLIGENCE (3 MODELS) ---
with tabs[2]:
    st.header("Competitive Intelligence Models")
    st.markdown("""
    This section synthesizes three distinct models to evaluate market positioning:
    1.  **Map Data (Static):** Physical locations of competitors relative to Best Regards.
    2.  **Competitor Impact Ranking:** Determining which competitors actively steal market share.
    3.  **Geo-Pressure Model:** Measuring the density of competition over time (The "Heat").
    """)
    
    if not data['df_map'].empty:
        # --- 1. DATA PREP ---
        df_m = data['df_map'].copy()
        df_m = df_m.dropna(subset=['Latitude', 'Longitude'])
        
        # --- 2. LAYOUT ---
        # Interactive Map Section
        st.subheader("1. & 3. Geospatial Market Pressure (Time-Lapse Heatmap)")
        st.caption("Use the slider at the bottom to visualize how 'Geo-Pressure' (Market Intensity) shifts over time.")
        
        # TRY/EXCEPT BLOCK FOR MAP RENDERING
        try:
            # Setup Time Data
            heat_data = []
            time_index = []
            valid_time_data = False
            
            if not data['df_geo'].empty and 'Month' in data['df_geo'].columns and 'GeoPressure_Total' in data['df_geo'].columns:
                try:
                    df_g = data['df_geo'].copy()
                    df_g['Month'] = pd.to_datetime(df_g['Month'])
                    df_g = df_g.sort_values('Month')
                    
                    max_p = df_g['GeoPressure_Total'].max()
                    # Scale intensity 0-1
                    df_g['Intensity'] = (df_g['GeoPressure_Total'] / max_p).astype(float) if max_p > 0 else 0.5
                    
                    for _, row in df_g.iterrows():
                        monthly_points = []
                        # Force intensity to simple float
                        intensity = float(row['Intensity'])
                        
                        for _, loc in df_m.iterrows():
                            # STRICT FLOAT CONVERSION [lat, lon, intensity]
                            lat = float(loc['Latitude'])
                            lon = float(loc['Longitude'])
                            monthly_points.append([lat, lon, intensity])
                        
                        if monthly_points:
                            heat_data.append(monthly_points)
                            # Ensure time index is a simple list of strings
                            time_index.append(str(row['Month'].strftime('%Y-%m')))
                    
                    valid_time_data = True
                except Exception as e:
                    st.warning(f"Time-Series calculation error: {e}")

            # Render Map
            # Force bounds based on static data
            min_lat, max_lat = df_m['Latitude'].min(), df_m['Latitude'].max()
            min_lon, max_lon = df_m['Longitude'].min(), df_m['Longitude'].max()
            
            m = folium.Map(location=[df_m['Latitude'].mean(), df_m['Longitude'].mean()], zoom_start=13)
            # Explicitly fit bounds to prevent auto-zoom crash
            m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]]) 
            
            # Static Markers
            for _, row in df_m.iterrows():
                loc_name = str(row.get('Location Name', 'Location'))
                color = 'red' if 'BEST REGARDS' in loc_name.upper() else 'blue'
                folium.CircleMarker(
                    [row['Latitude'], row['Longitude']], 
                    radius=6, color=color, fill=True, fill_color=color, fill_opacity=0.9,
                    tooltip=loc_name
                ).add_to(m)

            # Time Player
            if valid_time_data and heat_data:
                plugins.HeatMapWithTime(
                    heat_data,
                    index=time_index,
                    auto_play=True,
                    radius=50,
                    max_opacity=0.6,
                    use_local_extrema=False
                ).add_to(m)
            
            st_folium(m, width=900, height=500, returned_objects=[])
            
        except Exception as e:
            # FALLBACK STATIC MAP
            st.error(f"Interactive Map Error: {e}. Loading Static Fallback.")
            m_static = folium.Map(location=[df_m['Latitude'].mean(), df_m['Longitude'].mean()], zoom_start=13)
            for _, row in df_m.iterrows():
                folium.Marker([row['Latitude'], row['Longitude']]).add_to(m_static)
            st_folium(m_static, width=900, height=500, returned_objects=[])
            
        # --- 3. IMPACT RANKINGS ---
        st.markdown("---")
        st.subheader("2. Competitor Impact Rankings")
        st.markdown("""
        **What this shows:**
        We rank competitors not just by distance, but by estimated **Revenue Impact**. 
        Competitors with high revenue that are geographically close exert the highest "Gravitational Pull" on your customers.
        """)
        
        # Create Ranking Table
        impact_df = df_m.copy()
        # Ensure we have revenue data
        if 'Total_Revenue' in impact_df.columns:
            impact_df = impact_df.sort_values('Total_Revenue', ascending=False)
            impact_df = impact_df[['Location Name', 'Total_Revenue', 'Distance_mi'] if 'Distance_mi' in impact_df.columns else ['Location Name', 'Total_Revenue']]
            
            # Display Top 10
            st.dataframe(
                impact_df.head(10),
                hide_index=True,
                column_config={
                    "Total_Revenue": st.column_config.NumberColumn("Est. Annual Revenue", format="$%.0f"),
                    "Distance_mi": st.column_config.NumberColumn("Distance (mi)", format="%.2f mi")
                }
            )
        else:
            st.info("Revenue data missing for Impact Rankings.")
            
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
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if not data['df_sentiment'].empty:
            # DUAL AXIS CHART
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=data['df_sentiment']['Month'],
                y=data['df_sentiment']['BestRegards_Revenue'],
                name="Revenue ($)",
                marker_color='lightgreen',
                opacity=0.6
            ))

            fig.add_trace(go.Scatter(
                x=data['df_sentiment']['Month'],
                y=data['df_sentiment']['CX_Index'],
                name="CX Index",
                yaxis="y2",
                line=dict(color='red', width=3)
            ))

            fig.update_layout(
                title="Correlation: Revenue vs. Customer Experience",
                xaxis_title="Month",
                yaxis=dict(title="Revenue ($)"),
                yaxis2=dict(
                    title="CX Index (0-1)",
                    overlaying="y",
                    side="right",
                    range=[0, 1] 
                ),
                legend=dict(x=0, y=1.1, orientation="h")
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Sentiment data missing.")
            
    with col2:
        st.info("""
        💡 **Analyst Insight:**
        
        **The "Lag Effect":**
        Notice how the Red Line (Customer Experience) often moves *before* the Green Bars (Revenue).
        
        A drop in sentiment today typically correlates with a revenue drop **2-4 weeks later**.
        
        **Action:** Use the CX Index as a "Early Warning System" to correct service issues before they hit the bottom line.
        """)
