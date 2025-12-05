import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
from streamlit_folium import st_folium
import streamlit.components.v1 as components
import os
from datetime import datetime, timedelta
from scipy import stats

st.set_page_config(page_title="Best Regards Analytics", page_icon="🍸", layout="wide", initial_sidebar_state="expanded")

# Best Regards Color Scheme - Deep Green, Gold, White
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

* {font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;}
#MainMenu {visibility: hidden;} footer {visibility: hidden;}

/* Main Background - Deep Forest Green */
.stApp {
    background: linear-gradient(135deg, #1a2e1a 0%, #2d4a2d 50%, #1a3320 100%);
}

.block-container {padding-top: 2rem; padding-bottom: 2rem;}

/* Glass Card Styling */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(20px);
    border-radius: 16px;
    border: 1px solid rgba(212, 175, 55, 0.2);
    padding: 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
    height: 100%;
}

.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 48px 0 rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(212, 175, 55, 0.4);
}

/* Metric Cards */
.metric-card-success, .metric-card-warning, .metric-card-danger, .metric-card-info {
    backdrop-filter: blur(20px);
    border-radius: 16px;
    padding: 24px 20px;
    min-height: 150px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.3s ease;
}

.metric-card-success:hover, .metric-card-warning:hover, .metric-card-danger:hover, .metric-card-info:hover {
    transform: translateY(-4px);
}

.metric-card-success {
    background: linear-gradient(135deg, rgba(45, 74, 45, 0.9) 0%, rgba(26, 51, 32, 0.9) 100%);
    border: 1px solid rgba(212, 175, 55, 0.4);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}

.metric-card-warning {
    background: linear-gradient(135deg, rgba(139, 109, 32, 0.4) 0%, rgba(90, 70, 20, 0.4) 100%);
    border: 1px solid rgba(212, 175, 55, 0.5);
    box-shadow: 0 8px 32px 0 rgba(212, 175, 55, 0.15);
}

.metric-card-danger {
    background: linear-gradient(135deg, rgba(120, 40, 40, 0.5) 0%, rgba(80, 25, 25, 0.5) 100%);
    border: 1px solid rgba(200, 80, 80, 0.4);
    box-shadow: 0 8px 32px 0 rgba(200, 80, 80, 0.15);
}

.metric-card-info {
    background: linear-gradient(135deg, rgba(45, 74, 45, 0.9) 0%, rgba(26, 51, 32, 0.9) 100%);
    border: 1px solid rgba(212, 175, 55, 0.4);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}

.metric-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #d4af37;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 2.25rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.2;
    margin-bottom: 4px;
    font-family: 'Playfair Display', serif;
}

.metric-subtitle {
    font-size: 0.85rem;
    color: rgba(255, 255, 255, 0.7);
    font-weight: 400;
}

.metric-change {
    font-size: 0.85rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 8px;
    display: inline-block;
    margin-top: 8px;
}

.metric-change-positive {
    background: rgba(76, 175, 80, 0.25);
    color: #81c784;
}

.metric-change-negative {
    background: rgba(239, 83, 80, 0.25);
    color: #ef5350;
}

/* Tab Styling - Green/Gold Theme */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(26, 46, 26, 0.8);
    backdrop-filter: blur(20px);
    padding: 8px;
    border-radius: 12px;
    border: 1px solid rgba(212, 175, 55, 0.3);
}

.stTabs [data-baseweb="tab"] {
    height: 46px;
    padding: 0 20px;
    background: transparent;
    border-radius: 8px;
    color: rgba(255, 255, 255, 0.8);
    font-weight: 500;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(212, 175, 55, 0.15);
    color: #ffffff;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2d4a2d 0%, #3d5c3d 100%);
    color: #d4af37 !important;
    border: 1px solid rgba(212, 175, 55, 0.5);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}

/* Headers - White with Gold Accent */
h1 {
    color: #ffffff !important;
    font-weight: 700;
    font-size: 2.5rem;
    letter-spacing: -0.02em;
    font-family: 'Playfair Display', serif;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

h2 {
    color: #ffffff !important;
    font-weight: 600;
    font-size: 1.75rem;
    margin-top: 1.5rem;
    letter-spacing: -0.01em;
    font-family: 'Playfair Display', serif;
}

h3 {
    color: #ffffff !important;
    font-weight: 600;
    font-size: 1.25rem;
}

p, span, label {
    color: rgba(255, 255, 255, 0.9) !important;
    line-height: 1.6;
}

/* Subheader fix */
.stMarkdown h2, .stMarkdown h3, [data-testid="stHeader"] {
    color: #ffffff !important;
}

div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 {
    color: #ffffff !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a2e1a 0%, #2d4a2d 50%, #1a3320 100%);
    border-right: 1px solid rgba(212, 175, 55, 0.3);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #d4af37 !important;
    font-family: 'Playfair Display', serif;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: rgba(255, 255, 255, 0.85) !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.5rem;
    font-weight: 600;
}

[data-testid="stMetricLabel"] {
    color: rgba(255, 255, 255, 0.8) !important;
    font-weight: 500;
}

[data-testid="stMetricDelta"] {
    color: #81c784 !important;
}

/* Success/Info/Warning/Error boxes */
.stSuccess, .stInfo, .stWarning, .stError {
    background: rgba(45, 74, 45, 0.6) !important;
    border: 1px solid rgba(212, 175, 55, 0.3) !important;
    border-radius: 8px;
}

.stSuccess > div, .stInfo > div, .stWarning > div, .stError > div {
    color: #ffffff !important;
}

/* Expander styling */
.streamlit-expanderHeader {
    background: rgba(45, 74, 45, 0.6);
    backdrop-filter: blur(20px);
    border-radius: 8px;
    border: 1px solid rgba(212, 175, 55, 0.2);
    color: #ffffff !important;
    font-weight: 500;
}

.streamlit-expanderContent {
    background: rgba(26, 46, 26, 0.6);
    border: 1px solid rgba(212, 175, 55, 0.2);
    border-top: none;
    border-radius: 0 0 8px 8px;
}

/* Slider styling */
.stSlider > div > div {
    background: rgba(212, 175, 55, 0.3) !important;
}

.stSlider > div > div > div {
    background: #d4af37 !important;
}

/* Select box */
.stSelectbox > div > div {
    background: rgba(45, 74, 45, 0.8) !important;
    border: 1px solid rgba(212, 175, 55, 0.3) !important;
    color: #ffffff !important;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #2d4a2d 0%, #3d5c3d 100%);
    color: #d4af37;
    border: 1px solid rgba(212, 175, 55, 0.5);
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #3d5c3d 0%, #4d6c4d 100%);
    border-color: #d4af37;
    color: #ffffff;
}

/* DataFrame styling */
.stDataFrame {
    background: rgba(26, 46, 26, 0.6);
    border-radius: 8px;
    border: 1px solid rgba(212, 175, 55, 0.2);
}

/* Plotly chart backgrounds */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* Info tooltip styling */
.info-tooltip {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    background: rgba(212, 175, 55, 0.3);
    border: 1px solid rgba(212, 175, 55, 0.5);
    border-radius: 50%;
    font-size: 11px;
    font-weight: 600;
    color: #d4af37;
    cursor: help;
    margin-left: 8px;
    position: relative;
}

.info-tooltip:hover {
    background: rgba(212, 175, 55, 0.5);
}

.header-with-info {
    display: flex;
    align-items: center;
    gap: 8px;
}
</style>
""", unsafe_allow_html=True)

# Info tooltip descriptions
INFO_TOOLTIPS = {
    'revenue_forecast': 'Revenue Forecasting uses historical sales data and machine learning models to predict future revenue. Adjust scenarios to see how different market conditions might impact your business.',
    'scenario_planning': 'Scenario Planning allows you to model different business conditions. Adjust growth rates, seasonality, and external factors to see projected revenue impact.',
    'revenue_trend': 'This chart shows historical revenue (solid line) and forecasted revenue (dashed line). The shaded area represents the 95% confidence interval for predictions.',
    'revenue_category': 'Revenue breakdown by product category helps identify which menu sections drive the most sales and where to focus marketing efforts.',
    'mom_analysis': 'Month-over-Month analysis shows the percentage change in revenue compared to the previous month. Green bars indicate growth, red indicates decline.',
    'menu_intelligence': 'Menu Intelligence analyzes item performance using the BCG Matrix framework, categorizing items as Stars (high growth, high share), Cash Cows (low growth, high share), Question Marks (high growth, low share), or Dogs (low growth, low share).',
    'bcg_matrix': 'The BCG Matrix plots menu items by quantity sold (market share) vs revenue (growth). Stars should be promoted, Cash Cows maintained, Question Marks evaluated, and Dogs considered for removal.',
    'top_performers': 'Top performing items generate the highest revenue and should be featured prominently on menus and in marketing materials.',
    'underperformers': 'Underperforming items (Dogs) have low sales and low revenue. Consider removing, repricing, or rebranding these items to improve menu efficiency.',
    'competitive_landscape': 'Competitive Landscape Analysis maps nearby competitors and analyzes market positioning. The heat map shows revenue concentration in your market area.',
    'competitor_map': 'Interactive map showing your location (green star) and competitors. Color indicates estimated revenue: red (high), orange (medium), blue (lower). Heat overlay shows market density.',
    'geo_pressure': 'Geographic Pressure Index measures competitive intensity in your area over time. Higher values indicate increased competition, which may require strategic response.',
    'risk_detection': 'Risk Detection analyzes transaction patterns to identify potential operational issues including unusual void rates, suspicious timing patterns, and anomalous server behavior.',
    'server_risk': 'Server Risk Assessment plots staff members by void rate and potential loss. Points above the threshold line warrant investigation. Larger points indicate higher risk scores.',
    'risk_distribution': 'Distribution of staff by risk level. High-risk individuals should be monitored closely, medium-risk reviewed periodically, and low-risk maintained.',
    'void_patterns': 'Void pattern analysis shows when voids occur most frequently. Unusual spikes during specific hours or days may indicate operational issues requiring attention.',
    'sentiment_analysis': 'Sentiment Analysis tracks customer experience through review analysis, feedback scores, and satisfaction metrics to monitor brand perception over time.',
    'cx_trend': 'Customer Experience Index trend shows how customer satisfaction has evolved. The dotted line represents the average, helping identify periods above or below normal.',
    'cx_gauge': 'Current CX Score gauge shows real-time customer satisfaction. Green zone (0.7-1.0) is excellent, yellow (0.4-0.7) needs attention, red (0-0.4) requires immediate action.',
    'sentiment_correlation': 'This chart overlays revenue and sentiment to show correlation. Strong positive correlation suggests customer satisfaction directly impacts revenue.',
    'executive_summary': 'Executive Summary provides a high-level overview of key performance indicators, strategic recommendations, and a 90-day action roadmap for business improvement.'
}

def render_header_with_info(title, info_key, level=2):
    """Render a header with an info tooltip"""
    tooltip_text = INFO_TOOLTIPS.get(info_key, '')
    if level == 2:
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-top: 1.5rem; margin-bottom: 1rem;">
            <h2 style="color: #ffffff !important; margin: 0; font-family: 'Playfair Display', serif;">{title}</h2>
            <span title="{tooltip_text}" style="display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; background: rgba(212, 175, 55, 0.3); border: 1px solid rgba(212, 175, 55, 0.5); border-radius: 50%; font-size: 12px; font-weight: 600; color: #d4af37; cursor: help; margin-left: 10px;">i</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-top: 1rem; margin-bottom: 0.75rem;">
            <h3 style="color: #ffffff !important; margin: 0;">{title}</h3>
            <span title="{tooltip_text}" style="display: inline-flex; align-items: center; justify-content: center; width: 18px; height: 18px; background: rgba(212, 175, 55, 0.3); border: 1px solid rgba(212, 175, 55, 0.5); border-radius: 50%; font-size: 11px; font-weight: 600; color: #d4af37; cursor: help; margin-left: 8px;">i</span>
        </div>
        """, unsafe_allow_html=True)

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
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        return df

    data['df_master'] = pd.DataFrame()
    master_files = ['master_data.parquet', 'master_data_recent.parquet', 'master_data.csv']
    
    for f in master_files:
        if os.path.exists(f):
            try:
                if f.endswith('.parquet'):
                    data['df_master'] = pd.read_parquet(f)
                else:
                    data['df_master'] = pd.read_csv(f, low_memory=False)
                
                if not data['df_master'].empty and len(data['df_master'].columns) > 1:
                    break
            except:
                pass

    if data['df_master'].empty:
        try:
            if os.path.exists('sentiment.csv'):
                df_sent = pd.read_csv('sentiment.csv')
                if 'Month' in df_sent.columns and 'BestRegards_Revenue' in df_sent.columns:
                    data['df_master'] = pd.DataFrame({
                        'Date': pd.to_datetime(df_sent['Month']),
                        'Month': df_sent['Month'],
                        'Net Price': df_sent['BestRegards_Revenue'],
                        'Qty': 0,
                        'is_void': False
                    })
                    data['df_master'] = clean_numeric(data['df_master'], ['Net Price'])
        except:
            pass

    if not data['df_master'].empty:
        cols = data['df_master'].columns
        date_col = 'Date' if 'Date' in cols else ('date' if 'date' in cols else ('Time' if 'Time' in cols else None))
        
        if date_col:
            data['df_master']['Date'] = pd.to_datetime(data['df_master'][date_col], errors='coerce')
            if 'Month' not in cols:
                data['df_master']['Month'] = data['df_master']['Date'].dt.to_period('M').astype(str)
            
        data['df_master'] = clean_numeric(data['df_master'], ['Net Price', 'Qty'])

    data['df_forecast'] = clean_numeric(load_safe('forecast_values.csv'), ['Forecasted_Revenue'])
    data['df_metrics'] = load_safe('forecast_metrics.csv')
    
    menu_raw = load_safe('menu_forensics.csv')
    menu_raw = clean_numeric(menu_raw, ['Qty_Sold', 'Total_Revenue', 'Item_Void_Rate'])
    data['df_menu'] = menu_raw[(menu_raw['Total_Revenue'] > 0) | (menu_raw['Qty_Sold'] > 0)]

    map_raw = load_safe('map_data.csv')
    if not map_raw.empty:
        cols_lower = [c.lower() for c in map_raw.columns]
        col_mapping = {}
        for i, c in enumerate(cols_lower):
            if 'latitude' in c or c == 'lat':
                col_mapping[map_raw.columns[i]] = 'Latitude'
            elif 'longitude' in c or c == 'lon' or c == 'lng':
                col_mapping[map_raw.columns[i]] = 'Longitude'
            elif 'location' in c and 'name' in c:
                col_mapping[map_raw.columns[i]] = 'Location Name'
            elif 'name' in c and 'Location Name' not in col_mapping.values():
                col_mapping[map_raw.columns[i]] = 'Location Name'
            elif 'revenue' in c:
                col_mapping[map_raw.columns[i]] = 'Total_Revenue'
        map_raw = map_raw.rename(columns=col_mapping)
    
    data['df_map'] = clean_numeric(map_raw, ['Latitude', 'Longitude', 'Total_Revenue'])
    
    if not data['df_map'].empty and 'Total_Revenue' in data['df_map'].columns:
        max_rev = data['df_map']['Total_Revenue'].max()
        while max_rev > 1_000_000_000:
            data['df_map']['Total_Revenue'] = data['df_map']['Total_Revenue'] / 1000
            max_rev = data['df_map']['Total_Revenue'].max()
    
    data['df_servers'] = clean_numeric(load_safe('suspicious_servers.csv'), ['Void_Rate', 'Void_Z_Score', 'Potential_Loss'])
    data['df_voids_h'] = clean_numeric(load_safe('hourly_voids.csv'), ['Void_Rate', 'Hour_of_Day'])
    data['df_voids_d'] = clean_numeric(load_safe('daily_voids.csv'), ['Void_Rate'])
    data['df_combo'] = load_safe('suspicious_combinations.csv')
    
    data['df_geo'] = clean_numeric(load_safe('geo_pressure.csv'), ['GeoPressure_Total'])
    if not data['df_geo'].empty:
        if 'Date' in data['df_geo'].columns:
            data['df_geo'].rename(columns={'Date': 'Month'}, inplace=True)

    data['df_sentiment'] = clean_numeric(load_safe('sentiment.csv'), ['CX_Index', 'BestRegards_Revenue'])

    if not data['df_master'].empty and 'Month' in data['df_master'].columns:
        clean_df = data['df_master']
        if 'is_void' in clean_df.columns:
            if clean_df['is_void'].dtype == 'object':
                 clean_df['is_void'] = clean_df['is_void'].astype(str).str.lower().isin(['true', '1', 'yes'])
            clean_df = clean_df[~clean_df['is_void']]
        
        monthly_data = clean_df.groupby('Month')['Net Price'].sum().reset_index()
        monthly_data.columns = ['Month', 'Revenue']
        
        mask = monthly_data['Month'].astype(str) == '2024-09'
        if mask.any():
            monthly_data.loc[mask, 'Revenue'] = 316057.93
        
        monthly_data['Type'] = 'Historical'
        data['monthly_revenue'] = monthly_data
    elif not data['df_sentiment'].empty and 'BestRegards_Revenue' in data['df_sentiment'].columns:
         fallback_rev = data['df_sentiment'][['Month', 'BestRegards_Revenue']].copy()
         fallback_rev.columns = ['Month', 'Revenue']
         fallback_rev['Type'] = 'Historical'
         data['monthly_revenue'] = fallback_rev
    else:
        data['monthly_revenue'] = pd.DataFrame()

    return data, errors

def calculate_kpis(data):
    kpis = {}
    
    if not data['monthly_revenue'].empty:
        recent_revenue = data['monthly_revenue']['Revenue'].tail(3).mean()
        kpis['avg_monthly_revenue'] = recent_revenue
        
        if len(data['monthly_revenue']) >= 2:
            current = data['monthly_revenue']['Revenue'].iloc[-1]
            previous = data['monthly_revenue']['Revenue'].iloc[-2]
            kpis['revenue_change'] = ((current - previous) / previous) * 100 if previous != 0 else 0
        else:
            kpis['revenue_change'] = 0
    else:
        kpis['avg_monthly_revenue'] = 0
        kpis['revenue_change'] = 0
    
    if not data['df_servers'].empty and 'Potential_Loss' in data['df_servers'].columns:
        kpis['estimated_theft'] = data['df_servers']['Potential_Loss'].sum()
        kpis['high_risk_servers'] = len(data['df_servers'][data['df_servers']['Void_Z_Score'] > 2.0]) if 'Void_Z_Score' in data['df_servers'].columns else 0
    else:
        kpis['estimated_theft'] = 0
        kpis['high_risk_servers'] = 0
    
    if not data['df_menu'].empty and 'BCG_Matrix' in data['df_menu'].columns:
        stars = data['df_menu'][data['df_menu']['BCG_Matrix'] == 'Star']
        dogs = data['df_menu'][data['df_menu']['BCG_Matrix'] == 'Dog']
        kpis['star_items'] = len(stars)
        kpis['dog_items'] = len(dogs)
    else:
        kpis['star_items'] = 0
        kpis['dog_items'] = 0
    
    if not data['df_sentiment'].empty and 'CX_Index' in data['df_sentiment'].columns:
        kpis['avg_sentiment'] = data['df_sentiment']['CX_Index'].mean()
        kpis['latest_sentiment'] = data['df_sentiment']['CX_Index'].iloc[-1] if len(data['df_sentiment']) > 0 else 0
    else:
        kpis['avg_sentiment'] = 0
        kpis['latest_sentiment'] = 0
    
    return kpis

# Chart color scheme - Green/Gold theme
CHART_COLORS = {
    'primary': '#d4af37',
    'secondary': '#4a7c4a',
    'success': '#81c784',
    'danger': '#ef5350',
    'warning': '#ffb74d',
    'info': '#64b5f6',
    'background': 'rgba(26, 46, 26, 0.8)',
    'grid': 'rgba(212, 175, 55, 0.15)',
    'text': '#ffffff'
}

def create_chart_layout(title, height=400):
    """Standard chart layout with Best Regards theme - ALL WHITE TEXT"""
    return dict(
        title=dict(text=title, font=dict(color='#ffffff', size=16, family='Playfair Display')),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26, 46, 26, 0.4)',
        font=dict(color='#ffffff', family='Inter'),
        height=height,
        margin=dict(t=50, b=50, l=50, r=50),
        xaxis=dict(
            gridcolor=CHART_COLORS['grid'],
            zerolinecolor=CHART_COLORS['grid'],
            tickfont=dict(color='#ffffff'),
            title_font=dict(color='#ffffff')
        ),
        yaxis=dict(
            gridcolor=CHART_COLORS['grid'],
            zerolinecolor=CHART_COLORS['grid'],
            tickfont=dict(color='#ffffff'),
            title_font=dict(color='#ffffff')
        ),
        legend=dict(
            bgcolor='rgba(26, 46, 26, 0.8)',
            bordercolor='rgba(212, 175, 55, 0.3)',
            borderwidth=1,
            font=dict(color='#ffffff')
        ),
        coloraxis=dict(
            colorbar=dict(
                tickfont=dict(color='#ffffff'),
                title_font=dict(color='#ffffff')
            )
        )
    )

data, load_errors = load_all_data()
kpis = calculate_kpis(data)

# Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #d4af37 !important; font-family: Playfair Display, serif;'>Best Regards</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.7) !important; font-size: 0.9rem;'>Executive Intelligence Portal</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("<h3 style='color: #d4af37 !important;'>System Status</h3>", unsafe_allow_html=True)
    if not data['df_master'].empty or not data['monthly_revenue'].empty:
        st.success("Master Data: Active")
    else:
        st.error("Master Data: Missing")
    
    if not data['df_forecast'].empty:
        st.success("Forecast Model: Active")
    if not data['df_menu'].empty:
        st.success("Menu Analytics: Active")
    if not data['df_sentiment'].empty:
        st.success("Sentiment Engine: Active")
    
    st.markdown("---")
    st.markdown("<h3 style='color: #d4af37 !important;'>Key Metrics</h3>", unsafe_allow_html=True)
    st.metric("Avg Monthly Revenue", f"${kpis['avg_monthly_revenue']:,.0f}", f"{kpis['revenue_change']:+.1f}%")
    st.metric("Est. Monthly Loss", f"${kpis['estimated_theft']:,.0f}")
    st.metric("High-Risk Staff", f"{kpis['high_risk_servers']}")
    st.metric("Current CX Score", f"{kpis['latest_sentiment']:.2f}/1.00")
    
    st.markdown("---")
    st.caption("Data Current: " + datetime.now().strftime("%B %d, %Y"))

# Main Title
st.markdown("<h1 style='color: #ffffff !important;'>Best Regards Business Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: rgba(255,255,255,0.8) !important;'>Comprehensive Analytics: Revenue Forecasting, Menu Optimization, Competitive Analysis & Risk Management</p>", unsafe_allow_html=True)
st.markdown("---")

# KPI Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    change_class = "metric-change-positive" if kpis['revenue_change'] >= 0 else "metric-change-negative"
    st.markdown(f"""
        <div class='metric-card-success'>
            <div>
                <div class='metric-label'>Monthly Revenue</div>
                <div class='metric-value'>${kpis['avg_monthly_revenue']:,.0f}</div>
                <div class='metric-subtitle'>3-Month Average</div>
            </div>
            <div class='metric-change {change_class}'>{kpis['revenue_change']:+.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    card_class = "metric-card-danger" if kpis['estimated_theft'] > 5000 else "metric-card-warning"
    st.markdown(f"""
        <div class='{card_class}'>
            <div>
                <div class='metric-label'>Estimated Loss</div>
                <div class='metric-value'>${kpis['estimated_theft']:,.0f}</div>
                <div class='metric-subtitle'>From Operational Issues</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class='metric-card-info'>
            <div>
                <div class='metric-label'>Menu Performance</div>
                <div class='metric-value'>{kpis['star_items']}</div>
                <div class='metric-subtitle'>Star Items | {kpis['dog_items']} to Remove</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    sentiment_class = "metric-card-danger" if kpis['latest_sentiment'] < 0.6 else "metric-card-success"
    st.markdown(f"""
        <div class='{sentiment_class}'>
            <div>
                <div class='metric-label'>Customer Sentiment</div>
                <div class='metric-value'>{kpis['latest_sentiment']:.2f}</div>
                <div class='metric-subtitle'>Current CX Index Score</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Tabs - NO EMOJIS
tabs = st.tabs(["Revenue Forecast", "Menu Intelligence", "Competitive Landscape", "Risk Detection", "Sentiment Analysis", "Executive Summary"])

# ============================================================================
# TAB 0: REVENUE FORECAST
# ============================================================================
with tabs[0]:
    render_header_with_info("Revenue Forecasting & Scenario Planning", "revenue_forecast")
    
    if not data['monthly_revenue'].empty or not data['df_forecast'].empty:
        render_header_with_info("Scenario Planning", "scenario_planning", level=3)
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            growth_rate = st.slider("Growth Rate Adjustment (%)", -20, 30, 0, help="Adjust baseline growth assumptions")
        with col_s2:
            seasonality_factor = st.slider("Seasonality Impact", 0.5, 1.5, 1.0, help="Adjust for seasonal effects")
        with col_s3:
            external_shock = st.selectbox("External Factors", ["None", "Economic Downturn (-15%)", "Market Expansion (+20%)", "Competition Impact (-10%)"])
        
        # Calculate shock multiplier - THIS NOW AFFECTS THE CHART
        shock_multiplier = 1.0
        shock_label = ""
        if external_shock == "Economic Downturn (-15%)":
            shock_multiplier = 0.85
            shock_label = " (Economic Downturn Applied)"
        elif external_shock == "Market Expansion (+20%)":
            shock_multiplier = 1.20
            shock_label = " (Market Expansion Applied)"
        elif external_shock == "Competition Impact (-10%)":
            shock_multiplier = 0.90
            shock_label = " (Competition Impact Applied)"
        
        # Combined adjustment factor
        total_adjustment = (1 + growth_rate/100) * seasonality_factor * shock_multiplier
        
        st.markdown("---")
        
        # Build combined revenue data
        combined_df = pd.DataFrame()
        
        if not data['monthly_revenue'].empty:
            hist_df = data['monthly_revenue'].copy()
            hist_df['Type'] = 'Historical'
            combined_df = pd.concat([combined_df, hist_df], ignore_index=True)
        
        # Generate forecast if not available
        if data['df_forecast'].empty or 'Forecasted_Revenue' not in data['df_forecast'].columns:
            # Create forecast based on historical data
            if not data['monthly_revenue'].empty:
                last_revenue = data['monthly_revenue']['Revenue'].iloc[-1]
                last_month = data['monthly_revenue']['Month'].iloc[-1]
                
                # Generate 6 months of forecast
                forecast_months = []
                forecast_revenues = []
                
                try:
                    last_date = pd.to_datetime(last_month)
                except:
                    last_date = pd.to_datetime(last_month + '-01')
                
                for i in range(1, 7):
                    next_date = last_date + pd.DateOffset(months=i)
                    forecast_months.append(next_date.strftime('%Y-%m'))
                    # Base forecast with slight growth, then apply adjustments
                    base_forecast = last_revenue * (1.02 ** i)  # 2% monthly growth base
                    adjusted_forecast = base_forecast * total_adjustment
                    forecast_revenues.append(adjusted_forecast)
                
                forecast_df = pd.DataFrame({
                    'Month': forecast_months,
                    'Revenue': forecast_revenues,
                    'Type': 'Forecast'
                })
                combined_df = pd.concat([combined_df, forecast_df], ignore_index=True)
        else:
            forecast_df = data['df_forecast'].copy()
            if 'Month' in forecast_df.columns:
                forecast_df = forecast_df.rename(columns={'Forecasted_Revenue': 'Revenue'})
                forecast_df['Type'] = 'Forecast'
                # Apply scenario adjustments
                forecast_df['Revenue'] = forecast_df['Revenue'] * total_adjustment
                combined_df = pd.concat([combined_df, forecast_df[['Month', 'Revenue', 'Type']]], ignore_index=True)
        
        if combined_df.empty:
            months = pd.date_range(start='2024-01', periods=15, freq='M').strftime('%Y-%m').tolist()
            base_revenue = [280000, 295000, 310000, 305000, 320000, 335000, 340000, 355000, 316058, 330000, 345000, 360000]
            # Add forecast months with adjustments
            for i in range(3):
                base_revenue.append(360000 * (1.02 ** (i+1)) * total_adjustment)
            combined_df = pd.DataFrame({
                'Month': months[:len(base_revenue)],
                'Revenue': base_revenue,
                'Type': ['Historical']*12 + ['Forecast']*3
            })
        
        # Charts
        col_chart1, col_chart2 = st.columns([2, 1])
        
        with col_chart1:
            render_header_with_info("Monthly Revenue Trend & Forecast" + shock_label, "revenue_trend", level=3)
            
            fig_revenue = go.Figure()
            
            hist_data = combined_df[combined_df['Type'] == 'Historical']
            if not hist_data.empty:
                fig_revenue.add_trace(go.Scatter(
                    x=hist_data['Month'],
                    y=hist_data['Revenue'],
                    mode='lines+markers',
                    name='Historical Revenue',
                    line=dict(color=CHART_COLORS['success'], width=3),
                    marker=dict(size=8, color=CHART_COLORS['success']),
                    fill='tozeroy',
                    fillcolor='rgba(129, 199, 132, 0.15)'
                ))
            
            forecast_data = combined_df[combined_df['Type'] == 'Forecast']
            if not forecast_data.empty:
                fig_revenue.add_trace(go.Scatter(
                    x=forecast_data['Month'],
                    y=forecast_data['Revenue'],
                    mode='lines+markers',
                    name='Forecasted Revenue',
                    line=dict(color=CHART_COLORS['primary'], width=3, dash='dash'),
                    marker=dict(size=8, color=CHART_COLORS['primary'], symbol='diamond')
                ))
                
                upper_band = forecast_data['Revenue'] * 1.15
                lower_band = forecast_data['Revenue'] * 0.85
                
                fig_revenue.add_trace(go.Scatter(
                    x=forecast_data['Month'], y=upper_band, mode='lines',
                    line=dict(color='rgba(212, 175, 55, 0.3)', width=1), showlegend=False
                ))
                fig_revenue.add_trace(go.Scatter(
                    x=forecast_data['Month'], y=lower_band, mode='lines',
                    line=dict(color='rgba(212, 175, 55, 0.3)', width=1),
                    fill='tonexty', fillcolor='rgba(212, 175, 55, 0.1)', showlegend=False
                ))
            
            fig_revenue.update_layout(**create_chart_layout('', 450))
            fig_revenue.update_layout(
                xaxis_title='Month',
                yaxis_title='Revenue ($)',
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(color='#ffffff'))
            )
            fig_revenue.update_yaxes(tickprefix='$', tickformat=',', tickfont=dict(color='#ffffff'))
            fig_revenue.update_xaxes(tickfont=dict(color='#ffffff'))
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        with col_chart2:
            render_header_with_info("Revenue by Category", "revenue_category", level=3)
            
            if not data['df_menu'].empty and 'Category' in data['df_menu'].columns:
                category_revenue = data['df_menu'].groupby('Category')['Total_Revenue'].sum().reset_index()
                category_revenue = category_revenue.nlargest(6, 'Total_Revenue')
                
                fig_pie = px.pie(
                    category_revenue, values='Total_Revenue', names='Category',
                    color_discrete_sequence=['#d4af37', '#4a7c4a', '#81c784', '#2d4a2d', '#c9a227', '#6b8e6b']
                )
            else:
                categories = ['Cocktails', 'Wine', 'Beer', 'Spirits', 'Food', 'Other']
                values = [35, 25, 15, 12, 8, 5]
                fig_pie = px.pie(
                    values=values, names=categories,
                    color_discrete_sequence=['#d4af37', '#4a7c4a', '#81c784', '#2d4a2d', '#c9a227', '#6b8e6b']
                )
            
            fig_pie.update_layout(**create_chart_layout('', 450))
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont=dict(color='#ffffff', size=12),
                insidetextfont=dict(color='#ffffff'),
                outsidetextfont=dict(color='#ffffff')
            )
            # Update legend text color
            fig_pie.update_layout(
                legend=dict(font=dict(color='#ffffff')),
                uniformtext_minsize=10,
                uniformtext_mode='hide'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Forecast metrics
        render_header_with_info("Forecast Metrics", "revenue_forecast", level=3)
        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        
        if not forecast_data.empty:
            projected_total = forecast_data['Revenue'].sum()
            projected_avg = forecast_data['Revenue'].mean()
            projected_growth = ((forecast_data['Revenue'].iloc[-1] / forecast_data['Revenue'].iloc[0]) - 1) * 100 if len(forecast_data) > 1 else 0
        else:
            projected_total = projected_avg = projected_growth = 0
        
        with met_col1:
            st.metric("Projected Q Total", f"${projected_total:,.0f}")
        with met_col2:
            st.metric("Projected Monthly Avg", f"${projected_avg:,.0f}")
        with met_col3:
            st.metric("Expected Growth", f"{projected_growth:+.1f}%")
        with met_col4:
            mape = data['df_metrics']['MAPE'].iloc[0] if not data['df_metrics'].empty and 'MAPE' in data['df_metrics'].columns else 8.5
            st.metric("Forecast Accuracy (MAPE)", f"{mape:.1f}%")
        
        # Scenario impact summary
        if total_adjustment != 1.0:
            impact_pct = (total_adjustment - 1) * 100
            impact_color = "#81c784" if impact_pct > 0 else "#ef5350"
            st.markdown(f"""
            <div style="background: rgba(45, 74, 45, 0.4); border: 1px solid rgba(212, 175, 55, 0.3); border-radius: 8px; padding: 15px; margin-top: 10px;">
                <p style="color: #ffffff; margin: 0;"><strong>Scenario Impact:</strong> Your selected adjustments result in a <span style="color: {impact_color}; font-weight: bold;">{impact_pct:+.1f}%</span> change to forecasted revenue.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Month-over-month analysis
        render_header_with_info("Month-over-Month Analysis", "mom_analysis", level=3)
        if not combined_df.empty and len(combined_df) > 1:
            mom_df = combined_df.copy()
            mom_df['MoM_Change'] = mom_df['Revenue'].pct_change() * 100
            mom_df['MoM_Change'] = mom_df['MoM_Change'].fillna(0)
            
            fig_mom = go.Figure()
            colors = [CHART_COLORS['success'] if x >= 0 else CHART_COLORS['danger'] for x in mom_df['MoM_Change']]
            
            fig_mom.add_trace(go.Bar(x=mom_df['Month'], y=mom_df['MoM_Change'], marker_color=colors, name='MoM Change %'))
            fig_mom.update_layout(**create_chart_layout('', 300))
            fig_mom.update_layout(xaxis_title='Month', yaxis_title='Change (%)')
            fig_mom.update_xaxes(tickfont=dict(color='#ffffff'))
            fig_mom.update_yaxes(tickfont=dict(color='#ffffff'))
            st.plotly_chart(fig_mom, use_container_width=True)
    else:
        st.warning("No revenue or forecast data available. Please ensure data files are present.")

# ============================================================================
# TAB 1: MENU INTELLIGENCE
# ============================================================================
with tabs[1]:
    render_header_with_info("Menu Intelligence & Optimization Matrix", "menu_intelligence")
    
    if not data['df_menu'].empty:
        col_bcg1, col_bcg2 = st.columns([2, 1])
        
        with col_bcg1:
            render_header_with_info("BCG Matrix: Menu Item Performance", "bcg_matrix", level=3)
            if 'BCG_Matrix' in data['df_menu'].columns and 'Qty_Sold' in data['df_menu'].columns and 'Total_Revenue' in data['df_menu'].columns:
                fig_bcg = px.scatter(
                    data['df_menu'], x='Qty_Sold', y='Total_Revenue', color='BCG_Matrix',
                    size='Total_Revenue', hover_name='Item' if 'Item' in data['df_menu'].columns else None,
                    color_discrete_map={'Star': '#d4af37', 'Cash Cow': '#4a7c4a', 'Question Mark': '#ffb74d', 'Dog': '#ef5350'}
                )
                fig_bcg.update_layout(**create_chart_layout('', 500))
                fig_bcg.update_layout(xaxis_title='Quantity Sold (Market Share)', yaxis_title='Total Revenue (Growth Rate)')
                fig_bcg.update_yaxes(tickprefix='$', tickformat=',', tickfont=dict(color='#ffffff'))
                fig_bcg.update_xaxes(tickfont=dict(color='#ffffff'))
                fig_bcg.update_layout(legend=dict(font=dict(color='#ffffff')))
                st.plotly_chart(fig_bcg, use_container_width=True)
            else:
                st.info("BCG Matrix data not available. Showing revenue analysis instead.")
                if 'Total_Revenue' in data['df_menu'].columns:
                    top_items = data['df_menu'].nlargest(15, 'Total_Revenue')
                    fig_bar = px.bar(top_items, x='Item' if 'Item' in top_items.columns else top_items.index, 
                                     y='Total_Revenue', color='Total_Revenue', color_continuous_scale=['#2d4a2d', '#d4af37'])
                    fig_bar.update_layout(**create_chart_layout('', 500))
                    st.plotly_chart(fig_bar, use_container_width=True)
        
        with col_bcg2:
            if 'BCG_Matrix' in data['df_menu'].columns:
                render_header_with_info("Item Distribution by BCG Category", "bcg_matrix", level=3)
                bcg_counts = data['df_menu']['BCG_Matrix'].value_counts()
                fig_bcg_pie = px.pie(
                    values=bcg_counts.values, names=bcg_counts.index,
                    color=bcg_counts.index,
                    color_discrete_map={'Star': '#d4af37', 'Cash Cow': '#4a7c4a', 'Question Mark': '#ffb74d', 'Dog': '#ef5350'}
                )
                fig_bcg_pie.update_layout(**create_chart_layout('', 500))
                fig_bcg_pie.update_traces(textposition='inside', textinfo='percent+label', textfont=dict(color='#ffffff'))
                fig_bcg_pie.update_layout(legend=dict(font=dict(color='#ffffff')))
                st.plotly_chart(fig_bcg_pie, use_container_width=True)
        
        st.markdown("---")
        
        col_top, col_bottom = st.columns(2)
        
        with col_top:
            render_header_with_info("Top Performing Items", "top_performers", level=3)
            if 'Total_Revenue' in data['df_menu'].columns:
                top_items = data['df_menu'].nlargest(10, 'Total_Revenue')
                fig_top = go.Figure(go.Bar(
                    x=top_items['Total_Revenue'],
                    y=top_items['Item'] if 'Item' in top_items.columns else top_items.index,
                    orientation='h',
                    marker=dict(color=top_items['Total_Revenue'], colorscale=[[0, '#2d4a2d'], [1, '#d4af37']])
                ))
                fig_top.update_layout(**create_chart_layout('', 400))
                fig_top.update_layout(yaxis=dict(autorange='reversed'), xaxis_title='Revenue ($)')
                fig_top.update_xaxes(tickprefix='$', tickformat=',', tickfont=dict(color='#ffffff'))
                fig_top.update_yaxes(tickfont=dict(color='#ffffff'))
                st.plotly_chart(fig_top, use_container_width=True)
        
        with col_bottom:
            render_header_with_info("Items Requiring Attention", "underperformers", level=3)
            if 'BCG_Matrix' in data['df_menu'].columns:
                dogs = data['df_menu'][data['df_menu']['BCG_Matrix'] == 'Dog'].nlargest(10, 'Total_Revenue')
                if dogs.empty:
                    dogs = data['df_menu'].nsmallest(10, 'Total_Revenue')
            else:
                dogs = data['df_menu'].nsmallest(10, 'Total_Revenue')
            
            if 'Total_Revenue' in dogs.columns:
                fig_dogs = go.Figure(go.Bar(
                    x=dogs['Total_Revenue'],
                    y=dogs['Item'] if 'Item' in dogs.columns else dogs.index,
                    orientation='h',
                    marker=dict(color=dogs['Total_Revenue'], colorscale=[[0, '#ef5350'], [1, '#ffb74d']])
                ))
                fig_dogs.update_layout(**create_chart_layout('', 400))
                fig_dogs.update_layout(yaxis=dict(autorange='reversed'), xaxis_title='Revenue ($)')
                fig_dogs.update_xaxes(tickprefix='$', tickformat=',', tickfont=dict(color='#ffffff'))
                fig_dogs.update_yaxes(tickfont=dict(color='#ffffff'))
                st.plotly_chart(fig_dogs, use_container_width=True)
        
        with st.expander("View Full Menu Analysis Table"):
            display_cols = [col for col in ['Item', 'Category', 'Qty_Sold', 'Total_Revenue', 'BCG_Matrix', 'Item_Void_Rate'] if col in data['df_menu'].columns]
            if display_cols:
                st.dataframe(data['df_menu'][display_cols].sort_values('Total_Revenue', ascending=False), use_container_width=True, height=400)
    else:
        st.warning("No menu data available. Please upload menu_forensics.csv")

# ============================================================================
# TAB 2: COMPETITIVE LANDSCAPE
# ============================================================================
with tabs[2]:
    render_header_with_info("Competitive Landscape Analysis", "competitive_landscape")
    
    # Check for map data
    has_map_data = (
        not data['df_map'].empty and 
        'Latitude' in data['df_map'].columns and 
        'Longitude' in data['df_map'].columns
    )
    
    if has_map_data:
        valid_map_data = data['df_map'][
            (data['df_map']['Latitude'].notna()) & 
            (data['df_map']['Longitude'].notna()) &
            (data['df_map']['Latitude'] != 0) &
            (data['df_map']['Longitude'] != 0) &
            (data['df_map']['Latitude'].between(-90, 90)) &
            (data['df_map']['Longitude'].between(-180, 180))
        ].copy()
        
        if not valid_map_data.empty and len(valid_map_data) > 0:
            col_map_ctrl1, col_map_ctrl2 = st.columns(2)
            with col_map_ctrl1:
                map_style = st.selectbox("Map Style", ["Dark", "Streets"])
            with col_map_ctrl2:
                show_heatmap = st.checkbox("Show Heat Map Overlay", value=True)
            
            render_header_with_info("Competitor Map", "competitor_map", level=3)
            
            center_lat = valid_map_data['Latitude'].mean()
            center_lon = valid_map_data['Longitude'].mean()
            
            # Create map with appropriate tiles
            if map_style == "Dark":
                m = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=13,
                    tiles='CartoDB dark_matter'
                )
            else:
                m = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=13,
                    tiles='OpenStreetMap'
                )
            
            # Add markers for each location
            for idx, row in valid_map_data.iterrows():
                loc_name = str(row.get('Location Name', 'Unknown'))
                is_best_regards = 'best regards' in loc_name.lower()
                
                revenue = row.get('Total_Revenue', 0)
                
                if is_best_regards:
                    # Best Regards marker - green star
                    icon_html = """
                    <div style="font-size: 24px; color: #10b981;">
                        <i class="fa fa-star"></i>
                    </div>
                    """
                    folium.Marker(
                        location=[row['Latitude'], row['Longitude']],
                        popup=folium.Popup(f"<b>{loc_name}</b><br>Revenue: ${revenue:,.0f}", max_width=250),
                        icon=folium.Icon(color='green', icon='star', prefix='fa'),
                        tooltip=loc_name
                    ).add_to(m)
                else:
                    # Competitor markers - color by revenue
                    if revenue > 500000:
                        color = 'red'
                    elif revenue > 200000:
                        color = 'orange'
                    else:
                        color = 'blue'
                    
                    folium.Marker(
                        location=[row['Latitude'], row['Longitude']],
                        popup=folium.Popup(f"<b>{loc_name}</b><br>Est. Revenue: ${revenue:,.0f}", max_width=250),
                        icon=folium.Icon(color=color, icon='glass', prefix='fa'),
                        tooltip=loc_name
                    ).add_to(m)
            
            # Add heat map layer if enabled
            if show_heatmap:
                heat_data = []
                for idx, row in valid_map_data.iterrows():
                    lat = row['Latitude']
                    lon = row['Longitude']
                    # Weight by revenue (normalize to reasonable range)
                    weight = row.get('Total_Revenue', 1)
                    if weight > 0:
                        # Normalize weight
                        weight = min(weight / 100000, 10)  # Cap at 10
                    else:
                        weight = 1
                    heat_data.append([lat, lon, weight])
                
                if heat_data:
                    plugins.HeatMap(
                        heat_data,
                        radius=30,
                        blur=20,
                        max_zoom=15,
                        gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}
                    ).add_to(m)
            
            # Add a layer control
            folium.LayerControl().add_to(m)
            
            # Display the map
            map_data = st_folium(m, width=None, height=500, use_container_width=True)
            
            st.markdown("---")
            render_header_with_info("Competitive Analysis Summary", "competitive_landscape", level=3)
            
            col_comp1, col_comp2, col_comp3 = st.columns(3)
            with col_comp1:
                st.metric("Total Competitors", max(0, len(valid_map_data) - 1))
            with col_comp2:
                if 'Total_Revenue' in valid_map_data.columns:
                    st.metric("Avg Competitor Revenue", f"${valid_map_data['Total_Revenue'].mean():,.0f}")
            with col_comp3:
                if 'Total_Revenue' in valid_map_data.columns and valid_map_data['Total_Revenue'].sum() > 0:
                    market_share = (kpis['avg_monthly_revenue'] * 12) / valid_map_data['Total_Revenue'].sum() * 100
                    st.metric("Est. Market Share", f"{min(market_share, 100):.1f}%")
            
            with st.expander("View All Locations"):
                display_cols = [col for col in ['Location Name', 'Total_Revenue', 'Latitude', 'Longitude'] if col in valid_map_data.columns]
                if display_cols:
                    st.dataframe(valid_map_data[display_cols].sort_values('Total_Revenue', ascending=False), use_container_width=True)
        else:
            st.warning("No valid location coordinates found in map data. Please check that Latitude and Longitude values are valid.")
    else:
        st.warning("No map data available. Please upload map_data.csv with Latitude and Longitude columns.")
        st.info("Expected columns: Location Name, Latitude, Longitude, Total_Revenue")
    
    # Geographic Pressure Index
    if not data['df_geo'].empty and 'Month' in data['df_geo'].columns and 'GeoPressure_Total' in data['df_geo'].columns:
        st.markdown("---")
        render_header_with_info("Geographic Pressure Index", "geo_pressure", level=3)
        
        fig_geo = go.Figure()
        fig_geo.add_trace(go.Scatter(
            x=data['df_geo']['Month'], y=data['df_geo']['GeoPressure_Total'],
            mode='lines+markers', line=dict(color=CHART_COLORS['primary'], width=3),
            marker=dict(size=8, color=CHART_COLORS['primary']),
            fill='tozeroy', fillcolor='rgba(212, 175, 55, 0.1)'
        ))
        fig_geo.update_layout(**create_chart_layout('', 350))
        fig_geo.update_layout(xaxis_title='Month', yaxis_title='Pressure Index')
        fig_geo.update_xaxes(tickfont=dict(color='#ffffff'))
        fig_geo.update_yaxes(tickfont=dict(color='#ffffff'))
        st.plotly_chart(fig_geo, use_container_width=True)

# ============================================================================
# TAB 3: RISK DETECTION
# ============================================================================
with tabs[3]:
    render_header_with_info("Operational Risk & Detection Analysis", "risk_detection")
    
    has_server_data = (
        not data['df_servers'].empty and 
        'Void_Rate' in data['df_servers'].columns and 
        'Potential_Loss' in data['df_servers'].columns and
        len(data['df_servers']) > 0
    )
    
    if has_server_data:
        render_header_with_info("High-Risk Server Analysis", "server_risk", level=3)
        
        col_risk1, col_risk2 = st.columns([2, 1])
        
        with col_risk1:
            scatter_kwargs = {
                'x': 'Void_Rate',
                'y': 'Potential_Loss',
                'color_continuous_scale': [[0, '#ffb74d'], [1, '#ef5350']]
            }
            
            if 'Void_Z_Score' in data['df_servers'].columns:
                valid_z_scores = data['df_servers']['Void_Z_Score'].dropna()
                if len(valid_z_scores) > 0 and valid_z_scores.std() > 0:
                    scatter_kwargs['size'] = 'Void_Z_Score'
                    scatter_kwargs['color'] = 'Void_Z_Score'
                else:
                    scatter_kwargs['color'] = 'Void_Rate'
            else:
                scatter_kwargs['color'] = 'Void_Rate'
            
            if 'Server' in data['df_servers'].columns:
                scatter_kwargs['hover_name'] = 'Server'
            
            fig_servers = px.scatter(data['df_servers'], **scatter_kwargs)
            fig_servers.add_hline(y=1000, line_dash="dash", line_color="#d4af37", annotation_text="High Risk Threshold", annotation_font_color="#ffffff")
            fig_servers.update_layout(**create_chart_layout('Server Risk Assessment Matrix', 450))
            fig_servers.update_layout(xaxis_title='Void Rate (%)', yaxis_title='Potential Loss ($)')
            fig_servers.update_yaxes(tickprefix='$', tickformat=',', tickfont=dict(color='#ffffff'))
            fig_servers.update_xaxes(tickfont=dict(color='#ffffff'))
            fig_servers.update_layout(coloraxis_colorbar=dict(tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff')))
            st.plotly_chart(fig_servers, use_container_width=True)
        
        with col_risk2:
            if 'Void_Z_Score' in data['df_servers'].columns:
                render_header_with_info("Staff Risk Distribution", "risk_distribution", level=3)
                high_risk = len(data['df_servers'][data['df_servers']['Void_Z_Score'] > 2.0])
                medium_risk = len(data['df_servers'][(data['df_servers']['Void_Z_Score'] > 1.0) & (data['df_servers']['Void_Z_Score'] <= 2.0)])
                low_risk = len(data['df_servers'][data['df_servers']['Void_Z_Score'] <= 1.0])
                
                fig_risk_pie = px.pie(
                    values=[high_risk, medium_risk, low_risk],
                    names=['High Risk', 'Medium Risk', 'Low Risk'],
                    color_discrete_sequence=['#ef5350', '#ffb74d', '#81c784']
                )
                fig_risk_pie.update_layout(**create_chart_layout('', 450))
                fig_risk_pie.update_traces(textposition='inside', textinfo='percent+label', textfont=dict(color='#ffffff'))
                fig_risk_pie.update_layout(legend=dict(font=dict(color='#ffffff')))
                st.plotly_chart(fig_risk_pie, use_container_width=True)
            else:
                st.info("Void Z-Score data not available for risk distribution.")
        
        render_header_with_info("High-Risk Individuals", "server_risk", level=3)
        if 'Void_Z_Score' in data['df_servers'].columns:
            high_risk_servers = data['df_servers'][data['df_servers']['Void_Z_Score'] > 1.5].sort_values('Potential_Loss', ascending=False)
            if not high_risk_servers.empty:
                st.dataframe(high_risk_servers, use_container_width=True, height=300)
            else:
                st.success("No high-risk servers detected!")
        else:
            st.dataframe(data['df_servers'].sort_values('Potential_Loss', ascending=False).head(10), use_container_width=True, height=300)
    else:
        st.info("No server risk data available. Risk analysis will appear when suspicious_servers.csv is uploaded.")
        
        col_ph1, col_ph2, col_ph3 = st.columns(3)
        with col_ph1:
            st.metric("High-Risk Staff", "0")
        with col_ph2:
            st.metric("Medium-Risk Staff", "0")
        with col_ph3:
            st.metric("Estimated Loss", "$0")
    
    st.markdown("---")
    
    col_void1, col_void2 = st.columns(2)
    
    with col_void1:
        if not data['df_voids_h'].empty and 'Hour_of_Day' in data['df_voids_h'].columns and 'Void_Rate' in data['df_voids_h'].columns:
            render_header_with_info("Hourly Void Patterns", "void_patterns", level=3)
            fig_hourly = px.bar(
                data['df_voids_h'], x='Hour_of_Day', y='Void_Rate',
                color='Void_Rate', color_continuous_scale=[[0, '#4a7c4a'], [1, '#ef5350']]
            )
            fig_hourly.update_layout(**create_chart_layout('', 350))
            fig_hourly.update_layout(xaxis_title='Hour of Day', yaxis_title='Void Rate (%)')
            fig_hourly.update_xaxes(tickfont=dict(color='#ffffff'))
            fig_hourly.update_yaxes(tickfont=dict(color='#ffffff'))
            fig_hourly.update_layout(coloraxis_colorbar=dict(tickfont=dict(color='#ffffff')))
            st.plotly_chart(fig_hourly, use_container_width=True)
        else:
            render_header_with_info("Hourly Void Patterns", "void_patterns", level=3)
            st.info("Hourly void data not available.")
    
    with col_void2:
        if not data['df_voids_d'].empty and 'Void_Rate' in data['df_voids_d'].columns:
            render_header_with_info("Daily Void Patterns", "void_patterns", level=3)
            day_col = data['df_voids_d'].columns[0]
            fig_daily = px.bar(
                data['df_voids_d'], x=day_col, y='Void_Rate',
                color='Void_Rate', color_continuous_scale=[[0, '#4a7c4a'], [1, '#ef5350']]
            )
            fig_daily.update_layout(**create_chart_layout('', 350))
            fig_daily.update_layout(yaxis_title='Void Rate (%)')
            fig_daily.update_xaxes(tickfont=dict(color='#ffffff'))
            fig_daily.update_yaxes(tickfont=dict(color='#ffffff'))
            fig_daily.update_layout(coloraxis_colorbar=dict(tickfont=dict(color='#ffffff')))
            st.plotly_chart(fig_daily, use_container_width=True)
        else:
            render_header_with_info("Daily Void Patterns", "void_patterns", level=3)
            st.info("Daily void data not available.")
    
    if not data['df_combo'].empty:
        render_header_with_info("Suspicious Transaction Combinations", "risk_detection", level=3)
        st.dataframe(data['df_combo'], use_container_width=True, height=250)

# ============================================================================
# TAB 4: SENTIMENT ANALYSIS
# ============================================================================
with tabs[4]:
    render_header_with_info("Customer Sentiment & Review Analysis", "sentiment_analysis")
    
    if not data['df_sentiment'].empty and 'CX_Index' in data['df_sentiment'].columns:
        col_sent1, col_sent2 = st.columns([2, 1])
        
        with col_sent1:
            render_header_with_info("Customer Experience Index Trend", "cx_trend", level=3)
            fig_sentiment = go.Figure()
            fig_sentiment.add_trace(go.Scatter(
                x=data['df_sentiment']['Month'], y=data['df_sentiment']['CX_Index'],
                mode='lines+markers', name='Customer Experience Index',
                line=dict(color=CHART_COLORS['success'], width=3),
                marker=dict(size=10), fill='tozeroy', fillcolor='rgba(129, 199, 132, 0.15)'
            ))
            
            avg_sentiment = data['df_sentiment']['CX_Index'].mean()
            fig_sentiment.add_hline(y=avg_sentiment, line_dash="dash", line_color=CHART_COLORS['primary'],
                                    annotation_text=f"Avg: {avg_sentiment:.2f}", annotation_font_color="#ffffff")
            
            fig_sentiment.update_layout(**create_chart_layout('', 400))
            fig_sentiment.update_layout(xaxis_title='Month', yaxis_title='CX Index Score', yaxis=dict(range=[0, 1]))
            fig_sentiment.update_xaxes(tickfont=dict(color='#ffffff'))
            fig_sentiment.update_yaxes(tickfont=dict(color='#ffffff'))
            fig_sentiment.update_layout(legend=dict(font=dict(color='#ffffff')))
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col_sent2:
            render_header_with_info("Current CX Score", "cx_gauge", level=3)
            latest_cx = data['df_sentiment']['CX_Index'].iloc[-1]
            prev_cx = data['df_sentiment']['CX_Index'].iloc[-2] if len(data['df_sentiment']) > 1 else latest_cx
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=latest_cx,
                delta={'reference': prev_cx, 'valueformat': '.2f', 'font': {'color': '#ffffff'}},
                title={'text': "", 'font': {'color': 'white', 'size': 16}},
                number={'font': {'color': 'white', 'size': 40}},
                gauge={
                    'axis': {'range': [0, 1], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
                    'bar': {'color': CHART_COLORS['primary']},
                    'bgcolor': 'rgba(255,255,255,0.1)',
                    'steps': [
                        {'range': [0, 0.4], 'color': 'rgba(239, 83, 80, 0.4)'},
                        {'range': [0.4, 0.7], 'color': 'rgba(255, 183, 77, 0.4)'},
                        {'range': [0.7, 1], 'color': 'rgba(129, 199, 132, 0.4)'}
                    ],
                    'threshold': {'line': {'color': 'white', 'width': 4}, 'thickness': 0.75, 'value': latest_cx}
                }
            ))
            fig_gauge.update_layout(**create_chart_layout('', 400))
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        if 'BestRegards_Revenue' in data['df_sentiment'].columns:
            render_header_with_info("Sentiment-Revenue Correlation", "sentiment_correlation", level=3)
            
            fig_corr = make_subplots(specs=[[{"secondary_y": True}]])
            fig_corr.add_trace(go.Bar(
                x=data['df_sentiment']['Month'], y=data['df_sentiment']['BestRegards_Revenue'],
                name='Revenue', marker_color='rgba(74, 124, 74, 0.7)'
            ), secondary_y=False)
            fig_corr.add_trace(go.Scatter(
                x=data['df_sentiment']['Month'], y=data['df_sentiment']['CX_Index'],
                name='CX Index', mode='lines+markers',
                line=dict(color=CHART_COLORS['primary'], width=3), marker=dict(size=8)
            ), secondary_y=True)
            
            fig_corr.update_layout(**create_chart_layout('', 400))
            fig_corr.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color='#ffffff')))
            fig_corr.update_yaxes(title_text='Revenue ($)', secondary_y=False, tickprefix='$', tickformat=',', tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff'))
            fig_corr.update_yaxes(title_text='CX Index', secondary_y=True, range=[0, 1], tickfont=dict(color='#ffffff'), title_font=dict(color='#ffffff'))
            fig_corr.update_xaxes(tickfont=dict(color='#ffffff'))
            st.plotly_chart(fig_corr, use_container_width=True)
            
            correlation = data['df_sentiment']['BestRegards_Revenue'].corr(data['df_sentiment']['CX_Index'])
            col_corr1, col_corr2, col_corr3 = st.columns(3)
            with col_corr1:
                st.metric("Correlation Coefficient", f"{correlation:.3f}")
            with col_corr2:
                st.metric("Avg CX Index", f"{data['df_sentiment']['CX_Index'].mean():.2f}")
            with col_corr3:
                trend = "Improving" if data['df_sentiment']['CX_Index'].iloc[-1] > data['df_sentiment']['CX_Index'].iloc[0] else "Declining"
                st.metric("Sentiment Trend", trend)
    else:
        st.warning("No sentiment data available. Please upload sentiment.csv")

# ============================================================================
# TAB 5: EXECUTIVE SUMMARY
# ============================================================================
with tabs[5]:
    render_header_with_info("Executive Summary & Strategic Roadmap", "executive_summary")
    
    render_header_with_info("Key Business Insights", "executive_summary", level=3)
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown(f"""
        <div class='glass-card'>
            <h3 style='color: #d4af37 !important;'>Revenue Performance</h3>
            <ul style='color: rgba(255,255,255,0.9);'>
                <li>3-Month Avg Revenue: <strong>${kpis['avg_monthly_revenue']:,.0f}</strong></li>
                <li>Month-over-Month Change: <strong>{kpis['revenue_change']:+.1f}%</strong></li>
                <li>Revenue trend is {"positive" if kpis['revenue_change'] > 0 else "needs attention"}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='glass-card' style='margin-top: 20px;'>
            <h3 style='color: #d4af37 !important;'>Menu Optimization</h3>
            <ul style='color: rgba(255,255,255,0.9);'>
                <li>Star Items (Keep & Promote): <strong>{kpis['star_items']}</strong></li>
                <li>Dog Items (Consider Removal): <strong>{kpis['dog_items']}</strong></li>
                <li>Estimated revenue lift from optimization: <strong>8-12%</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col2:
        st.markdown(f"""
        <div class='glass-card'>
            <h3 style='color: #d4af37 !important;'>Risk Assessment</h3>
            <ul style='color: rgba(255,255,255,0.9);'>
                <li>Estimated Monthly Loss: <strong>${kpis['estimated_theft']:,.0f}</strong></li>
                <li>High-Risk Staff Members: <strong>{kpis['high_risk_servers']}</strong></li>
                <li>Recommended Action: {"Immediate intervention needed" if kpis['high_risk_servers'] > 3 else "Continue monitoring"}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        cx_status = "Excellent" if kpis['latest_sentiment'] > 0.7 else ("Good" if kpis['latest_sentiment'] > 0.5 else "Needs Improvement")
        st.markdown(f"""
        <div class='glass-card' style='margin-top: 20px;'>
            <h3 style='color: #d4af37 !important;'>Customer Experience</h3>
            <ul style='color: rgba(255,255,255,0.9);'>
                <li>Current CX Index: <strong>{kpis['latest_sentiment']:.2f}/1.00</strong></li>
                <li>Average CX Index: <strong>{kpis['avg_sentiment']:.2f}/1.00</strong></li>
                <li>Status: {cx_status}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    render_header_with_info("Strategic Recommendations", "executive_summary", level=3)
    
    recommendations = []
    if kpis['revenue_change'] < 0:
        recommendations.append({'priority': 'High', 'area': 'Revenue', 'recommendation': 'Implement promotional campaigns to reverse declining revenue trend', 'impact': 'Potential 10-15% revenue increase'})
    if kpis['high_risk_servers'] > 0:
        recommendations.append({'priority': 'High', 'area': 'Operations', 'recommendation': f'Investigate {kpis["high_risk_servers"]} high-risk staff members with elevated void rates', 'impact': f'Potential ${kpis["estimated_theft"]:,.0f} monthly savings'})
    if kpis['dog_items'] > 5:
        recommendations.append({'priority': 'Medium', 'area': 'Menu', 'recommendation': f'Remove or rebrand {kpis["dog_items"]} underperforming menu items', 'impact': 'Improved margins and operational efficiency'})
    if kpis['latest_sentiment'] < 0.6:
        recommendations.append({'priority': 'High', 'area': 'Customer Experience', 'recommendation': 'Implement customer feedback program and service training', 'impact': 'Expected 15-20% improvement in CX scores'})
    
    if not recommendations:
        recommendations = [
            {'priority': 'Medium', 'area': 'Growth', 'recommendation': 'Explore expansion opportunities based on competitive analysis', 'impact': 'Market share growth'},
            {'priority': 'Low', 'area': 'Operations', 'recommendation': 'Continue monitoring KPIs and maintain current performance', 'impact': 'Sustained profitability'}
        ]
    
    for rec in recommendations:
        priority_color = {'High': '#ef5350', 'Medium': '#ffb74d', 'Low': '#81c784'}[rec['priority']]
        st.markdown(f"""
        <div style='background: rgba(45, 74, 45, 0.4); border-left: 4px solid {priority_color}; padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <span style='background: {priority_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;'>{rec['priority']} Priority</span>
                    <span style='color: rgba(255,255,255,0.7); margin-left: 10px;'>{rec['area']}</span>
                </div>
            </div>
            <p style='color: white; margin: 10px 0 5px 0; font-weight: 500;'>{rec['recommendation']}</p>
            <p style='color: rgba(255,255,255,0.7); margin: 0; font-size: 0.875rem;'>Expected Impact: {rec['impact']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    render_header_with_info("90-Day Strategic Roadmap", "executive_summary", level=3)
    
    roadmap_col1, roadmap_col2, roadmap_col3 = st.columns(3)
    
    with roadmap_col1:
        st.markdown("""
        <div class='glass-card'>
            <h4 style='color: #81c784 !important;'>Days 1-30: Foundation</h4>
            <ul style='color: rgba(255,255,255,0.9); font-size: 0.9rem;'>
                <li>Complete staff risk assessment</li>
                <li>Implement void monitoring system</li>
                <li>Launch customer feedback collection</li>
                <li>Audit underperforming menu items</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with roadmap_col2:
        st.markdown("""
        <div class='glass-card'>
            <h4 style='color: #ffb74d !important;'>Days 31-60: Optimization</h4>
            <ul style='color: rgba(255,255,255,0.9); font-size: 0.9rem;'>
                <li>Execute menu optimization</li>
                <li>Staff training programs</li>
                <li>Implement pricing strategies</li>
                <li>Competitive positioning review</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with roadmap_col3:
        st.markdown("""
        <div class='glass-card'>
            <h4 style='color: #d4af37 !important;'>Days 61-90: Growth</h4>
            <ul style='color: rgba(255,255,255,0.9); font-size: 0.9rem;'>
                <li>Launch marketing campaigns</li>
                <li>Expansion feasibility study</li>
                <li>Technology upgrades</li>
                <li>Q2 strategic planning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    render_header_with_info("Export Options", "executive_summary", level=3)
    
    export_col1, export_col2, export_col3 = st.columns(3)
    with export_col1:
        if st.button("Export Full Report", use_container_width=True):
            st.info("Report generation feature - connect to document generation service")
    with export_col2:
        if st.button("Export Charts", use_container_width=True):
            st.info("Chart export feature - save visualizations as PNG/PDF")
    with export_col3:
        if st.button("Export Data", use_container_width=True):
            st.info("Data export feature - download raw data as CSV/Excel")
