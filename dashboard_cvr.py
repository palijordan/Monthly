import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# Configuración inicial de la página
st.set_page_config(
    page_title="CVR Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def inject_css():
    st.markdown("""
    <style>
    .stApp { background-color: white !important; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { color: black !important; }
    .stApp p, .stApp div, .stApp span, .stApp label { color: black !important; }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .metric-title { color: #000000 !important; font-size: 1rem; margin-bottom: 0.5rem; font-weight: 600; }
    .metric-value { color: #000000 !important; font-size: 2.5rem; font-weight: 700; margin: 0; }
    .metric-delta { font-size: 0.9rem; margin-top: 0.5rem; font-weight: 600; }
    .metric-delta.positive { color: #28a745; }
    .metric-delta.negative { color: #dc3545; }
    .platform-table {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .platform-table table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
    }
    .platform-table th,
    .platform-table td {
        padding: 12px;
        text-align: left;
        border-bottom: 1px solid #ddd;
        color: black !important;
    }
    .platform-table th {
        background-color: #f8f9fa;
        font-weight: 600;
    }
    .platform-table tr:hover { background-color: #f5f5f5; }
    .positive-change { color: #28a745; font-weight: 600; }
    .negative-change { color: #dc3545; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Archivo {file_path} no encontrado. Por favor, coloca tu archivo CSV en el directorio.")
        st.stop()
    
    # Convert date column
    df['ds'] = pd.to_datetime(df['ds'])
    df['month'] = df['ds'].dt.month
    df['month_name'] = df['ds'].dt.strftime('%B')
    
    return df

def create_metric_card(title, value, delta=None, delta_type=None):
    delta_class = ""
    if delta_type == "positive":
        delta_class = "positive"
    elif delta_type == "negative":
        delta_class = "negative"
    
    delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>' if delta else ''
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

def calculate_conversion_rate(df, step_from, step_to):
    """Calculate conversion rate between two funnel steps"""
    total_from = df[step_from].sum()
    total_to = df[step_to].sum()
    return (total_to / total_from * 100) if total_from > 0 else 0

# Paleta de colores suaves y elegantes
color_palette = [
    "#8884d9",  # violeta
    "#a3a1fb",  # violeta claro
    "#6fc2d0",  # celeste
    "#81c99d",  # verde
    "#b6e3c6",  # verde claro
    "#e2e2f6",  # gris lavanda
    "#b5b5e2",  # lavanda medio
    "#c3e6e8",  # celeste muy claro
]

# Aplicar CSS
inject_css()

# Cargar datos
try:
    df = load_data("bquxjob_145dbef4_1977e15505c.csv")
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

if df.empty:
    st.error("No hay datos disponibles para mostrar")
    st.stop()

# Site selector at the top
st.title("CVR Analysis Dashboard")
sites = ["General"] + sorted(df['site'].unique().tolist())
selected_site = st.selectbox("Seleccionar Site", sites, key="site_selector")

# Filter data by site if not General
if selected_site == "General":
    filtered_df = df.copy()
    site_suffix = ""
else:
    filtered_df = df[df['site'] == selected_site].copy()
    site_suffix = f" - {selected_site}"

# Get data for April and May
df_abril = filtered_df[filtered_df['month'] == 4]
df_mayo = filtered_df[filtered_df['month'] == 5]

if df_abril.empty or df_mayo.empty:
    st.error("No hay datos suficientes para abril y mayo")
    st.stop()

# Calculate main metrics
cvr_abril = calculate_conversion_rate(df_abril, 'payments_sessions', 'congrats_sessions')
cvr_mayo = calculate_conversion_rate(df_mayo, 'payments_sessions', 'congrats_sessions')
cvr_change = cvr_mayo - cvr_abril

trafico_abril = df_abril['payments_sessions'].sum()
trafico_mayo = df_mayo['payments_sessions'].sum()
trafico_change = ((trafico_mayo - trafico_abril) / trafico_abril * 100) if trafico_abril > 0 else 0

# Calculate sites with decline (only for General view)
sites_con_caida = 0
total_sites = 0
if selected_site == "General":
    for site in df['site'].unique():
        site_abril = df[(df['month'] == 4) & (df['site'] == site)]
        site_mayo = df[(df['month'] == 5) & (df['site'] == site)]
        
        if not site_abril.empty and not site_mayo.empty:
            cvr_site_abril = calculate_conversion_rate(site_abril, 'payments_sessions', 'congrats_sessions')
            cvr_site_mayo = calculate_conversion_rate(site_mayo, 'payments_sessions', 'congrats_sessions')
            
            if cvr_site_mayo < cvr_site_abril:
                sites_con_caida += 1
            total_sites += 1

# Top metrics row
st.markdown("### Métricas Principales" + site_suffix)
col1, col2, col3 = st.columns(3)

with col1:
    delta_type = "negative" if cvr_change < 0 else "positive"
    delta_text = f"{cvr_change:+.2f}pp vs Abril"
    st.markdown(create_metric_card(
        "CVR General",
        f"{cvr_mayo:.2f}%",
        delta_text,
        delta_type
    ), unsafe_allow_html=True)

with col2:
    delta_type = "positive" if trafico_change > 0 else "negative"
    delta_text = f"{trafico_change:+.2f}% vs Abril"
    st.markdown(create_metric_card(
        "Tráfico Total",
        f"{trafico_mayo:,.0f}",
        delta_text,
        delta_type
    ), unsafe_allow_html=True)

with col3:
    if selected_site == "General":
        st.markdown(create_metric_card(
            "Sites con Caída",
            f"{sites_con_caida}",
            f"de {total_sites} sites totales"
        ), unsafe_allow_html=True)
    else:
        # Show site-specific metric
        congrats_mayo = df_mayo['congrats_sessions'].sum()
        st.markdown(create_metric_card(
            "Conversiones Mayo",
            f"{congrats_mayo:,.0f}",
            f"Total conversiones"
        ), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Platform Analysis Table
st.markdown("### Análisis por Plataforma")
st.markdown("Comparación de conversión y tráfico por plataforma")

platform_data = []
total_trafico_mayo = df_mayo['payments_sessions'].sum()

for platform in filtered_df['plataforma'].unique():
    # April data
    platform_abril = df_abril[df_abril['plataforma'] == platform]
    platform_mayo = df_mayo[df_mayo['plataforma'] == platform]
    
    if not platform_abril.empty and not platform_mayo.empty:
        # CVR calculation
        cvr_abril_plat = calculate_conversion_rate(platform_abril, 'payments_sessions', 'congrats_sessions')
        cvr_mayo_plat = calculate_conversion_rate(platform_mayo, 'payments_sessions', 'congrats_sessions')
        cvr_change_plat = cvr_mayo_plat - cvr_abril_plat
        
        # Traffic calculation
        trafico_abril_plat = platform_abril['payments_sessions'].sum()
        trafico_mayo_plat = platform_mayo['payments_sessions'].sum()
        trafico_change_plat = ((trafico_mayo_plat - trafico_abril_plat) / trafico_abril_plat * 100) if trafico_abril_plat > 0 else 0
        
        # Share calculation
        share = (trafico_mayo_plat / total_trafico_mayo * 100) if total_trafico_mayo > 0 else 0
        
        platform_data.append({
            'Plataforma': platform,
            'CVR Mayo': f"{cvr_mayo_plat:.2f}%",
            'Cambio CVR': f"{cvr_change_plat:+.2f}pp",
            'Tráfico Mayo': f"{trafico_mayo_plat:,.0f}",
            'Cambio Tráfico': f"{trafico_change_plat:+.2f}%",
            'Share': f"{share:.1f}%",
            'cvr_change_num': cvr_change_plat,
            'trafico_change_num': trafico_change_plat
        })

if platform_data:
    platform_df = pd.DataFrame(platform_data)
    
    # Sort by share descending
    platform_df['share_num'] = platform_df['Share'].str.replace('%', '').astype(float)
    platform_df = platform_df.sort_values('share_num', ascending=False)
    
    # Create HTML table with conditional formatting
    html_rows = []
    for _, row in platform_df.iterrows():
        cvr_class = "negative-change" if row['cvr_change_num'] < 0 else "positive-change"
        trafico_class = "negative-change" if row['trafico_change_num'] < 0 else "positive-change"
        
        html_row = f"""
        <tr>
            <td>{row['Plataforma']}</td>
            <td>{row['CVR Mayo']}</td>
            <td class="{cvr_class}">{row['Cambio CVR']}</td>
            <td>{row['Tráfico Mayo']}</td>
            <td class="{trafico_class}">{row['Cambio Tráfico']}</td>
            <td>{row['Share']}</td>
        </tr>
        """
        html_rows.append(html_row)
    
    table_html = f"""
    <div class="platform-table">
        <table>
            <thead>
                <tr>
                    <th>Plataforma</th>
                    <th>CVR Mayo</th>
                    <th>Cambio CVR</th>
                    <th>Tráfico Mayo</th>
                    <th>Cambio Tráfico</th>
                    <th>Share</th>
                </tr>
            </thead>
            <tbody>
                {''.join(html_rows)}
            </tbody>
        </table>
    </div>
    """
    
    st.markdown(table_html, unsafe_allow_html=True)

# Daily trend chart
st.markdown("### Tendencia Diaria de CVR")

daily_data = []
for date in sorted(filtered_df['ds'].dt.date.unique()):
    date_df = filtered_df[filtered_df['ds'].dt.date == date]
    cvr = calculate_conversion_rate(date_df, 'payments_sessions', 'congrats_sessions')
    month_name = date_df['month_name'].iloc[0] if not date_df.empty else ""
    
    daily_data.append({
        'Fecha': date,
        'CVR': cvr,
        'Mes': month_name
    })

daily_df = pd.DataFrame(daily_data)

if not daily_df.empty:
    fig_daily = px.line(
        daily_df,
        x='Fecha',
        y='CVR',
        color='Mes',
        markers=True,
        color_discrete_sequence=['#8884d9', '#81c99d']
    )
    
    fig_daily.update_layout(
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        xaxis=dict(
            title="Fecha",
            tickfont=dict(color='black'),
            title_font=dict(color='black')
        ),
        yaxis=dict(
            title="CVR (%)",
            tickfont=dict(color='black'),
            title_font=dict(color='black'),
            gridcolor='lightgray'
        ),
        legend=dict(font=dict(color='black'))
    )
    
    st.plotly_chart(fig_daily, use_container_width=True)

# Funnel comparison
st.markdown("### Comparación de Embudo Abril vs Mayo")

funnel_data = []
steps = ['shipping_sessions', 'review_sessions', 'payments_sessions', 'congrats_sessions']
step_names = ['Shipping', 'Review', 'Payments', 'Congrats']

for month_df, month_name in [(df_abril, 'Abril'), (df_mayo, 'Mayo')]:
    for step, step_name in zip(steps, step_names):
        total = month_df[step].sum()
        funnel_data.append({
            'Mes': month_name,
            'Etapa': step_name,
            'Sesiones': total
        })

funnel_df = pd.DataFrame(funnel_data)

fig_funnel = px.bar(
    funnel_df,
    x='Etapa',
    y='Sesiones',
    color='Mes',
    barmode='group',
    color_discrete_sequence=['#8884d9', '#81c99d'],
    text='Sesiones'
)

fig_funnel.update_layout(
    height=400,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black'),
    legend=dict(font=dict(color='black')),
    xaxis=dict(
        title="Etapas del Embudo",
        tickfont=dict(color='black'),
        title_font=dict(color='black')
    ),
    yaxis=dict(
        title="Sesiones",
        tickfont=dict(color='black'),
        title_font=dict(color='black'),
        gridcolor='lightgray'
    )
)

fig_funnel.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
st.plotly_chart(fig_funnel, use_container_width=True)
