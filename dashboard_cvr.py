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

def calculate_status(change):
    if change > 0.01:
        return "Crece"
    elif change < -0.01:
        return "Caída"
    else:
        return "Estable"

# Aplicar CSS
inject_css()

# Cargar datos
df = load_data('bquxjob_145dbef4_1977e15505c.csv')
if df is None:
    st.stop()


if df.empty:
    st.error("No hay datos disponibles para mostrar")
    st.stop()

# Título y fecha de actualización
st.title("Reporte de Conversión\nMayo vs Abril 2024")
st.write(f"Última actualización: 17/6/2025")

# Site selector at the top
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
df_abril = filtered_df[filtered_df['mes'] == 'Abril']
df_mayo = filtered_df[filtered_df['mes'] == 'Mayo']

#Handle empty dataframes
if df_abril.empty and df_mayo.empty:
    st.warning("No data available for April and May based on current filters.")
    st.stop()
elif df_abril.empty:
    st.warning("No data available for April based on current filters.")
elif df_mayo.empty:
    st.warning("No data available for May based on current filters.")

if selected_site == "General":
    # Calculate main metrics
    cvr_abril = df_abril['CVR_GENERAL'].mean() * 100 if not df_abril.empty else 0
    cvr_mayo =  df_mayo['CVR_GENERAL'].mean() * 100 if not df_mayo.empty else 0
    cvr_change = cvr_mayo - cvr_abril

    trafico_abril = df_abril['trafico'].sum() if not df_abril.empty else 0
    trafico_mayo = df_mayo['trafico'].sum() if not df_mayo.empty else 0
    trafico_change = ((trafico_mayo - trafico_abril) / trafico_abril * 100) if trafico_abril > 0 else 0

    # Calculate sites with decline (only for General view)
    sites_con_caida = 0
    total_sites = len(df['site'].unique())
    for site in df['site'].unique():
        site_abril = df[(df['mes'] == 'Abril') & (df['site'] == site)]
        site_mayo = df[(df['mes'] == 'Mayo') & (df['site'] == site)]

        if not site_abril.empty and not site_mayo.empty:
            cvr_site_abril = site_abril['CVR_GENERAL'].mean() * 100
            cvr_site_mayo = site_mayo['CVR_GENERAL'].mean() * 100

            if cvr_site_mayo < cvr_site_abril:
                sites_con_caida += 1

    # Main metrics row
    st.markdown("### Resumen")
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
        st.markdown(create_metric_card(
            "Sites con Caída",
            f"{sites_con_caida}",
            f"de {total_sites} sites totales"
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Performance por Site (formato tabla HTML)
    st.markdown("### Performance por Site")
    st.markdown("Comparación mensual de conversión por marketplace")

    site_data = []
    for site in df['site'].unique():
        site_df_abril = df_abril[df_abril['site'] == site]
        site_df_mayo = df_mayo[df_mayo['site'] == site]

        cvr_abril_site = site_df_abril['CVR_GENERAL'].mean() * 100 if not site_df_abril.empty else 0
        cvr_mayo_site = site_df_mayo['CVR_GENERAL'].mean() * 100 if not site_df_mayo.empty else 0
        cvr_change_site = cvr_mayo_site - cvr_abril_site
        trafico_abril_site = site_df_abril['trafico'].sum() if not site_df_abril.empty else 0
        trafico_mayo_site = site_df_mayo['trafico'].sum() if not site_df_mayo.empty else 0
        trafico_change_site = ((trafico_mayo_site - trafico_abril_site) / trafico_abril_site * 100) if trafico_abril_site > 0 else 0

        status = calculate_status(cvr_change_site/100)

        # Find step with the largest variation
        step_variation = {}
        for step in ['STEP_1', 'STEP_2', 'STEP_3']:
            step_abril = site_df_abril[step].mean() if not site_df_abril.empty else 0
            step_mayo = site_df_mayo[step].mean() if not site_df_mayo.empty else 0
            step_variation[step] = step_mayo - step_abril

        most_var_step = max(step_variation, key=step_variation.get)
        step_name_mapping = {'STEP_1': 'Ship - Pay', 'STEP_2': 'Pay - Rev', 'STEP_3': 'Rev - Cong'}
        most_var_step_name = step_name_mapping.get(most_var_step, 'Unknown')

        site_data.append({
            'Site': site,
            'CVR Abril': cvr_abril_site,
            'CVR Mayo': cvr_mayo_site,
            'Cambio CVR': cvr_change_site,
            'Cambio Tráfico': trafico_change_site,
            'Step Mayor Variación': most_var_step_name,
            'Status': status
        })

    if site_data:
        site_df = pd.DataFrame(site_data)
        site_df = site_df.sort_values('CVR Mayo', ascending=False).reset_index(drop=True)

        fig = px.bar(
            site_df,
            x='Site',
            y=['CVR Abril', 'CVR Mayo'],
            barmode='group',
            labels={'value': 'CVR (%)', 'variable': 'Mes'},
            color_discrete_sequence=['#8884d9', '#81c99d']
        )
        fig.update_layout(
            title="CVR por Site - Abril vs Mayo",
            xaxis_title="Site",
            yaxis_title="CVR (%)",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            legend=dict(title='Mes')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay datos de performance por site para mostrar")
    st.markdown("<br>", unsafe_allow_html=True)


# Platform Analysis Table
st.markdown("### Análisis por Plataforma")
st.markdown("Comparación de conversión y tráfico por plataforma")

platform_data = []
total_trafico_mayo = df_mayo['trafico'].sum() if not df_mayo.empty else 0

# Orden deseado de plataformas
platform_order = ['/mobile/android', '/mobile/ios', '/web/desktop', '/web/mobile']

# Si hay plataformas fuera del orden, las agregamos al final
all_platforms = filtered_df['platform'].unique().tolist()
ordered_platforms = [p for p in platform_order if p in all_platforms] + [p for p in all_platforms if p not in platform_order]

for platform in ordered_platforms:
    # April data
    platform_abril = df_abril[df_abril['platform'] == platform]
    platform_mayo = df_mayo[df_mayo['platform'] == platform]

    # Handle empty platform dataframes
    cvr_abril_plat = platform_abril['CVR_GENERAL'].mean() * 100 if not platform_abril.empty else 0
    cvr_mayo_plat = platform_mayo['CVR_GENERAL'].mean() * 100 if not platform_mayo.empty else 0
    cvr_change_plat = cvr_mayo_plat - cvr_abril_plat

    trafico_abril_plat = platform_abril['trafico'].sum() if not platform_abril.empty else 0
    trafico_mayo_plat = platform_mayo['trafico'].sum() if not platform_mayo.empty else 0
    trafico_change_plat = ((trafico_mayo_plat - trafico_abril_plat) / trafico_abril_plat * 100) if trafico_abril_plat > 0 else 0

    share = (trafico_mayo_plat / total_trafico_mayo * 100) if total_trafico_mayo > 0 else 0

    platform_data.append({
        'Plataforma': platform,
        'CVR Mayo (%)': round(cvr_mayo_plat, 2),
        'Cambio CVR (pp)': round(cvr_change_plat, 2),
        'Tráfico Mayo': int(trafico_mayo_plat),
        'Cambio Tráfico (%)': round(trafico_change_plat, 2),
        'Share (%)': round(share, 1)
    })

if platform_data:
    platform_df = pd.DataFrame(platform_data)
    # Sort by platform order
    platform_df['platform_order'] = platform_df['Plataforma'].apply(lambda x: ordered_platforms.index(x) if x in ordered_platforms else 999)
    platform_df = platform_df.sort_values('platform_order').reset_index(drop=True)
    platform_df = platform_df.drop(columns=['platform_order'])
    st.dataframe(platform_df, use_container_width=True)
else:
    st.info("No hay datos de plataformas para mostrar.")

st.markdown("### Evolución Semanal: Rev - Cong")
st.markdown("Comparación de performance por semana entre Abril y Mayo")
weekly_data = filtered_df.groupby(['mes', 'semana_category'])['STEP_3'].mean().unstack()
# Asegura que las columnas estén en el orden correcto y existan
week_cols = ['semana_1_abril', 'semana_2_abril', 'semana_3_abril', 'semana_4_abril',
             'semana_1_mayo', 'semana_2_mayo', 'semana_3_mayo', 'semana_4_mayo', 'semana_5_mayo']
existing_cols = [col for col in week_cols if col in weekly_data.columns]
weekly_data = weekly_data[existing_cols]
weekly_data = weekly_data.stack().reset_index(name='STEP_3').rename(columns={'semana_category': 'Semana'})
weekly_data['STEP_3'] = weekly_data['STEP_3'] * 100

fig_rev_cong = px.bar(
    weekly_data,
    x='Semana',
    y='STEP_3',
    color='mes',
    barmode='group',
    category_orders={"Semana": week_cols},
    color_discrete_sequence=['#A0C4FF', '#CAFFBF']
)

fig_rev_cong.update_layout(
    height=400,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black'),
    xaxis=dict(
        title="Semana",
        tickfont=dict(color='black'),
        title_font=dict(color='black'),
        categoryorder='array',
        categoryarray=week_cols
    ),
    yaxis=dict(
        title="Rev - Cong (%)",
        tickfont=dict(color='black'),
        title_font=dict(color='black'),
        gridcolor='lightgray'
    ),
    legend=dict(font=dict(color='black'))

)

fig_rev_cong.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
st.plotly_chart(fig_rev_cong, use_container_width=True)