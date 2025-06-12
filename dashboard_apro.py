import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

# Configuración inicial de la página
st.set_page_config(
    page_title="Payment Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_css():
    st.markdown("""
    <style>
    .stApp { background-color: white !important; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { color: black !important; }
    .stApp p, .stApp div, .stApp span, .stApp label { color: black !important; }
    .stSidebar { background-color: #f8f9fa !important; }
    .stSidebar .stSelectbox label,
    .stSidebar .stSelectbox div,
    .stSidebar h1, .stSidebar h2, .stSidebar h3,
    .stSidebar p, .stSidebar span { color: black !important; }
    .stSidebar .stSelectbox > div > div { color: black !important; background-color: white !important; }
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
    .metric-value { color: #000000 !important; font-size: 2rem; font-weight: 700; margin: 0; }
    .metric-delta { color: #FF7F50; font-size: 0.8rem; margin-top: 0.5rem; }
    .rechazos-container {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .rechazos-container table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
    }
    .rechazos-container th,
    .rechazos-container td {
        padding: 12px;
        text-align: left;
        border-bottom: 1px solid #ddd;
        color: black !important;
    }
    .rechazos-container th {
        background-color: #f8f9fa;
        font-weight: 600;
    }
    .rechazos-container tr:hover { background-color: #f5f5f5; }
    </style>
    """, unsafe_allow_html=True)

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.warning("Archivo no encontrado. Usando datos de ejemplo.")
        df = pd.read_csv("Monthly-Apro-05vs04.csv")
    # Limpiar y procesar datos
    df.columns = [col.lower() for col in df.columns]  # <-- agrega esto
    df = df[df['mes'].notnull() & (df['mes'].str.lower() != 'null')]
    df['mes'] = df['mes'].str.strip().str.lower()
    # Convertir amounts
    if df['incoming_amt'].dtype == 'object':
        df['incoming_amt'] = df['incoming_amt'].astype(str).str.replace(',', '.').astype(float)
    df['is_approved'] = df['status'].str.lower() == 'approved'
    return df

def create_metric_card(title, value, delta=None):
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ''
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

def calculate_approval_rate(df, month):
    df_month = df[df['mes'] == month]
    if df_month.empty:
        return 0
    total = df_month['incoming_amt'].sum()
    approved = df_month[df_month['is_approved']]['incoming_amt'].sum()
    return (approved / total) * 100 if total > 0 else 0
# Aplicar CSS
inject_css()

# Cargar datos
try:
    df = load_data("Monthly-Apro-05vs04.csv")
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

if df.empty:
    st.error("No hay datos disponibles para mostrar")
    st.stop()

# Sidebar
st.sidebar.header("🔧 Filtros")
sites = ["General"] + sorted(df['site'].unique().tolist())
selected_site = st.sidebar.selectbox("Seleccionar Site", sites)

# Filtrar datos
filtered_df = df if selected_site == "General" else df[df['site'] == selected_site]

if filtered_df.empty:
    st.error(f"No hay datos disponibles para {selected_site}")
    st.stop()

# Diccionario para mapear nombres de meses a número de mes (soporta inglés y español)
meses_map = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
    'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
}

# Limpiar y filtrar meses válidos
meses_validos = [m for m in filtered_df['mes'].unique() if m and m not in ['null', '']]
# Ordenar por número de mes
meses_ordenados = sorted(
    meses_validos,
    key=lambda x: meses_map.get(x.lower(), 0)
)
if len(meses_ordenados) < 2:
    st.error("No hay suficientes meses para comparar.")
    st.stop()
mes_anterior, mes_actual = meses_ordenados[-2], meses_ordenados[-1]

# Header principal
st.subheader(f"📊 Aprobación - {selected_site}")
col1, col2 = st.columns(2)

anterior_rate = calculate_approval_rate(filtered_df, mes_anterior)
actual_rate = calculate_approval_rate(filtered_df, mes_actual)
variation = ((actual_rate - anterior_rate) / anterior_rate * 100) if anterior_rate != 0 else 0

with col1:
    st.markdown(create_metric_card(
        f"Tasa de Aprobación - {mes_anterior.capitalize()}",
        f"{anterior_rate:.1f}%"
    ), unsafe_allow_html=True)

with col2:
    variation_symbol = "↑" if variation >= 0 else "↓"
    variation_color = "#28a745" if variation >= 0 else "#dc3545"
    value_html = f'''
    <div style="display: flex; flex-direction: row; align-items: center; justify-content: center;">
        <span style="font-size:2rem; color:#000; font-weight:700;">{actual_rate:.1f}%</span>
        <span style="color:{variation_color}; font-size:1rem; margin-left:12px; font-weight:600;">
            {variation_symbol} {abs(variation):.1f}%
        </span>
    </div>
    '''
    st.markdown(create_metric_card(
        f"Tasa de Aprobación - {mes_actual.capitalize()}",
        value_html,
        delta=None
    ), unsafe_allow_html=True)
st.divider()

# Sección 2: Gráficos

st.subheader("💳 Distribución de Medios de Pago (%)")
if not filtered_df.empty:
    payment_data = (
        filtered_df.groupby(['mes', 'medio_de_pago'])['incoming_amt'].sum().reset_index()
    )
    total_by_mes = (
        filtered_df.groupby('mes')['incoming_amt'].sum().reset_index().rename(columns={'incoming_amt': 'Total_mes'})
    )
    payment_data = payment_data.merge(total_by_mes, on='mes')
    payment_data['Porcentaje'] = (payment_data['incoming_amt'] / payment_data['Total_mes']) * 100

    # Solo mostrar los dos meses más recientes
    payment_data = payment_data[payment_data['mes'].isin([mes_anterior, mes_actual])]

    if not payment_data.empty:
        fig = px.bar(
            payment_data,
            x='medio_de_pago',
            y='Porcentaje',
            color='mes',
            barmode='group',
            color_discrete_sequence=['#20B2AA', '#4682B4'],
            category_orders={'mes': [mes_anterior, mes_actual]}
        )
        fig.update_layout(
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            legend=dict(font=dict(color='black'))
        )
        fig.update_xaxes(
            tickfont=dict(color='black'),
            title_font=dict(color='black'),
        )
        fig.update_yaxes(
            title_text="Porcentaje (%)",
            tickfont=dict(color='black'),
            title_font=dict(color='black'),
            gridcolor='lightgray'
        )
        fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay datos suficientes para mostrar el gráfico")
else:
    st.info("No hay datos disponibles")

st.subheader("🏦 Share de Medios de Pagos Ecosistémicos")

eco_medios = ["account_money", "tc mp", "digital_currency"]

eco_data = []
for mes in [mes_anterior, mes_actual]:
    mes_data = filtered_df[filtered_df['mes'] == mes]
    if not mes_data.empty:
        total_mes = mes_data['incoming_amt'].sum()
        eco_total = mes_data[mes_data['medio_de_pago'].isin(eco_medios)]['incoming_amt'].sum()
        proportion = (eco_total / total_mes * 100) if total_mes > 0 else 0
        eco_data.append({'mes': mes.capitalize(), 'porcentaje': proportion})

if eco_data:
    eco_df = pd.DataFrame(eco_data)
    col_eco, _ = st.columns([1, 1])
    with col_eco:
        fig = px.bar(
            eco_df,
            x='mes',
            y='porcentaje',
            color_discrete_sequence=["#FF7F50"]
        )
        max_height = eco_df['porcentaje'].max() * 1.05
        fig.update_layout(
            height=300,
            width=500,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        fig.update_yaxes(range=[0, max_height])
        fig.update_xaxes(
            title_text="Mes",
            tickfont=dict(color='black'),
            title_font=dict(color='black')
        )
        fig.update_yaxes(
            title_text="Porcentaje (%)",
            tickfont=dict(color='black'),
            title_font=dict(color='black'),
            gridcolor='lightgray'
        )
        for i, row in eco_df.iterrows():
            fig.add_annotation(
                x=row['mes'],
                y=row['porcentaje'],
                text=f"{row['porcentaje']:.1f}%",
                showarrow=False,
                yshift=10,
                font=dict(color='black')
            )
        st.plotly_chart(fig, use_container_width=False)
else:
    st.info("No hay datos disponibles para medios ecosistémicos")

st.divider()

# Sección 3: Análisis de Rechazos
st.subheader("❌ Análisis de Rechazos")

rejected_df = filtered_df[filtered_df['status'] != 'approved']

if not rejected_df.empty:
    rechazos_data = []
    medios_unicos = rejected_df['medio_de_pago'].unique()
    for medio in medios_unicos:
        anterior_rechazos = rejected_df[(rejected_df['mes'] == mes_anterior) & (rejected_df['medio_de_pago'] == medio)]['incoming_amt'].sum()
        actual_rechazos = rejected_df[(rejected_df['mes'] == mes_actual) & (rejected_df['medio_de_pago'] == medio)]['incoming_amt'].sum()
        anterior_total = rejected_df[rejected_df['mes'] == mes_anterior]['incoming_amt'].sum()
        actual_total = rejected_df[rejected_df['mes'] == mes_actual]['incoming_amt'].sum()
        anterior_pct = (anterior_rechazos / anterior_total * 100) if anterior_total > 0 else 0
        actual_pct = (actual_rechazos / actual_total * 100) if actual_total > 0 else 0
        variacion = actual_pct - anterior_pct
        rechazos_data.append({
            f'{mes_anterior.capitalize()} (%)': f"{anterior_pct:.1f}%",
            f'{mes_actual.capitalize()} (%)': f"{actual_pct:.1f}%",
            'medio de pago': medio,
            'Variación': f"{variacion:+.1f}%"
        })
    if rechazos_data:
        rechazos_df = pd.DataFrame(rechazos_data)
        rechazos_df['Actual_num'] = rechazos_df[f'{mes_actual.capitalize()} (%)'].str.replace('%', '').astype(float)
        rechazos_df = rechazos_df.sort_values('Actual_num', ascending=False).reset_index(drop=True)
        rechazos_df = rechazos_df.drop(columns=['Actual_num'])
        def highlight_variacion(val):
            try:
                num = float(val.replace('%', '').replace('+', ''))
                if abs(num) > 1:
                    return f"<b>{val}</b>"
            except Exception:
                pass
            return val
        rechazos_df['Variación'] = rechazos_df['Variación'].apply(highlight_variacion)
        rechazos_df = rechazos_df[['medio de pago', f'{mes_anterior.capitalize()} (%)', f'{mes_actual.capitalize()} (%)', 'Variación']]
        st.markdown(f"""
        <div class="rechazos-container">
            {rechazos_df.to_html(escape=False, index=False)}
            <p style="color: #666; font-size: 0.9rem; margin-top: 1rem; font-style: italic;">
                *Los porcentajes muestran la participación de cada medio de pago en el total de rechazos del mes.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No hay datos de rechazos para mostrar")
else:
    st.info("No hay rechazos registrados en el período seleccionado")

st.divider()

st.subheader("📊 Status de Compras No Aprobadas")

not_approved_df = filtered_df[(filtered_df['status'] != 'Approved') & (filtered_df['status'] != 'Pending')]

if not not_approved_df.empty:
    # Gráfico 1: Distribución por Status
    status_share = (
        not_approved_df.groupby(['mes', 'status'])['incoming_amt']
        .sum()
        .reset_index()
    )
    total_by_mes = (
        not_approved_df.groupby('mes')['incoming_amt']
        .sum()
        .reset_index()
        .rename(columns={'incoming_amt': 'Total_mes'})
    )
    status_share = status_share.merge(total_by_mes, on='mes')
    status_share['Porcentaje'] = (status_share['incoming_amt'] / status_share['Total_mes']) * 100
    # Solo mostrar los dos meses más recientes
    status_share = status_share[status_share['mes'].isin([mes_anterior, mes_actual])]
    color_palette = [
        "#ffe5cc", "#ffd8b3", "#ffcc99", "#ffb380", "#ff9966",
        "#e6a87c", "#d9a066", "#bfa380", "#a67c52", "#8c6e54"
    ]
    fig = px.bar(
        status_share,
        x='mes',
        y='Porcentaje',
        color='status',
        text=status_share['Porcentaje'].apply(lambda x: f"{x:.1f}%"),
        barmode='stack',
        color_discrete_sequence=color_palette,
        category_orders={'mes': [mes_anterior, mes_actual]}
    )
    fig.update_layout(
        yaxis=dict(
            title='Porcentaje (%)',
            range=[0, 100],
            gridcolor='lightgray',
            tickfont=dict(color='black'),
            title_font=dict(color='black')
        ),
        xaxis=dict(
            title='Mes',
            tickfont=dict(color='black'),
            title_font=dict(color='black')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        legend_title_text='Status',
        legend=dict(font=dict(color='black')),
        height=400
    )
    fig.update_traces(textposition='inside', textfont_color='black')
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Cuadro: Distribución de Buyer Type en Compras No Aprobadas
    st.subheader("👤 Buyer Type en Compras No Aprobadas")
    if 'buyer_type' in not_approved_df.columns:
        buyer_share = (
            not_approved_df.groupby(['mes', 'buyer_type'])['incoming_amt']
            .sum()
            .reset_index()
        )
        buyer_share = buyer_share.merge(total_by_mes, on='mes')
        buyer_share['Porcentaje'] = (buyer_share['incoming_amt'] / buyer_share['Total_mes']) * 100
        buyer_share = buyer_share[buyer_share['mes'].isin([mes_anterior, mes_actual])]

        # Mostrar como tabla con barras horizontales embebidas (visualización tipo "bar chart in table")
        def bar_html(pct, color="#4682B4"):
            width = min(max(pct, 0), 100)
            return f'''
                <div style="background:#f0f0f0; border-radius:4px; width:100%; height:18px; position:relative;">
                    <div style="background:{color}; width:{width}%; height:100%; border-radius:4px;"></div>
                    <span style="position:absolute; left:8px; top:0; color:#222; font-size:0.85rem; font-weight:600;">{pct:.1f}%</span>
                </div>
            '''

        table_html = "<table style='width:100%; border-collapse:collapse;'>"
        # Quitar el título de la columna de buyer_type (dejar celda vacía)
        table_html += "<tr><th style='width:2px;'></th>"
        table_html += f"<th style='text-align:left;'>{mes_anterior.capitalize()}</th>"
        table_html += f"<th style='text-align:left;'>{mes_actual.capitalize()}</th></tr>"

        for buyer in sorted(buyer_share['buyer_type'].unique()):
            row = buyer_share[buyer_share['buyer_type'] == buyer]
            pct_ant = row[row['mes'] == mes_anterior]['Porcentaje'].values[0] if mes_anterior in row['mes'].values else 0
            pct_act = row[row['mes'] == mes_actual]['Porcentaje'].values[0] if mes_actual in row['mes'].values else 0
            # Añadir espacio de 1cm entre la columna de buyer y las barras
            # Usar white-space:nowrap para evitar quiebre de línea en buyer_type
            table_html += (
                f"<tr>"
                f"<td style='white-space:nowrap'>{buyer}</td>"
                f"<td style='padding-left:2cm'>{bar_html(pct_ant, '#20B2AA')}</td>"
                f"<td style='padding-left:2cm'>{bar_html(pct_act, '#FF7F50')}</td>"
                f"</tr>"
            )

        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("No hay columna 'buyer_type' en los datos.")

    st.divider()

    # Gráfico 3: Distribución por spender_type
    st.subheader("💸 Distribución de Spender Type en Compras No Aprobadas")
    if 'spender_type' in not_approved_df.columns:
        # Filtrar spender_types distintos a 'non_spender'
        spender_filtered = not_approved_df[not_approved_df['spender_type'].str.lower() != 'non_spender']
        spender_share = (
            spender_filtered.groupby(['mes', 'spender_type'])['incoming_amt']
            .sum()
            .reset_index()
        )
        spender_share = spender_share.merge(total_by_mes, on='mes')
        spender_share['Porcentaje'] = (spender_share['incoming_amt'] / spender_share['Total_mes']) * 100
        spender_share = spender_share[spender_share['mes'].isin([mes_anterior, mes_actual])]


        pivot_bar = spender_share.pivot(index='spender_type', columns='mes', values='Porcentaje').fillna(0)
        all_types = sorted(spender_share['spender_type'].unique())
        pivot_bar = pivot_bar.reindex(all_types)
        bar_df = pivot_bar.reset_index().melt(id_vars='spender_type', var_name='mes', value_name='Porcentaje')

        fig_bar = px.bar(
            bar_df,
            x='spender_type',
            y='Porcentaje',
            color='mes',
            barmode='group',
            color_discrete_sequence=['#2c5aa0 ', "#772e0f"],
            category_orders={'mes': [mes_anterior, mes_actual], 'spender_type': all_types}
        )
        fig_bar.update_layout(
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            legend=dict(font=dict(color='black')),
            xaxis_title="Spender Type",
            yaxis_title="Porcentaje (%)"
        )
        fig_bar.update_xaxes(
            tickfont=dict(color='black'),
            title_font=dict(color='black'),
        )
        fig_bar.update_yaxes(
            tickfont=dict(color='black'),
            title_font=dict(color='black'),
            gridcolor='lightgray'
        )
        fig_bar.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)
