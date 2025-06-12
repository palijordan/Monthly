import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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
    # Normalizar nombres de columnas a minúsculas
    df.columns = [col.lower() for col in df.columns]
    # Limpiar y procesar datos
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

# Paleta de colores suaves y elegantes para todo el dashboard
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

st.markdown("<br>", unsafe_allow_html=True)

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
            color_discrete_sequence=color_palette[:2],
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
        st.markdown("<br>", unsafe_allow_html=True)
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
            color_discrete_sequence=[color_palette[2]]
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
        st.markdown("<br>", unsafe_allow_html=True)
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
        st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("No hay datos de rechazos para mostrar")
else:
    st.info("No hay rechazos registrados en el período seleccionado")

st.subheader("📊 Status de Compras No Aprobadas")

not_approved_df = filtered_df[(filtered_df['status'] != 'approved') & (filtered_df['status'] != 'pending')]

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
    fig_status = px.bar(
        status_share,
        x='mes',
        y='Porcentaje',
        color='status',
        text=status_share['Porcentaje'].apply(lambda x: f"{x:.1f}%"),
        barmode='stack',
        color_discrete_sequence=color_palette,
        category_orders={'mes': [mes_anterior, mes_actual]}
    )
    fig_status.update_layout(
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
    fig_status.update_traces(textposition='inside', textfont_color='black')
    st.plotly_chart(fig_status, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)


    # Gráfico : Evolución de Spender Type en Compras No Aprobadas
    st.subheader("💸 Evolución de Spender Type en Compras No Aprobadas")
    if 'spender_type' in not_approved_df.columns:
        # Excluir 'non_spender'
        filtered_spender_df = not_approved_df[not_approved_df['spender_type'] != 'non_spender']
        # Filtrar solo los dos meses a comparar
        spender_share = (
            filtered_spender_df[filtered_spender_df['mes'].isin([mes_anterior, mes_actual])]
            .groupby(['spender_type', 'mes'])['incoming_amt']
            .sum()
            .reset_index()
        )
        # Calcular el total de cada mes para porcentaje
        total_by_mes_spender = (
            filtered_spender_df[filtered_spender_df['mes'].isin([mes_anterior, mes_actual])]
            .groupby('mes')['incoming_amt']
            .sum()
            .reset_index()
            .rename(columns={'incoming_amt': 'Total_mes'})
        )
        spender_share = spender_share.merge(total_by_mes_spender, on='mes')
        spender_share['Porcentaje'] = (spender_share['incoming_amt'] / spender_share['Total_mes']) * 100

        # Pivotear para tener una columna por mes
        pivot_spender = spender_share.pivot(index='spender_type', columns='mes', values='Porcentaje').fillna(0)
        # Ordenar spender_type por el valor de mes_actual
        if mes_anterior in pivot_spender.columns and mes_actual in pivot_spender.columns:
            pivot_spender = pivot_spender[[mes_anterior, mes_actual]]
        pivot_spender = pivot_spender.sort_values(by=mes_actual, ascending=False)
        # Renombrar columnas para mostrar nombres bonitos
        pivot_spender = pivot_spender.rename(columns={mes_anterior: mes_anterior.capitalize(), mes_actual: mes_actual.capitalize()})

        # Calcular el valor máximo para ajustar el eje y
        max_val = pivot_spender.max().max()
        y_max = min(100, max_val + 15)

        # Crear gráfico de líneas para comparar la evolución de spender_type entre los dos meses
        pivot_spender_reset = pivot_spender.reset_index()
        fig = go.Figure()
        color_count = len(color_palette)
        for idx, row in pivot_spender_reset.iterrows():
            val1 = row[mes_anterior.capitalize()]
            val2 = row[mes_actual.capitalize()]
            diff = abs(val2 - val1)
            # Negrita si la variación es mayor a 5%
            if diff > 5:
                text1 = f"<b>{val1:.1f}%</b>"
                text2 = f"<b>{val2:.1f}%</b>"
            else:
                text1 = f"{val1:.1f}%"
                text2 = f"{val2:.1f}%"
            fig.add_trace(go.Scatter(
                x=[mes_anterior.capitalize(), mes_actual.capitalize()],
                y=[val1, val2],
                mode='lines+markers+text',
                name=row['spender_type'],
                text=[text1, text2],
                textposition=["middle left", "middle right"],
                line=dict(width=3, color=color_palette[idx % color_count]),
                marker=dict(size=10, color=color_palette[idx % color_count]),
                textfont=dict(color='black'),
                hovertemplate='%{y:.1f}%'
            ))

        fig.update_layout(
            xaxis=dict(title='Mes', tickfont=dict(color='black'), title_font=dict(color='black')),
            yaxis=dict(title='Porcentaje (%)', range=[0, y_max], gridcolor='lightgray', tickfont=dict(color='black'), title_font=dict(color='black')),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            legend_title_text='Spender Type',
            legend=dict(font=dict(color='black')),
            height=400,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("No hay columna 'spender_type' en los datos.")


    # Gráfico 2: Distribución por buyer_type (barras horizontales, comparación Abril vs Mayo)
    st.subheader("👤 Distribución de Buyer Type en Compras No Aprobadas")
    if 'buyer_type' in not_approved_df.columns:
        # Filtrar solo los dos meses a comparar
        buyer_share = (
            not_approved_df[not_approved_df['mes'].isin([mes_anterior, mes_actual])]
            .groupby(['buyer_type', 'mes'])['incoming_amt']
            .sum()
            .reset_index()
        )
        # Calcular el total de cada mes para porcentaje
        total_by_mes = (
            not_approved_df[not_approved_df['mes'].isin([mes_anterior, mes_actual])]
            .groupby('mes')['incoming_amt']
            .sum()
            .reset_index()
            .rename(columns={'incoming_amt': 'Total_mes'})
        )
        buyer_share = buyer_share.merge(total_by_mes, on='mes')
        buyer_share['Porcentaje'] = (buyer_share['incoming_amt'] / buyer_share['Total_mes']) * 100

        # Pivotear para tener una columna por mes
        pivot_df = buyer_share.pivot(index='buyer_type', columns='mes', values='Porcentaje').fillna(0)
        # Ordenar buyer_type por el valor de mes_actual
        if mes_anterior in pivot_df.columns and mes_actual in pivot_df.columns:
            pivot_df = pivot_df[[mes_anterior, mes_actual]]
        pivot_df = pivot_df.rename(columns={mes_anterior: mes_anterior.capitalize(), mes_actual: mes_actual.capitalize()})
        pivot_df = pivot_df.sort_values(by=mes_actual.capitalize(), ascending=False)
        pivot_df_reset = pivot_df.reset_index()

        # Gráfico de barras horizontales
        import plotly.graph_objects as go
        fig = go.Figure()
        for i, mes in enumerate([mes_anterior.capitalize(), mes_actual.capitalize()]):
            fig.add_trace(go.Bar(
                y=pivot_df_reset['buyer_type'],
                x=pivot_df_reset[mes],
                name=mes,
                orientation='h',
                marker_color=color_palette[i % len(color_palette)],
                text=[f"{v:.1f}%" for v in pivot_df_reset[mes]],
                textposition='auto'
            ))

        fig.update_layout(
            barmode='group',
            xaxis=dict(title='Porcentaje (%)', range=[0, 100], gridcolor='lightgray', tickfont=dict(color='black'), title_font=dict(color='black')),
            yaxis=dict(title='Buyer Type', tickfont=dict(color='black'), title_font=dict(color='black')),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            legend_title_text='Mes',
            legend=dict(font=dict(color='black')),
            height=350,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("No hay columna 'buyer_type' en los datos.")
