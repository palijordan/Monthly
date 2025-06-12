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
    /* Fondo blanco para toda la aplicación */
    .stApp {
        background-color: white !important;
    }
    
    /* Títulos y subtítulos en negro */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: black !important;
    }
    
    /* Texto general en negro */
    .stApp p, .stApp div, .stApp span, .stApp label {
        color: black !important;
    }
    
    /* Sidebar específico - más agresivo */
    .stSidebar {
        background-color: #f8f9fa !important;
    }
    
    .stSidebar .stSelectbox label,
    .stSidebar .stSelectbox div,
    .stSidebar h1, .stSidebar h2, .stSidebar h3,
    .stSidebar p, .stSidebar span {
        color: black !important;
    }
    
    .stSidebar .stSelectbox > div > div {
        color: black !important;
        background-color: white !important;
    }
    

    /* CSS más específico y menos invasivo */
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    
    .metric-title {
        color: #000000 !important;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .metric-value {
        color: #000000 !important;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-delta {
        color: #FF7F50;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
    
    /* Nuevo estilo para la tabla de rechazos */
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
    
    .rechazos-container tr:hover {
        background-color: #f5f5f5;
    }
</style>
    """, unsafe_allow_html=True)


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.warning("Archivo 'aprobacion.csv' no encontrado. Usando datos de ejemplo.")
        df = ("aprobacion.csv")
    
    # Limpiar y procesar datos
    df['mes'] = df['mes'].str.lower().str.strip().str.replace(' ', '_')
    df = df[df['mes'].isin(['abril', 'mayo'])]
    
    # Convertir amounts
    if df['Incoming_amt'].dtype == 'object':
        df['Incoming_amt'] = df['Incoming_amt'].str.replace(',', '.').astype(float)
    
    df['is_approved'] = df['Status'] == 'Approved'
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
    
    total = df_month['Incoming_amt'].sum()
    approved = df_month[df_month['is_approved']]['Incoming_amt'].sum()
    return (approved / total) * 100 if total > 0 else 0

# Aplicar CSS
inject_css()

# Cargar datos
try:
    df = load_data("aprobacion.csv")
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

# Verificar que tenemos datos
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

# Header principal - CORREGIDO: Usar st.title en lugar de markdown

st.subheader(f"📊 Aprobación - {selected_site}")
col1, col2 = st.columns(2)

abril_rate = calculate_approval_rate(filtered_df, 'abril')
mayo_rate = calculate_approval_rate(filtered_df, 'mayo')
variation = ((mayo_rate - abril_rate) / abril_rate * 100) if abril_rate != 0 else 0

with col1:
    st.markdown(create_metric_card(
        "Tasa de Aprobación - Abril", 
        f"{abril_rate:.1f}%"
    ), unsafe_allow_html=True)

with col2:
    variation_symbol = "↑" if variation >= 0 else "↓"
    variation_color = "#28a745" if variation >= 0 else "#dc3545"
    # Centrar el valor y el texto de variación, y asegurar color correcto
    value_html = f'''
    <div style="display: flex; flex-direction: row; align-items: center; justify-content: center;">
        <span style="font-size:2rem; color:#000; font-weight:700;">{mayo_rate:.1f}%</span>
        <span style="color:{variation_color}; font-size:1rem; margin-left:12px; font-weight:600;">
            {variation_symbol} {abs(variation):.1f}%
        </span>
    </div>
    '''
    st.markdown(create_metric_card(
        "Tasa de Aprobación - Mayo", 
        value_html,
        delta=None  # No usar delta, ya que el texto está incluido en el valor
    ), unsafe_allow_html=True)

# Separador 
st.divider()
# Sección 2: Gráficos

st.subheader("💳 Distribución de Medios de Pago (%)")
# Verificar que tenemos datos para el gráfico
if not filtered_df.empty:
    # Calcular el porcentaje que representa cada medio de pago sobre el total de Incoming_amt por mes
    payment_data = (
        filtered_df.groupby(['mes', 'Medio_de_pago'])['Incoming_amt'].sum().reset_index()
    )
    total_by_mes = (
        filtered_df.groupby('mes')['Incoming_amt'].sum().reset_index().rename(columns={'Incoming_amt': 'Total_mes'})
    )
    payment_data = payment_data.merge(total_by_mes, on='mes')
    payment_data['Porcentaje'] = (payment_data['Incoming_amt'] / payment_data['Total_mes']) * 100

    if not payment_data.empty:
        # Crear gráfico de barras agrupadas con porcentajes
        fig = px.bar(
            payment_data,
            x='Medio_de_pago',
            y='Porcentaje',
            color='mes',
            barmode='group',
            color_discrete_sequence=['#20B2AA', '#4682B4'],
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
            gridcolor='lightgray'  # <-- líneas horizontales en gris
        )
        
        # Mostrar valores en las barras como etiquetas (usando text_auto de plotly)
        fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay datos suficientes para mostrar el gráfico")
else:
    st.info("No hay datos disponibles")

st.subheader("🏦 Share de Medios de Pagos Ecosistémicos")

eco_medios = ["account_money", "tc mp", "digital_currency"]

# Calcular proporción de medios ecosistémicos
eco_data = []
for mes in ['abril', 'mayo']:
    mes_data = filtered_df[filtered_df['mes'] == mes]
    if not mes_data.empty:
        total_mes = mes_data['Incoming_amt'].sum()
        eco_total = mes_data[mes_data['Medio_de_pago'].isin(eco_medios)]['Incoming_amt'].sum()
        proportion = (eco_total / total_mes * 100) if total_mes > 0 else 0
        eco_data.append({'mes': mes.capitalize(), 'porcentaje': proportion})

if eco_data:
    eco_df = pd.DataFrame(eco_data)
    
    # Gráfico más compacto, usando columnas de Streamlit
    col_eco, _ = st.columns([1, 1])
    with col_eco:
        fig = px.bar(
            eco_df,
            x='mes',
            y='porcentaje',
            color_discrete_sequence=["#FF7F50"]
        )
        
        max_height = eco_df['porcentaje'].max() * 1.05  # Aumentar un 5% la altura máxima de las barras
        fig.update_layout(
            height=300,
            width=500,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        #busco que los valores del eje y del grafico sea un 5% mas que el maximo de los porcentajes
        fig.update_yaxes(range=[0, max_height])  # Ajustar el rango del eje Y
        
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
        
        # Agregar valores en las barras
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

# Separador
st.divider()
# Sección 3: Análisis de Rechazos - CORREGIDO: Usar DataFrame de Streamlit
st.subheader("❌ Análisis de Rechazos")

rejected_df = filtered_df[filtered_df['Status'] != 'Approved']

if not rejected_df.empty:
    rechazos_data = []
    medios_unicos = rejected_df['Medio_de_pago'].unique()
    
    for medio in medios_unicos:
        abril_rechazos = rejected_df[(rejected_df['mes'] == 'abril') & (rejected_df['Medio_de_pago'] == medio)]['Incoming_amt'].sum()
        mayo_rechazos = rejected_df[(rejected_df['mes'] == 'mayo') & (rejected_df['Medio_de_pago'] == medio)]['Incoming_amt'].sum()
        
        abril_total = rejected_df[rejected_df['mes'] == 'abril']['Incoming_amt'].sum()
        mayo_total = rejected_df[rejected_df['mes'] == 'mayo']['Incoming_amt'].sum()
        
        abril_pct = (abril_rechazos / abril_total * 100) if abril_total > 0 else 0
        mayo_pct = (mayo_rechazos / mayo_total * 100) if mayo_total > 0 else 0
        
        variacion = mayo_pct - abril_pct
        
        rechazos_data.append({
            'Medio de Pago': medio,
            'Abril (%)': f"{abril_pct:.1f}%",
            'Mayo (%)': f"{mayo_pct:.1f}%",
            'Variación': f"{variacion:+.1f}%"
        })
    
    if rechazos_data:
        rechazos_df = pd.DataFrame(rechazos_data)

        # Ordenar de mayor a menor por la columna 'Mayo (%)'
        rechazos_df['Mayo_num'] = rechazos_df['Mayo (%)'].str.replace('%', '').astype(float)
        rechazos_df = rechazos_df.sort_values('Mayo_num', ascending=False).reset_index(drop=True)
        rechazos_df = rechazos_df.drop(columns=['Mayo_num'])

        # Resaltar en negrita los valores de variación mayores al 1%
        def highlight_variacion(val):
            try:
                num = float(val.replace('%', '').replace('+', ''))
                if abs(num) > 1:
                    return f"<b>{val}</b>"
            except Exception:
                pass
            return val

        rechazos_df['Variación'] = rechazos_df['Variación'].apply(highlight_variacion)

        # Mostrar como tabla HTML dentro de un contenedor estilizado
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


#creo un grafico que me muestre el share de status

st.subheader("📊 Status de Compras No Aprobadas")

# Filtrar solo los no aprobados
not_approved_df = filtered_df[(filtered_df['Status'] != 'Approved') & (filtered_df['Status'] != 'Pending')]

if not not_approved_df.empty:
    # Agrupar por mes y status, sumar Incoming_amt
    status_share = (
        not_approved_df.groupby(['mes', 'Status'])['Incoming_amt']
        .sum()
        .reset_index()
    )
    # Calcular el total de Incoming_amt no aprobado por mes
    total_by_mes = (
        not_approved_df.groupby('mes')['Incoming_amt']
        .sum()
        .reset_index()
        .rename(columns={'Incoming_amt': 'Total_mes'})
    )
    # Merge para calcular el porcentaje
    status_share = status_share.merge(total_by_mes, on='mes')
    status_share['Porcentaje'] = (status_share['Incoming_amt'] / status_share['Total_mes']) * 100


    color_palette = [
        "#ffe5cc",  # muy claro, naranja pastel
        "#ffd8b3",  # naranja muy suave
        "#ffcc99",  # naranja claro
        "#ffb380",  # naranja pastel medio
        "#ff9966",  # naranja suave
        "#e6a87c",  # marrón claro opaco
        "#d9a066",  # marrón claro, formal
        "#bfa380",  # marrón grisáceo claro
        "#a67c52",  # marrón opaco, formal
        "#8c6e54"   # marrón grisáceo más oscuro
    ]

    fig = px.bar(
        status_share,
        x='mes',
        y='Porcentaje',
        color='Status',
        text=status_share['Porcentaje'].apply(lambda x: f"{x:.1f}%"),
        barmode='stack',
        color_discrete_sequence=color_palette,
        category_orders={'mes': ['abril', 'mayo']}
    )
    fig.update_layout(
        yaxis=dict(
            title='Porcentaje (%)',
            range=[0, 100],
            gridcolor='lightgray',  # Líneas horizontales grises
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

else:
    st.info("No hay compras no aprobadas para mostrar.")