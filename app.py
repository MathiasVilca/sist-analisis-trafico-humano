"""
app.py — Dashboard interactivo del Sistema de Análisis de Tráfico Urbano.

Flujo end-to-end:
  1. El usuario sube un video de tráfico.
  2. Se corre vision_pipeline.py (YOLOv8 + ByteTrack) sobre el video.
  3. Se corre metrics.py para calcular flujo vehicular y nivel de congestión.
  4. Se corre llm_reporter.py (Ollama / LLaMA 3.2) para generar recomendaciones.
  5. Se muestra todo: video anotado, gráfico de flujo interactivo y reporte de IA.
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / "scripts"
OUTPUTS_DIR = BASE_DIR / "outputs"
sys.path.insert(0, str(SCRIPTS_DIR))

import vision_pipeline   # noqa: E402
import metrics           # noqa: E402
import llm_reporter      # noqa: E402


st.set_page_config(
    page_title="Análisis de Tráfico Urbano — Lima",
    page_icon="🚦",
    layout="wide",
)

COLOR_NIVEL = {"Bajo": "#43A047", "Medio": "#FB8C00", "Alto": "#E53935"}


@st.cache_resource(show_spinner=False)
def cargar_modelo_yolo():
    """Carga YOLOv8 una sola vez por sesión de Streamlit (es lento y pesado)."""
    return vision_pipeline.cargar_modelo()


def guardar_video_temporal(archivo_subido) -> Path:
    """Guarda el video subido por el usuario en un archivo temporal en disco,
    ya que OpenCV necesita una ruta de archivo, no un objeto en memoria."""
    sufijo = Path(archivo_subido.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=sufijo)
    tmp.write(archivo_subido.read())
    tmp.close()
    return Path(tmp.name)


def construir_grafico_flujo(metricas: dict) -> go.Figure:
    """Gráfico de barras interactivo (Plotly) del flujo vehicular por segmentos."""
    flujo = metricas.get("flujo_por_segmento", {})
    segmentos, valores, colores, niveles = [], [], [], []

    for clave, datos in sorted(flujo.items(), key=lambda kv: int(kv[0].replace("segmento_", ""))):
        segmento = int(clave.replace("segmento_", ""))
        nivel = datos.get("congestion", "Bajo")
        inicio = segmento * metricas.get("intervalo_segundos", 15)
        fin = inicio + metricas.get("intervalo_segundos", 15)
        segmentos.append(f"{inicio}-{fin}s")
        valores.append(datos.get("detecciones", 0))
        colores.append(COLOR_NIVEL.get(nivel, "#43A047"))
        niveles.append(nivel)

    fig = go.Figure(
        go.Bar(
            x=segmentos, y=valores, marker_color=colores,
            text=valores, textposition="outside",
            customdata=niveles,
            hovertemplate="%{x}<br>Vehículos simultáneos: %{y}<br>Congestión: %{customdata}<extra></extra>",
        )
    )
    intervalo = metricas.get("intervalo_segundos", 15)
    fig.update_layout(
        title=f"Flujo vehicular por segmento de {intervalo} segundos",
        xaxis_title="Segmento de tiempo",
        yaxis_title="Vehículos simultáneos (promedio)",
        showlegend=False,
        margin=dict(t=50, b=10),
    )
    return fig


def ejecutar_pipeline_completo(ruta_video: Path, fuente_nombre: str, modelo):
    """Corre vision_pipeline → metrics → llm_reporter en secuencia, actualizando
    la interfaz con el progreso de cada etapa."""
    OUTPUTS_DIR.mkdir(exist_ok=True)

    estado = st.empty()
    barra = st.progress(0.0)

    # 1) Visión + tracking
    estado.info("🔍 Detectando y siguiendo vehículos con YOLOv8 + ByteTrack...")
    resultado_vision = vision_pipeline.procesar_video(
        ruta_video, carpeta_salida=OUTPUTS_DIR, modelo=modelo,
        progress_callback=lambda frac: barra.progress(frac * 0.7),
    )

    # 2) Métricas
    estado.info("📊 Calculando flujo vehicular y nivel de congestión...")
    barra.progress(0.8)
    metricas = metrics.generar_metricas(
        resultado_vision["csv_path"], carpeta_salida=OUTPUTS_DIR,
        fps_efectivos=resultado_vision["fps_efectivos"],
        fuente_video=fuente_nombre,
    )

    # 3) Reporte con IA (Ollama)
    estado.info("🧠 Generando recomendaciones con LLaMA 3.2 (Ollama local)...")
    barra.progress(0.9)
    try:
        reporte = llm_reporter.generar_reporte_texto(
            metricas["_json_path"], carpeta_salida=OUTPUTS_DIR
        )
    except llm_reporter.OllamaNoDisponibleError as e:
        reporte = None
        st.session_state["error_ollama"] = str(e)

    barra.progress(1.0)
    estado.success("✓ Procesamiento completo.")

    st.session_state["resultado_vision"] = resultado_vision
    st.session_state["metricas"] = metricas
    st.session_state["reporte"] = reporte


def render_resultados():
    """Muestra video anotado, métricas, gráfico interactivo y reporte de IA."""
    metricas = st.session_state["metricas"]
    resultado_vision = st.session_state["resultado_vision"]
    reporte = st.session_state.get("reporte")

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Vehículos únicos detectados", metricas["total_detecciones"])
    col2.metric("Nivel de congestión", metricas["nivel_congestion"])
    intervalo = metricas.get("intervalo_segundos", 15)
    segmento_punta = metricas.get("segmento_punta", 0)
    inicio = segmento_punta * intervalo
    fin = inicio + intervalo
    col3.metric("Segmento pico", f"{inicio}-{fin}s")

    st.divider()
    video_col, chart_col = st.columns([1, 1.2])

    with video_col:
        st.subheader("🎥 Video anotado")
        st.video(resultado_vision["video_path"])

    with chart_col:
        st.subheader("📈 Flujo vehicular por segmentos de tiempo")
        st.plotly_chart(construir_grafico_flujo(metricas), use_container_width=True)

        st.subheader("🚗 Distribución por tipo de vehículo")
        dist_df = pd.DataFrame([
            {"Tipo": tipo.capitalize(), "Vehículos": d["detecciones"], "Porcentaje": f"{d['porcentaje']}%"}
            for tipo, d in metricas["distribucion_tipos"].items()
        ])
        st.dataframe(dist_df, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("🧠 Reporte y recomendaciones (LLaMA 3.2 vía Ollama)")
    if reporte:
        st.markdown(reporte.split("=" * 60)[-1].strip() or reporte)
    else:
        st.warning(
            "No se pudo generar el reporte de IA porque Ollama no respondió. "
            "Verifica que esté corriendo (`ollama serve`) y que el modelo esté "
            "descargado (`ollama pull llama3.2`), luego vuelve a procesar el video."
        )
        if "error_ollama" in st.session_state:
            with st.expander("Detalle técnico del error"):
                st.code(st.session_state["error_ollama"])


def main():
    st.title("🚦 Sistema de Análisis de Tráfico Urbano — Lima")
    st.markdown(
        "Sube un video de tráfico. El sistema detecta y sigue vehículos con "
        "**YOLOv8 + ByteTrack**, calcula el flujo vehicular y genera "
        "recomendaciones con **LLaMA 3.2** corriendo localmente en Ollama."
    )

    archivo_subido = st.file_uploader(
        "Selecciona un video (mp4, avi, mov, mkv)",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if archivo_subido is not None:
        st.video(archivo_subido)
        procesar = st.button("▶️ Procesar video", type="primary")

        if procesar:
            ruta_temporal = guardar_video_temporal(archivo_subido)
            modelo = cargar_modelo_yolo()
            try:
                ejecutar_pipeline_completo(ruta_temporal, archivo_subido.name, modelo)
            finally:
                ruta_temporal.unlink(missing_ok=True)

    if "metricas" in st.session_state:
        render_resultados()
    elif archivo_subido is None:
        st.info("Sube un video para comenzar el análisis.")


if __name__ == "__main__":
    main()
