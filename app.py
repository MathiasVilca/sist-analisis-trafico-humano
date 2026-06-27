import json
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
FLOW_IMAGE_PATH = OUTPUTS_DIR / "flujo_por_minuto.png"
REPORT_PATH = OUTPUTS_DIR / "report.txt"


st.set_page_config(
    page_title="Análisis de Tráfico Humano",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_metrics(path: Path) -> dict:
    """Carga las métricas previamente generadas en outputs/metrics.json."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_report(path: Path) -> str:
    """Carga el reporte textual previamente generado en outputs/report.txt."""
    return path.read_text(encoding="utf-8")


def format_percent(value: float) -> str:
    """Convierte valores decimales a porcentaje legible."""
    return f"{value * 100:.1f}%"


def build_distribution_dataframe(metrics: dict) -> pd.DataFrame:
    """Construye una tabla con la distribución por tipo de vehículo."""
    distribution = metrics.get("distribucion_tipos", {})
    rows = []

    for vehicle_type, data in distribution.items():
        rows.append(
            {
                "Tipo": vehicle_type.capitalize(),
                "Detecciones": data.get("detecciones", 0),
                "Porcentaje": f"{data.get('porcentaje', 0):.1f}%",
            }
        )

    return pd.DataFrame(rows)


def build_flow_dataframe(metrics: dict) -> pd.DataFrame:
    """Construye una tabla con el flujo vehicular por minuto."""
    flow = metrics.get("flujo_por_minuto", {})
    rows = []

    for minute_key, data in flow.items():
        minute_number = minute_key.replace("minuto_", "")
        rows.append(
            {
                "Minuto": int(minute_number) if minute_number.isdigit() else minute_key,
                "Detecciones": data.get("detecciones", 0),
                "Congestión": data.get("congestion", "Sin dato"),
            }
        )

    return pd.DataFrame(rows).sort_values("Minuto") if rows else pd.DataFrame()


def render_missing_file_message(path: Path) -> None:
    st.error(f"No se encontró el archivo requerido: `{path}`")
    st.info(
        "Ejecuta primero los scripts de generación de métricas/reporte o verifica "
        "que la carpeta `outputs/` contenga los archivos esperados."
    )


def render_sidebar(metrics: dict) -> None:
    st.sidebar.header("📊 Métricas")

    st.sidebar.metric(
        "Total de detecciones",
        f"{metrics.get('total_detecciones', 0):,}".replace(",", " "),
    )
    st.sidebar.metric(
        "Confianza promedio",
        format_percent(float(metrics.get("confianza_promedio", 0))),
    )
    st.sidebar.metric(
        "Nivel de congestión",
        metrics.get("nivel_congestion", "Sin dato"),
    )
    st.sidebar.metric(
        "Minuto punta",
        f"Minuto {metrics.get('minuto_punta', 'N/D')}",
        f"{metrics.get('detecciones_en_punta', 0)} detecciones",
    )

    st.sidebar.divider()
    st.sidebar.subheader("🚗 Distribución por tipo")

    distribution_df = build_distribution_dataframe(metrics)
    if distribution_df.empty:
        st.sidebar.warning("No hay datos de distribución disponibles.")
    else:
        st.sidebar.dataframe(distribution_df, hide_index=True, use_container_width=True)

    note = metrics.get("nota")
    if note:
        st.sidebar.divider()
        st.sidebar.caption(note)


def render_upload_section() -> None:
    st.subheader("🎥 Subir video")
    uploaded_video = st.file_uploader(
        "Selecciona un video para una futura ejecución del modelo",
        type=["mp4", "avi", "mov", "mkv"],
        help=(
            "En esta versión la app solo muestra datos previos de outputs/. "
            "El video subido todavía no se procesa con el modelo pesado."
        ),
    )

    if uploaded_video is not None:
        st.success(f"Video cargado visualmente: {uploaded_video.name}")
        st.video(uploaded_video)
    else:
        st.info("Aún no se ha subido un video. Se muestran los resultados previos disponibles.")


def render_main_panel(metrics: dict, report: str) -> None:
    st.header("📄 Reporte de análisis")

    source = metrics.get("fuente_video")
    if source:
        st.caption(f"Fuente de datos: {source}")

    summary_col_1, summary_col_2, summary_col_3 = st.columns(3)
    summary_col_1.metric("Detecciones", f"{metrics.get('total_detecciones', 0):,}".replace(",", " "))
    summary_col_2.metric("Congestión", metrics.get("nivel_congestion", "Sin dato"))
    summary_col_3.metric("Pico", f"Minuto {metrics.get('minuto_punta', 'N/D')}")

    st.divider()

    graph_col, flow_col = st.columns([1.25, 1])

    with graph_col:
        st.subheader("📈 Flujo por minuto")
        if FLOW_IMAGE_PATH.exists():
            st.image(str(FLOW_IMAGE_PATH), use_column_width=True)
        else:
            render_missing_file_message(FLOW_IMAGE_PATH)

    with flow_col:
        st.subheader("⏱️ Tabla de flujo")
        flow_df = build_flow_dataframe(metrics)
        if flow_df.empty:
            st.warning("No hay datos de flujo por minuto disponibles.")
        else:
            st.dataframe(flow_df, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("🧠 Reporte generado")
    st.text_area(
        "Contenido de outputs/report.txt",
        value=report,
        height=360,
        label_visibility="collapsed",
    )


def main() -> None:
    st.title("🚦 Sistema de Análisis de Tráfico Humano")
    st.markdown(
        "Interfaz visual para consultar métricas, flujo por minuto y reporte generado "
        "a partir de los archivos previos ubicados en `outputs/`."
    )

    required_files = [METRICS_PATH, REPORT_PATH]
    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        for path in missing_files:
            render_missing_file_message(path)
        st.stop()

    metrics = load_metrics(METRICS_PATH)
    report = load_report(REPORT_PATH)

    render_sidebar(metrics)

    upload_col, status_col = st.columns([1.2, 1])
    with upload_col:
        render_upload_section()

    with status_col:
        st.subheader("✅ Estado de datos")
        st.success("Métricas previas cargadas correctamente.")
        st.write(f"**Métricas:** `{METRICS_PATH.relative_to(BASE_DIR)}`")
        st.write(f"**Gráfica:** `{FLOW_IMAGE_PATH.relative_to(BASE_DIR)}`")
        st.write(f"**Reporte:** `{REPORT_PATH.relative_to(BASE_DIR)}`")

    st.divider()
    render_main_panel(metrics, report)


if __name__ == "__main__":
    main()