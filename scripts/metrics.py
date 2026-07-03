"""
metrics.py
----------
Calcula métricas de tráfico a partir del CSV generado por vision_pipeline.py.
Lee detections.csv y produce metrics.json con:
  - Total de vehículos únicos por tipo (usando track_id de ByteTrack)
  - Flujo vehicular por minuto (vehículos únicos presentes en promedio por frame)
  - Minuto de mayor congestión (hora punta)
  - Nivel de congestión general (Bajo / Medio / Alto)
  - Distribución porcentual por tipo de vehículo
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURACIÓN POR DEFECTO (usada solo si se ejecuta metrics.py directo)
# ─────────────────────────────────────────────

CSV_ENTRADA    = "outputs/detections.csv"
JSON_SALIDA    = "outputs/metrics.json"
GRAFICA_SALIDA = "outputs/flujo_por_intervalo.png"

# FPS efectivos por defecto (video_original_fps / SALTO_FRAMES).
# vision_pipeline.py calcula el valor real y app.py lo pasa explícitamente;
# este valor solo se usa como fallback si metrics.py se corre solo.
FPS_EFECTIVOS_DEFAULT = 6

# Duración del segmento para agrupar el flujo de congestión.
# Por defecto usamos 15 segundos para gráficos de congestión más finos.
INTERVALO_SEGUNDOS_DEFAULT = 15

# Umbrales para clasificar el nivel de congestión.
# IMPORTANTE: ahora que contamos vehículos ÚNICOS presentes por frame (gracias a
# ByteTrack) en vez de detecciones-con-duplicados-por-frame, estos números son
# muchísimo más bajos que en la v1. Son un punto de partida razonable para avenidas
# de 2-3 carriles; ajústalos con datos reales de tus videos si hace falta.
UMBRAL_BAJO  = 5    # menos de 5 vehículos únicos/frame en promedio → congestión baja
UMBRAL_MEDIO = 12   # entre 5 y 12                                  → congestión media
                     # 12 o más                                     → congestión alta


# ─────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────

def cargar_detecciones(ruta):
    """
    Carga el CSV generado por vision_pipeline.py.
    Verifica que tenga las columnas esperadas y descarta detecciones sin
    track_id asignado (track_id == -1), ya que no se pueden atribuir a un
    vehículo único.
    """
    print(f"[1/5] Cargando detecciones desde {ruta}...")
    df = pd.read_csv(ruta)

    columnas_esperadas = {"frame", "track_id", "tipo", "confianza", "x1", "y1", "x2", "y2"}
    if not columnas_esperadas.issubset(df.columns):
        raise ValueError(f"El CSV no tiene las columnas esperadas. Encontradas: {list(df.columns)}")

    df = df[df["track_id"] != -1]
    if df.empty:
        raise ValueError(
            "No se encontraron vehículos con track_id válido en el CSV. "
            "Verifica que vision_pipeline.py haya corrido con tracking activo."
        )

    print(f"      {len(df)} detecciones cargadas.")
    print(f"      Frames únicos: {df['frame'].nunique()}")
    return df


def calcular_distribucion(df):
    """
    Calcula la distribución de VEHÍCULOS ÚNICOS por tipo (un track_id cuenta una sola vez).
    """
    print("\n[2/5] Calculando distribución por tipo de vehículo...")
    conteo_df = df.drop_duplicates(subset=["track_id"])
    conteo = conteo_df["tipo"].value_counts()
    total  = len(conteo_df)

    distribucion = {}
    for tipo, cantidad in conteo.items():
        porcentaje = round(cantidad / total * 100, 1) if total else 0.0
        distribucion[tipo] = {
            "detecciones": int(cantidad),
            "porcentaje":  porcentaje
        }
        print(f"      {tipo:<8} {cantidad:>6} vehículos únicos  ({porcentaje:.1f}%)")

    return distribucion


def calcular_flujo_por_intervalo(df, fps_efectivos, intervalo_segundos=INTERVALO_SEGUNDOS_DEFAULT):
    """
    Agrupa las detecciones por intervalos del video y calcula,
    para cada segmento, el promedio de vehículos únicos presentes
    simultáneamente por frame.

    Cómo funciona:
    - frame / fps_efectivos = segundo del video
    - segundo / intervalo_segundos = índice de segmento
    - Por cada (segmento, frame) contamos track_id únicos presentes
    - Promediamos esos conteos dentro de cada segmento
    """
    print(f"\n[3/5] Calculando flujo vehicular por cada {intervalo_segundos} segundos...")
    if not fps_efectivos or fps_efectivos <= 0:
        fps_efectivos = FPS_EFECTIVOS_DEFAULT

    df = df.copy()
    df["segundo"] = df["frame"] / fps_efectivos
    df["intervalo"] = (df["segundo"] // intervalo_segundos).astype(int)

    vehiculos_por_frame = (
        df.groupby(["intervalo", "frame"])["track_id"]
          .nunique()
          .reset_index(name="vehiculos_presentes")
    )
    flujo = (
        vehiculos_por_frame.groupby("intervalo")["vehiculos_presentes"]
          .mean()
          .reset_index(name="detecciones")
    )
    flujo["detecciones"] = flujo["detecciones"].round().astype(int)

    for _, row in flujo.iterrows():
        nivel = clasificar_congestion(row["detecciones"])
        inicio = int(row["intervalo"] * intervalo_segundos)
        fin = inicio + intervalo_segundos
        print(
            f"      Segmento {int(row['intervalo'])} "
            f"({inicio}s - {fin}s): {int(row['detecciones'])} vehículos/frame (prom.) → {nivel}"
        )

    return flujo


def clasificar_congestion(vehiculos_por_frame):
    """
    Clasifica el nivel de congestión según el promedio de vehículos únicos
    presentes simultáneamente por frame en ese minuto.
    """
    if vehiculos_por_frame < UMBRAL_BAJO:
        return "Bajo"
    elif vehiculos_por_frame < UMBRAL_MEDIO:
        return "Medio"
    else:
        return "Alto"


def encontrar_intervalo_punta(flujo):
    """
    Identifica el segmento con mayor cantidad de vehículos simultáneos.
    """
    idx_max = flujo["detecciones"].idxmax()
    intervalo_punta = int(flujo.loc[idx_max, "intervalo"])
    detecciones_punta = int(flujo.loc[idx_max, "detecciones"])
    return intervalo_punta, detecciones_punta


def generar_grafica(flujo, grafica_salida, fuente_video="Video procesado",
                      intervalo_segundos=INTERVALO_SEGUNDOS_DEFAULT):
    """
    Genera una gráfica de barras del flujo vehicular por intervalo.
    Colorea las barras según el nivel de congestión.
    """
    print(f"\n[4/5] Generando gráfica de flujo por cada {intervalo_segundos} segundos...")

    colores = []
    for det in flujo["detecciones"]:
        nivel = clasificar_congestion(det)
        if nivel == "Alto":
            colores.append("#E53935")
        elif nivel == "Medio":
            colores.append("#FB8C00")
        else:
            colores.append("#43A047")

    xlabels = []
    for intervalo in flujo["intervalo"]:
        inicio = int(intervalo * intervalo_segundos)
        fin = inicio + intervalo_segundos
        xlabels.append(f"{inicio}-{fin}s")

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(flujo["intervalo"], flujo["detecciones"], color=colores, edgecolor="white", width=0.6)

    for bar, val in zip(bars, flujo["detecciones"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(int(val)), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(y=UMBRAL_BAJO,  color="#43A047", linestyle="--", alpha=0.5)
    ax.axhline(y=UMBRAL_MEDIO, color="#FB8C00", linestyle="--", alpha=0.5)

    ax.set_xlabel(f"Segmentos de {intervalo_segundos} segundos", fontsize=11)
    ax.set_ylabel("Vehículos simultáneos (prom.)", fontsize=11)
    ax.set_title(f"Flujo vehicular por segmento de {intervalo_segundos}s\n{fuente_video}",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(flujo["intervalo"])
    ax.set_xticklabels(xlabels)
    ax.set_ylim(0, max(flujo["detecciones"].max() * 1.3, 1))

    from matplotlib.patches import Patch
    leyenda = [
        Patch(color="#43A047", label="Congestión Baja"),
        Patch(color="#FB8C00", label="Congestión Media"),
        Patch(color="#E53935", label="Congestión Alta"),
    ]
    ax.legend(handles=leyenda, loc="upper right", fontsize=9)

    plt.tight_layout()
    Path(grafica_salida).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(grafica_salida, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Gráfica guardada en: {grafica_salida}")


def guardar_json(distribucion, flujo, intervalo_punta, detecciones_punta,
                  confianza_promedio, json_salida, fuente_video="Video procesado",
                  intervalo_segundos=INTERVALO_SEGUNDOS_DEFAULT):
    """
    Guarda todas las métricas en metrics.json. Este archivo es la entrada
    para llm_reporter.py.
    """
    print("\n[5/5] Guardando métricas en JSON...")

    flujo_dict = {}
    for _, row in flujo.iterrows():
        intervalo = int(row["intervalo"])
        dets      = int(row["detecciones"])
        flujo_dict[f"segmento_{intervalo}"] = {
            "detecciones": dets,
            "congestion":  clasificar_congestion(dets)
        }

    minuto_punta = int((intervalo_punta * intervalo_segundos) // 60)
    nivel_general = clasificar_congestion(detecciones_punta)

    metricas = {
        "fuente_video":         fuente_video,
        "total_detecciones":    int(sum(d["detecciones"] for d in distribucion.values())),
        "confianza_promedio":   round(float(confianza_promedio), 3),
        "distribucion_tipos":   distribucion,
        "flujo_por_segmento":   flujo_dict,
        "flujo_por_minuto":     {
            f"minuto_{minuto_punta}": {
                "detecciones": int(flujo["detecciones"].mean()),
                "congestion":  clasificar_congestion(int(flujo["detecciones"].mean()))
            }
        },
        "intervalo_segundos":   intervalo_segundos,
        "segmento_punta":       intervalo_punta,
        "minuto_punta":         minuto_punta,
        "detecciones_en_punta": detecciones_punta,
        "nivel_congestion":     nivel_general,
        "nota": (
            "El conteo usa vehículos únicos (track_id de ByteTrack), no "
            "detecciones brutas por frame."
        )
    }

    Path(json_salida).parent.mkdir(parents=True, exist_ok=True)
    with open(json_salida, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    print(f"      JSON guardado en: {json_salida}")
    return metricas


def imprimir_resumen(metricas):
    print("\n" + "="*50)
    print("RESUMEN DE MÉTRICAS")
    print("="*50)
    print(f"Fuente         : {metricas['fuente_video']}")
    print(f"Total vehículos: {metricas['total_detecciones']}")
    print(f"Confianza prom.: {metricas['confianza_promedio']:.1%}")
    print(f"Segmento punta : Segmento {metricas['segmento_punta']} "
          f"({metricas['intervalo_segundos']}s) — "
          f"{metricas['detecciones_en_punta']} vehículos simultáneos")
    print(f"Nivel congest. : {metricas['nivel_congestion']}")
    print()
    print("Distribución:")
    for tipo, datos in metricas["distribucion_tipos"].items():
        barra = "█" * int(datos["porcentaje"] / 3)
        print(f"  {tipo:<8} {datos['detecciones']:>6}  ({datos['porcentaje']:5.1f}%)  {barra}")
    print("="*50)


# ─────────────────────────────────────────────
# FUNCIÓN REUTILIZABLE (usada por app.py)
# ─────────────────────────────────────────────

def generar_metricas(csv_entrada, carpeta_salida="outputs", fps_efectivos=None,
                      intervalo_segundos=INTERVALO_SEGUNDOS_DEFAULT,
                      fuente_video="Video procesado"):
    """
    Pipeline completo de métricas a partir del CSV de detecciones.

    Parámetros
    ----------
    csv_entrada : str o Path — CSV generado por vision_pipeline.py.
    carpeta_salida : str o Path — carpeta donde se guardan metrics.json y la gráfica.
    fps_efectivos : FPS reales del muestreo (fps_original / salto_frames).
                    Si no se pasa, usa FPS_EFECTIVOS_DEFAULT (menos preciso).
    fuente_video : texto descriptivo para mostrar en el reporte/gráfica.

    Retorna
    -------
    dict con las métricas (el mismo que se guarda en metrics.json) + rutas de archivos.
    """
    carpeta_salida = Path(carpeta_salida)
    json_salida    = str(carpeta_salida / "metrics.json")
    grafica_salida = str(carpeta_salida / "flujo_por_intervalo.png")

    df = cargar_detecciones(csv_entrada)
    distribucion = calcular_distribucion(df)
    flujo = calcular_flujo_por_intervalo(df, fps_efectivos or FPS_EFECTIVOS_DEFAULT,
                                           intervalo_segundos=intervalo_segundos)
    intervalo_punta, detecciones_punta = encontrar_intervalo_punta(flujo)
    generar_grafica(flujo, grafica_salida, fuente_video,
                    intervalo_segundos=intervalo_segundos)

    metricas = guardar_json(
        distribucion, flujo, intervalo_punta, detecciones_punta,
        df["confianza"].mean(), json_salida, fuente_video,
        intervalo_segundos=intervalo_segundos,
    )
    imprimir_resumen(metricas)

    metricas["_json_path"] = json_salida
    metricas["_grafica_path"] = grafica_salida
    return metricas


def main():
    print("\n" + "="*50)
    print("SISTEMA DE ANÁLISIS DE TRÁFICO — LIMA")
    print("Metrics v1.0")
    print("="*50 + "\n")
    generar_metricas(
        CSV_ENTRADA,
        carpeta_salida="outputs",
        fps_efectivos=FPS_EFECTIVOS_DEFAULT,
        intervalo_segundos=INTERVALO_SEGUNDOS_DEFAULT,
    )
    print("\n✓ metrics.json listo para llm_reporter.py\n")


if __name__ == "__main__":
    main()
