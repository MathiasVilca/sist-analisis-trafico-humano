"""
metrics.py
----------
Calcula métricas de tráfico a partir del CSV generado por vision_pipeline.py.
Lee detections.csv y produce metrics.json con:
  - Total de detecciones por tipo
  - Flujo vehicular por minuto
  - Minuto de mayor congestión (hora punta)
  - Nivel de congestión general (Bajo / Medio / Alto)
  - Distribución porcentual por tipo de vehículo
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

CSV_ENTRADA  = "outputs/detections.csv"
JSON_SALIDA  = "outputs/metrics.json"
GRAFICA_SALIDA = "outputs/flujo_por_minuto.png"

# FPS efectivos del pipeline (video original / SALTO_FRAMES)
FPS_EFECTIVOS = 30

# Umbrales para clasificar el nivel de congestión
# (detecciones por minuto — recordar que incluyen duplicados por frame)
UMBRAL_BAJO  = 15   # menos de 200 det/min → congestión baja
UMBRAL_MEDIO = 30   # entre 200 y 500      → congestión media
                     # más de 500            → congestión alta


# ─────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────

def cargar_detecciones(ruta):
    """
    Carga el CSV generado por vision_pipeline.py.
    Verifica que tenga las columnas esperadas.
    """
    print(f"[1/5] Cargando detecciones desde {ruta}...")
    df = pd.read_csv(ruta)

    columnas_esperadas = {"frame", "track_id", "tipo", "confianza", "x1", "y1", "x2", "y2"}
    if not columnas_esperadas.issubset(df.columns):
        raise ValueError(f"El CSV no tiene las columnas esperadas. Encontradas: {list(df.columns)}")
    df = df[df["track_id"]!=-1] #solo cargamos 
    print(f"      {len(df)} detecciones cargadas.")
    print(f"      Frames únicos: {df['frame'].nunique()}")
    return df


def calcular_distribucion(df):
    """
    Calcula la distribución de detecciones por tipo de vehículo.
    Retorna un dict con cantidad y porcentaje por tipo.
    """
    print("\n[2/5] Calculando distribución por tipo de vehículo...")
    conteo_df = df.drop_duplicates(subset=['track_id']) #solo toma en cuenta la aparicion de un vehiculo
    conteo = conteo_df["tipo"].value_counts()
    total  = len(conteo_df)

    distribucion = {}
    for tipo, cantidad in conteo.items():
        porcentaje = round(cantidad / total * 100, 1)
        distribucion[tipo] = {
            "detecciones": int(cantidad),
            "porcentaje":  porcentaje
        }
        print(f"      {tipo:<8} {cantidad:>6} detecciones  ({porcentaje:.1f}%)")

    return distribucion


def calcular_flujo_por_minuto(df, fps_efectivos):
    """
    Agrupa las detecciones por minuto del video.
    
    Cómo funciona:
    - Cada fila del CSV tiene el número de frame
    - frame / fps_efectivos = segundo del video
    - segundo / 60 = minuto del video
    - Agrupamos detecciones por minuto para ver el flujo temporal
    """
    print("\n[3/5] Calculando flujo vehicular por minuto...")
    #revertido (error :P)
    df = df.copy()
    df["segundo"] = df["frame"] / fps_efectivos
    df["minuto"]  = (df["segundo"] // 60).astype(int)

    #ahora cuenta cuantos ids unicos existen por minuto
    autos_por_frame = df.groupby(['minuto', 'frame'])['track_id'].nunique().reset_index(name='autos_presentes')
    flujo = autos_por_frame.groupby('minuto')['autos_presentes'].mean().reset_index(name='detecciones')
    flujo['detecciones'] = flujo['detecciones'].astype(int)
    for _, row in flujo.iterrows():
        nivel = clasificar_congestion(row["detecciones"])
        print(f"      Minuto {int(row['minuto'])}: {int(row['detecciones'])} detecciones → {nivel}")

    return flujo


def clasificar_congestion(detecciones_por_minuto):
    """
    Clasifica el nivel de congestión según las detecciones por minuto.
    Los umbrales están calibrados para detecciones con duplicados por frame.
    """
    if detecciones_por_minuto < UMBRAL_BAJO:
        return "Bajo"
    elif detecciones_por_minuto < UMBRAL_MEDIO:
        return "Medio"
    else:
        return "Alto"


def encontrar_hora_punta(flujo):
    """
    Identifica el minuto con mayor cantidad de detecciones.
    Ese es el momento de mayor congestión en el video.
    """
    idx_max = flujo["detecciones"].idxmax()
    minuto_punta = int(flujo.loc[idx_max, "minuto"])
    detecciones_punta = int(flujo.loc[idx_max, "detecciones"])
    return minuto_punta, detecciones_punta


def generar_grafica(flujo):
    """
    Genera una gráfica de barras del flujo vehicular por minuto.
    Colorea las barras según el nivel de congestión.
    """
    print("\n[4/5] Generando gráfica de flujo por minuto...")

    colores = []
    for det in flujo["detecciones"]:
        nivel = clasificar_congestion(det)
        if nivel == "Alto":
            colores.append("#E53935")   # rojo
        elif nivel == "Medio":
            colores.append("#FB8C00")   # naranja
        else:
            colores.append("#43A047")   # verde

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(flujo["minuto"], flujo["detecciones"], color=colores, edgecolor="white", width=0.6)

    # Etiquetas sobre cada barra
    for bar, val in zip(bars, flujo["detecciones"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(int(val)), ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Líneas de umbral
    ax.axhline(y=UMBRAL_BAJO,  color="#43A047", linestyle="--", alpha=0.5, label=f"Umbral Bajo ({UMBRAL_BAJO})")
    ax.axhline(y=UMBRAL_MEDIO, color="#FB8C00", linestyle="--", alpha=0.5, label=f"Umbral Medio ({UMBRAL_MEDIO})")

    ax.set_xlabel("Minuto del video", fontsize=11)
    ax.set_ylabel("Detecciones", fontsize=11)
    ax.set_title("Flujo vehicular por minuto\nÓvalo Higuereta — Lima (5 PM)", fontsize=12, fontweight="bold")
    ax.set_xticks(flujo["minuto"])
    ax.set_xticklabels([f"Min {m}" for m in flujo["minuto"]])
    ax.legend(fontsize=9)
    ax.set_ylim(0, flujo["detecciones"].max() * 1.2)

    # Leyenda de colores
    from matplotlib.patches import Patch
    leyenda = [
        Patch(color="#43A047", label="Congestión Baja"),
        Patch(color="#FB8C00", label="Congestión Media"),
        Patch(color="#E53935", label="Congestión Alta"),
    ]
    ax.legend(handles=leyenda, loc="upper right", fontsize=9)

    plt.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig(GRAFICA_SALIDA, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Gráfica guardada en: {GRAFICA_SALIDA}")


def guardar_json(distribucion, flujo, minuto_punta, detecciones_punta, confianza_promedio):
    """
    Guarda todas las métricas en metrics.json.
    Este archivo es la entrada para llm_reporter.py.
    """
    print("\n[5/5] Guardando métricas en JSON...")

    flujo_dict = {}
    for _, row in flujo.iterrows():
        minuto = int(row["minuto"])
        dets   = int(row["detecciones"])
        flujo_dict[f"minuto_{minuto}"] = {
            "detecciones":  dets,
            "congestion":   clasificar_congestion(dets)
        }

    nivel_general = clasificar_congestion(detecciones_punta)

    metricas = {
        "fuente_video":         "Óvalo Higuereta, Lima — 5 PM (hora punta)",
        "total_detecciones":    int(sum(d["detecciones"] for d in distribucion.values())),
        "confianza_promedio":   round(float(confianza_promedio), 3),
        "distribucion_tipos":   distribucion,
        "flujo_por_minuto":     flujo_dict,
        "minuto_punta":         minuto_punta,
        "detecciones_en_punta": detecciones_punta,
        "nivel_congestion":     nivel_general,
        "nota": (
            "Las detecciones incluyen duplicados por frame (mismo vehículo "
            "aparece en múltiples frames). ByteTrack para conteo único se "
            "implementa en la siguiente entrega."
        )
    }

    Path("outputs").mkdir(exist_ok=True)
    with open(JSON_SALIDA, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    print(f"      JSON guardado en: {JSON_SALIDA}")
    return metricas


def imprimir_resumen(metricas):
    print("\n" + "="*50)
    print("RESUMEN DE MÉTRICAS")
    print("="*50)
    print(f"Fuente         : {metricas['fuente_video']}")
    print(f"Total detecc.  : {metricas['total_detecciones']}")
    print(f"Confianza prom.: {metricas['confianza_promedio']:.1%}")
    print(f"Minuto punta   : Minuto {metricas['minuto_punta']} "
          f"({metricas['detecciones_en_punta']} detecciones)")
    print(f"Nivel congest. : {metricas['nivel_congestion']}")
    print()
    print("Distribución:")
    for tipo, datos in metricas["distribucion_tipos"].items():
        barra = "█" * int(datos["porcentaje"] / 3)
        print(f"  {tipo:<8} {datos['detecciones']:>6}  ({datos['porcentaje']:5.1f}%)  {barra}")
    print("="*50)
    print(f"\n✓ metrics.json listo para llm_reporter.py\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*50)
    print("SISTEMA DE ANÁLISIS DE TRÁFICO — LIMA")
    print("Metrics v1.0")
    print("="*50 + "\n")

    # 1. Cargar CSV
    df = cargar_detecciones(CSV_ENTRADA)

    # 2. Distribución por tipo
    distribucion = calcular_distribucion(df)

    # 3. Flujo por minuto
    flujo = calcular_flujo_por_minuto(df, FPS_EFECTIVOS)

    # 4. Hora punta
    minuto_punta, detecciones_punta = encontrar_hora_punta(flujo)
    print(f"\n      → Hora punta: Minuto {minuto_punta} "
          f"({detecciones_punta} detecciones, nivel: {clasificar_congestion(detecciones_punta)})")

    # 5. Gráfica
    generar_grafica(flujo)

    # 6. Guardar JSON
    metricas = guardar_json(
        distribucion, flujo,
        minuto_punta, detecciones_punta,
        df["confianza"].mean()
    )

    # 7. Resumen
    imprimir_resumen(metricas)


if __name__ == "__main__":
    main()
