import json
import ollama
from pathlib import Path

JSON_ENTRADA   = "outputs/metrics.json"
REPORTE_SALIDA = "outputs/report.txt"
MODELO         = "llama3.2"

def cargar_metricas(ruta):
    print(f"[1/3] Cargando métricas desde {ruta}...")
    with open(ruta, "r", encoding="utf-8") as f:
        metricas = json.load(f)
    print("      Métricas cargadas.")
    return metricas

def construir_prompt(metricas):
    tipos = ", ".join([
        f"{t} {d['porcentaje']}%"
        for t, d in metricas["distribucion_tipos"].items()
    ])
    return (
        f"Analiza este tráfico en Lima, Perú ({metricas['fuente_video']}):\n"
        f"- Vehículos detectados: {metricas['total_detecciones']}\n"
        f"- Tipos: {tipos}\n"
        f"- Congestión: {metricas['nivel_congestion']}\n"
        f"- Minuto de mayor congestión: minuto {metricas['minuto_punta']}\n\n"
        f"Escribe en español un párrafo de resumen y dos recomendaciones "
        f"concretas para mejorar el flujo vehicular. Sé breve y directo."
    )

def llamar_llm(prompt):
    print("[2/3] Generando reporte con Ollama (local)...")
    respuesta = ollama.chat(
        model=MODELO,
        messages=[
            {"role": "system", "content": "Eres un analista de tráfico urbano en Lima, Perú."},
            {"role": "user",   "content": prompt}
        ]
    )
    print("      Reporte generado.")
    return respuesta["message"]["content"]

def guardar_reporte(reporte, metricas):
    print(f"[3/3] Guardando reporte en {REPORTE_SALIDA}...")
    Path("outputs").mkdir(exist_ok=True)
    encabezado = (
        "=" * 60 + "\n"
        "REPORTE DE ANÁLISIS DE TRÁFICO URBANO\n"
        f"Fuente  : {metricas['fuente_video']}\n"
        "Modelo  : LLaMA 3.2 (Ollama — local)\n"
        "Curso   : Computación Gráfica — UNI\n"
        "Equipo  : Dery Gonzales, Ariana Mercado, Mathias Vilca\n"
        "=" * 60 + "\n\n"
    )
    with open(REPORTE_SALIDA, "w", encoding="utf-8") as f:
        f.write(encabezado)
        f.write(reporte)
    print(f"      Guardado en: {REPORTE_SALIDA}")

def imprimir_reporte(reporte):
    print("\n" + "=" * 60)
    print("REPORTE GENERADO POR LLM (LLaMA 3.2 — Ollama local)")
    print("=" * 60)
    print(reporte)
    print("=" * 60)

def main():
    metricas = cargar_metricas(JSON_ENTRADA)
    prompt   = construir_prompt(metricas)
    reporte  = llamar_llm(prompt)
    guardar_reporte(reporte, metricas)
    imprimir_reporte(reporte)
    print("\n✓ Listo. El reporte está en outputs/report.txt\n")

if __name__ == "__main__":
    main()