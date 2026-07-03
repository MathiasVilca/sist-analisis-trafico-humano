"""
llm_reporter.py
----------------
Genera un reporte en lenguaje natural (resumen + recomendaciones) a partir
de outputs/metrics.json, usando un modelo LLaMA 3.2 corriendo localmente en Ollama.

Requisito: Ollama debe estar instalado y corriendo (`ollama serve`), y el
modelo debe estar descargado (`ollama pull llama3.2`).
"""

import json
from pathlib import Path

import ollama

JSON_ENTRADA   = "outputs/metrics.json"
REPORTE_SALIDA = "outputs/report.txt"
MODELO         = "llama3.2"


class OllamaNoDisponibleError(RuntimeError):
    """Se lanza cuando no se puede contactar al servidor de Ollama o falta el modelo."""
    pass


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
        f"- Vehículos únicos detectados: {metricas['total_detecciones']}\n"
        f"- Tipos: {tipos}\n"
        f"- Congestión: {metricas['nivel_congestion']}\n"
        f"- Minuto de mayor congestión: minuto {metricas['minuto_punta']}\n\n"
        f"Escribe en español un párrafo de resumen y dos recomendaciones "
        f"concretas para mejorar el flujo vehicular. Sé breve y directo."
    )


def llamar_llm(prompt, modelo=MODELO):
    print("[2/3] Generando reporte con Ollama (local)...")
    try:
        respuesta = ollama.chat(
            model=modelo,
            messages=[
                {"role": "system", "content": "Eres un analista de tráfico urbano en Lima, Perú."},
                {"role": "user",   "content": prompt}
            ]
        )
    except Exception as e:
        # Cubre: Ollama no está corriendo (connection refused) y modelo no descargado (404).
        raise OllamaNoDisponibleError(
            "No se pudo generar el reporte con Ollama. Verifica que:\n"
            "  1) Ollama esté corriendo (`ollama serve`)\n"
            f"  2) El modelo esté descargado (`ollama pull {modelo}`)\n"
            f"Detalle técnico: {e}"
        ) from e

    print("      Reporte generado.")
    return respuesta["message"]["content"]


def guardar_reporte(reporte, metricas, reporte_salida):
    print(f"[3/3] Guardando reporte en {reporte_salida}...")
    Path(reporte_salida).parent.mkdir(parents=True, exist_ok=True)
    encabezado = (
        "=" * 60 + "\n"
        "REPORTE DE ANÁLISIS DE TRÁFICO URBANO\n"
        f"Fuente  : {metricas['fuente_video']}\n"
        "Modelo  : LLaMA 3.2 (Ollama — local)\n"
        "Curso   : Computación Gráfica — UNI\n"
        "Equipo  : Dery Gonzales, Ariana Mercado, Mathias Vilca\n"
        "=" * 60 + "\n\n"
    )
    with open(reporte_salida, "w", encoding="utf-8") as f:
        f.write(encabezado)
        f.write(reporte)
    print(f"      Guardado en: {reporte_salida}")


def imprimir_reporte(reporte):
    print("\n" + "=" * 60)
    print("REPORTE GENERADO POR LLM (LLaMA 3.2 — Ollama local)")
    print("=" * 60)
    print(reporte)
    print("=" * 60)


# ─────────────────────────────────────────────
# FUNCIÓN REUTILIZABLE (usada por app.py)
# ─────────────────────────────────────────────

def generar_reporte_texto(metrics_path="outputs/metrics.json", carpeta_salida="outputs",
                           modelo=MODELO):
    """
    Genera el reporte de IA a partir de un metrics.json ya calculado.

    Retorna el texto completo del reporte (con encabezado incluido) y también
    lo guarda en <carpeta_salida>/report.txt.

    Lanza OllamaNoDisponibleError si Ollama no responde o falta el modelo,
    para que app.py pueda mostrar un mensaje claro en vez de un traceback.
    """
    reporte_salida = str(Path(carpeta_salida) / "report.txt")
    metricas = cargar_metricas(metrics_path)
    prompt   = construir_prompt(metricas)
    reporte  = llamar_llm(prompt, modelo=modelo)
    guardar_reporte(reporte, metricas, reporte_salida)
    return reporte


def main():
    reporte = generar_reporte_texto(JSON_ENTRADA, carpeta_salida="outputs")
    imprimir_reporte(reporte)
    print("\n✓ Listo. El reporte está en outputs/report.txt\n")


if __name__ == "__main__":
    main()
