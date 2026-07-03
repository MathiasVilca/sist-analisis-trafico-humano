import cv2
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

VIDEO_ENTRADA  = "data/videos/lima_ovalo_higuereta.mp4"
VIDEO_SALIDA   = "outputs/video_anotado.mp4"
CSV_SALIDA     = "outputs/detections.csv"

# Clases de COCO que nos interesan (índice : nombre)
# YOLO usa los índices del dataset COCO — estos son los vehículos
CLASES_VEHICULO = {
    2:  "auto",
    3:  "moto",
    5:  "bus",
    7:  "camion"
}

# Colores por tipo de vehículo para los bounding boxes (BGR)
COLORES = {
    "auto":   (0, 200, 0),    # verde
    "moto":   (0, 165, 255),  # naranja
    "bus":    (255, 50, 50),  # azul
    "camion": (0, 0, 220)     # rojo
}

# Confianza mínima para aceptar una detección (0.0 a 1.0)
CONFIANZA_MIN = 0.4

# Procesar 1 de cada N frames (5 = ~6fps si el video es 30fps)
# Reduce el tiempo de procesamiento sin perder mucha información
SALTO_FRAMES = 5


# ─────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────

def cargar_modelo():
    """
    Carga YOLOv8n (nano) preentrenado en COCO.
    La primera vez descarga el modelo (~6MB) automáticamente.
    Usamos 'nano' porque es el más rápido y suficiente para este proyecto.
    Versiones más grandes: yolov8s, yolov8m, yolov8l (más precisas, más lentas)
    """
    print("[1/4] Cargando modelo YOLOv8...")
    modelo = YOLO("yolov8n.pt")
    print("      Modelo cargado correctamente.")
    return modelo


def abrir_video(ruta):
    """
    Abre el video con OpenCV y devuelve el objeto VideoCapture.
    También retorna el total de frames y los FPS originales.
    """
    cap = cv2.VideoCapture(ruta)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {ruta}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    ancho        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[2/4] Video abierto: {ruta}")
    print(f"      Resolución : {ancho}x{alto}")
    print(f"      FPS        : {fps:.1f}")
    print(f"      Frames     : {total_frames}")
    print(f"      Duración   : {total_frames/fps:.1f} segundos")

    return cap, fps, ancho, alto, total_frames


def preparar_salida(ruta, fps, ancho, alto, salto_frames=SALTO_FRAMES):
    """
    Prepara el VideoWriter para guardar el video anotado.
    Usamos mp4v como codec — compatible con Windows.
    """
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(ruta, fourcc, fps / salto_frames, (ancho, alto))
    return writer


def detectar_vehiculos(modelo, frame):
    """
    Corre YOLOv8 sobre un frame y filtra solo los vehículos que nos interesan.
    Retorna lista de detecciones: cada una es un dict con tipo, confianza y bbox.
    
    La bbox (bounding box) es [x1, y1, x2, y2] — esquina superior izquierda
    y esquina inferior derecha del rectángulo que rodea al vehículo.
    """
    # model() retorna una lista de resultados, uno por imagen procesada
    resultados = modelo.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0] #ahora usa model.track con bytetracker ByteTrack
    detecciones = []

    for box in resultados.boxes:
        clase_id   = int(box.cls[0])
        confianza  = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else -1 #Se guarda una id si la tiene

        # Ignorar si no es un vehículo de interés o confianza muy baja
        if clase_id not in CLASES_VEHICULO:
            continue
        if confianza < CONFIANZA_MIN:
            continue

        tipo = CLASES_VEHICULO[clase_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detecciones.append({
            "track_id":   track_id, #se guarda id
            "tipo":       tipo,
            "confianza":  round(confianza, 3),
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2
        })

    return detecciones


def anotar_frame(frame, detecciones, num_frame, conteo_total):
    """
    Dibuja los bounding boxes y etiquetas sobre el frame.
    También muestra el conteo acumulado por tipo en la esquina superior.
    """
    for det in detecciones:
        tipo      = det["tipo"]
        confianza = det["confianza"]
        track_id = det["track_id"]
        color     = COLORES[tipo]
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Etiqueta con tipo y confianza
        etiqueta = f"{tipo} #{track_id} {confianza:.0%}"
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(etiqueta) * 9, y1), color, -1)
        cv2.putText(frame, etiqueta, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Panel de conteo en esquina superior izquierda
    panel_y = 25
    cv2.rectangle(frame, (5, 5), (220, 120), (0, 0, 0), -1)
    cv2.putText(frame, f"Frame: {num_frame}", (10, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    panel_y += 22
    for tipo, set_ids in conteo_total.items():
        color = COLORES[tipo]
        cv2.putText(frame, f"{tipo}: {len(set_ids)}", (10, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        panel_y += 22

    return frame


def guardar_csv(registros, csv_salida):
    """
    Guarda todas las detecciones en un CSV estructurado.
    Este CSV es la entrada para metrics.py y llm_reporter.py.
    
    Columnas: frame, tipo, confianza, x1, y1, x2, y2
    """
    Path(csv_salida).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(registros)
    df.to_csv(csv_salida, index=False)
    print(f"\n      CSV guardado en: {csv_salida}")
    print(f"      Total de detecciones registradas: {len(df)}")
    return df


def imprimir_resumen(df, total_frames_procesados):
    """
    Imprime un resumen de las detecciones al finalizar el procesamiento.
    """
    print("\n" + "="*50)
    print("RESUMEN DE DETECCIONES")
    print("="*50)
    print(f"Frames procesados : {total_frames_procesados}")
    print(f"Total detecciones : {len(df)}")
    print()
    print("Distribución por tipo de vehículo:")
    conteo = df["tipo"].value_counts()
    total  = len(df)
    for tipo, cantidad in conteo.items():
        porcentaje = cantidad / total * 100
        barra = "█" * int(porcentaje / 3)
        print(f"  {tipo:<8} {cantidad:>5}  ({porcentaje:5.1f}%)  {barra}")
    print()
    print(f"Confianza promedio: {df['confianza'].mean():.1%}")
    print("="*50)


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL (reutilizable desde app.py o CLI)
# ─────────────────────────────────────────────

def procesar_video(ruta_video, carpeta_salida="outputs", modelo=None,
                    salto_frames=SALTO_FRAMES, progress_callback=None):
    """
    Ejecuta el pipeline completo de visión sobre un video:
    detección + tracking (ByteTrack) + anotación + export a CSV.

    Parámetros
    ----------
    ruta_video : str o Path — video de entrada (mp4/avi/mov).
    carpeta_salida : str o Path — carpeta donde se guardan video_anotado.mp4 y detections.csv.
    modelo : instancia de YOLO ya cargada (opcional). Si no se pasa, se carga una nueva
             (en Streamlit conviene cargarla una sola vez con @st.cache_resource y reusarla).
    salto_frames : cada cuántos frames se procesa (5 = ~6 FPS si el video es 30 FPS).
    progress_callback : función opcional callback(fraccion_0_a_1) para reportar avance
                         (usada por app.py para actualizar una barra de progreso).

    Retorna
    -------
    dict con: csv_path, video_path, fps_original, fps_efectivos, frames_procesados, df
    """
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    video_salida = str(carpeta_salida / "video_anotado.mp4")
    csv_salida   = str(carpeta_salida / "detections.csv")

    if modelo is None:
        modelo = cargar_modelo()

    cap, fps, ancho, alto, total_frames = abrir_video(str(ruta_video))
    writer = preparar_salida(video_salida, fps, ancho, alto, salto_frames)

    fps_efectivos = fps / salto_frames if fps else 0

    print(f"\n[3/4] Procesando video (1 de cada {salto_frames} frames)...")

    registros    = []
    conteo_total = {"auto": set(), "moto": set(), "bus": set(), "camion": set()}
    num_frame    = 0
    frames_procesados = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        num_frame += 1
        if num_frame % salto_frames != 0:
            continue

        frames_procesados += 1
        detecciones = detectar_vehiculos(modelo, frame)

        for det in detecciones:
            tipo = det["tipo"]
            track_id = det["track_id"]
            if track_id != -1:
                conteo_total[tipo].add(track_id)
            registros.append({
                "frame": num_frame,
                "track_id": track_id,
                "tipo": tipo,
                "confianza": det["confianza"],
                "x1": det["x1"], "y1": det["y1"],
                "x2": det["x2"], "y2": det["y2"],
            })

        frame_anotado = anotar_frame(frame, detecciones, num_frame, conteo_total)
        writer.write(frame_anotado)

        if total_frames > 0 and progress_callback is not None:
            progress_callback(min(num_frame / total_frames, 1.0))

        if frames_procesados % 30 == 0:
            progreso = num_frame / total_frames * 100 if total_frames else 0
            total_detectados = sum(len(s) for s in conteo_total.values())
            print(f"  Frame {num_frame:>5}/{total_frames}  ({progreso:5.1f}%)  "
                  f"Vehículos únicos hasta ahora: {total_detectados}")

    cap.release()
    writer.release()

    df = guardar_csv(registros, csv_salida)
    imprimir_resumen(df, frames_procesados)

    if progress_callback is not None:
        progress_callback(1.0)

    return {
        "csv_path": csv_salida,
        "video_path": video_salida,
        "fps_original": fps,
        "fps_efectivos": fps_efectivos,
        "frames_procesados": frames_procesados,
        "df": df,
    }


def main():
    print("\n" + "="*50)
    print("SISTEMA DE ANÁLISIS DE TRÁFICO — LIMA")
    print("Vision Pipeline v1.0")
    print("="*50 + "\n")

    resultado = procesar_video(VIDEO_ENTRADA, carpeta_salida="outputs")

    print(f"\n✓ Video anotado guardado en : {resultado['video_path']}")
    print(f"✓ CSV guardado en           : {resultado['csv_path']}")
    print(f"✓ FPS efectivos             : {resultado['fps_efectivos']:.2f}")
    print("\nListo. Ahora puedes correr metrics.py\n")


if __name__ == "__main__":
    main()
