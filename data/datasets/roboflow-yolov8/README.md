# Dataset Roboflow exportado para YOLOv8

Aqui debe ir el dataset exportado desde Roboflow en formato YOLOv8.

Estructura esperada:

```txt
data/datasets/roboflow-yolov8/
  data.yaml
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
```

## Pasos en Roboflow

1. Abrir el proyecto etiquetado en Roboflow.
2. Ir a `Versions` y generar una version del dataset.
3. Elegir `Export Dataset`.
4. Seleccionar formato `YOLOv8`.
5. Descargar como ZIP y descomprimir el contenido en esta carpeta.
6. Verificar que `data.yaml` apunte a `train/images`, `valid/images` y `test/images`.

Las imagenes y labels generadas por Roboflow pueden ser pesadas; por eso no se versionan por defecto. Si el dataset pesa mucho, subir el ZIP o la carpeta exportada a Drive y compartir el enlace en el issue.
