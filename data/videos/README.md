# Videos de prueba

Carpeta usada para guardar los videos cortos de trafico en Lima solicitados en el issue #3.

## Videos locales

Los archivos MP4 no se suben a Git porque pueden ser pesados. Actualmente se tienen estos videos en `data/videos/`:

| Archivo | Tamano aproximado | Zona |
| --- | ---: | --- |
| `Av-Javier-Prado-La-Victoria-Lima-Peru.mp4` | 10.8 MB | Av. Javier Prado / La Victoria |
| `La-Via-mas-peligrosa-de-Lima-Via-de-Evit.mp4` | 22.6 MB | Via de Evitamiento / Panamericana |
| `lima_ovalo_higuereta.mp4` | 49.3 MB | Ovalo Higuereta |

Drive = https://drive.google.com/drive/folders/1b3MT8F8GIlYvBiYq6GTokvpm7Ud1xyMr?usp=sharing

## Uso

El pipeline actual usa por defecto:

```txt
data/videos/lima_ovalo_higuereta.mp4
```

Para procesar otro video, cambiar `VIDEO_ENTRADA` en `scripts/vision_pipeline.py`.
