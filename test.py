from ultralytics import YOLO
import cv2
import pandas as pd

print("✓ ultralytics OK")
print("✓ opencv OK")
print("✓ pandas OK")

# con esto descargué YOLOv8n automáticamente la primera vez
model = YOLO("yolov8n.pt")
print("✓ YOLO modelo cargado OK")