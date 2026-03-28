<<<<<<< HEAD

import os, sys

dest = os.path.join(os.path.dirname(__file__), "yolov8n.onnx")
if os.path.exists(dest):
    print(f"✓ yolov8n.onnx already exists ({os.path.getsize(dest)//1024} KB) — nothing to do.")
    sys.exit(0)

try:
    from ultralytics import YOLO
    print("Downloading yolov8n.pt and exporting to ONNX...")
    model = YOLO("yolov8n.pt")          # downloads ~6 MB .pt from ultralytics CDN
    model.export(format="onnx", imgsz=640, simplify=True)
    # ultralytics exports to yolov8n.onnx in cwd
    if os.path.exists("yolov8n.onnx"):
        os.rename("yolov8n.onnx", dest)
    print(f"✓ Saved to {dest}")
except ImportError:
    print("ultralytics not installed. Run:  pip install ultralytics")
    print("Then re-run this script.")
    sys.exit(1)
=======
"""
Run this once to download and export the YOLOv8n model used by `solar_analysis.py`.

    python download_model.py

Requires: pip install ultralytics
"""

import os
import sys


ROOT = os.path.dirname(__file__)
MODELS_DIR = os.path.join(ROOT, "models")
ONNX_PATH = os.path.join(MODELS_DIR, "yolov8n.onnx")
PT_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.exists(ONNX_PATH):
        size_kb = os.path.getsize(ONNX_PATH) // 1024
        print(f"yolov8n.onnx already exists ({size_kb} KB); nothing to do.")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Run: pip install ultralytics")
        print("Then re-run this script.")
        return 1

    print("Downloading yolov8n.pt and exporting to ONNX...")
    model = YOLO("yolov8n.pt")
    model.export(format="onnx", imgsz=640, simplify=True)

    if os.path.exists("yolov8n.onnx"):
        os.replace("yolov8n.onnx", ONNX_PATH)
    if os.path.exists("yolov8n.pt") and not os.path.exists(PT_PATH):
        os.replace("yolov8n.pt", PT_PATH)

    print(f"Saved ONNX model to {ONNX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
>>>>>>> 66e2e02 (update struct)
