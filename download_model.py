
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
