"""
train_yolo11.py

Usage:
    python train_yolo11.py

This script:
- scans datasets/iue_yolo_ready for images/labels (train/val/test),
- builds a data.yaml (train/val/test paths, nc, names),
- loads a pretrained YOLOv11 model (by default yolo11n.pt) and starts training.
"""

import os
import argparse

# ---------- Config ----------
DEFAULT_DATA_DIR = "E:\PythonProjects\RecyclingRobot\Yolo\datasets\iue_yolo_ready"   # <-- change if needed
OUTPUT_DATA_YAML = os.path.join(DEFAULT_DATA_DIR, "data.yaml")
DEFAULT_MODEL = "yolo11s.pt"   # tiny / fast. other options: yolo11s.pt, yolo11m.pt, ...
EPOCHS = 50
IMG_SIZE = 640
BATCH = 15
DEVICE = "0"  # GPU device id or "cpu"
# ----------------------------

def main(args):
    from ultralytics import YOLO

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)  # load pretrained checkpoint (or path)
    print("Starting training...")
    model.train(data=args.output_yaml,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="YOLO dataset root (images/, labels/).")
    ap.add_argument("--output-yaml", default=OUTPUT_DATA_YAML, help="Path to write data.yaml")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Pretrained YOLOv11 checkpoint (e.g. yolo11n.pt)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--imgsz", type=int, default=IMG_SIZE)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--device", default=DEVICE, help="GPU id (e.g. '0') or 'cpu'")
    ap.add_argument("--exp-name", default="iue_yolo11_exp", help="Experiment name (results saved under runs/train/)")
    args = ap.parse_args()
    main(args)
