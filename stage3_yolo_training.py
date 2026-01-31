#!/usr/bin/env python
"""
Stage-3: YOLOv8 training for primary bone tumor localization
"""

import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_yaml", required=True,
                        help="YOLO dataset yaml")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default="stage3_yolo_runs")
    parser.add_argument("--name", default="btxrd_detection")
    args = parser.parse_args()

    cmd = [
        "yolo", "detect", "train",
        f"data={args.data_yaml}",
        f"model={args.weights}",
        f"imgsz={args.imgsz}",
        f"epochs={args.epochs}",
        f"batch={args.batch}",
        f"project={args.project}",
        f"name={args.name}",
        "optimizer=AdamW",
        "lr0=0.0005",
        "cos_lr=True",
        "patience=25",
        "box=7.5",
        "cls=0.5",
        "dfl=1.5",
        "amp=True"
    ]

    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
