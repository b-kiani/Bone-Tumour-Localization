#!/usr/bin/env python
"""
Figure 6: Qualitative detection examples
"""

import argparse
import cv2
import os
import pandas as pd

def draw_box(img, row, color):
    x1, y1, x2, y2 = map(int, [row.x1, row.y1, row.x2, row.y2])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--img_root", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    examples = {
        "tp_clear": df[(df.gt == 1) & (df.iou > 0.5)].iloc[0],
        "tp_subtle": df[(df.gt == 1) & (df.score < 0.7)].iloc[0],
        "fn": df[(df.gt == 1) & (df.iou < 0.1)].iloc[0],
        "fp": df[(df.gt == 0) & (df.score > 0.8)].iloc[0],
    }

    for name, row in examples.items():
        img = cv2.imread(os.path.join(args.img_root, row.image))
        draw_box(img, row, (0, 255, 0) if "tp" in name else (0, 0, 255))
        cv2.imwrite(os.path.join(args.outdir, f"fig6_{name}.png"), img)

if __name__ == "__main__":
    main()
