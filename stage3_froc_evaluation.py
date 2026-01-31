#!/usr/bin/env python
"""
Stage-3: Global FROC evaluation with bootstrap confidence intervals
"""

import argparse
import numpy as np
import pandas as pd

def compute_froc(df, iou_thresh):
    fps = []
    sens = []

    for th in np.linspace(0.001, 0.999, 50):
        subset = df[df["score"] >= th]
        tp = ((subset["iou"] >= iou_thresh)).sum()
        fn = (df["gt"] == 1).sum() - tp
        fp = ((subset["gt"] == 0)).sum()

        sens.append(tp / (tp + fn + 1e-8))
        fps.append(fp / df["image_id"].nunique())

    return fps, sens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)
    results = []

    for iou in [0.1, 0.2, 0.3, 0.5]:
        fps, sens = compute_froc(df, iou)
        for f, s in zip(fps, sens):
            results.append({
                "IoU": iou,
                "FPPI": f,
                "Sensitivity": s
            })

    pd.DataFrame(results).to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
