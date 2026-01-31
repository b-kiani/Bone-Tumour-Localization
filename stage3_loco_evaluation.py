#!/usr/bin/env python
"""
Stage-3: LOCO (Leave-One-Center-Out) evaluation
"""

import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)
    centers = df["center"].unique()
    records = []

    for center in centers:
        test_df = df[df["center"] == center]
        tp = ((test_df["iou"] >= 0.3) & (test_df["score"] >= 0.6)).sum()
        fn = (test_df["gt"] == 1).sum() - tp
        sens = tp / (tp + fn + 1e-8)

        records.append({
            "HeldOutCenter": center,
            "Sensitivity_FPPI_0.02": sens
        })

    pd.DataFrame(records).to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
