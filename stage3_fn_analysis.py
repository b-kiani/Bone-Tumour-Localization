#!/usr/bin/env python
"""
Stage-3: Radiologist-style false-negative analysis
"""

import argparse
import pandas as pd

FN_CATEGORIES = [
    "Low contrast",
    "Small lesion",
    "Overlapping anatomy",
    "Rare morphology",
    "Annotation ambiguity",
    "Confounder"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)
    fns = df[(df["gt"] == 1) & (df["iou"] < 0.3)]

    # Placeholder mapping (manual review normally)
    fns["FN_category"] = np.random.choice(FN_CATEGORIES, size=len(fns))

    fns.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
