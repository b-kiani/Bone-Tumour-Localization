#!/usr/bin/env python
"""
Figure 7: False-negative category distribution
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fn_csv", required=True)
    parser.add_argument("--out_png", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.fn_csv)
    counts = df["FN_category"].value_counts()

    plt.figure(figsize=(7,5))
    counts.plot(kind="bar")
    plt.ylabel("Number of FN cases")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
