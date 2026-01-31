#!/usr/bin/env python
"""
Export training convergence plots
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_history(csv_path, out_path, ylabel):
    df = pd.read_csv(csv_path)
    plt.figure()
    plt.plot(df.iloc[:, 0])
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--tag", default="stage")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{args.tag}_training_curve.png")

    plot_history(args.history, out, ylabel="Loss / AUC")

if __name__ == "__main__":
    main()
