"""
Plot long-context eval results produced by run_long_context.py.

Produces two figures for the manuscript's long-context section:
  1. NIAH depth x context-length accuracy heatmap (one panel per checkpoint/arch).
  2. Accuracy vs context length, per task, one line per checkpoint/arch -- the key
     plot for "does loss-equivalence survive as context grows?".

Usage:
    python plot_long_context.py results/*.json --outdir figures/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load(paths):
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def plot_niah_heatmaps(runs, outpath):
    runs = [r for r in runs if any(s["task"] == "niah_single" for s in r["scores"])]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4), squeeze=False)
    for ax, r in zip(axes[0], runs):
        pts = [s for s in r["scores"] if s["task"] == "niah_single"]
        lengths = sorted({s["context_length"] for s in pts})
        depths = sorted({s["depth"] for s in pts})
        grid = np.full((len(depths), len(lengths)), np.nan)
        for s in pts:
            i = depths.index(s["depth"]); j = lengths.index(s["context_length"])
            grid[i, j] = s["accuracy"]
        im = ax.imshow(grid, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn", origin="lower")
        ax.set_xticks(range(len(lengths))); ax.set_xticklabels([f"{l//1024}k" for l in lengths])
        ax.set_yticks(range(len(depths))); ax.set_yticklabels([f"{d:.2f}" for d in depths])
        ax.set_xlabel("Context length"); ax.set_ylabel("Needle depth")
        ax.set_title(r.get("label") or r["checkpoint"].split("/")[-1], fontsize=10)
    fig.colorbar(im, ax=axes[0].tolist(), label="Retrieval accuracy", fraction=0.025)
    fig.suptitle("Needle-in-a-Haystack: retrieval accuracy by depth and context length", y=1.02)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"wrote {outpath}")


def _wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (k successes of n)."""
    if n == 0:
        return (float("nan"), float("nan"))
    import math
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def plot_acc_vs_length(runs, outpath):
    # Collect tasks present across runs.
    tasks = sorted({s["task"] for r in runs for s in r["scores"]})
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.5 * len(tasks), 4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for r in runs:
            # Pool successes and samples per length (NIAH single aggregates 5 depths -> n=100).
            ksum, nsum = defaultdict(int), defaultdict(int)
            for s in r["scores"]:
                if s["task"] == task:
                    n = s.get("n", 20)
                    ksum[s["context_length"]] += round(s["accuracy"] * n)
                    nsum[s["context_length"]] += n
            if not nsum:
                continue
            xs = sorted(nsum)
            ys = np.array([ksum[x] / nsum[x] for x in xs])
            band = np.array([_wilson(ksum[x], nsum[x]) for x in xs])
            line, = ax.plot(xs, ys, marker="o", label=r.get("label") or r["checkpoint"].split("/")[-1])
            ax.fill_between(xs, band[:, 0], band[:, 1], color=line.get_color(), alpha=0.15, linewidth=0)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Context length (tokens)"); ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.02, 1.02); ax.set_title(task, fontsize=10); ax.grid(alpha=0.3)
        ax.axvline(8192, ls="--", color="gray", alpha=0.6)  # training seq len
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("Long-context accuracy vs context length (dashed = 8192 training length; "
                 "bands = Wilson 95% CI)", y=1.02)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"wrote {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="result JSON files")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    runs = load(args.results)
    plot_niah_heatmaps(runs, f"{args.outdir}/fig_niah_heatmap.png")
    plot_acc_vs_length(runs, f"{args.outdir}/fig_longctx_acc_vs_length.png")


if __name__ == "__main__":
    main()
