#!/usr/bin/env python3
"""Plot ADMM convergence from a training report (Fig. 3).

Left: role-local objectives f_i per round (geometric mean across agents with the
min-max range). Right: normalised primal residual
(sum_i ||x_i - z_i||^2)^(1/2) / ||x_1||. Both on log axes; the x axis is the
ADMM round, the colour is the cumulative communication per node.

    python tools/plot_convergence.py results/hmas-admm-train.json --out convergence.pdf
"""
import argparse
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--out", default="convergence.pdf")
    a = ap.parse_args()

    rep = json.load(open(a.report))
    rounds = rep["training"]["rounds"]
    per_node_bytes = rep["training"]["comm_bytes_per_round_per_node"]
    t = [r["round"] + 1 for r in rounds]
    gb = [per_node_bytes * k / 1e9 for k in t]
    fs = [[ag["f"] for ag in r["agents"] if ag.get("f")] for r in rounds]
    gm = [math.exp(sum(math.log(v) for v in f) / len(f)) if f else float("nan") for f in fs]
    lo = [min(f) if f else float("nan") for f in fs]
    hi = [max(f) if f else float("nan") for f in fs]
    res = [r.get("normalized_primal_residual") or float("nan") for r in rounds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.6))
    ax1.fill_between(t, lo, hi, alpha=0.25)
    sc = ax1.scatter(t, gm, c=gb, cmap="viridis", zorder=3)
    ax1.plot(t, gm, lw=1)
    ax1.set_ylabel(r"local objective $f_i$")
    ax2.scatter(t, res, c=gb, cmap="viridis", zorder=3)
    ax2.plot(t, res, lw=1)
    ax2.set_ylabel("normalized primal residual")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("ADMM round")
    fig.colorbar(sc, ax=[ax1, ax2], label="traffic per node (GB)")
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

