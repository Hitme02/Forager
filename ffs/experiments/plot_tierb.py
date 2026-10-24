"""Four-panel Tier-B figure: the transfer boundary, and the boundary-aware policy that fixes it.

(a) Diminishing-value regime -- MVT's (and GuardedMVT's) time-to-objective speed-up over a
    tuned fixed timeout vs heterogeneity: the abstract crossover reproduces in the live harness.
(b) Constant-hazard regime -- solve-rate: plain MVT collapses (the transfer boundary);
    GuardedMVT does not.
(c) Cross-regime robustness (the design win) -- ONE policy must serve BOTH regimes. A fixed
    timeout tuned for one regime fails in the other; plain MVT collapses under constant hazard;
    GuardedMVT is strong in both with a single configuration.
(d) Burstiness -- the adaptive estimator (and GuardedMVT on top of it) recover the loss that
    the recent-slope estimator suffers as the collection stream becomes bursty.
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C_FIXED = "#444444"
C_SLOPE = "#0072B2"
C_MVT = "#D55E00"
C_GUARD = "#009E73"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="results/tierb")
    ap.add_argument("--out", default="results/tierb.png")
    args = ap.parse_args()

    dim = pd.read_csv(f"{args.prefix}_diminishing.csv")
    con = pd.read_csv(f"{args.prefix}_constant.csv")
    rob = pd.read_csv(f"{args.prefix}_robustness.csv")
    bur = pd.read_csv(f"{args.prefix}_burstiness.csv")

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))

    # (a) diminishing-value regime: the abstract advantage transfers (MVT and GuardedMVT)
    a = ax[0, 0]
    a.axhline(0, color=C_FIXED, lw=1)
    a.plot(dim["heterogeneity"], dim["mvt_speedup_pct"], color=C_MVT, marker="o", lw=2,
           label="MVT (adaptive)")
    a.plot(dim["heterogeneity"], dim["guarded_speedup_pct"], color=C_GUARD, marker="D", lw=2,
           label="GuardedMVT")
    a.set_xlabel(r"Branch heterogeneity  $\sigma$")
    a.set_ylabel("Speed-up over tuned fixed (%)")
    a.set_title("(a) Diminishing value: advantage transfers")
    a.legend(loc="upper left", fontsize=9)

    # (b) constant-hazard regime: MVT collapses, GuardedMVT does not
    b = ax[0, 1]
    b.plot(con["heterogeneity"], [1.0] * len(con), color=C_FIXED, marker="s", lw=2,
           label="tuned fixed")
    b.plot(con["heterogeneity"], con["mvt_solve_rate"], color=C_MVT, marker="o", lw=2,
           label="MVT (adaptive)")
    b.plot(con["heterogeneity"], con["guarded_solve_rate"], color=C_GUARD, marker="D", lw=2,
           label="GuardedMVT")
    b.set_ylim(-0.05, 1.1)
    b.set_xlabel(r"Branch heterogeneity  $\sigma$")
    b.set_ylabel("Objective solve-rate")
    b.set_title("(b) Constant hazard: the boundary, and the fix")
    b.legend(loc="center right", fontsize=9)

    # (c) cross-regime robustness bars at one sigma: one policy, both regimes
    c = ax[1, 0]
    labels = ["fixed\n(tuned const)", "fixed\n(tuned dimin)", "MVT\n(adaptive)", "GuardedMVT"]
    dim_t = rob["diminishing_ticks"].to_numpy(dtype=float)
    con_t = rob["constant_ticks"].to_numpy(dtype=float)
    x = np.arange(len(labels))
    w = 0.38
    c.bar(x - w / 2, dim_t, w, color=C_MVT, label="diminishing regime")
    c.bar(x + w / 2, np.nan_to_num(con_t), w, color=C_SLOPE, label="constant regime")
    ymax = float(np.nanmax([np.nanmax(dim_t), np.nanmax(con_t)]))
    for i, v in enumerate(con_t):      # mark the collapse (no solve) explicitly
        if not np.isfinite(v):
            c.text(x[i] + w / 2, ymax * 0.5, "✗\ncollapse", ha="center", va="center",
                   color=C_SLOPE, fontsize=9, fontweight="bold")
    c.set_xticks(x)
    c.set_xticklabels(labels, fontsize=8.5)
    c.set_ylabel("Time-to-objective (ticks, lower=better)")
    sigma = rob["heterogeneity"].iloc[0] if "heterogeneity" in rob else ""
    c.set_title(f"(c) One policy, both regimes  ($\\sigma$={sigma})")
    c.legend(loc="upper center", fontsize=9)

    # (d) burstiness: adaptive estimator (and GuardedMVT) recover the loss
    d = ax[1, 1]
    xb = bur["burst_p"].astype(str)
    d.plot(xb, bur["fixed_ticks"], color=C_FIXED, marker="s", lw=2, label="tuned fixed")
    d.plot(xb, bur["mvt_slope_ticks"], color=C_SLOPE, marker="^", lw=2, label="MVT recent-slope")
    d.plot(xb, bur["mvt_adaptive_ticks"], color=C_MVT, marker="o", lw=2, label="MVT adaptive")
    d.plot(xb, bur["guarded_ticks"], color=C_GUARD, marker="D", lw=2, label="GuardedMVT")
    d.invert_xaxis()
    d.set_xlabel(r"Collection burstiness  $p$  (smaller = burstier)")
    d.set_ylabel("Time-to-objective (ticks, lower=better)")
    d.set_title("(d) Burstiness: adaptive estimator recovers")
    d.legend(loc="upper left", fontsize=9)

    fig.suptitle("Tier B: a boundary-aware foraging scheduler driving a real tool-executing "
                 "agent harness over a sandboxed cyber range", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=130)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
