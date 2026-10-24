"""Four-panel ablation figure: switch cost, burstiness (adaptive estimator), scaling, R-window."""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

C_FIXED = "#444444"
C_SLOPE = "#0072B2"
C_BAYES = "#009E73"
C_ADAPT = "#D55E00"
C_SHORT = "#CC79A7"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--out", default="results/ablations.png")
    args = ap.parse_args()
    d = args.dir

    sc = pd.read_csv(f"{d}/ablation_switch_cost.csv")
    bu = pd.read_csv(f"{d}/ablation_burstiness.csv")
    rw = pd.read_csv(f"{d}/ablation_r_window.csv")
    scal = pd.read_csv(f"{d}/ablation_scaling.csv")

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    # (a) switch cost -> MVT speed-up shrinks as migrating gets expensive.
    a = ax[0, 0]
    a.axhline(0, color=C_FIXED, lw=1)
    a.plot(sc["switch_cost"], sc["mvt_slope_speedup_pct"], color=C_SLOPE, marker="s",
           label="MVT (recent-slope)")
    a.plot(sc["switch_cost"], sc["mvt_bayes_speedup_pct"], color=C_BAYES, marker="D",
           label="MVT (Bayesian)")
    a.set_xlabel("Switch cost  $s$  (compute units)")
    a.set_ylabel("MVT speed-up vs tuned fixed timeout (%)")
    a.set_title("(a) Advantage shrinks as switching gets expensive")
    a.grid(True, alpha=0.3); a.legend(fontsize=8)

    # (b) burstiness -> short window collapses; adaptive estimator recovers solvability.
    b = ax[0, 1]
    b.plot(bu["burst_p"], bu["fixed_solve_rate"], color=C_FIXED, marker="o",
           label="Fixed timeout (signal-agnostic)")
    b.plot(bu["burst_p"], bu["mvt_slope_short_solve_rate"], color=C_SHORT, marker="x",
           ls="--", label="MVT short window (hl=8)")
    b.plot(bu["burst_p"], bu["mvt_adaptive_solve_rate"], color=C_ADAPT, marker="*",
           markersize=10, label="MVT adaptive (ours)")
    b.set_xscale("log")
    b.invert_xaxis()      # left -> right = burstier
    b.set_xlabel("Hit rate  $p$   (← burstier;  $p{=}1$ is smooth)")
    b.set_ylabel("Solve rate")
    b.set_ylim(-0.03, 1.05)
    b.set_title("(b) Burstiness-adaptive estimator restores robustness")
    b.grid(True, alpha=0.3); b.legend(fontsize=8)

    # (c) scaling -> near-linear speed-up; MVT below fixed at every scale.
    c = ax[1, 0]
    c.plot(scal["n_workers"], scal["fixed_timeout*_ticks"], color=C_FIXED, marker="o",
           label="Fixed timeout (tuned)")
    c.plot(scal["n_workers"], scal["mvt_slope_ticks"], color=C_SLOPE, marker="s",
           label="MVT (recent-slope)")
    c.plot(scal["n_workers"], scal["mvt_bayes_ticks"], color=C_BAYES, marker="D",
           label="MVT (Bayesian)")
    c.set_xscale("log", base=2); c.set_yscale("log")
    c.set_xlabel("Workers")
    c.set_ylabel("Time-to-objective (ticks)")
    c.set_title("(c) Near-linear scaling; MVT wins at every worker count")
    c.grid(True, alpha=0.3, which="both"); c.legend(fontsize=8)

    # (d) R-window -> MVT insensitive to the frontier-rate memory (robustness).
    dd = ax[1, 1]
    dd.axhline(0, color=C_FIXED, lw=1)
    dd.plot(rw["r_half_life_visits"], rw["mvt_slope_speedup_pct"], color=C_SLOPE,
            marker="s", label="MVT (recent-slope)")
    dd.plot(rw["r_half_life_visits"], rw["mvt_bayes_speedup_pct"], color=C_BAYES,
            marker="D", label="MVT (Bayesian)")
    dd.set_xscale("log")
    dd.set_xlabel("Frontier-rate memory  (EWMA half-life, visits)")
    dd.set_ylabel("MVT speed-up vs tuned fixed timeout (%)")
    dd.set_ylim(0, max(15, rw["mvt_slope_speedup_pct"].max() * 1.3))
    dd.set_title("(d) Insensitive to the $R$-estimator window")
    dd.grid(True, alpha=0.3); dd.legend(fontsize=8)

    fig.suptitle("FFS ablations  ($\\sigma{=}0.9$, 20 seeds, fixed timeout tuned per config)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
