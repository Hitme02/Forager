"""Plot the heterogeneity sweep: time-to-objective and MVT speedup vs heterogeneity."""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Consistent, colour-blind-safe-ish styling across the two panels.
STYLE = {
    "fixed_timeout*": ("Fixed timeout (tuned)", "#444444", "o", "-"),
    "mvt_slope":      ("MVT (recent-slope)",    "#0072B2", "s", "-"),
    "mvt_bayes":      ("MVT (Bayesian)",         "#009E73", "D", "-"),
    "top_k":          ("Top-K (prioritise)",     "#E69F00", "^", "--"),
    "ucb":            ("UCB bandit",             "#CC79A7", "v", "--"),
    "restless_index": ("Restless index (Gittins)", "#D55E00", "P", "--"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/sweep.csv")
    ap.add_argument("--out", default="results/sweep.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Panel 1: time-to-objective vs heterogeneity (lower is better).
    for pol, (label, color, marker, ls) in STYLE.items():
        sub = df[df["policy"] == pol].sort_values("heterogeneity")
        if sub.empty:
            continue
        ax1.errorbar(sub["heterogeneity"], sub["ticks_mean"],
                     yerr=sub["ticks_std"], label=label, color=color,
                     marker=marker, linestyle=ls, capsize=3, markersize=5)
    ax1.set_xlabel("Branch-return heterogeneity  (log-normal $\\sigma$)")
    ax1.set_ylabel("Time-to-objective  (ticks, lower = better)")
    ax1.set_title("Time-to-objective vs heterogeneity")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, framealpha=0.9)

    # Panel 2: MVT speedup over the tuned fixed timeout (the hypothesis).
    adv = df[["heterogeneity", "speedup_mvt_slope_pct",
              "speedup_mvt_bayes_pct"]].drop_duplicates().sort_values("heterogeneity")
    ax2.axhline(0, color="#444444", lw=1)
    ax2.plot(adv["heterogeneity"], adv["speedup_mvt_slope_pct"],
             color="#0072B2", marker="s", label="MVT (recent-slope)")
    ax2.plot(adv["heterogeneity"], adv["speedup_mvt_bayes_pct"],
             color="#009E73", marker="D", label="MVT (Bayesian)")
    ax2.set_xlabel("Branch-return heterogeneity  (log-normal $\\sigma$)")
    ax2.set_ylabel("MVT speedup over tuned fixed timeout  (%)")
    ax2.set_title("MVT advantage grows with heterogeneity")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.annotate("MVT faster", xy=(0.02, 0.96), xycoords="axes fraction",
                 fontsize=8, va="top", color="#0072B2")
    ax2.annotate("fixed faster", xy=(0.02, 0.04), xycoords="axes fraction",
                 fontsize=8, va="bottom", color="#444444")

    fig.suptitle("Foraging-Based Frontier Scheduling: MVT branch abandonment vs baselines",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
