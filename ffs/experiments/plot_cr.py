"""Camera-ready figures: vector PDF, drawn at print size (LNCS text width 12.2 cm) so that
no lettering is below 6 pt, no in-figure titles (captions live in LaTeX), panels labelled
(a), (b), ... Colour-blind-safe palette with distinct markers and line styles, so the
figures stay legible in greyscale print.

  fig_speedup.pdf    speedup over the per-level tuned timeout vs heterogeneity, 95% CI bands
                     (a) abstract testbed, (b) Tier-B diminishing-value regime
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUT = "paper/atis2026/figs"
TEXTWIDTH_IN = 12.2 / 2.54

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "legend.fontsize": 6.8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.1, "lines.markersize": 3.6,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "mathtext.fontset": "cm", "legend.frameon": False,
    "legend.handlelength": 1.8,
})
# Okabe-Ito palette
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "sky": "#56B4E9", "black": "#222222", "grey": "#777777"}


def _label(ax, s):
    ax.set_title(s, loc="left", fontsize=7.5, pad=2)


def _top_legend(fig, ax_list, ncol):
    seen, hs, ls = set(), [], []
    for a in ax_list:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                hs.append(h)
                ls.append(l)
    fig.legend(hs, ls, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 1.0),
               columnspacing=0.9, handletextpad=0.4, handlelength=1.6, fontsize=6.4)


def fig_speedup():
    a = pd.read_csv("results/cr/abstract_stats.csv")
    t = pd.read_csv("results/cr/tierb_stats.csv")
    t = t[t.regime == "diminishing"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 1.6))
    series = {
        "a": [("mvt_slope", "Slope MVT", C["blue"], "s", "-"),
              ("guarded_mvt", "GuardedMVT", C["green"], "D", "-"),
              ("asha*", "ASHA", C["orange"], "^", "--"),
              ("median*", "Median", C["purple"], "v", "--")],
        "b": [("mvt_adaptive", "Adaptive MVT", C["sky"], "o", "-"),
              ("guarded_mvt", "GuardedMVT", C["green"], "D", "-"),
              ("guarded_mvt_learned", "Learned GuardedMVT", C["red"], "x", ":")],
    }
    for ax, df, key, ylim in ((axes[0], a, "a", (-40, 27)), (axes[1], t, "b", (-32, 18))):
        for pol, name, col, mk, ls in series[key]:
            d = df[df.policy == pol].sort_values("sigma")
            ax.plot(d.sigma, d.speedup_pct, ls, color=col, marker=mk, label=name)
            ax.fill_between(d.sigma, d.ci_lo, d.ci_hi, color=col, alpha=0.15, lw=0)
        ax.axhline(0, color=C["black"], lw=0.6)
        ax.set_xlabel(r"Branch-value spread $\sigma$")
        ax.set_xticks([0, 0.3, 0.6, 0.9, 1.2])
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.35)
    m = a[(a.policy == "median*") & (a.sigma <= 0.6)].speedup_pct
    axes[0].text(0.03, 0.9, f"Median, $\\sigma\\leq0.6$: ${m.min():.0f}$ to ${m.max():.0f}$%",
                 transform=axes[0].transAxes, fontsize=6.3, color=C["purple"])
    _label(axes[0], "(a) Abstract testbed")
    _label(axes[1], "(b) Agent harness")
    axes[0].set_ylabel("Speedup (%)")
    _top_legend(fig, axes, ncol=6)
    fig.tight_layout(pad=0.3, w_pad=1.2, rect=(0, 0, 1, 0.88))
    path = os.path.join(OUT, "fig_speedup.pdf")
    fig.savefig(path)
    print("saved", path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_speedup()
