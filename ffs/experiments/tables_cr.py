"""Emit the camera-ready LaTeX table bodies straight from results/cr/*.csv (no hand copying).

    python -m ffs.experiments.tables_cr      # writes paper/atis2026/tables/*.tex
"""

from __future__ import annotations

import os
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

OUT = "paper/atis2026/tables"
LEVELS = [0.0, 0.3, 0.6, 0.9, 1.2]


def _f(x, d=1):
    """Half-up rounding of the shortest repr, so 136.85 -> 136.9 as in the submission."""
    if x is None or not np.isfinite(x):
        return "--"
    q = Decimal(1).scaleb(-d)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def _pct(x, d=1):
    return "--" if not np.isfinite(x) else f"${x:+.{d}f}$"


def table_sweep() -> str:
    s = pd.read_csv("results/cr/abstract_stats.csv")
    ref = s.groupby("sigma").ref_ticks_mean.first()
    rows = [("Fixed timeout (tuned)", None)]
    names = [("MaxRetry (tuned)", "max_retry*"), ("Median stopping (tuned)", "median*"),
             ("ASHA (tuned)", "asha*"), ("TopK (priority only)", "top_k"),
             ("UCB (bandit)", "ucb"), ("Restless index (Gittins)", "restless_index"),
             ("MVT (recent-slope)", "mvt_slope"), ("MVT (Bayesian)", "mvt_bayes"),
             ("GuardedMVT", "guarded_mvt")]
    tab = {p: s[s.policy == p].set_index("sigma").ticks_mean for _, p in names}
    best = {sg: min([ref[sg]] + [tab[p][sg] for _, p in names]) for sg in LEVELS}

    def cell(v, sg):
        t = _f(v)
        return f"\\textbf{{{t}}}" if abs(v - best[sg]) < 0.05 else t

    lines = ["Fixed timeout (tuned) & " + " & ".join(cell(ref[sg], sg) for sg in LEVELS) + r" \\"]
    for i, (label, p) in enumerate(names):
        if p == "mvt_slope":
            lines.append(r"\midrule")
        lines.append(f"{label} & " + " & ".join(cell(tab[p][sg], sg) for sg in LEVELS) + r" \\")
    m = s[s.policy == "mvt_slope"].set_index("sigma")
    lines.append(r"\midrule")
    lines.append(r"\emph{MVT vs tuned (\%)} & " +
                 " & ".join(_pct(m.speedup_pct[sg]) for sg in LEVELS) + r" \\")
    lines.append(r"\emph{\quad 95\% interval} & " +
                 " & ".join(f"{{\\scriptsize$[{m.ci_lo[sg]:.1f},{m.ci_hi[sg]:.1f}]$}}"
                            for sg in LEVELS) + r" \\")
    return "\n".join(lines)


def table_robust() -> str:
    r = pd.read_csv("results/cr/robustness.csv")
    agg = pd.read_csv("results/cr/scale_aggregate.csv").set_index("policy")
    order = [
        ("fixed (tuned constant, t=8)", r"Fixed, tuned for constant ($t{=}8$)",
         "fixed t=8 (tuned constant)"),
        ("fixed (tuned diminishing, t=60)",
         r"Fixed, tuned for diminishing ($t{=}60$)",
         "fixed t=60 (tuned diminishing)"),
        ("fixed (minimax, t=12)", r"Fixed, best for both ($t{=}12$)",
         "fixed t=12 (minimax)"),
        ("max_retry (minimax, k=8)", r"MaxRetry, best for both ($k{=}8$)",
         "max_retry k=8 (minimax)"),
        ("median (minimax, g=2)", r"Median stopping, best for both", "median g=2 (minimax)"),
        ("asha (minimax, r0=2,eta=2)", r"ASHA, best for both", "asha r0=2,eta=2 (minimax)"),
        ("mvt_adaptive", r"Plain MVT (adaptive)", "mvt_adaptive"),
        ("mvt_slope", r"Plain MVT (recent-slope)", "mvt_slope"),
        ("guarded_mvt", r"\textbf{GuardedMVT}", "guarded_mvt"),
        ("guarded_mvt_learned", r"GuardedMVT, learned", "guarded_mvt_learned"),
        ("guarded_mvt_dyn", r"GuardedMVT, dynamic budget", "guarded_mvt_dyn"),
    ]
    lines = []
    for key, label, akey in order:
        row = r[r.policy == key].iloc[0]
        cells = []
        for reg in ("diminishing", "constant"):
            sr, t = row[f"{reg}_solve_rate"], row[f"{reg}_ticks"]
            if sr == 0:
                cells.append("fails")
            elif sr < 1:
                cells.append(f"{_f(t)} ({100 * sr:.0f}\\%)")
            else:
                cells.append(_f(t))
        n = int(agg.loc[akey, "solved_all"])
        cells.append(f"{n}/9")
        if key == "guarded_mvt":
            lines.append(r"\midrule")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


LLM_NAMES = [("scripted", "Scripted tactic"), ("qwen2.5:7b", "Qwen2.5-7B"),
             ("llama3.1:8b", "Llama-3.1-8B"), ("gemma2:9b", "Gemma-2-9B"),
             ("mistral:7b", "Mistral-7B"), ("qwen2.5:1.5b", "Qwen2.5-1.5B")]


def table_llm() -> str:
    s = pd.read_csv("results/cr/llm/llm_summary_prod.csv")
    s["tuned"] = s["tuned"].fillna("")
    d = pd.read_csv("results/cr/llm/llm_decisions_prod.csv")
    lat = pd.read_csv("results/cr/llm/llm_decisions.csv")      # first, uncached pass
    lines = []
    for key, label in LLM_NAMES:
        g = s[s.brain == key]
        dd = d[d.brain == key].set_index("regime")
        cells = [label]
        if key == "scripted":
            cells += ["0 / 0", "0 / 0", "--"]
        else:
            ll = lat[lat.brain == key]
            ms = float((ll.latency_ms * ll.model_calls).sum() / ll.model_calls.sum())
            cells += [f"{_f(dd.wasted_pct['diminishing'], 0)} / {_f(dd.wasted_pct['constant'], 0)}",
                      f"{_f(dd.fallback_pct['diminishing'], 0)} / "
                      f"{_f(dd.fallback_pct['constant'], 0)}", f"{ms:.0f}"]
        dim = g[g.regime == "diminishing"].set_index("policy")
        con = g[g.regime == "constant"].set_index("policy")
        gm = dim.loc["guarded_mvt"]
        cells.append(f"${gm.speedup_pct:+.1f}$ $[{gm.ci_lo:+.0f}, {gm.ci_hi:+.0f}]$")
        t_con = con.loc["fixed*", "tuned"].replace("t=", "")
        cells += [f"{_f(con.loc['fixed*', 'ticks'], 0)} ({t_con})",
                  _f(con.loc['guarded_mvt', 'ticks'], 0),
                  ("--" if key == "scripted" else
                   _f(con.loc['fixed (scripted-tuned)', 'ticks'], 0))]
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("sweep", table_sweep), ("robust", table_robust), ("llm", table_llm)):
        body = fn()
        with open(os.path.join(OUT, f"{name}.tex"), "w") as f:
            f.write(body + "\n")
        print(f"--- {name}\n{body}")


if __name__ == "__main__":
    main()
