"""Ablation suite -- turns the Phase-1 core result into a full results section.

Each ablation doubles as a research question / threat-to-validity test from the proposal:

  1. switch_cost   : MVT's advantage should SHRINK as the switch cost s rises
                     (when migrating is expensive, staying beats moving).
  2. burstiness    : does MVT survive BURSTY gains? (a single exploit flips a branch
                     discontinuously, violating the smooth-concavity MVT assumes.)
  3. r_window      : robustness of MVT to the R-estimator's memory (non-stationary R).
  4. scaling       : worker-count scaling -- time-to-objective and load balance
                     (utilisation), the PAGP parallel-scaling study.

All comparisons are at equal compute budget, with the fixed-timeout baseline TUNED per
configuration (fairness). Default heterogeneity is held high (sigma=0.9), the regime where
MVT wins, so the ablations characterise *how* that win depends on each factor.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from ..scheduler.engine import EngineConfig
from ..simulator.environment import EnvConfig
from ..simulator.heterogeneity import SurfaceConfig
from .run import DEFAULT_POLICIES, run_seeds, tune_fixed_timeout

TIMEOUTS = [5, 8, 12, 16, 20, 25, 30, 40, 60]
MVT_SPECS = {k: DEFAULT_POLICIES[k] for k in ("mvt_slope", "mvt_bayes")}

# Estimator variants for the burstiness ablation: a short fixed window (fragile under
# burst) vs the burstiness-adaptive estimator (the mechanism contribution).
SLOPE_SHORT = {"type": "mvt", "estimator": "recent_slope",
               "estimator_kw": {"half_life": 8.0}, "min_samples": 3, "hysteresis": 0.05}
ADAPTIVE = {"type": "mvt", "estimator": "adaptive",
            "estimator_kw": {"base_half_life": 6.0, "max_half_life": 80.0,
                             "hit_half_life": 20.0}, "min_samples": 3, "hysteresis": 0.05}


def _mean_ticks(df: pd.DataFrame) -> float:
    s = df.loc[df["solved"], "ticks"]
    return float(s.mean()) if len(s) else np.nan


def _mean(df: pd.DataFrame, col: str) -> float:
    s = df.loc[df["solved"], col]
    return float(s.mean()) if len(s) else np.nan


# --------------------------------------------------------------------------- #
def ablation_switch_cost(seeds, sigma=0.9, costs=(1, 3, 5, 10, 20)) -> pd.DataFrame:
    rows = []
    for s in costs:
        surf = SurfaceConfig(n_branches=40, heterogeneity=sigma)
        env = EnvConfig(switch_cost=float(s), target_frac=0.5)
        eng = EngineConfig(n_workers=8)
        best_t, _ = tune_fixed_timeout(surf, env, eng, seeds, TIMEOUTS)
        ref = run_seeds({"type": "fixed_timeout", "timeout": best_t}, surf, env, eng, seeds)
        base = _mean_ticks(ref)
        row = {"switch_cost": s, "fixed_tuned_ticks": base, "tuned_timeout": best_t}
        for name, spec in MVT_SPECS.items():
            df = run_seeds(spec, surf, env, eng, seeds)
            mt = _mean_ticks(df)
            row[f"{name}_ticks"] = mt
            row[f"{name}_speedup_pct"] = 100.0 * (base - mt) / base if base > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def ablation_burstiness(seeds, sigma=0.9, burst_ps=(1.0, 0.5, 0.2, 0.1, 0.05)) -> pd.DataFrame:
    """Burstiness robustness, and the payoff of the adaptive estimator.

    Compares, as the signal gets burstier: the tuned fixed timeout (signal-agnostic, so
    robust), a short-fixed-window MVT (fragile), and the burstiness-adaptive MVT. Reports
    solve-rate (can the policy even reach the objective) and time-to-objective."""
    rows = []
    for p in burst_ps:
        # p = 1.0 is effectively smooth; smaller p = rarer, larger jumps = burstier.
        model = "smooth" if p >= 1.0 else "bursty"
        surf = SurfaceConfig(n_branches=40, heterogeneity=sigma, noise_model=model, burst_p=p)
        env = EnvConfig(switch_cost=5.0, target_frac=0.5, max_ticks=5000)
        eng = EngineConfig(n_workers=8)
        best_t, _ = tune_fixed_timeout(surf, env, eng, seeds, TIMEOUTS)
        ref = run_seeds({"type": "fixed_timeout", "timeout": best_t}, surf, env, eng, seeds)
        base = _mean_ticks(ref)
        row = {"burst_p": p, "noise": model,
               "fixed_solve_rate": float(ref["solved"].mean()), "fixed_ticks": base}
        for name, spec in (("mvt_slope_short", SLOPE_SHORT), ("mvt_adaptive", ADAPTIVE)):
            df = run_seeds(spec, surf, env, eng, seeds)
            mt = _mean_ticks(df)
            row[f"{name}_solve_rate"] = float(df["solved"].mean())
            row[f"{name}_ticks"] = mt
            row[f"{name}_speedup_pct"] = 100.0 * (base - mt) / base if base > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def ablation_r_window(seeds, sigma=0.9, half_lives=(5, 10, 20, 40, 80)) -> pd.DataFrame:
    surf = SurfaceConfig(n_branches=40, heterogeneity=sigma)
    env = EnvConfig(switch_cost=5.0, target_frac=0.5)
    best_t, _ = tune_fixed_timeout(surf, env, EngineConfig(n_workers=8), seeds, TIMEOUTS)
    ref = run_seeds({"type": "fixed_timeout", "timeout": best_t}, surf, env,
                    EngineConfig(n_workers=8), seeds)
    base = _mean_ticks(ref)
    rows = []
    for hl in half_lives:
        eng = EngineConfig(n_workers=8, frontier_half_life_entries=float(hl))
        row = {"r_half_life_visits": hl, "fixed_tuned_ticks": base}
        for name, spec in MVT_SPECS.items():
            df = run_seeds(spec, surf, env, eng, seeds)
            mt = _mean_ticks(df)
            row[f"{name}_ticks"] = mt
            row[f"{name}_speedup_pct"] = 100.0 * (base - mt) / base if base > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def ablation_scaling(seeds, sigma=0.9, workers=(2, 4, 8, 16, 32)) -> pd.DataFrame:
    surf = SurfaceConfig(n_branches=64, heterogeneity=sigma)
    env = EnvConfig(switch_cost=5.0, target_frac=0.5)
    rows = []
    for w in workers:
        eng = EngineConfig(n_workers=int(w))
        best_t, _ = tune_fixed_timeout(surf, env, eng, seeds, TIMEOUTS)
        row = {"n_workers": w}
        for name, spec in {"fixed_timeout*": {"type": "fixed_timeout", "timeout": best_t},
                           **MVT_SPECS}.items():
            df = run_seeds(spec, surf, env, eng, seeds)
            row[f"{name}_ticks"] = _mean_ticks(df)
            row[f"{name}_util"] = _mean(df, "utilization")
            row[f"{name}_compute"] = _mean(df, "compute")
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="FFS ablation suite")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=0.9)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    runners = {
        "switch_cost": lambda: ablation_switch_cost(seeds, args.sigma),
        "burstiness": lambda: ablation_burstiness(seeds, args.sigma),
        "r_window": lambda: ablation_r_window(seeds, args.sigma),
        "scaling": lambda: ablation_scaling(seeds, args.sigma),
    }
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    for name, fn in runners.items():
        print(f"\n===== ablation: {name} (sigma={args.sigma}, {args.seeds} seeds) =====")
        df = fn()
        path = f"{args.outdir}/ablation_{name}.csv"
        df.to_csv(path, index=False)
        print(df.to_string(index=False))
        print(f"saved {path}")


if __name__ == "__main__":
    main()
