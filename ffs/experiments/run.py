"""Experiment driver: single runs, seed aggregation, baseline tuning, and the
heterogeneity sweep that tests the central hypothesis.

Central hypothesis (restated): at equal compute budget, MVT abandonment reaches the
objective faster than a (well-tuned) fixed timeout, and its advantage *grows* with
branch-return heterogeneity.

The sweep therefore reports, per heterogeneity level, the time-to-objective of each
policy -- with the fixed-timeout baseline **tuned per level** (its best static budget),
so MVT is never compared against a strawman.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from ..scheduler.engine import EngineConfig, RunResult, Scheduler
from ..scheduler.policies import Policy, make_policy
from ..simulator.environment import EnvConfig, Environment
from ..simulator.heterogeneity import SurfaceConfig


# --------------------------------------------------------------------------- #
# Single episode and seed aggregation
# --------------------------------------------------------------------------- #
def run_episode(
    policy_spec: dict,
    surface_cfg: SurfaceConfig,
    env_cfg: EnvConfig,
    engine_cfg: EngineConfig,
    seed: int,
    trace: bool = False,
) -> RunResult:
    """One episode. The *same* seed gives the same surface to every policy, so policy
    differences are not confounded by surface differences."""
    rng = np.random.default_rng(seed)
    env = Environment(surface_cfg, env_cfg, rng)
    policy = make_policy(policy_spec)
    sched = Scheduler(env, policy, engine_cfg, rng, trace=trace)
    return sched.run()


def run_seeds(
    policy_spec: dict,
    surface_cfg: SurfaceConfig,
    env_cfg: EnvConfig,
    engine_cfg: EngineConfig,
    seeds: list[int],
) -> pd.DataFrame:
    rows = []
    for s in seeds:
        r = run_episode(policy_spec, surface_cfg, env_cfg, engine_cfg, s)
        row = {k: v for k, v in asdict(r).items() if k != "history"}
        row["seed"] = s
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Fair baseline tuning
# --------------------------------------------------------------------------- #
def tune_fixed_timeout(
    surface_cfg: SurfaceConfig,
    env_cfg: EnvConfig,
    engine_cfg: EngineConfig,
    seeds: list[int],
    candidates: list[float],
) -> tuple[float, float]:
    """Pick the fixed timeout with the lowest mean time-to-objective (ties -> smaller).

    Returns (best_timeout, best_mean_ticks). This is the 'tune each baseline per
    scenario' mitigation for the tuned-baseline-fairness threat to validity.
    """
    best_t, best_mean = candidates[0], float("inf")
    for t in candidates:
        df = run_seeds(
            {"type": "fixed_timeout", "timeout": float(t)},
            surface_cfg, env_cfg, engine_cfg, seeds,
        )
        mean_ticks = df.loc[df["solved"], "ticks"].mean()
        if not np.isfinite(mean_ticks):
            mean_ticks = env_cfg.max_ticks  # penalise non-solving settings
        if mean_ticks < best_mean - 1e-9:
            best_mean, best_t = mean_ticks, float(t)
    return best_t, best_mean


# --------------------------------------------------------------------------- #
# Heterogeneity sweep -- the key experiment
# --------------------------------------------------------------------------- #
DEFAULT_POLICIES: dict[str, dict] = {
    "mvt_bayes": {"type": "mvt", "estimator": "bayesian",
                  "estimator_kw": {"prior_mean": "auto", "prior_strength": 2.0,
                                   "obs_var": 0.25, "half_life": 8.0},
                  "min_samples": 3, "hysteresis": 0.05},
    "mvt_slope": {"type": "mvt", "estimator": "recent_slope",
                  "estimator_kw": {"half_life": 8.0},
                  "min_samples": 3, "hysteresis": 0.05},
    "top_k": {"type": "top_k"},
    "ucb": {"type": "ucb", "c": 1.0, "margin": 0.05},
    # fixed_timeout is added per-level after tuning; oracle_mvt is reported separately
    # as an efficiency upper bound (it is rate-optimal and does not chase sub-R* tails).
}


def heterogeneity_sweep(
    levels: list[float],
    seeds: list[int],
    base_surface: SurfaceConfig,
    env_cfg: EnvConfig,
    engine_cfg: EngineConfig,
    timeout_candidates: list[float],
    policies: dict[str, dict] | None = None,
) -> pd.DataFrame:
    policies = policies or DEFAULT_POLICIES
    records = []

    for sigma in levels:
        surf = replace(base_surface, heterogeneity=float(sigma))

        # Tune the headline baseline for THIS heterogeneity level.
        best_t, _ = tune_fixed_timeout(surf, env_cfg, engine_cfg, seeds, timeout_candidates)
        level_policies = dict(policies)
        level_policies["fixed_timeout*"] = {"type": "fixed_timeout", "timeout": best_t}

        for pname, spec in level_policies.items():
            df = run_seeds(spec, surf, env_cfg, engine_cfg, seeds)
            solved = df[df["solved"]]
            rec = {
                "heterogeneity": float(sigma),
                "policy": pname,
                "solve_rate": float(df["solved"].mean()),
                "ticks_mean": float(solved["ticks"].mean()) if len(solved) else np.nan,
                "ticks_std": float(solved["ticks"].std(ddof=1)) if len(solved) > 1 else np.nan,
                "compute_mean": float(solved["compute"].mean()) if len(solved) else np.nan,
                "util_mean": float(solved["utilization"].mean()) if len(solved) else np.nan,
                "eff_mean": float(solved["findings_per_kcompute"].mean()) if len(solved) else np.nan,
                "switches_mean": float(solved["n_switches"].mean()) if len(solved) else np.nan,
                "tuned_timeout": best_t if "fixed_timeout" in pname else np.nan,
            }
            records.append(rec)

    return pd.DataFrame.from_records(records)


def add_relative_advantage(df: pd.DataFrame, ref: str = "fixed_timeout*",
                           targets: tuple[str, ...] = ("mvt_slope", "mvt_bayes")) -> pd.DataFrame:
    """Add each MVT variant's % speedup in time-to-objective over the tuned fixed timeout.

    A positive value means MVT reaches the objective faster than the *best* fixed timeout
    for that heterogeneity level. The hypothesis predicts this rising with heterogeneity."""
    out = []
    for sigma, grp in df.groupby("heterogeneity"):
        r = grp.set_index("policy")["ticks_mean"]
        row = {"heterogeneity": sigma}
        for target in targets:
            if ref in r.index and target in r.index and np.isfinite(r[ref]) and r[ref] > 0:
                row[f"speedup_{target}_pct"] = 100.0 * (r[ref] - r[target]) / r[ref]
            else:
                row[f"speedup_{target}_pct"] = np.nan
        out.append(row)
    return df.merge(pd.DataFrame(out), on="heterogeneity", how="left")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Foraging-Based Frontier Scheduling experiments")
    ap.add_argument("--seeds", type=int, default=30, help="number of random seeds")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--branches", type=int, default=40)
    ap.add_argument("--switch-cost", type=float, default=5.0)
    ap.add_argument("--target-frac", type=float, default=0.5)
    ap.add_argument("--noise", choices=["smooth", "bursty"], default="smooth")
    ap.add_argument("--levels", type=float, nargs="+",
                    default=[0.0, 0.3, 0.6, 0.9, 1.2])
    ap.add_argument("--out", type=str, default="results/sweep.csv")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    base_surface = SurfaceConfig(
        n_branches=args.branches, mean_A=10.0, heterogeneity=0.6,
        mean_tau=25.0, tau_coupling=0.5, noise_model=args.noise,
    )
    env_cfg = EnvConfig(switch_cost=args.switch_cost, objective="coverage",
                        target_frac=args.target_frac)
    engine_cfg = EngineConfig(n_workers=args.workers)
    timeout_candidates = [5, 8, 12, 16, 20, 25, 30, 40, 60, 80]

    df = heterogeneity_sweep(args.levels, seeds, base_surface, env_cfg, engine_cfg,
                             timeout_candidates)
    df = add_relative_advantage(df)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    print(f"\nSaved {args.out}\n")
    cols = ["heterogeneity", "policy", "solve_rate", "ticks_mean", "ticks_std",
            "eff_mean", "switches_mean"]
    print(df[cols].to_string(index=False))
    print("\nMVT speedup (%) vs tuned fixed timeout, by heterogeneity "
          "(positive = MVT faster):")
    adv = df[["heterogeneity", "speedup_mvt_slope_pct",
              "speedup_mvt_bayes_pct"]].drop_duplicates()
    print(adv.to_string(index=False))


if __name__ == "__main__":
    main()
