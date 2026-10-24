"""Tier-B external-validity experiment: does the MVT advantage survive a real agent harness?

Tier B drives the EXACT SAME scheduler + MVT policy that produced the abstract and NASim
results through a bounded, tool-executing agent harness against a sandboxed cyber range
(`cyber_range.CyberRange`). Everything above the `work_one` seam is unchanged; only the
backend is now an agent taking grounded actions on services. Three studies, mirroring the
abstract experiments so the comparison is like-for-like:

1. `sweep_diminishing` -- the headline. Coverage objective over a heterogeneous range whose
   services yield a CONCAVE collection stream (the diminishing value MVT needs). Reports MVT's
   time-to-objective speedup over a per-level-tuned fixed timeout, across the heterogeneity
   knob sigma. Prediction (and result): MVT trails at low sigma, crosses over, and wins at
   high sigma -- the abstract advantage TRANSFERS to the live harness.

2. `sweep_constant` -- the transfer boundary. The same sweep over a CONSTANT-HAZARD range
   (memoryless foothold access, uniform value). Here the marginal-rate signal is
   uninformative, so MVT either over-abandons (its solve-rate collapses) or, tuned
   conservatively, solves but is dominated by a tuned fixed timeout. Reported via solve-rate
   and time, it delineates exactly where foraging-optimal stopping stops helping -- the same
   boundary found in NASim, now in a real agent harness.

3. `burstiness_study` -- ties Tier B to the project's adaptive-estimator contribution. As the
   collection stream becomes bursty (rare large jumps), recent-slope MVT degrades below the
   fixed baseline; the burstiness-adaptive estimator recovers most of the loss. The
   smooth-concavity threat and its fix both reproduce in the live setting.

The default range is the in-process `MockSandbox` (no Docker, no live traffic): a faithful
model of the range that exercises the whole scheduler->harness->grounded-progress pipeline.
`--live` instead drives the Dockerised, isolated fixture (gated; see TIERB.md) on the flag
(crown-jewel) objective -- the real-tool demonstration, run only in an authorised lab.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..scheduler.engine import EngineConfig, Scheduler
from ..scheduler.policies import make_policy
from .cyber_range import RangeConfig, make_mock_range

# One global configuration per policy, used across every heterogeneity level (only the
# fixed-timeout baseline is tuned per level -- the fairness rule from the abstract
# experiments). The adaptive estimator is the project's burstiness-robust marginal estimator;
# min_samples and hysteresis are the keep-exploring / anti-churn guards, set once for the
# whole study.
_EST = {"base_half_life": 8.0, "max_half_life": 24.0}
MVT_ADAPTIVE = {"type": "mvt", "estimator": "adaptive", "estimator_kw": _EST,
                "min_samples": 6, "hysteresis": 0.3}
# GuardedMVT: the boundary-aware policy. Same MVT rule, but applied only where the observed
# signal is diminishing; elsewhere it commits a budget (budget_cap) instead of churning. One
# config serves BOTH regimes -- the design contribution.
GUARDED = {"type": "guarded_mvt", "estimator": "adaptive", "estimator_kw": _EST,
           "min_samples": 6, "hysteresis": 0.3,
           "decline_frac": 0.5, "peak_floor_frac": 0.05, "budget_cap": 16.0}
MVT_SLOPE = {"type": "mvt", "estimator": "recent_slope",
             "estimator_kw": {"half_life": 8.0}, "min_samples": 6, "hysteresis": 0.3}
TIMEOUTS = [8, 12, 16, 25, 40, 60, 80]

_ENGINE = dict(n_workers=8, entry_window=5)


def _run(spec, *, sigma, regime, burst_p, seeds, switch_cost, target_frac, max_ticks):
    ticks, switches, solved = [], [], 0
    for s in range(seeds):
        cfg = RangeConfig(switch_cost=switch_cost, max_ticks=max_ticks,
                          objective="coverage", target_frac=target_frac)
        rng = make_mock_range(cfg, n_services=40, flag_services=2, sigma=sigma,
                              regime=regime, burst_p=burst_p, seed=s)
        sch = Scheduler(rng, make_policy(dict(spec)), EngineConfig(**_ENGINE),
                        np.random.default_rng(1000 + s))
        r = sch.run()
        if r.solved:
            solved += 1
            ticks.append(r.ticks)
            switches.append(r.n_switches)
    return (solved / seeds,
            float(np.mean(ticks)) if ticks else np.nan,
            float(np.mean(switches)) if switches else np.nan)


def _tuned_fixed(*, sigma, regime, burst_p, seeds, switch_cost, target_frac, max_ticks):
    best_t, best_m = None, np.inf
    for t in TIMEOUTS:
        sr, mt, _ = _run({"type": "fixed_timeout", "timeout": float(t)}, sigma=sigma,
                         regime=regime, burst_p=burst_p, seeds=seeds, switch_cost=switch_cost,
                         target_frac=target_frac, max_ticks=max_ticks)
        if sr >= 0.95 and mt < best_m:
            best_m, best_t = mt, t
    return best_t, best_m


def _sweep(regime, levels, seeds, switch_cost, target_frac, max_ticks) -> pd.DataFrame:
    """Per-σ: tuned fixed baseline, plain MVT, and the boundary-aware GuardedMVT."""
    rows = []
    for sigma in levels:
        kw = dict(sigma=sigma, regime=regime, burst_p=1.0, seeds=seeds,
                  switch_cost=switch_cost, target_frac=target_frac, max_ticks=max_ticks)
        best_t, best_m = _tuned_fixed(**kw)
        sr_m, mt_m, _ = _run(MVT_ADAPTIVE, **kw)
        sr_g, mt_g, _ = _run(GUARDED, **kw)
        sp_m = 100.0 * (best_m - mt_m) / best_m if (best_m and np.isfinite(mt_m)) else np.nan
        sp_g = 100.0 * (best_m - mt_g) / best_m if (best_m and np.isfinite(mt_g)) else np.nan
        rows.append({"heterogeneity": sigma, "fixed_tuned_timeout": best_t,
                     "fixed_ticks": best_m,
                     "mvt_solve_rate": sr_m, "mvt_ticks": mt_m, "mvt_speedup_pct": sp_m,
                     "guarded_solve_rate": sr_g, "guarded_ticks": mt_g,
                     "guarded_speedup_pct": sp_g})
    return pd.DataFrame(rows)


def sweep_diminishing(levels, seeds, switch_cost, target_frac, max_ticks) -> pd.DataFrame:
    return _sweep("diminishing", levels, seeds, switch_cost, target_frac, max_ticks)


def sweep_constant(levels, seeds, switch_cost, target_frac, max_ticks) -> pd.DataFrame:
    return _sweep("constant", levels, seeds, switch_cost, target_frac, max_ticks)


def robustness(sigma, seeds, switch_cost, target_frac, max_ticks) -> pd.DataFrame:
    """The design win: ONE policy must serve BOTH regimes without knowing which it faces.

    A fixed timeout tuned for one regime fails in the other (the optimal budget differs ~8x);
    plain MVT wins under diminishing value but collapses under constant hazard; GuardedMVT is
    strong in both with a single configuration. Reported at one heterogeneity level."""
    def both(spec):
        sd, td, _ = _run(spec, sigma=sigma, regime="diminishing", burst_p=1.0, seeds=seeds,
                         switch_cost=switch_cost, target_frac=target_frac, max_ticks=max_ticks)
        sc, tc, _ = _run(spec, sigma=sigma, regime="constant", burst_p=1.0, seeds=seeds,
                         switch_cost=switch_cost, target_frac=target_frac, max_ticks=max_ticks)
        return sd, td, sc, tc

    # fixed timeouts tuned for each regime (so each is a genuine per-regime champion)
    t_dim, _ = _tuned_fixed(sigma=sigma, regime="diminishing", burst_p=1.0, seeds=seeds,
                            switch_cost=switch_cost, target_frac=target_frac, max_ticks=max_ticks)
    t_con, _ = _tuned_fixed(sigma=sigma, regime="constant", burst_p=1.0, seeds=seeds,
                            switch_cost=switch_cost, target_frac=target_frac, max_ticks=max_ticks)
    policies = [
        (f"fixed (tuned constant, t={t_con})", {"type": "fixed_timeout", "timeout": float(t_con)}),
        (f"fixed (tuned diminishing, t={t_dim})", {"type": "fixed_timeout", "timeout": float(t_dim)}),
        ("MVT (adaptive)", MVT_ADAPTIVE),
        ("GuardedMVT", GUARDED),
    ]
    rows = []
    for name, spec in policies:
        sd, td, sc, tc = both(spec)
        rows.append({"policy": name, "heterogeneity": sigma,
                     "diminishing_solve_rate": sd, "diminishing_ticks": td,
                     "constant_solve_rate": sc, "constant_ticks": tc})
    return pd.DataFrame(rows)


def burstiness_study(burst_levels, sigma, seeds, switch_cost, target_frac,
                     max_ticks) -> pd.DataFrame:
    rows = []
    for bp in burst_levels:
        kw = dict(sigma=sigma, regime="diminishing", burst_p=bp, seeds=seeds,
                  switch_cost=switch_cost, target_frac=target_frac, max_ticks=max_ticks)
        best_t, best_m = _tuned_fixed(**kw)
        _, mt_slope, _ = _run(MVT_SLOPE, **kw)
        _, mt_adapt, _ = _run(MVT_ADAPTIVE, **kw)
        _, mt_guard, _ = _run(GUARDED, **kw)
        rows.append({"burst_p": bp, "fixed_ticks": best_m, "mvt_slope_ticks": mt_slope,
                     "mvt_adaptive_ticks": mt_adapt, "guarded_ticks": mt_guard})
    return pd.DataFrame(rows)


def run_live_flag(seeds: int) -> None:
    """Drive the gated, isolated Docker fixture on the crown-jewel flag objective.

    Requires a running `docker/` fixture on an internal (no-egress) network and the operator's
    authorisation acknowledgement. See TIERB.md. Not part of the default experiment."""
    from .cyber_range import make_docker_range

    rng = make_docker_range(RangeConfig(objective="flag"), acknowledged=True)
    sch = Scheduler(rng, make_policy(dict(GUARDED)), EngineConfig(**_ENGINE),
                    np.random.default_rng(0))
    r = sch.run()
    print(f"[live] flag captured={r.solved} ticks={r.ticks} switches={r.n_switches}")
    rng.teardown()


def main() -> None:
    ap = argparse.ArgumentParser(description="Tier-B live-harness external-validity study")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--switch-cost", type=float, default=5.0)
    ap.add_argument("--target-frac", type=float, default=0.5)
    ap.add_argument("--max-ticks", type=int, default=12000)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.9, 1.2])
    ap.add_argument("--burst-levels", type=float, nargs="+", default=[1.0, 0.4, 0.2, 0.1])
    ap.add_argument("--burst-sigma", type=float, default=0.9)
    ap.add_argument("--robustness-sigma", type=float, default=1.2)
    ap.add_argument("--out-prefix", default="results/tierb")
    ap.add_argument("--live", action="store_true",
                    help="drive the gated isolated Docker fixture on the flag objective")
    args = ap.parse_args()

    if args.live:
        run_live_flag(args.seeds)
        return

    common = dict(seeds=args.seeds, switch_cost=args.switch_cost,
                  target_frac=args.target_frac, max_ticks=args.max_ticks)
    dim = sweep_diminishing(args.levels, **common)
    con = sweep_constant(args.levels, **common)
    rob = robustness(args.robustness_sigma, **common)
    bur = burstiness_study(args.burst_levels, args.burst_sigma, **common)

    dim.to_csv(f"{args.out_prefix}_diminishing.csv", index=False)
    con.to_csv(f"{args.out_prefix}_constant.csv", index=False)
    rob.to_csv(f"{args.out_prefix}_robustness.csv", index=False)
    bur.to_csv(f"{args.out_prefix}_burstiness.csv", index=False)

    print("\n=== Tier B: DIMINISHING-value regime (MVT & GuardedMVT advantage transfers) ===")
    print(dim.to_string(index=False))
    print("\n=== Tier B: CONSTANT-hazard regime (MVT collapses; GuardedMVT does not) ===")
    print(con.to_string(index=False))
    print(f"\n=== Tier B: cross-regime robustness at sigma={args.robustness_sigma} "
          "(one policy, both regimes) ===")
    print(rob.to_string(index=False))
    print("\n=== Tier B: burstiness (adaptive estimator recovers the loss) ===")
    print(bur.to_string(index=False))
    print(f"\nSaved {args.out_prefix}_{{diminishing,constant,robustness,burstiness}}.csv")


if __name__ == "__main__":
    main()
