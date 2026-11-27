"""NASim external-validity experiment: does the abstract MVT advantage transfer?

Runs the SAME scheduler + MVT policy that produced the abstract results against a real
network-attack simulator (NASim), via the segment-as-branch testbed, and compares it to a
per-configuration-tuned fixed timeout across a heterogeneity sweep. The fixed timeout is
tuned per level (fairness), exactly as in the abstract experiments.

Finding (see README): MVT functions correctly here -- the opportunity-rate R lets workers
migrate off depleting segments -- but a well-tuned fixed timeout remains competitive or
better across regimes (host- vs segment-branch, uniform vs diminishing value, breadth vs
value objective). This delineates the transfer boundary: foraging-optimal stopping needs
steep diminishing *value* and a clean progress signal, strongly present in the abstract
model and only weakly in NASim-style exploitation (constant-hazard exploits, uniform host
value). A negative/nuanced but genuine characterization of when foraging rules transfer.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..scheduler.engine import EngineConfig, Scheduler
from ..scheduler.policies import make_policy
from .brains import SubnetBrain
from .nasim_target import RealismConfig, make_segmented_nasim_target

MVT = {"type": "mvt", "estimator": "adaptive",
       "estimator_kw": {"base_half_life": 20.0, "max_half_life": 20.0},
       "min_samples": 2, "hysteresis": 0.1}
# GuardedMVT: the same MVT estimator config as above, plus the boundary-aware guard
# (apply the MVT leaving rule only where the signal is diminishing; elsewhere commit a
# budget). On NASim's constant-hazard exploitation the guard's committed-budget fallback
# should keep GuardedMVT from the premature-abandonment that hurts plain MVT here.
GUARDED = {"type": "guarded_mvt", "estimator": "adaptive",
           "estimator_kw": {"base_half_life": 20.0, "max_half_life": 20.0},
           "min_samples": 2, "hysteresis": 0.1,
           "decline_frac": 0.5, "peak_floor_frac": 0.05, "budget_cap": 16.0}
TIMEOUTS = [10, 16, 24, 35, 55]


def _run(spec, sigma, seeds, cfg_kw, engine_kw):
    ticks, solved = [], 0
    for s in range(seeds):
        cfg = RealismConfig(**cfg_kw)
        tgt = make_segmented_nasim_target(SubnetBrain(), cfg, n_subnets=12,
                                          hosts_per_subnet=5, num_services=6,
                                          heterogeneity=sigma, seed=s)
        sch = Scheduler(tgt, make_policy(spec), EngineConfig(**engine_kw),
                        np.random.default_rng(1000 + s))
        r = sch.run()
        if r.solved:
            ticks.append(r.ticks)
            solved += 1
    return solved / seeds, (float(np.mean(ticks)) if ticks else np.nan)


def sweep(levels, seeds, switch_cost, objective, value_decay):
    cfg_kw = dict(switch_cost=switch_cost, max_ticks=15000, target_frac=0.5,
                  objective=objective, value_decay=value_decay)
    engine_kw = dict(n_workers=4, entry_window=20)
    rows = []
    for sigma in levels:
        best_t, best_m = None, np.inf
        for t in TIMEOUTS:
            sr, mt = _run({"type": "fixed_timeout", "timeout": float(t)}, sigma, seeds,
                          cfg_kw, engine_kw)
            if sr >= 0.95 and mt < best_m:
                best_m, best_t = mt, t
        sr, mt = _run(MVT, sigma, seeds, cfg_kw, engine_kw)
        speedup = 100.0 * (best_m - mt) / best_m if (best_m and np.isfinite(mt)) else np.nan
        gsr, gmt = _run(GUARDED, sigma, seeds, cfg_kw, engine_kw)
        gspeedup = 100.0 * (best_m - gmt) / best_m if (best_m and np.isfinite(gmt)) else np.nan
        rows.append({"heterogeneity": sigma, "fixed_tuned_timeout": best_t,
                     "fixed_ticks": best_m, "mvt_solve_rate": sr, "mvt_ticks": mt,
                     "mvt_speedup_pct": speedup, "guarded_solve_rate": gsr,
                     "guarded_ticks": gmt, "guarded_speedup_pct": gspeedup})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="NASim external-validity sweep")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--switch-cost", type=float, default=5.0)
    ap.add_argument("--objective", choices=["count", "value"], default="count")
    ap.add_argument("--value-decay", type=float, default=1.0)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.4, 0.8, 1.2, 1.6])
    ap.add_argument("--out", default="results/nasim_sweep.csv")
    args = ap.parse_args()
    df = sweep(args.levels, args.seeds, args.switch_cost, args.objective, args.value_decay)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
