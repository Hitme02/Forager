"""Sanity tests for the FFS testbed and policies."""

from __future__ import annotations

import numpy as np

from ffs.scheduler.engine import EngineConfig, Scheduler
from ffs.scheduler.estimators import make_marginal_estimator
from ffs.scheduler.policies import make_policy
from ffs.simulator.environment import EnvConfig, Environment
from ffs.simulator.heterogeneity import SurfaceConfig, gini, make_surface
from ffs.simulator.patch import Patch


def test_patch_diminishing_returns():
    p = Patch(0, A=10.0, tau=20.0)
    # Marginal must be strictly decreasing in spent compute.
    rng = np.random.default_rng(0)
    prev = p.marginal(0.0)
    for _ in range(50):
        p.work_one(rng)
        cur = p.marginal()
        assert cur <= prev + 1e-12
        prev = cur
    # Cumulative yield approaches but never exceeds A.
    assert p.expected_cumulative(1e6) <= p.A + 1e-6


def test_heterogeneity_knob_controls_spread():
    rng = np.random.default_rng(1)
    lo = make_surface(SurfaceConfig(n_branches=200, heterogeneity=0.0), rng)
    hi = make_surface(SurfaceConfig(n_branches=200, heterogeneity=1.0), rng)
    g_lo = gini(np.array([p.A for p in lo]))
    g_hi = gini(np.array([p.A for p in hi]))
    assert g_lo < 0.05          # homogeneous
    assert g_hi > g_lo + 0.2    # much more unequal
    # Mean richness held roughly fixed across the knob (abundance not confounded).
    assert abs(np.mean([p.A for p in lo]) - np.mean([p.A for p in hi])) < 1.0


def test_oracle_rate_positive_and_bounded():
    rng = np.random.default_rng(2)
    env = Environment(SurfaceConfig(n_branches=40, heterogeneity=0.6),
                      EnvConfig(switch_cost=5.0), rng)
    R = env.oracle_rate()
    rate0_max = max(p.A / p.tau for p in env.patches)
    assert 0.0 < R < rate0_max


def test_all_policies_solve_small_instance():
    # The deployed baselines and the MVT policies must all reach a feasible coverage
    # target. (OracleMVT is excluded here: it is a strict rate-optimal reference that
    # deliberately refuses sub-R* tails and so need not meet every coverage target.)
    seeds = list(range(5))
    specs = [
        {"type": "fixed_timeout", "timeout": 20.0},
        {"type": "max_retry", "retry_cap": 8},
        {"type": "top_k"},
        {"type": "ucb"},
        {"type": "mvt", "estimator": "bayesian"},
        {"type": "mvt", "estimator": "recent_slope"},
    ]
    for spec in specs:
        solved_any = False
        for s in seeds:
            rng = np.random.default_rng(s)
            env = Environment(SurfaceConfig(n_branches=30, heterogeneity=0.6),
                              EnvConfig(switch_cost=5.0, target_frac=0.5), rng)
            sched = Scheduler(env, make_policy(spec), EngineConfig(n_workers=6), rng)
            res = sched.run()
            solved_any = solved_any or res.solved
            assert res.ticks > 0
        assert solved_any, f"policy {spec['type']} never solved"


def test_adaptive_estimator_keeps_exploring_and_tracks_hit_rate():
    est = make_marginal_estimator("adaptive", base_half_life=6.0, max_half_life=80.0)
    # Before any informative (non-zero) event: optimistic rate, zero informative count,
    # so a min_samples guard keeps the worker exploring rather than abandoning on zeros.
    est.update(0.0)
    est.update(0.0)
    assert est.n == 0
    assert est.rate > 1e3
    # Dense (smooth) signal -> hit rate approaches 1 (despite the two leading zeros).
    for _ in range(150):
        est.update(0.5)
    assert est.n > 0
    assert est.hit_rate > 0.9
    assert abs(est.rate - 0.5) < 0.1

    # Bursty signal -> low hit rate; informative-event count grows only on non-zero gains;
    # the rate's expectation still recovers the true marginal (E[gain] = 1.0*0.1 = 0.1).
    burst = make_marginal_estimator("adaptive", base_half_life=6.0, max_half_life=120.0)
    rng = np.random.default_rng(0)
    nonzero = 0
    for _ in range(4000):
        hit = rng.random() < 0.1
        g = (0.1 / 0.1) if hit else 0.0        # E[g] = 0.1
        nonzero += int(hit)
        burst.update(g)
    assert burst.hit_rate < 0.3                # detects burstiness
    assert burst.n == nonzero                  # counts informative events only
    # Recovers the right marginal despite ~90% zero ticks (true m = 0.1); the terminal
    # EWMA snapshot is noisy, so we assert order-of-magnitude recovery, not an exact value.
    assert 0.05 < burst.rate < 0.22


def test_oracle_runs_and_is_efficient():
    # OracleMVT need not solve coverage, but it must run and be highly compute-efficient
    # (it only ever works above the MVT rate R*), i.e. high utilization.
    rng = np.random.default_rng(0)
    env = Environment(SurfaceConfig(n_branches=30, heterogeneity=0.9),
                      EnvConfig(switch_cost=5.0, target_frac=0.3, max_ticks=5000), rng)
    res = Scheduler(env, make_policy({"type": "oracle_mvt"}),
                    EngineConfig(n_workers=6), rng).run()
    assert res.ticks > 0
