"""Tests for the GPU-parallel batch foraging simulator."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ffs.gpu.batch_sim import BatchConfig, pick_device, simulate


def test_batch_sim_runs_and_solves():
    cfg = BatchConfig(n_episodes=256, n_workers=8, sigma=0.6, target_yield=120.0)
    r = simulate({"type": "fixed_timeout", "timeout": 20.0}, cfg, device=torch.device("cpu"))
    assert r["solve_rate"] > 0.95
    assert r["mean_tto"] > 0


def test_mvt_advantage_grows_with_heterogeneity():
    # On the GPU batch sim, MVT should trail a fixed timeout at low heterogeneity and
    # overtake it at high heterogeneity -- the central hypothesis, reproduced in a second
    #, data-parallel implementation (corroborating the CPU engine).
    dev = torch.device("cpu")  # deterministic & CI-friendly
    fixed = {"type": "fixed_timeout", "timeout": 20.0}
    mvt = {"type": "mvt", "half_life": 8.0, "min_samples": 3, "hysteresis": 0.05}

    def speedup(sigma):
        cfg = BatchConfig(n_episodes=2048, n_workers=8, sigma=sigma, target_yield=120.0)
        rf = simulate(fixed, cfg, device=dev, seed=0)
        rm = simulate(mvt, cfg, device=dev, seed=0)
        return 100.0 * (rf["mean_tto"] - rm["mean_tto"]) / rf["mean_tto"]

    lo, hi = speedup(0.0), speedup(1.2)
    assert lo < 0        # fixed wins when branches are homogeneous
    assert hi > 0        # MVT wins when branches are heterogeneous
    assert hi > lo + 10  # and the advantage grows substantially


def test_pick_device_returns_device():
    d = pick_device()
    assert d.type in ("cuda", "mps", "cpu")
