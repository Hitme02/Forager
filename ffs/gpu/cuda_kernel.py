"""Hand-written Numba CUDA kernel: thread-per-episode foraging simulator.

A second GPU implementation, complementary to the PyTorch batch simulator (`batch_sim.py`).
Where PyTorch parallelises SIMD-style *across* episodes (one tensor lane per episode, one
fused kernel per tick), this maps **one CUDA thread to one whole episode**: each thread runs
the entire tick loop for its episode in registers/local memory, and thousands of threads run
concurrently across the GPU's SMs. This is the canonical "embarrassingly parallel Monte-Carlo
on CUDA" pattern and is what a PAGP Unit-III/V hand-written-kernel deliverable looks like.

Runs on any NVIDIA GPU (including Google Colab's T4 -- see `COLAB.md`). It is validated on
machines without a GPU via Numba's CUDA simulator (`NUMBA_ENABLE_CUDASIM=1`), which executes
the identical kernel in Python, so the logic is checked before it reaches real hardware.

The model and policies match `batch_sim.py` exactly (i.i.d.-patch MVT, fixed-timeout vs MVT
with recent-slope marginal and opportunity-rate R), so the two GPU paths corroborate each
other and the CPU engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import cuda, float32, int32
from numba.cuda.random import (create_xoroshiro128p_states,
                               xoroshiro128p_normal_float32,
                               xoroshiro128p_uniform_float32)

MAXW = 32  # compile-time cap on workers per episode (local arrays are fixed-size)


@cuda.jit(device=True)
def _draw_patch(rng, tid, sigma, mu, mean_A, mean_tau, coupling):
    """Sample one fresh patch (A, tau) from the heterogeneity distribution."""
    if sigma <= 0.0:
        A = mean_A
    else:
        A = math.exp(mu + sigma * xoroshiro128p_normal_float32(rng, tid))
    tau = coupling * (mean_tau * (A / mean_A)) + (1.0 - coupling) * mean_tau
    if tau < 1.0:
        tau = 1.0
    return A, tau


@cuda.jit
def forage_kernel(rng, out_tto, n_eps, W, sigma, mu, mean_A, mean_tau, coupling,
                  switch_cost, target, max_ticks, entry_window, alpha, r_alpha,
                  policy_code, timeout, min_samples, hyst, noise_code, burst_p):
    e = cuda.grid(1)
    if e >= n_eps:
        return

    A = cuda.local.array(MAXW, float32)
    tau = cuda.local.array(MAXW, float32)
    spent = cuda.local.array(MAXW, float32)
    time_on = cuda.local.array(MAXW, float32)
    marg = cuda.local.array(MAXW, float32)
    marg_n = cuda.local.array(MAXW, float32)
    state = cuda.local.array(MAXW, int32)          # 0 working, 1 switching
    switch_rem = cuda.local.array(MAXW, float32)
    entry_ticks = cuda.local.array(MAXW, float32)
    entry_yield = cuda.local.array(MAXW, float32)

    for w in range(W):
        a, t = _draw_patch(rng, e, sigma, mu, mean_A, mean_tau, coupling)
        A[w] = a; tau[w] = t
        spent[w] = 0.0; time_on[w] = 0.0; marg[w] = 0.0; marg_n[w] = 0.0
        state[w] = 0; switch_rem[w] = 0.0; entry_ticks[w] = 0.0; entry_yield[w] = 0.0

    f_fresh = 0.0; f_fresh_n = 0.0; R = 0.0; total = 0.0
    tto = float(max_ticks)

    for tick in range(1, max_ticks + 1):
        # A. switching countdown; arrivals draw a fresh patch
        for w in range(W):
            if state[w] == 1:
                switch_rem[w] -= 1.0
                if switch_rem[w] <= 0.0:
                    a, t = _draw_patch(rng, e, sigma, mu, mean_A, mean_tau, coupling)
                    A[w] = a; tau[w] = t
                    spent[w] = 0.0; time_on[w] = 0.0; marg[w] = 0.0; marg_n[w] = 0.0
                    entry_ticks[w] = 0.0; entry_yield[w] = 0.0
                    state[w] = 0

        # B. working workers produce; accumulate entry-phase rate for R
        entry_sum = 0.0; entry_cnt = 0.0
        for w in range(W):
            if state[w] == 0:
                m = (A[w] / tau[w]) * math.exp(-spent[w] / tau[w])
                if noise_code == 0:
                    gain = m
                else:
                    if xoroshiro128p_uniform_float32(rng, e) < burst_p:
                        gain = m / burst_p
                    else:
                        gain = 0.0
                spent[w] += 1.0; time_on[w] += 1.0; total += gain
                if marg_n[w] == 0.0:
                    marg[w] = gain
                else:
                    marg[w] = (1.0 - alpha) * marg[w] + alpha * gain
                marg_n[w] += 1.0
                if entry_ticks[w] < entry_window:
                    entry_yield[w] += gain
                    entry_ticks[w] += 1.0
                    if entry_ticks[w] >= entry_window:
                        entry_sum += entry_yield[w] / entry_window
                        entry_cnt += 1.0

        # C. update frontier opportunity rate R
        if entry_cnt > 0.0:
            sample = entry_sum / entry_cnt
            if f_fresh_n == 0.0:
                f_fresh = sample
            else:
                f_fresh = (1.0 - r_alpha) * f_fresh + r_alpha * sample
            f_fresh_n += 1.0
            R = f_fresh * (entry_window / (entry_window + switch_cost))

        # D. objective
        if total >= target:
            tto = float(tick)
            break

        # E. abandonment
        for w in range(W):
            if state[w] == 0:
                ab = False
                if policy_code == 0:
                    if time_on[w] >= timeout:
                        ab = True
                else:
                    if marg_n[w] >= min_samples and marg[w] <= R * (1.0 - hyst):
                        ab = True
                if ab:
                    state[w] = 1
                    switch_rem[w] = switch_cost

    out_tto[e] = tto


# --------------------------------------------------------------------------- #
# Host driver
# --------------------------------------------------------------------------- #
@dataclass
class CudaConfig:
    n_episodes: int = 65536
    n_workers: int = 8
    sigma: float = 0.6
    mean_A: float = 10.0
    mean_tau: float = 25.0
    tau_coupling: float = 0.5
    switch_cost: float = 5.0
    target_yield: float = 120.0
    max_ticks: int = 2000
    entry_window: int = 5
    noise: str = "smooth"
    burst_p: float = 0.2


def simulate_cuda(policy: dict, cfg: CudaConfig, *, seed: int = 0,
                  threads_per_block: int = 128) -> dict:
    """Launch the CUDA kernel (or its simulator) and return time-to-objective results."""
    assert cfg.n_workers <= MAXW, f"n_workers must be <= {MAXW}"
    n = cfg.n_episodes
    mu = math.log(cfg.mean_A) - 0.5 * cfg.sigma ** 2
    hl = float(policy.get("half_life", 8.0))
    alpha = 1.0 - 0.5 ** (1.0 / max(hl, 1e-6))
    r_alpha = 1.0 - 0.5 ** (1.0 / 15.0)
    policy_code = 0 if policy["type"] == "fixed_timeout" else 1
    noise_code = 0 if cfg.noise == "smooth" else 1

    rng = create_xoroshiro128p_states(n, seed=seed)
    out = cuda.device_array(n, dtype=np.float32)
    blocks = (n + threads_per_block - 1) // threads_per_block
    forage_kernel[blocks, threads_per_block](
        rng, out, n, cfg.n_workers, float(cfg.sigma), float(mu), float(cfg.mean_A),
        float(cfg.mean_tau), float(cfg.tau_coupling), float(cfg.switch_cost),
        float(cfg.target_yield), int(cfg.max_ticks), float(cfg.entry_window),
        float(alpha), float(r_alpha), int(policy_code),
        float(policy.get("timeout", 20.0)), float(policy.get("min_samples", 3)),
        float(policy.get("hysteresis", 0.05)), int(noise_code), float(cfg.burst_p),
    )
    tto = out.copy_to_host()
    solved = tto < cfg.max_ticks
    return {
        "tto": tto,
        "solved": solved,
        "mean_tto": float(tto[solved].mean()) if solved.any() else float("nan"),
        "solve_rate": float(solved.mean()),
    }
