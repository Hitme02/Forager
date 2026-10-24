"""GPU-parallel batch Monte-Carlo simulator for foraging-based frontier scheduling.

PAGP core artifact (Units III & V). The abstract foraging model is pure arithmetic, and the
thousands of independent episodes a heterogeneity sweep needs are embarrassingly parallel.
This runs ``B`` episodes x ``W`` workers entirely as PyTorch tensor ops, so the whole sweep
executes in lockstep on the GPU (MPS on Apple Silicon, CUDA on NVIDIA) with no Python loop
over episodes, workers, or branches -- only the unavoidable loop over simulation ticks, each
tick a handful of fused tensor kernels.

Model. Each worker forages an i.i.d. stream of patches drawn from the heterogeneity
distribution -- Charnov's original Marginal Value Theorem setting (a forager encountering
patches sampled from the environment). A patch has concave return g(c)=A(1-e^{-c/tau}) with
marginal (A/tau)e^{-c/tau}; richness A is log-normal with spread ``sigma`` (the heterogeneity
knob), mean held fixed; tau is coupled to A. A worker harvests its patch, abandons it per the
policy, pays a switch cost, then draws a fresh patch. This is the pure-foraging counterpart of
the finite-shared-pool CPU scheduler, and reproduces the same hypothesis (MVT vs fixed, and
the advantage growing with heterogeneity), so the two corroborate each other.

Policies (vectorised):
  - fixed_timeout : abandon when time-on-patch >= timeout.
  - mvt           : abandon when the estimated marginal (recent-slope EWMA) falls to the
                    frontier opportunity rate R, R estimated from fresh-patch entry-phase
                    intake (the same mechanism as the CPU engine).

Objective: cumulative validated yield across the pool reaches ``target_yield``; the recorded
outcome is time-to-objective (tick count) per episode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class BatchConfig:
    n_episodes: int = 4096
    n_workers: int = 8
    sigma: float = 0.6          # heterogeneity (log-normal spread of richness A)
    mean_A: float = 10.0
    mean_tau: float = 25.0
    tau_coupling: float = 0.5
    switch_cost: float = 5.0
    target_yield: float = 120.0  # cumulative pool yield that ends the episode
    max_ticks: int = 2000
    entry_window: int = 5
    noise: str = "smooth"        # "smooth" | "bursty"
    burst_p: float = 0.2


def _draw_patches(n: int, shape, cfg: BatchConfig, gen, device):
    """Sample (A, tau) for `shape` fresh patches from the heterogeneity distribution."""
    if cfg.sigma <= 0:
        A = torch.full(shape, cfg.mean_A, device=device)
    else:
        mu = math.log(cfg.mean_A) - 0.5 * cfg.sigma ** 2
        A = torch.exp(mu + cfg.sigma * torch.randn(shape, generator=gen, device=device))
    coupled = cfg.mean_tau * (A / cfg.mean_A)
    tau = cfg.tau_coupling * coupled + (1.0 - cfg.tau_coupling) * cfg.mean_tau
    tau = tau.clamp_min(1.0)
    return A, tau


def simulate(policy: dict, cfg: BatchConfig, *, device=None, seed: int = 0) -> dict:
    """Run a batch of foraging episodes on the GPU; return time-to-objective tensors."""
    device = device or pick_device()
    gen = torch.Generator(device=device).manual_seed(seed)
    B, W = cfg.n_episodes, cfg.n_workers
    shape = (B, W)
    f = lambda v: torch.full(shape, float(v), device=device)  # noqa: E731

    # --- policy params ---
    kind = policy["type"]
    timeout = float(policy.get("timeout", 20.0))
    hl = float(policy.get("half_life", 8.0))
    alpha = 1.0 - 0.5 ** (1.0 / max(hl, 1e-6))
    min_samples = int(policy.get("min_samples", 3))
    hyst = float(policy.get("hysteresis", 0.05))

    # --- per-(episode,worker) state ---
    A, tau = _draw_patches(B * W, shape, cfg, gen, device)
    spent = torch.zeros(shape, device=device)
    time_on = torch.zeros(shape, device=device)
    marg = torch.zeros(shape, device=device)
    marg_n = torch.zeros(shape, device=device)
    state = torch.zeros(shape, device=device)          # 0 working, 1 switching
    switch_rem = torch.zeros(shape, device=device)
    entry_ticks = torch.zeros(shape, device=device)
    entry_yield = torch.zeros(shape, device=device)

    # --- per-episode state ---
    f_fresh = torch.zeros(B, device=device)            # EWMA of fresh entry-phase rate
    f_fresh_n = torch.zeros(B, device=device)
    R = torch.zeros(B, device=device)
    total_yield = torch.zeros(B, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    tto = torch.full((B,), float(cfg.max_ticks), device=device)

    k, s = cfg.entry_window, cfg.switch_cost
    r_alpha = 1.0 - 0.5 ** (1.0 / 15.0)                 # R EWMA over entries

    for tick in range(1, cfg.max_ticks + 1):
        active = ~done                                  # [B]
        act2 = active.unsqueeze(1)                      # [B,1] broadcast

        # A. switching countdown (from the previous tick), then arrivals draw fresh patches.
        # Everything is branchless (masked `where`) -- no `.any()`/`.item()` sync per tick,
        # which is what lets the whole batch stay resident on the GPU across all ticks.
        switching = state == 1
        switch_rem = switch_rem - switching.float()
        arrived = switching & (switch_rem <= 0)
        nA, nTau = _draw_patches(0, shape, cfg, gen, device)
        A = torch.where(arrived, nA, A)
        tau = torch.where(arrived, nTau, tau)
        z = torch.zeros_like(spent)
        spent = torch.where(arrived, z, spent)
        time_on = torch.where(arrived, z, time_on)
        marg = torch.where(arrived, z, marg)
        marg_n = torch.where(arrived, z, marg_n)
        entry_ticks = torch.where(arrived, z, entry_ticks)
        entry_yield = torch.where(arrived, z, entry_yield)
        state = torch.where(arrived, z, state)          # -> working

        # B. working workers produce one tick of validated progress
        working = (state == 0) & act2
        wf = working.float()
        m = (A / tau) * torch.exp(-spent / tau)         # expected marginal g'(c)
        if cfg.noise == "smooth":
            gain = m * wf
        else:                                           # bursty: Bernoulli jumps, E[gain]=m
            hit = (torch.rand(shape, generator=gen, device=device) < cfg.burst_p).float()
            gain = (m / cfg.burst_p) * hit * wf
        spent = spent + wf
        time_on = time_on + wf
        total_yield = total_yield + gain.sum(dim=1)

        # recent-slope marginal EWMA (init to first sample)
        first = working & (marg_n == 0)
        marg = torch.where(first, gain,
                           torch.where(working, (1 - alpha) * marg + alpha * gain, marg))
        marg_n = marg_n + wf

        # entry-phase accumulation -> per-episode fresh rate -> R
        in_entry = working & (entry_ticks < k)
        entry_yield = entry_yield + gain * in_entry.float()
        entry_ticks = entry_ticks + in_entry.float()
        completed = in_entry & (entry_ticks >= k)
        rate = (entry_yield / k) * completed.float()             # [B,W]
        cnt = completed.float().sum(dim=1)                       # [B]
        ssum = rate.sum(dim=1)                                   # [B]
        has = cnt > 0
        sample = torch.where(has, ssum / cnt.clamp_min(1), f_fresh)
        newf = torch.where(f_fresh_n == 0, sample, (1 - r_alpha) * f_fresh + r_alpha * sample)
        f_fresh = torch.where(has, newf, f_fresh)
        f_fresh_n = f_fresh_n + has.float()
        R = f_fresh * (k / (k + s))

        # C. objective check -> record time-to-objective
        newly = active & (total_yield >= cfg.target_yield)
        tto = torch.where(newly, torch.tensor(float(tick), device=device), tto)
        done = done | newly
        # Early-exit check is itself a GPU->CPU sync, so do it only occasionally.
        if tick % 128 == 0 and bool(done.all()):
            break

        # D. abandonment decisions
        if kind == "fixed_timeout":
            abandon = working & (time_on >= timeout)
        elif kind == "mvt":
            thr = (R * (1.0 - hyst)).unsqueeze(1)
            abandon = working & (marg_n >= min_samples) & (marg <= thr)
        else:
            raise ValueError(f"unknown policy type: {kind!r}")
        abandon = abandon & act2
        state = torch.where(abandon, torch.ones_like(state), state)
        switch_rem = torch.where(abandon, torch.full_like(switch_rem, s), switch_rem)

    solved = tto < cfg.max_ticks
    return {
        "tto": tto,                      # [B] time-to-objective (ticks)
        "solved": solved,                # [B] bool
        "mean_tto": float(tto[solved].mean()) if bool(solved.any()) else float("nan"),
        "solve_rate": float(solved.float().mean()),
        "device": str(device),
    }
