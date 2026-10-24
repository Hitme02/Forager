"""The attack-surface environment: a frontier of branches plus foraging economics.

The environment owns:
- the set of branches (patches) the agent may work,
- the **switching cost** `s` (the "travel time": compute to abandon a branch, pull a
  new frontier task, and re-establish reconnaissance state before progress resumes),
- the **objective** (when the episode is considered solved),
- the ground-truth **MVT-optimal environment rate** R* (for the oracle policy and as
  an analysis reference). The agent never sees R*; it must estimate its own R online.

Objective modes
---------------
- "coverage"  : reach a target fraction of the total available validated yield,
                i.e. cumulative realised progress >= target_frac * sum_i A_i.
                This is the natural "harvest the attack surface" objective and the
                one for which minimising time == maximising average intake rate ==
                exactly what MVT optimises.
- "flag"      : one designated high-value branch holds the objective; the episode is
                solved when that branch's realised yield crosses a threshold. A
                capture-the-flag style objective (used for robustness / ablation).

Time vs compute
---------------
With W workers running in parallel, one simulation tick advances wall-clock time by 1
and spends up to W units of compute (a switching worker spends compute but yields
nothing that tick). "time-to-objective" is the tick count at which the objective is
first met; "compute-to-objective" is W * ticks. Reporting both keeps the parallel
(wall-clock) and the efficiency (per-compute) stories separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .heterogeneity import SurfaceConfig, gini, make_surface
from .patch import Patch


@dataclass
class EnvConfig:
    """Environment-level configuration."""

    switch_cost: float = 5.0          # s: compute to migrate + re-establish recon
    objective: str = "coverage"       # "coverage" | "flag"
    target_frac: float = 0.60         # coverage: fraction of total yield to harvest
    flag_branch: str = "richest"      # flag: "richest" | "random"
    flag_frac: float = 0.80           # flag: fraction of that branch's A to extract
    max_ticks: int = 100_000          # safety cap on episode length

    def __post_init__(self) -> None:
        if self.objective not in ("coverage", "flag"):
            raise ValueError(f"unknown objective: {self.objective!r}")
        if self.switch_cost < 0:
            raise ValueError("switch_cost must be >= 0")


class Environment:
    """A concrete attack surface instance for one episode."""

    def __init__(
        self,
        surface_cfg: SurfaceConfig,
        env_cfg: EnvConfig,
        rng: np.random.Generator,
    ) -> None:
        self.surface_cfg = surface_cfg
        self.cfg = env_cfg
        self.rng = rng
        self.patches: list[Patch] = make_surface(surface_cfg, rng)

        self.total_yield_available = float(sum(p.A for p in self.patches))
        self.gini = gini(np.array([p.A for p in self.patches]))

        # Objective target in units of realised cumulative yield.
        if env_cfg.objective == "coverage":
            self.target_yield = env_cfg.target_frac * self.total_yield_available
            self.flag_id: int | None = None
        else:  # flag
            if env_cfg.flag_branch == "richest":
                self.flag_id = int(np.argmax([p.A for p in self.patches]))
            else:
                self.flag_id = int(rng.integers(len(self.patches)))
            self.target_yield = env_cfg.flag_frac * self.patches[self.flag_id].A

        self._oracle_rate: float | None = None

    # ------------------------------------------------------------------ #
    # Objective bookkeeping
    # ------------------------------------------------------------------ #
    @property
    def total_yielded(self) -> float:
        return float(sum(p.yielded for p in self.patches))

    def objective_met(self) -> bool:
        if self.cfg.objective == "coverage":
            return self.total_yielded >= self.target_yield
        else:
            assert self.flag_id is not None
            return self.patches[self.flag_id].yielded >= self.target_yield

    def progress_fraction(self) -> float:
        """How far toward the objective, in [0, 1] (clipped)."""
        if self.cfg.objective == "coverage":
            denom = self.target_yield
            num = self.total_yielded
        else:
            assert self.flag_id is not None
            denom = self.target_yield
            num = self.patches[self.flag_id].yielded
        return float(min(1.0, num / denom)) if denom > 0 else 1.0

    # ------------------------------------------------------------------ #
    # Oracle: MVT-optimal long-run environment rate R*
    # ------------------------------------------------------------------ #
    def oracle_rate(self) -> float:
        """Self-consistent MVT environment rate R* for this (realised) surface.

        For exponential patches g_i(c) = A_i (1 - e^{-c/tau_i}), the MVT leaving rule
        g_i'(t*) = R gives, for every branch with initial rate A_i/tau_i > R:

            t_i*(R) = tau_i * ln( A_i / (tau_i * R) )          (leaving time)
            y_i*(R) = A_i - tau_i * R                          (yield at leaving)

        and the optimal environment rate R* is the fixed point

            R* = sum_i y_i*(R*) / sum_i ( t_i*(R*) + s )

        taken over the branches worth entering (A_i/tau_i > R*). We solve the scalar
        equation F(R) = R * sum(t_i* + s) - sum(y_i*) = 0 with Brent's method.

        This is cached; it depends only on the (fixed) patch parameters and s.
        """
        if self._oracle_rate is not None:
            return self._oracle_rate

        A = np.array([p.A for p in self.patches])
        tau = np.array([p.tau for p in self.patches])
        s = self.cfg.switch_cost
        rate0 = A / tau                       # initial marginal rate of each branch

        def F(R: float) -> float:
            enter = rate0 > R                 # branches worth entering at rate R
            if not np.any(enter):
                # Nothing worth entering: environment rate "should" be lower.
                return R  # F(R) > 0 => pushes root toward smaller R
            t_star = tau[enter] * np.log(A[enter] / (tau[enter] * R))
            y_star = A[enter] - tau[enter] * R
            return R * np.sum(t_star + s) - np.sum(y_star)

        lo = 1e-9
        hi = float(rate0.max())               # R* < max initial rate
        # F(lo) < 0 (yields dominate), F(hi) > 0 (no branch worth entering): bracketed.
        try:
            self._oracle_rate = float(brentq(F, lo, hi, xtol=1e-9, maxiter=200))
        except ValueError:
            # Degenerate surface; fall back to a safe small rate.
            self._oracle_rate = float(rate0.mean() * 0.1)
        return self._oracle_rate

    def reset(self) -> None:
        for p in self.patches:
            p.reset()
