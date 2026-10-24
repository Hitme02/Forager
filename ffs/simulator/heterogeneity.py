"""Generate an attack surface (a set of branches) with *tunable heterogeneity*.

Branch-return heterogeneity is the independent variable of the central hypothesis:

    "MVT abandonment's advantage over fixed-timeout grows with branch-return
     heterogeneity."

So the generator must turn a single scalar knob into a set of branch richnesses
{A_i} whose spread is controlled, while holding the *total* available yield roughly
constant (so that higher heterogeneity means 'the same reward, more unequally
distributed' rather than 'more reward overall' -- otherwise the comparison confounds
heterogeneity with abundance).

Parameterisation
----------------
We draw richnesses A_i from a log-normal distribution:

    A_i = exp(mu + sigma * Z_i),    Z_i ~ N(0, 1)

- `sigma` is the **heterogeneity knob**. sigma = 0 => all branches identical
  (homogeneous). Larger sigma => a few very rich branches and many poor ones
  (heavy-tailed, the regime where which-branch-to-abandon-when actually matters).
- `mu` is then chosen so that the *expected mean richness* is fixed at `mean_A`,
  i.e. mu = log(mean_A) - sigma^2 / 2, keeping total yield comparable across the
  sweep. After sampling we additionally rescale so the realised mean hits mean_A
  exactly, removing sampling noise from the sweep axis.

The time-constants tau_i can be coupled to richness (richer branches take longer to
mature -- `tau_coupling`) or drawn independently. Coupling is the more realistic and
more interesting case: it means a rich branch is not merely "more", it also has a
different rate profile, so a single fixed timeout cannot be right for both rich and
poor branches at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .patch import Patch


@dataclass
class SurfaceConfig:
    """Configuration for generating an attack surface."""

    n_branches: int = 40
    mean_A: float = 10.0          # target mean branch richness (abundance, held fixed)
    heterogeneity: float = 0.6    # sigma of the log-normal over A_i (the knob)
    mean_tau: float = 25.0        # mean return time-constant
    tau_coupling: float = 0.5     # 0 => tau independent of A; 1 => tau scales with A
    tau_spread: float = 0.2       # log-normal sigma on the independent part of tau
    prior_noise: float = 0.4      # log-normal sigma of the recon prior vs true A
    noise_model: str = "smooth"   # "smooth" | "bursty"
    burst_p: float = 0.2

    def __post_init__(self) -> None:
        if self.n_branches < 1:
            raise ValueError("n_branches must be >= 1")
        if self.heterogeneity < 0:
            raise ValueError("heterogeneity (sigma) must be >= 0")
        if self.noise_model not in ("smooth", "bursty"):
            raise ValueError(f"unknown noise_model: {self.noise_model!r}")
        if not (0.0 < self.burst_p <= 1.0):
            raise ValueError("burst_p must be in (0, 1]")


def make_surface(cfg: SurfaceConfig, rng: np.random.Generator) -> list[Patch]:
    """Build a list of Patches with the configured heterogeneity."""
    n = cfg.n_branches
    sigma = cfg.heterogeneity

    # --- richnesses A_i: log-normal with mean held at mean_A ---
    if sigma == 0.0:
        A = np.full(n, cfg.mean_A, dtype=float)
    else:
        mu = np.log(cfg.mean_A) - 0.5 * sigma**2
        A = np.exp(mu + sigma * rng.standard_normal(n))
        A *= cfg.mean_A / A.mean()        # rescale so realised mean == mean_A exactly

    # --- time-constants tau_i: partly coupled to richness, partly independent ---
    # Coupled part: richer branches mature proportionally slower.
    coupled = cfg.mean_tau * (A / cfg.mean_A)
    independent = np.exp(
        np.log(cfg.mean_tau) - 0.5 * cfg.tau_spread**2
        + cfg.tau_spread * rng.standard_normal(n)
    )
    tau = cfg.tau_coupling * coupled + (1.0 - cfg.tau_coupling) * independent
    tau = np.clip(tau, 1.0, None)         # keep tau positive and well-behaved

    # --- recon prior: a noisy, order-preserving-ish signal of true richness A ---
    # The scheduler sees prior_score (not A) when choosing which branch to enter.
    prior = A * np.exp(
        -0.5 * cfg.prior_noise**2 + cfg.prior_noise * rng.standard_normal(n)
    )

    patches = [
        Patch(
            patch_id=i,
            A=float(A[i]),
            tau=float(tau[i]),
            prior_score=float(prior[i]),
            noise_model=cfg.noise_model,
            burst_p=cfg.burst_p,
        )
        for i in range(n)
    ]
    return patches


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array -- a reportable heterogeneity measure.

    0 = perfectly equal branches, -> 1 = all yield concentrated in one branch.
    Useful as an *outcome* descriptor of the generated surface, independent of the
    sigma knob (two surfaces with the same sigma can realise different Gini).
    """
    v = np.sort(np.asarray(values, dtype=float))
    n = v.size
    if n == 0 or v.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * v)) / (n * v.sum()) - (n + 1.0) / n)
