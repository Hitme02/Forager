"""Attack-branch model as a foraging patch with diminishing returns.

Each attack branch is a "patch". Working the branch costs compute; the cumulative
*validated progress* it yields follows a concave (diminishing-returns) curve:

    g_i(c) = A_i * (1 - exp(-c / tau_i))

with marginal (instantaneous) return rate:

    g_i'(c) = (A_i / tau_i) * exp(-c / tau_i)

- A_i    : asymptotic yield -- the total validated progress available on the branch
           (a "rich" branch has large A_i; a dead branch has small A_i).
- tau_i  : return time-constant -- how quickly the branch pays out (small tau = fast,
           front-loaded payout; large tau = slow to mature).

The marginal rate g_i'(c) is monotonically decreasing in spent compute c, which is
exactly the diminishing-returns assumption the Marginal Value Theorem (MVT) relies on.

Realised progress is stochastic around this expected curve. Two noise models are
supported, both with expectation equal to the smooth marginal g_i'(c):

- "smooth" : realised gain per compute unit == g_i'(c)              (deterministic)
- "bursty" : realised gain arrives as Bernoulli(p) jumps of size g_i'(c)/p
             (small p => rarer, larger jumps => higher variance). This is the
             threat-to-validity case: a single exploit can flip a branch
             discontinuously, violating the smooth-concavity assumption.

Note the separation of concerns that makes the model well-posed: the *spent compute*
c_i is deterministic (it increments by one per compute unit actually worked), so it
alone drives position on the return curve. Only the *observed yield* is noisy. An
estimator therefore sees noisy yields and must recover the latent marginal rate --
which is the real technical problem the paper studies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Patch:
    """A single attack branch modelled as a foraging patch."""

    patch_id: int
    A: float              # asymptotic yield (branch richness)
    tau: float            # return time-constant (payout speed)
    prior_score: float = 0.0      # noisy recon signal of richness (known a priori)
    noise_model: str = "smooth"   # "smooth" | "bursty"
    burst_p: float = 0.2          # Bernoulli success prob per unit compute (bursty)

    # --- mutable state (per episode) ---
    spent: float = 0.0            # cumulative compute c_i invested so far
    yielded: float = 0.0          # cumulative *realised* validated progress

    def reset(self) -> None:
        self.spent = 0.0
        self.yielded = 0.0

    # ------------------------------------------------------------------ #
    # Latent (ground-truth) quantities -- known to the oracle, not the agent
    # ------------------------------------------------------------------ #
    def expected_cumulative(self, c: float) -> float:
        """g_i(c): expected cumulative yield after spending compute c."""
        return self.A * (1.0 - np.exp(-c / self.tau))

    def marginal(self, c: float | None = None) -> float:
        """g_i'(c): expected marginal return rate at spent compute c.

        Defaults to the current spent compute, i.e. the branch's *current* rate.
        """
        if c is None:
            c = self.spent
        return (self.A / self.tau) * np.exp(-c / self.tau)

    @property
    def remaining_potential(self) -> float:
        """Expected yield still extractable from the branch: A_i - g_i(spent)."""
        return self.A - self.expected_cumulative(self.spent)

    # ------------------------------------------------------------------ #
    # Dynamics -- what a worker observes when it spends one compute unit
    # ------------------------------------------------------------------ #
    def work_one(self, rng: np.random.Generator) -> float:
        """Spend one unit of compute on this branch; return the realised gain.

        Expectation of the returned gain equals the smooth marginal g_i'(spent).
        """
        m = self.marginal(self.spent)          # expected gain at current position
        if self.noise_model == "smooth":
            gain = m
        elif self.noise_model == "bursty":
            # Bernoulli jump with expectation m: E[gain] = p * (m / p) = m.
            p = self.burst_p
            gain = (m / p) if (rng.random() < p) else 0.0
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"unknown noise_model: {self.noise_model!r}")

        self.spent += 1.0
        self.yielded += gain
        return gain

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Patch(id={self.patch_id}, A={self.A:.3f}, tau={self.tau:.3f}, "
            f"spent={self.spent:.0f}, yielded={self.yielded:.3f})"
        )
