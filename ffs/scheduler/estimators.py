"""Online estimators -- the real technical content of the MVT rule.

Two quantities must be estimated from noisy, bursty, non-stationary observations:

1. g'(c), the marginal return rate of the branch a worker is currently on.
   This is noisy early (few samples) and the burst noise model can make a depleted
   branch suddenly pay out. Two estimators are provided:

     - RecentSlopeEstimator : an exponentially-weighted moving average (EWMA) of
       realised per-compute gains. Simple, fast, no uncertainty model. This is the
       "recent-slope" option from the proposal.

     - BayesianMarginalEstimator : a conjugate Normal-mean estimator whose prior
       *defaults to keep-exploring* -- the prior mean is optimistic, so with few
       samples the estimate stays high and the worker does not abandon a branch
       prematurely. As evidence accrues the posterior mean converges to the observed
       intake. It also exposes posterior uncertainty, enabling an optional optimism
       bonus (upper-confidence marginal) to further delay premature abandonment.
       This is the "Bayesian posterior" option from the proposal.

2. R, the frontier opportunity rate -- the long-run rate a worker would earn by
   abandoning and working the best alternative, *net of the switching cost s*.
   FrontierRateEstimator tracks it as a discounted ratio of realised yield to total
   compute spent across the pool, where "total compute" includes compute burnt while
   switching. That denominator is what nets out the travel cost: when the pool churns
   a lot, the realised environment rate falls automatically. Discounting (an EWMA)
   handles non-stationarity -- R drifts as the agent learns the surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Marginal-rate estimators  (per worker, reset on each branch switch)
# --------------------------------------------------------------------------- #
class MarginalEstimator:
    """Interface for estimating a branch's current marginal return rate g'(c)."""

    def reset(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def update(self, gain: float) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def rate(self) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def n(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class RecentSlopeEstimator(MarginalEstimator):
    """EWMA of realised per-compute gains. ``half_life`` sets the memory window."""

    half_life: float = 8.0
    _ewma: float = 0.0
    _count: int = 0
    _alpha: float = 0.0

    def __post_init__(self) -> None:
        # EWMA decay so that a sample's weight halves every `half_life` steps.
        self._alpha = 1.0 - 0.5 ** (1.0 / max(self.half_life, 1e-6))

    def reset(self) -> None:
        self._ewma = 0.0
        self._count = 0

    def update(self, gain: float) -> None:
        self._count += 1
        if self._count == 1:
            self._ewma = gain
        else:
            self._ewma = (1.0 - self._alpha) * self._ewma + self._alpha * gain

    @property
    def rate(self) -> float:
        return float(self._ewma)

    @property
    def n(self) -> int:
        return self._count


@dataclass
class BayesianMarginalEstimator(MarginalEstimator):
    """Conjugate Normal-mean estimate of the marginal rate with a keep-exploring prior.

    Model: observed gains ~ Normal(m, obs_var), with a Normal prior on m centred at an
    optimistic `prior_mean` with pseudo-count `prior_strength`. Posterior mean is the
    pseudo-count-weighted blend of prior and sample mean, so early on (few samples) the
    optimistic prior dominates and the worker keeps exploring the branch; later the data
    dominate. `optimism` adds k posterior standard deviations as an upper-confidence
    bonus (0 = plain posterior mean).
    """

    prior_mean: float | str = "auto"  # optimistic initial rate; "auto" = first gain
    prior_strength: float = 2.0    # pseudo-observations backing the prior
    obs_var: float = 0.25          # assumed observation variance (of a single gain)
    optimism: float = 0.0          # UCB bonus in posterior-std units
    half_life: float = 8.0         # decay of the likelihood, so it tracks CURRENT g'(c)

    _dsum: float = 0.0             # discounted sum of gains
    _dcount: float = 0.0           # discounted effective sample size
    _count: int = 0
    _prior: float = 1.0
    _gamma: float = 0.0

    def __post_init__(self) -> None:
        # A fixed numeric prior is set now; "auto" is deferred to the first gain so the
        # keep-exploring prior is *calibrated* to the branch's own entry rate rather
        # than a hand-set constant that may not match the surface's scale.
        if self.prior_mean != "auto":
            self._prior = float(self.prior_mean)
        # Discount so older observations fade: the estimate tracks the *current* marginal
        # of a declining branch rather than its visit-long average (which lags and would
        # make the policy abandon far too late).
        self._gamma = 0.5 ** (1.0 / max(self.half_life, 1e-6))

    def reset(self) -> None:
        self._dsum = self._dcount = 0.0
        self._count = 0

    def update(self, gain: float) -> None:
        if self._count == 0 and self.prior_mean == "auto":
            self._prior = gain          # calibrate the keep-exploring prior on entry
        self._dsum = self._gamma * self._dsum + gain
        self._dcount = self._gamma * self._dcount + 1.0
        self._count += 1

    @property
    def _post_mean(self) -> float:
        k0, m0 = self.prior_strength, self._prior
        return (k0 * m0 + self._dsum) / (k0 + self._dcount)

    @property
    def _post_std(self) -> float:
        # Posterior variance of the mean under the Normal-Normal model.
        return float(np.sqrt(self.obs_var / (self.prior_strength + self._dcount)))

    @property
    def rate(self) -> float:
        return float(self._post_mean + self.optimism * self._post_std)

    @property
    def n(self) -> int:
        return self._count


@dataclass
class AdaptiveMarginalEstimator(MarginalEstimator):
    """Burstiness-adaptive marginal estimator -- the mechanism contribution.

    Bursty validated-progress (a single exploit flips a branch from flat to solved) makes
    a fixed short window mis-read g'(c): most ticks are zero, which looks like depletion, so
    the worker abandons a live branch prematurely. A fixed long window is robust to bursts
    but sluggish on genuinely smooth, depleting branches. Neither fixed choice is right
    across the burstiness the signal actually exhibits.

    This estimator adapts online. It tracks the **hit rate** p_hat -- the EWMA fraction of
    ticks with a non-zero gain -- and sets the effective smoothing half-life inversely to
    it:

        eff_half_life = clip( base_half_life / max(p_hat, p_floor),
                              base_half_life, max_half_life )

    so the rate EWMA spans roughly a *constant number of informative events* regardless of
    burstiness: tight and responsive when the signal is dense (p_hat -> 1 => eff = base),
    broad and stable when it is bursty (p_hat small => eff grows). The rate itself is the
    EWMA of gains *including zeros*, whose expectation is the true marginal m irrespective
    of burstiness; only its variance is controlled by the adaptive window.

    `n` reports the count of **informative (non-zero) events**, not raw ticks, so a policy's
    `min_samples` guard means "enough real evidence of depletion" -- under heavy burstiness
    that correctly waits for several jumps before permitting abandonment, and under a smooth
    signal it reduces to the ordinary sample count. Until the first non-zero gain the rate is
    reported optimistically (keep-exploring), so no branch is abandoned on zeros alone.
    """

    base_half_life: float = 6.0     # responsive window when the signal is dense
    max_half_life: float = 80.0     # cap for very bursty signals
    hit_half_life: float = 20.0     # memory of the hit-rate estimate p_hat
    p_floor: float = 0.02           # floor on p_hat (bounds eff_half_life)
    optimistic_rate: float = 1e6    # reported rate before any informative event
    # Seed for p_hat. None (as submitted) seeds it from the first tick's hit/miss; a value
    # (e.g. 1.0) seeds it optimistically, so a branch whose first action yields nothing (an
    # agent's premature step) is not read as bursty. Identical when the first tick is a hit.
    hit_init: float | None = None

    _rate: float = 0.0
    _phat: float = 0.0
    _nonzero: int = 0
    _total: int = 0
    _inited: bool = False
    _a_hit: float = 0.0

    def __post_init__(self) -> None:
        self._a_hit = 1.0 - 0.5 ** (1.0 / max(self.hit_half_life, 1e-6))

    def reset(self) -> None:
        self._rate = 0.0
        self._phat = 0.0
        self._nonzero = self._total = 0
        self._inited = False

    def update(self, gain: float) -> None:
        self._total += 1
        hit = 1.0 if gain > 1e-12 else 0.0
        if self._total == 1:
            self._phat = hit if self.hit_init is None else (
                (1.0 - self._a_hit) * self.hit_init + self._a_hit * hit)
        else:
            self._phat = (1.0 - self._a_hit) * self._phat + self._a_hit * hit
        if hit:
            self._nonzero += 1
        eff_hl = min(self.max_half_life, self.base_half_life / max(self._phat, self.p_floor))
        alpha = 1.0 - 0.5 ** (1.0 / max(eff_hl, 1e-6))
        if not self._inited:
            self._rate = gain
            self._inited = True
        else:
            self._rate = (1.0 - alpha) * self._rate + alpha * gain

    @property
    def rate(self) -> float:
        # Keep exploring until a branch has produced at least one informative event.
        return float(self._rate) if self._nonzero > 0 else float(self.optimistic_rate)

    @property
    def n(self) -> int:
        return self._nonzero

    @property
    def hit_rate(self) -> float:
        return float(self._phat)


def make_marginal_estimator(name: str, **kw) -> MarginalEstimator:
    name = name.lower()
    if name in ("recent_slope", "recent-slope", "ewma"):
        allowed = {"half_life"}
        return RecentSlopeEstimator(**{k: v for k, v in kw.items() if k in allowed})
    if name in ("bayesian", "bayes"):
        allowed = {"prior_mean", "prior_strength", "obs_var", "optimism", "half_life"}
        return BayesianMarginalEstimator(**{k: v for k, v in kw.items() if k in allowed})
    if name in ("adaptive", "burst_adaptive", "adaptive_slope"):
        allowed = {"base_half_life", "max_half_life", "hit_half_life", "p_floor",
                   "optimistic_rate", "hit_init"}
        return AdaptiveMarginalEstimator(**{k: v for k, v in kw.items() if k in allowed})
    raise ValueError(f"unknown marginal estimator: {name!r}")


# --------------------------------------------------------------------------- #
# Frontier opportunity-rate estimator  (shared across the pool)
# --------------------------------------------------------------------------- #
@dataclass
class FrontierRateEstimator:
    """Online estimate of the MVT environment opportunity rate R, from fresh-branch entry.

    R is what a worker would earn by abandoning its current branch and working the best
    available *fresh* branch, net of the switch cost s. We estimate it directly and
    continuously from the **early-phase intake** workers achieve just after entering a
    branch -- the rate a fresh branch pays before it depletes -- rather than from completed
    visits:

        f_fresh = EWMA over branch-entries of ( yield in the first `entry_window`
                  productive ticks ) / entry_window
        R       = f_fresh * entry_window / (entry_window + switch_cost)

    The discount `entry_window / (entry_window + s)` prices in travel: switching costs s
    ticks of nothing before the fresh branch's early intake begins. A worker leaves its
    current branch when its marginal falls below this R.

    Why entry-phase, not completed-visit, rate. The completed-visit average is dragged
    down by the low-marginal *tails* a worker grinds before leaving, so it underestimates
    what switching is worth and the worker never migrates (a chicken-and-egg: no switches
    -> no visit data -> R stays low -> no switches). The entry-phase rate is observed on
    every single branch entry, needs no completed visits to bootstrap, and correctly
    reflects the high intake a fresh branch offers -- so a worker stuck on a depleting
    branch's tail sees its marginal drop below R and migrates. Discounting (an EWMA over
    entries) tracks the non-stationary R as the surface is harvested and fresh branches
    become scarcer.
    """

    switch_cost: float = 5.0
    entry_window: int = 5          # productive ticks that define a branch's "fresh" phase
    half_life_entries: float = 15.0
    # "ticks" (as submitted): the entry phase is the first `entry_window` ticks on a branch.
    # "productive": it lasts until `entry_window` ticks with non-zero gain have been seen, and
    # its rate is yield / (elapsed ticks + switch_cost), so an agent's reconnaissance ramp is
    # charged as travel time instead of diluting the fresh-branch intake. When every tick is
    # productive the two modes coincide exactly.
    entry_mode: str = "ticks"
    _f: float = 0.0                # EWMA of fresh entry-phase intake rate
    _n: int = 0                    # number of recorded entry-phase samples
    _alpha: float = 0.0

    def __post_init__(self) -> None:
        self._alpha = 1.0 - 0.5 ** (1.0 / max(self.half_life_entries, 1e-6))

    def reset(self) -> None:
        self._f = 0.0
        self._n = 0

    def record_entry_phase(self, entry_rate: float, elapsed: int | None = None) -> None:
        """Record a freshly entered branch's entry-phase intake.

        "ticks" mode: `entry_rate` is the mean gain over the first `entry_window` ticks.
        "productive" mode: `entry_rate` is the entry-phase yield and `elapsed` the ticks it
        took; it is stored rescaled so that `rate` returns yield / (elapsed + switch_cost)."""
        if self.entry_mode == "productive" and elapsed is not None:
            k, s = self.entry_window, self.switch_cost
            entry_rate = entry_rate * (k + s) / (k * (elapsed + s))
        self._f = entry_rate if self._n == 0 else (1.0 - self._alpha) * self._f + self._alpha * entry_rate
        self._n += 1

    # Kept for API compatibility; completed-visit data is no longer used for R.
    def record_entry_rate(self, entry_gain: float) -> None:  # pragma: no cover
        pass

    def record_visit(self, visit_yield: float, residence_time: float) -> None:  # pragma: no cover
        pass

    @property
    def rate(self) -> float:
        if self._n == 0:
            return 0.0             # keep exploring until a fresh-phase rate is observed
        return float(self._f * self.entry_window / (self.entry_window + self.switch_cost))
