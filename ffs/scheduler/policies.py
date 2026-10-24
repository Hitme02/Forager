"""Abandonment-and-migration policies (the compared schedulers' decision rules).

Every policy answers two questions for a parallel worker:

  * select_branch(worker, available) -> which frontier branch to enter next
  * should_abandon(worker, R)        -> abandon the current branch now?

To isolate the *contribution* (the MVT abandonment rule), branch **selection** is held
identical across the core policies: greedily enter the available branch with the
highest recon prior score. Only the **abandonment** rule differs. (Top-K and UCB are
deliberate exceptions: they are fundamentally selection/activation strategies, included
to test whether *stopping* adds value beyond *prioritising*, and to pre-empt the
"isn't this just a bandit?" review.)

Policies:
  FixedTimeout   - abandon after a static per-branch compute budget. THE headline baseline.
  MaxRetry       - abandon after R consecutive near-zero-gain ticks (a static retry cap).
  TopK           - only ever work the K highest-prior branches; abandon only when a branch
                   is effectively depleted (priority, no *early* stopping).
  UCB            - index-based activation: work the branch with the best UCB index; switch
                   when another available branch's index beats the current one. The bandit
                   comparator.
  MVT            - abandon when the estimated marginal g'(c) falls to the estimated frontier
                   opportunity rate R. THE contribution.
  GuardedMVT     - MVT's rule applied only where the signal is *diminishing* (discovered
                   online), with a committed-budget fallback elsewhere. Boundary-aware: keeps
                   MVT's win under diminishing value and does not collapse under constant
                   hazard, with one config for both regimes.
  OracleMVT      - MVT using the *true* marginal and the analytic R* (upper-bound reference).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..simulator.patch import Patch
from .estimators import MarginalEstimator, make_marginal_estimator


class WorkerView(Protocol):
    """The subset of worker state policies rely on (duck-typed from engine.Worker)."""

    wid: int
    patch: Patch | None
    time_on_patch: int
    last_gain: float
    marg_est: MarginalEstimator | None


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #
class Policy:
    name = "base"

    def reset(self, patches: list[Patch], rng: np.random.Generator) -> None:
        self._patches = patches
        self._rng = rng

    # Called when a worker is (re)assigned to a branch -- set up per-worker state.
    def on_assign(self, worker: WorkerView) -> None:
        pass

    # Called every tick the worker produced a (possibly zero) gain.
    def observe(self, worker: WorkerView, gain: float) -> None:
        pass

    def select_branch(
        self, worker: WorkerView, available: list[Patch], memory: "BranchMemory"
    ) -> Patch | None:
        """Default selection, shared by the core policies to isolate the abandon rule.

        Prefer *fresh* (unvisited) branches ranked by recon prior; only if none are
        available fall back to re-entering the least-depleted visited branch (ranked by
        the scheduler's remembered recent intake rate). This keeps selection identical
        across FixedTimeout/MaxRetry/MVT, prevents churn on depleted branches, and still
        guarantees the objective is reachable.
        """
        if not available:
            return None
        unvisited = [p for p in available if p.patch_id not in memory.visited]
        if unvisited:
            return max(unvisited, key=lambda p: p.prior_score)
        return max(available, key=lambda p: memory.recent_rate.get(p.patch_id, 0.0))

    def should_abandon(
        self, worker: WorkerView, R: float, memory: "BranchMemory"
    ) -> bool:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Static baselines
# --------------------------------------------------------------------------- #
@dataclass
class FixedTimeout(Policy):
    """Abandon a branch after a fixed compute budget. The deployed heuristic."""

    timeout: float = 20.0
    name: str = field(default="fixed_timeout", init=False)

    def should_abandon(self, worker: WorkerView, R: float, memory: "BranchMemory") -> bool:
        return worker.time_on_patch >= self.timeout


@dataclass
class MaxRetry(Policy):
    """Abandon after `retry_cap` consecutive near-zero-gain ticks (static retry/depth cap)."""

    retry_cap: int = 8
    eps: float = 1e-3
    name: str = field(default="max_retry", init=False)
    _fails: dict = field(default_factory=dict, init=False)

    def on_assign(self, worker: WorkerView) -> None:
        self._fails[worker.wid] = 0

    def observe(self, worker: WorkerView, gain: float) -> None:
        if gain <= self.eps:
            self._fails[worker.wid] = self._fails.get(worker.wid, 0) + 1
        else:
            self._fails[worker.wid] = 0

    def should_abandon(self, worker: WorkerView, R: float, memory: "BranchMemory") -> bool:
        return self._fails.get(worker.wid, 0) >= self.retry_cap


@dataclass
class TopK(Policy):
    """Priority + exhaust: work branches in recon-prior order, abandon only when a branch
    is effectively *depleted* -- never an early stop. This is the "prioritise but do not
    stop" baseline: it isolates whether MVT's *early* abandonment (leaving a branch while
    it still pays, because something better is available) adds value over merely choosing
    good branches and working them out.

    `k` optionally restricts the eligible set to the top-K branches (an ablation; the
    headline configuration sets k >= n_branches so eligibility never blocks the target)."""

    k: int = 10_000
    deplete_eps: float = 1e-2
    name: str = field(default="top_k", init=False)
    _allowed: set = field(default_factory=set, init=False)

    def reset(self, patches: list[Patch], rng: np.random.Generator) -> None:
        super().reset(patches, rng)
        ranked = sorted(patches, key=lambda p: p.prior_score, reverse=True)
        self._allowed = {p.patch_id for p in ranked[: min(self.k, len(ranked))]}

    def on_assign(self, worker: WorkerView) -> None:
        # Use a short-window slope purely to detect depletion.
        worker.marg_est = make_marginal_estimator("recent_slope", half_life=6.0)

    def observe(self, worker: WorkerView, gain: float) -> None:
        if worker.marg_est is not None:
            worker.marg_est.update(gain)

    def select_branch(
        self, worker: WorkerView, available: list[Patch], memory: "BranchMemory"
    ) -> Patch | None:
        pool = [p for p in available if p.patch_id in self._allowed] or available
        if not pool:
            return None
        unvisited = [p for p in pool if p.patch_id not in memory.visited]
        if unvisited:
            return max(unvisited, key=lambda p: p.prior_score)
        return max(pool, key=lambda p: memory.recent_rate.get(p.patch_id, 0.0))

    def should_abandon(self, worker: WorkerView, R: float, memory: "BranchMemory") -> bool:
        est = worker.marg_est
        # Only abandon once the branch looks genuinely depleted (not an early stop).
        return est is not None and est.n >= 4 and est.rate <= self.deplete_eps


@dataclass
class UCB(Policy):
    """UCB1-style index activation over branches treated as (restless) arms.

    Shared per-branch running mean and pull count across the pool. A worker switches
    when the best available branch's index exceeds its current branch's index by a
    hysteresis margin (prevents thrashing). Restless/Whittle is left as future work.
    """

    c: float = 1.0
    margin: float = 0.05
    name: str = field(default="ucb", init=False)
    _mean: dict = field(default_factory=dict, init=False)
    _pulls: dict = field(default_factory=dict, init=False)
    _total: int = field(default=0, init=False)

    def reset(self, patches: list[Patch], rng: np.random.Generator) -> None:
        super().reset(patches, rng)
        # Optimistic init from the recon prior (normalised to a per-compute scale).
        scale = max(np.mean([p.prior_score for p in patches]), 1e-9)
        self._mean = {p.patch_id: p.prior_score / scale for p in patches}
        self._pulls = {p.patch_id: 0 for p in patches}
        self._total = 0

    def observe(self, worker: WorkerView, gain: float) -> None:
        if worker.patch is None:
            return
        pid = worker.patch.patch_id
        self._pulls[pid] += 1
        self._total += 1
        n = self._pulls[pid]
        self._mean[pid] += (gain - self._mean[pid]) / n   # running mean

    def _index(self, pid: int) -> float:
        n = self._pulls[pid]
        if n == 0:
            return float("inf")                            # force initial exploration
        bonus = self.c * np.sqrt(np.log(max(self._total, 2)) / n)
        return self._mean[pid] + bonus

    def select_branch(
        self, worker: WorkerView, available: list[Patch], memory: "BranchMemory"
    ) -> Patch | None:
        if not available:
            return None
        return max(available, key=lambda p: self._index(p.patch_id))

    def should_abandon(self, worker: WorkerView, R: float, memory: "BranchMemory") -> bool:
        if worker.patch is None:
            return False
        cur = self._index(worker.patch.patch_id)
        # Compare against the best *other* branch we could switch to.
        best_other = -float("inf")
        for p in self._patches:
            if p.patch_id == worker.patch.patch_id:
                continue
            best_other = max(best_other, self._index(p.patch_id))
        return best_other > cur + self.margin


# --------------------------------------------------------------------------- #
# The contribution: MVT abandonment
# --------------------------------------------------------------------------- #
@dataclass
class MVT(Policy):
    """Abandon when estimated marginal g'(c) falls to the frontier opportunity rate R.

    continue while  g'(c) >  R
    abandon/migrate when  g'(c) <= R

    `estimator` selects the marginal estimator ("recent_slope" | "bayesian"); its
    kwargs are forwarded. `min_samples` enforces a floor of observations before a
    branch can be abandoned (the keep-exploring guard against noisy early estimates).
    `hysteresis` requires the marginal to fall a fraction below R to prevent churn.
    """

    estimator: str = "bayesian"
    estimator_kw: dict = field(default_factory=dict)
    min_samples: int = 3
    hysteresis: float = 0.0        # abandon when g'(c) <= R * (1 - hysteresis)
    name: str = field(default="mvt", init=False)

    def on_assign(self, worker: WorkerView) -> None:
        worker.marg_est = make_marginal_estimator(self.estimator, **self.estimator_kw)

    def observe(self, worker: WorkerView, gain: float) -> None:
        if worker.marg_est is not None:
            worker.marg_est.update(gain)

    def should_abandon(self, worker: WorkerView, R: float, memory: "BranchMemory") -> bool:
        est = worker.marg_est
        if est is None or est.n < self.min_samples:
            return False
        return est.rate <= R * (1.0 - self.hysteresis)


@dataclass
class GuardedMVT(MVT):
    """MVT abandonment, gated by an online test for the regime where the rule is valid.

    MVT's leaving rule is optimal only when a branch's return *diminishes* with compute. On a
    constant-hazard branch (memoryless access, uniform value) the marginal signal is
    uninformative: the naive rule reads a low rate and abandons before the hazard pays off,
    and -- as the NASim and Tier-B experiments show -- its solve-rate can collapse. GuardedMVT
    removes that failure mode by discovering the regime *online, per branch*, from the signal
    the scheduler already estimates:

      * It tracks the running peak of the smoothed marginal on the current branch.
      * A *diminishing* signal is present once the marginal has fallen to `decline_frac` of a
        peak that was genuinely productive (peak > `peak_floor_frac` x R). A concave branch
        reaches this at compute ~ tau*ln(1/decline_frac) regardless of its richness; a
        not-yet-productive or memoryless branch never does (its peak stays ~0).
      * Where the diminishing signal IS present, it applies the exact MVT rule (abandon when
        g'(c) <= R). Where it is NOT, it falls back to a committed budget `budget_cap`,
        degrading gracefully to a fixed-timeout-like commitment instead of churning off a
        branch whose value has not yet arrived.

    The result is a policy that keeps MVT's advantage under diminishing returns and does not
    collapse under constant hazard -- a boundary-aware scheduler, tuned by observation rather
    than by hand per environment."""

    decline_frac: float = 0.5        # "declined" := smoothed rate <= decline_frac * peak
    peak_floor_frac: float = 0.05    # peak must exceed this * R to count as productive
    budget_cap: float = 16.0         # committed budget when no diminishing evidence is seen
    name: str = field(default="guarded_mvt", init=False)

    eps: float = 1e-9                # a gain above this counts as "productive this tick"

    def reset(self, patches: list[Patch], rng: np.random.Generator) -> None:
        super().reset(patches, rng)
        self._peak: dict[int, float] = {}
        self._stall: dict[int, int] = {}

    def on_assign(self, worker: WorkerView) -> None:
        super().on_assign(worker)
        self._peak[worker.wid] = 0.0
        self._stall[worker.wid] = 0

    def observe(self, worker: WorkerView, gain: float) -> None:
        super().observe(worker, gain)
        est = worker.marg_est
        if est is not None:
            self._peak[worker.wid] = max(self._peak.get(worker.wid, 0.0), est.rate)
        # Ticks since the branch last produced a gain -> a regime-agnostic "stalled" signal.
        if gain > self.eps:
            self._stall[worker.wid] = 0
        else:
            self._stall[worker.wid] = self._stall.get(worker.wid, 0) + 1

    def should_abandon(self, worker: WorkerView, R: float, memory: "BranchMemory") -> bool:
        est = worker.marg_est
        peak = self._peak.get(worker.wid, 0.0)
        # (1) MVT leaving rule -- applied only where a *diminishing* signal is present: the
        # branch has been genuinely productive (enough INFORMATIVE samples) and its smoothed
        # marginal has fallen from its peak. This is the foraging value-add: leave a still-
        # somewhat-productive branch early to migrate to a richer frontier.
        if (est is not None and est.n >= self.min_samples
                and peak > self.peak_floor_frac * R and est.rate <= self.decline_frac * peak
                and est.rate <= R * (1.0 - self.hysteresis)):
            return True
        # (2) Committed-budget / stall fallback -- RAW-TICK based, so it stays active even when
        # the branch yields too few informative events to feed the marginal estimator (exactly
        # the constant-hazard case, where `est.n` -- informative-event count -- never reaches
        # min_samples). Leave once the branch has produced nothing for `budget_cap` ticks. This
        # covers both the uncracked memoryless branch (commit a budget, then migrate) and the
        # exhausted branch (migrate promptly), without ever cutting off a rich branch still
        # paying out (its stall counter keeps resetting while it yields).
        return self._stall.get(worker.wid, 0) >= self.budget_cap


@dataclass
class OracleMVT(Policy):
    """Strict rate-optimal reference: abandon when the *true* marginal g'(c) falls to the
    analytic MVT environment rate R* (computed from the known surface and switch cost).

    This is the clairvoyant MVT policy -- it knows each branch's g'(c) exactly and uses
    the exact R*, so it is the rate-maximizing upper bound a real (estimating) MVT policy
    can only approach. By construction it refuses to harvest branches below R*; on a
    coverage objective whose target exceeds the yield extractable above R* it will not
    reach the target (a property of rate-optimality, not a scheduler bug). It is therefore
    reported as a per-compute *efficiency* reference, not entered in the time-to-objective
    comparison, and the engine sets oracle_R to env.oracle_rate()."""

    oracle_R: float = 0.0
    name: str = field(default="oracle_mvt", init=False)

    def should_abandon(self, worker: WorkerView, R: float, memory: "BranchMemory") -> bool:
        if worker.patch is None:
            return False
        threshold = self.oracle_R if self.oracle_R > 0 else R
        return worker.patch.marginal() <= threshold


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def make_policy(spec: dict) -> Policy:
    """Build a policy from a spec dict, e.g. {"type": "mvt", "estimator": "bayesian"}."""
    spec = dict(spec)
    kind = spec.pop("type").lower()
    table = {
        "fixed_timeout": FixedTimeout,
        "max_retry": MaxRetry,
        "top_k": TopK,
        "ucb": UCB,
        "mvt": MVT,
        "guarded_mvt": GuardedMVT,
        "oracle_mvt": OracleMVT,
    }
    if kind not in table:
        raise ValueError(f"unknown policy type: {kind!r}")
    return table[kind](**spec)
