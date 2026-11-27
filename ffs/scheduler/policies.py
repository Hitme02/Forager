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
    # Stall-budget mode. "static" uses `budget_cap` as a fixed tick count (the submitted
    # configuration). "dynamic" re-derives the budget online from the pool's own completed
    # stalls (zero-gain runs that ended in a gain): budget = budget_kappa * mean completed-stall
    # length, clipped to [budget_min, budget_max], with `budget_cap` as the warm start until
    # `budget_warmup` completed stalls have been seen. budget_kappa is dimensionless (a multiple
    # of the observed wait), so the budget rescales with the target's hazard instead of being a
    # hand-set tick count. Since completed stalls are truncated at the current budget, the
    # update has a non-zero fixed point only for kappa > 2 (at kappa = 3 it settles at 1.8-2.0x
    # the mean of a geometric wait for per-tick hazards 0.05-0.2); kappa <= 2 collapses.
    budget_mode: str = "static"
    budget_kappa: float = 3.0
    budget_min: float = 4.0
    budget_max: float = 256.0
    budget_warmup: int = 5
    name: str = field(default="guarded_mvt", init=False)

    eps: float = 1e-9                # a gain above this counts as "productive this tick"

    def reset(self, patches: list[Patch], rng: np.random.Generator) -> None:
        super().reset(patches, rng)
        if self.budget_mode not in ("static", "dynamic"):
            raise ValueError(f"unknown budget_mode: {self.budget_mode!r}")
        self._peak: dict[int, float] = {}
        self._stall: dict[int, int] = {}
        self._stall_sum = 0.0            # total length of completed stalls (pool-wide)
        self._stall_n = 0                # number of completed stalls (pool-wide)

    @property
    def current_budget(self) -> float:
        """The stall budget in force now (static, or the online dynamic estimate)."""
        if self.budget_mode == "static" or self._stall_n < self.budget_warmup:
            return float(self.budget_cap)
        b = self.budget_kappa * self._stall_sum / self._stall_n
        return float(min(self.budget_max, max(self.budget_min, b)))

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
            run = self._stall.get(worker.wid, 0)
            if run > 0:                  # a stall just ended in a gain: one completed wait
                self._stall_sum += run
                self._stall_n += 1
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
        return self._stall.get(worker.wid, 0) >= self.current_budget


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


@dataclass
class RestlessIndex(Policy):
    """Gittins-style index policy for deteriorating (restless) arms -- the restless-bandit
    comparator a reviewer asks for.

    Our branches are a bandit whose arms only deteriorate while active and freeze while
    passive (a branch's spent compute advances only when a worker is on it). For such
    deteriorating-reward arms the Gittins index reduces to the arm's *current* expected
    reward rate, so the index-optimal rule is greedy on the current marginal. We estimate
    each branch's current rate with a shared recency-weighted (EWMA) mean of observed gains,
    work the highest-index available branch, and switch when another branch's index beats
    the current one's by a hysteresis `margin` (which prevents thrashing). Unvisited
    branches carry an optimistic (infinite) index so each is tried once.

    The point of including it: unlike MVT it has no notion of the frontier opportunity rate,
    the switch cost, or the completion objective -- it only ranks arms against one another,
    and so never stops early on opportunity cost. It therefore isolates, together with UCB,
    whether a selection/index policy alone can match MVT's stopping.
    """

    half_life: float = 8.0
    margin: float = 0.05
    name: str = field(default="restless_index", init=False)
    _rate: dict = field(default_factory=dict, init=False)
    _seen: set = field(default_factory=set, init=False)
    _alpha: float = field(default=0.0, init=False)

    def reset(self, patches: list[Patch], rng: np.random.Generator) -> None:
        super().reset(patches, rng)
        self._alpha = 1.0 - 0.5 ** (1.0 / max(self.half_life, 1e-6))
        self._rate = {p.patch_id: 0.0 for p in patches}
        self._seen = set()

    def observe(self, worker: WorkerView, gain: float) -> None:
        if worker.patch is None:
            return
        pid = worker.patch.patch_id
        if pid not in self._seen:
            self._rate[pid] = gain               # initialise on first observation
            self._seen.add(pid)
        else:
            self._rate[pid] = (1.0 - self._alpha) * self._rate[pid] + self._alpha * gain

    def _index(self, pid: int) -> float:
        return float("inf") if pid not in self._seen else self._rate[pid]

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
        best_other = -float("inf")
        for p in self._patches:
            if p.patch_id == worker.patch.patch_id:
                continue
            best_other = max(best_other, self._index(p.patch_id))
        return best_other > cur + self.margin


# --------------------------------------------------------------------------- #
# Published early-stopping comparators (hyper-parameter-optimisation literature)
# --------------------------------------------------------------------------- #
class _BranchCurves:
    """Per-branch learning curves (cumulative yield after each unit of branch compute).

    Shared bookkeeping for the early-stopping comparators below: a branch is treated like a
    training run whose "learning curve" is its cumulative validated yield as a function of
    compute spent on it (across visits). Prefix sums make running averages O(1)."""

    def __init__(self) -> None:
        self.cum: dict[int, list[float]] = {}      # pid -> cumulative yield after k units
        self.prefix: dict[int, list[float]] = {}   # pid -> prefix sums of `cum`

    def add(self, pid: int, gain: float) -> int:
        c = self.cum.setdefault(pid, [])
        p = self.prefix.setdefault(pid, [])
        y = (c[-1] if c else 0.0) + gain
        c.append(y)
        p.append((p[-1] if p else 0.0) + y)
        return len(c)                                # compute spent on pid so far

    def length(self, pid: int) -> int:
        return len(self.cum.get(pid, ()))


@dataclass
class MedianStopping(Policy):
    """Median stopping rule (Golovin et al., Google Vizier, KDD 2017), adapted to branches.

    Vizier stops a trial at step s if its best objective so far is strictly worse than the
    median of the running averages of the other trials' objectives up to step s. Here a
    branch's objective at compute c is its cumulative yield y(c) (monotone, so "best so far" is
    the current value); after a grace period of `grace` units a branch is abandoned when y(c)
    falls below the median, over other branches already worked for at least c units, of their
    running averages (1/c) sum_{k<=c} y(k). At least `min_peers` such branches are required for a
    comparison. A branch that is never out-performed is worked until it is effectively depleted
    (the TopK depletion test), so no branch is worked forever. Selection is the shared default.
    """

    grace: int = 4
    min_peers: int = 3
    deplete_eps: float = 1e-2
    name: str = field(default="median_stopping", init=False)

    def reset(self, patches: list[Patch], rng: np.random.Generator) -> None:
        super().reset(patches, rng)
        self._curves = _BranchCurves()
        self._stop: dict[int, bool] = {}

    def on_assign(self, worker: WorkerView) -> None:
        worker.marg_est = make_marginal_estimator("recent_slope", half_life=6.0)
        self._stop[worker.wid] = False

    def observe(self, worker: WorkerView, gain: float) -> None:
        if worker.patch is None:
            return
        if worker.marg_est is not None:
            worker.marg_est.update(gain)
        pid = worker.patch.patch_id
        c = self._curves.add(pid, gain)
        if c < self.grace:
            return
        peers = []
        for qid, pre in self._curves.prefix.items():
            if qid != pid and len(pre) >= c:
                peers.append(pre[c - 1] / c)
        if len(peers) >= self.min_peers:
            y = self._curves.cum[pid][-1]
            self._stop[worker.wid] = y < float(np.median(peers))

    def should_abandon(self, worker: WorkerView, R: float, memory: "BranchMemory") -> bool:
        if self._stop.get(worker.wid, False):
            return True
        est = worker.marg_est
        return est is not None and est.n >= 4 and est.rate <= self.deplete_eps


@dataclass
class ASHA(Policy):
    """Asynchronous successive halving (Li et al., MLSys 2020), adapted to branches.

    ASHA allocates compute to many configurations in rungs: budgets r0, r0*eta, r0*eta^2, ...
    and, asynchronously as workers free up, promotes a configuration to the next rung only if it
    ranks in the top 1/eta of the configurations that have reached its current rung. The
    asynchronous form fits a parallel pool directly. Here a configuration is a branch, its
    budget is the compute spent on it, and its score at a rung is its cumulative yield there.
    When a worker's branch reaches rung k it continues only if it is in the top
    max(1, floor(n_k / eta)) of the n_k scores recorded at rung k so far (itself included);
    otherwise it is abandoned. Past the last rung a branch is worked until depleted (TopK test).
    """

    r0: float = 4.0
    eta: float = 3.0
    max_budget: float = 400.0
    deplete_eps: float = 1e-2
    name: str = field(default="asha", init=False)

    def reset(self, patches: list[Patch], rng: np.random.Generator) -> None:
        super().reset(patches, rng)
        self._curves = _BranchCurves()
        rungs, r = [], float(self.r0)
        while r <= self.max_budget:
            rungs.append(int(round(r)))
            r *= self.eta
        self._rungs = sorted(set(rungs))
        self._rung_scores: dict[int, list[float]] = {r: [] for r in self._rungs}
        self._recorded: set[tuple[int, int]] = set()
        self._stop: dict[int, bool] = {}

    def on_assign(self, worker: WorkerView) -> None:
        worker.marg_est = make_marginal_estimator("recent_slope", half_life=6.0)
        self._stop[worker.wid] = False

    def observe(self, worker: WorkerView, gain: float) -> None:
        if worker.patch is None:
            return
        if worker.marg_est is not None:
            worker.marg_est.update(gain)
        pid = worker.patch.patch_id
        c = self._curves.add(pid, gain)
        if c in self._rung_scores and (pid, c) not in self._recorded:
            self._recorded.add((pid, c))
            score = self._curves.cum[pid][-1]
            scores = self._rung_scores[c]
            scores.append(score)
            k = max(1, int(np.floor(len(scores) / self.eta)))
            threshold = sorted(scores, reverse=True)[k - 1]
            self._stop[worker.wid] = score < threshold

    def should_abandon(self, worker: WorkerView, R: float, memory: "BranchMemory") -> bool:
        if self._stop.get(worker.wid, False):
            return True
        est = worker.marg_est
        return est is not None and est.n >= 4 and est.rate <= self.deplete_eps


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
        "restless_index": RestlessIndex,
        "gittins": RestlessIndex,
        "mvt": MVT,
        "guarded_mvt": GuardedMVT,
        "oracle_mvt": OracleMVT,
        "median_stopping": MedianStopping,
        "asha": ASHA,
    }
    if kind not in table:
        raise ValueError(f"unknown policy type: {kind!r}")
    return table[kind](**spec)
