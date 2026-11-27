"""Discrete-event parallel pentest scheduler.

A pool of W workers forages the attack surface in parallel. Each tick advances
wall-clock time by one and spends up to W units of compute. A worker is in one of:

  working   - spending compute on its branch, producing (noisy) validated progress
  switching - paying the switch cost s after abandoning, producing nothing
  idle      - no branch available to enter (rare; only when the frontier is exhausted)

The policy controls two decisions -- which branch to enter (select_branch) and when to
abandon the current one (should_abandon) -- but all mechanics, accounting, the frontier
opportunity-rate estimate R, and the stop condition live here, identically for every
policy. That is what makes the comparison fair: only the decision rule varies.

This deterministic, seeded discrete-event model is the MVP testbed for the *science*
(the MVT claim). The Worker/Pool abstraction is intentionally thin so Phase 2 can back
it with real threads/processes, and later MPI workers across nodes, without touching
the policies or the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..simulator.environment import Environment
from ..simulator.patch import Patch
from .estimators import FrontierRateEstimator, MarginalEstimator
from .policies import OracleMVT, Policy


@dataclass
class Worker:
    wid: int
    patch: Patch | None = None
    state: str = "idle"              # "working" | "switching" | "idle"
    switch_remaining: float = 0.0
    time_on_patch: int = 0
    last_gain: float = 0.0
    entry_ticks: int = 0             # productive ticks counted toward the entry phase
    entry_yield: float = 0.0         # yield accumulated during the entry phase
    entry_elapsed: int = 0           # all ticks elapsed in the entry phase ("productive" mode)
    marg_est: MarginalEstimator | None = None   # policy-owned scratch


class BranchMemory:
    """What the scheduler has learned about each branch (shared across workers)."""

    def __init__(self, patches: list[Patch], half_life: float = 6.0) -> None:
        self.visited: set[int] = set()
        self.recent_rate: dict[int, float] = {}
        self.assigned: set[int] = set()
        self._alpha = 1.0 - 0.5 ** (1.0 / max(half_life, 1e-6))

    def on_gain(self, pid: int, gain: float) -> None:
        self.visited.add(pid)
        if pid not in self.recent_rate:
            self.recent_rate[pid] = gain
        else:
            self.recent_rate[pid] = (
                (1.0 - self._alpha) * self.recent_rate[pid] + self._alpha * gain
            )


@dataclass
class EngineConfig:
    n_workers: int = 8
    entry_window: int = 5                      # productive ticks defining a branch's fresh phase
    frontier_half_life_entries: float = 15.0   # EWMA memory of R, in branch entries
    entry_mode: str = "ticks"                  # "ticks" (submitted) | "productive" (ramp-aware)


@dataclass
class RunResult:
    solved: bool
    ticks: int                       # wall-clock time-to-objective
    compute: int                     # total compute spent (incl. switching)
    productive_compute: int          # compute that actually worked a branch
    switch_compute: int              # compute burnt switching
    n_switches: int                  # number of abandon-and-migrate events
    final_progress: float            # objective progress fraction at stop
    utilization: float               # productive / total compute
    findings_per_kcompute: float     # realised yield per 1000 compute (efficiency)
    history: dict = field(default_factory=dict)   # per-tick traces (optional)


class Scheduler:
    """Runs one episode of a policy against one environment instance."""

    def __init__(
        self,
        env: Environment,
        policy: Policy,
        cfg: EngineConfig,
        rng: np.random.Generator,
        trace: bool = False,
    ) -> None:
        self.env = env
        self.policy = policy
        self.cfg = cfg
        self.rng = rng
        self.trace = trace

        self.workers = [Worker(wid=i) for i in range(cfg.n_workers)]
        self.memory = BranchMemory(env.patches)
        self.frontier = FrontierRateEstimator(
            switch_cost=env.cfg.switch_cost,
            entry_window=cfg.entry_window,
            half_life_entries=cfg.frontier_half_life_entries,
            entry_mode=cfg.entry_mode,
        )

        policy.reset(env.patches, rng)
        if isinstance(policy, OracleMVT):
            policy.oracle_R = env.oracle_rate()

    # ------------------------------------------------------------------ #
    def _available(self) -> list[Patch]:
        """Branches not currently assigned to any worker."""
        return [p for p in self.env.patches if p.patch_id not in self.memory.assigned]

    def _assign(self, worker: Worker) -> None:
        avail = self._available()
        choice = self.policy.select_branch(worker, avail, self.memory)
        if choice is None:
            worker.state = "idle"
            worker.patch = None
            return
        worker.patch = choice
        worker.state = "working"
        worker.time_on_patch = 0
        worker.entry_ticks = 0
        worker.entry_yield = 0.0
        worker.entry_elapsed = 0
        self.memory.assigned.add(choice.patch_id)
        self.policy.on_assign(worker)

    def _release(self, worker: Worker) -> None:
        if worker.patch is not None:
            self.memory.assigned.discard(worker.patch.patch_id)

    def _do_work(self, w: Worker) -> float:
        """Spend one compute unit on w's branch; update all bookkeeping; return gain."""
        gain = w.patch.work_one(self.rng)
        w.last_gain = gain
        w.time_on_patch += 1
        # Accumulate the branch's "fresh phase" intake to feed the opportunity-rate R.
        k = self.frontier.entry_window
        if self.cfg.entry_mode == "productive":
            if w.entry_ticks < k:
                w.entry_yield += gain
                w.entry_elapsed += 1
                if gain > 1e-12:
                    w.entry_ticks += 1
                    if w.entry_ticks == k:
                        self.frontier.record_entry_phase(w.entry_yield, w.entry_elapsed)
        elif w.entry_ticks < k:
            w.entry_yield += gain
            w.entry_ticks += 1
            if w.entry_ticks == k:
                self.frontier.record_entry_phase(w.entry_yield / k)
        self.memory.on_gain(w.patch.patch_id, gain)
        self.policy.observe(w, gain)
        return gain

    # ------------------------------------------------------------------ #
    def run(self) -> RunResult:
        env, pol = self.env, self.policy
        max_ticks = env.cfg.max_ticks

        # Initial assignment of all workers.
        for w in self.workers:
            self._assign(w)

        productive = switch = n_switches = 0
        hist_R: list[float] = []
        hist_progress: list[float] = []
        hist_util: list[float] = []

        tick = 0
        solved = env.objective_met()
        while not solved and tick < max_ticks:
            tick += 1
            tick_yield = 0.0
            tick_compute = 0
            tick_productive = 0

            # --- 1. advance every worker one tick (exactly one compute unit each) ---
            for w in self.workers:
                if w.state == "switching":
                    w.switch_remaining -= 1
                    tick_compute += 1
                    switch += 1
                    if w.switch_remaining <= 0:
                        self._assign(w)      # arrives now; starts working next tick
                    continue
                if w.state == "idle":
                    self._assign(w)          # idle consumed no compute; may work now
                if w.state == "working":
                    gain = self._do_work(w)
                    tick_yield += gain
                    tick_compute += 1
                    tick_productive += 1
                    productive += 1

            # --- 2. read the current frontier opportunity rate R ---
            R = self.frontier.rate

            # --- 3. check objective before abandonment decisions ---
            if env.objective_met():
                solved = True

            # --- 4. abandonment / migration decisions ---
            if not solved:
                for w in self.workers:
                    if w.state == "working" and pol.should_abandon(w, R, self.memory):
                        self._release(w)
                        w.state = "switching"
                        w.switch_remaining = env.cfg.switch_cost
                        n_switches += 1

            if self.trace:
                hist_R.append(R)
                hist_progress.append(env.progress_fraction())
                hist_util.append(tick_productive / max(len(self.workers), 1))

        total_compute = productive + switch
        util = productive / total_compute if total_compute else 0.0
        yield_total = env.total_yielded
        eff = 1000.0 * yield_total / total_compute if total_compute else 0.0

        history = {}
        if self.trace:
            history = {"R": hist_R, "progress": hist_progress, "utilization": hist_util}

        return RunResult(
            solved=solved,
            ticks=tick,
            compute=total_compute,
            productive_compute=productive,
            switch_compute=switch,
            n_switches=n_switches,
            final_progress=env.progress_fraction(),
            utilization=util,
            findings_per_kcompute=eff,
            history=history,
        )
