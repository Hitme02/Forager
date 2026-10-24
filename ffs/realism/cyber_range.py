"""A sandboxed cyber range presented to the EXISTING MVT scheduler as a foraging surface.

This is the Tier-B analogue of `nasim_target.py`: it cashes in the same design claim one
more time, now against a real tool-executing agent harness. `ServiceBranch` structurally
matches `simulator.patch.Patch` (`patch_id`, `prior_score`, `spent`, `work_one`) and
`CyberRange` matches the parts of `simulator.environment.Environment` the scheduler uses
(`patches`, `cfg.switch_cost`, `cfg.max_ticks`, `objective_met`, `progress_fraction`,
`total_yielded`). So `Scheduler(CyberRange(...), policy, EngineConfig(...), rng)` runs the
very same parallel engine, policies, estimators and opportunity-rate machinery that produced
the simulated and NASim results -- now driving the bounded agent harness over a sandbox.

Parallel model: one shared range; each worker is a concurrent agent committed to one service
(a branch). A tick advances wall-clock by one and lets each working worker take one bounded,
grounded action on its service via the harness. Validated progress g(c) is scored from
observed service state (recon milestones, foothold, artifact collection, crown-jewel read)
through the grounded `ProgressLedger` plus continuous collection value -- never agent claims.
The switch cost is paid as idle ticks while a migrating worker re-establishes context.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agent_harness import AgentHarness, Brain
from .progress import ProgressLedger
from .sandbox import Sandbox
from .services import ServiceModel

# Tier-B milestone weights are deliberately LIGHT (unlike the NASim foothold-counting
# weights). Here recon/foothold milestones only *gate and ground* progress and give the
# estimators early informative events; the yield that drives the forage/abandon decision is
# the continuous collection stream (`Evidence.value`). Heavy foothold weights would inject a
# large one-time spike into the per-branch marginal estimate and make MVT refuse to abandon a
# depleted branch -- the signal must reflect the diminishing collection return, not the entry.
TIERB_WEIGHTS: dict[str, float] = {
    "host_discovered": 0.1,
    "service_enumerated": 0.1,
    "vuln_identified": 0.2,
    "credential_obtained": 0.2,
    "foothold_shell": 0.5,
    "privilege_escalation": 0.5,
    "lateral_reachable": 0.2,
    "objective_flag": 500.0,
}


@dataclass
class RangeConfig:
    switch_cost: float = 4.0          # idle ticks to re-establish context on a new service
    max_ticks: int = 8000
    objective: str = "coverage"       # "coverage" (harvest a value fraction) | "flag" (CTF)
    target_frac: float = 0.5          # coverage: fraction of total collectable value to reach

    def __post_init__(self) -> None:
        if self.objective not in ("coverage", "flag"):
            raise ValueError(f"unknown objective: {self.objective!r}")
        if self.switch_cost < 0:
            raise ValueError("switch_cost must be >= 0")


class ServiceBranch:
    """One range service, foraged as a patch. Satisfies the scheduler's branch contract."""

    def __init__(self, patch_id: int, service: ServiceModel, harness: AgentHarness) -> None:
        self.patch_id = patch_id
        self.service = service
        self.harness = harness
        self.prior_score = float(service.prior_score)
        # mutable per-episode state
        self.spent = 0.0
        self.cumulative_score = 0.0     # full grounded gain (milestones + collection): the
        #                                 signal the scheduler's estimator sees
        self.collection_value = 0.0     # harvested artifact value only: the coverage objective
        self.ledger = ProgressLedger(weights=dict(TIERB_WEIGHTS))

    def work_one(self, rng: np.random.Generator) -> float:
        """One grounded agent action on this service; return the validated-progress delta.

        The returned gain is the full grounded signal (recon/foothold milestones + collection
        value) so the marginal estimator sees the real front-loaded, diminishing shape. The
        coverage *objective*, however, counts only harvested artifact value -- the diminishing
        stream -- so it is not won by racing to foothold many services (a breadth race a short
        fixed timeout trivially wins); it is exactly the simulator's g(c)-coverage objective.
        """
        self.spent += 1.0
        ev = self.harness.step(self.service, rng)
        if not ev.grounded:
            return 0.0
        self.collection_value += ev.value
        delta = ev.value
        if ev.milestone:
            delta += self.ledger.record(ev.milestone, ev.target_ref, True)
        self.cumulative_score += delta
        return float(delta)


class CyberRange:
    """A sandboxed range presented to the scheduler as a heterogeneous foraging frontier."""

    def __init__(self, sandbox: Sandbox, cfg: RangeConfig, brain: Brain | None = None) -> None:
        self.sandbox = sandbox
        self.cfg = cfg
        self.harness = AgentHarness(sandbox, brain)

        self.patches: list[ServiceBranch] = [
            ServiceBranch(i, svc, self.harness) for i, svc in enumerate(sandbox.services())
        ]
        # Total collectable value (for the coverage objective), from each service's potential.
        self.total_value = float(sum(getattr(b.service, "max_value", 0.0)
                                     for b in self.patches))
        self.target_value = cfg.target_frac * self.total_value

    # --- Environment-compatible interface used by the scheduler ---
    @property
    def total_yielded(self) -> float:
        # Full grounded yield (for efficiency reporting); the scheduler reads this.
        return float(sum(b.cumulative_score for b in self.patches))

    @property
    def collected_value(self) -> float:
        # Harvested artifact value only -- what the coverage objective is measured on.
        return float(sum(b.collection_value for b in self.patches))

    def objective_met(self) -> bool:
        if self.cfg.objective == "flag":
            return self.sandbox.flag_captured()
        return self.collected_value >= self.target_value

    def progress_fraction(self) -> float:
        if self.cfg.objective == "flag":
            return 1.0 if self.sandbox.flag_captured() else min(
                0.99, self.collected_value / self.target_value if self.target_value > 0 else 0.0)
        if self.target_value <= 0:
            return 1.0
        return float(min(1.0, self.collected_value / self.target_value))

    def teardown(self) -> None:
        self.sandbox.teardown()


def make_mock_range(cfg: RangeConfig | None = None, *, n_services: int = 12,
                    flag_services: int = 2, sigma: float = 0.8,
                    regime: str = "diminishing", burst_p: float = 1.0,
                    brain: Brain | None = None, seed: int = 0) -> CyberRange:
    """Build an in-process (no-Docker) range -- the default, runnable here and in CI.

    `sigma` is the branch-heterogeneity knob (log-normal richness, mean held fixed); `regime`
    selects the diminishing-value vs constant-hazard contrast under test. This is the live
    analogue of the simulator's heterogeneity sweep."""
    from .agent_harness import HarvestBrain
    from .sandbox import MockSandbox

    cfg = cfg or RangeConfig()
    sandbox = MockSandbox(n_services=n_services, flag_services=flag_services, sigma=sigma,
                          regime=regime, burst_p=burst_p, seed=seed)
    return CyberRange(sandbox, cfg, brain or HarvestBrain())


def make_docker_range(cfg: RangeConfig | None = None, *, acknowledged: bool = False,
                      network_name: str = "ffs_tierb_isolated",
                      compose_project: str = "ffs_tierb",
                      brain: Brain | None = None) -> CyberRange:
    """Build a range backed by the live, isolated Docker fixture (gated; see TIERB.md).

    Requires an explicit authorisation acknowledgement and a running `docker/` fixture on an
    internal (no-egress) network. Never the default; not exercised by the test suite."""
    from .sandbox import DockerSandbox

    cfg = cfg or RangeConfig(objective="flag")
    sandbox = DockerSandbox(compose_project=compose_project, network_name=network_name,
                            acknowledged=acknowledged)
    return CyberRange(sandbox, cfg, brain)
