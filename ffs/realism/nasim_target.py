"""NASim-backed attack surface that the EXISTING MVT scheduler can forage, unchanged.

The design claim of the whole project is that the scheduler and the MVT policy are
agent/environment-agnostic. We cash that in here: `NASimBranch` structurally matches
`simulator.patch.Patch` (it has `patch_id`, `prior_score`, `spent`, `work_one`) and
`NASimTarget` matches the parts of `simulator.environment.Environment` the scheduler uses
(`patches`, `cfg.switch_cost`, `cfg.max_ticks`, `objective_met`, `progress_fraction`,
`total_yielded`). So `Scheduler(NASimTarget(...), policy, EngineConfig(...), rng)` runs the
very same parallel engine, policies, estimators and frontier-rate machinery that produced
the simulated results -- now against a real network-attack simulator.

Parallel model: one shared NASim network; each worker is a concurrent attack thread
committed to one target host (a branch). A tick advances wall-clock by one and lets each
working worker take one grounded NASim action on its host. Validated progress g(c) is
scored from observed state transitions (foothold, privilege escalation, enumeration,
lateral reach, sensitive capture) via the grounded `ProgressLedger` -- never agent claims.
The switch cost is paid as idle ticks while a migrating worker re-establishes context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._nasim_compat import nasim
from .progress import ProgressLedger

_FOOTHOLD_SCORE = 10.0   # constant reward for gaining access to a host (breadth objective)

# NASim action class name -> grounded milestone (and its ledger weight name).
_SCAN_MILESTONE = {
    "ServiceScan": "service_enumerated",
    "OSScan": "host_discovered",
    "SubnetScan": "lateral_reachable",
}


@dataclass
class RealismConfig:
    switch_cost: float = 3.0          # idle ticks to re-establish context on a new host
    max_ticks: int = 4000
    target_frac: float = 0.40         # objective target (fraction of hosts, or of value)
    objective: str = "count"          # "count" (breadth) | "value" (coverage of value)
    value_decay: float = 1.0          # per-branch value of the k-th foothold = base*decay^k
    #                                   decay<1 => diminishing VALUE within a branch (the
    #                                   property MVT needs); decay=1 => uniform value.


class NASimBranch:
    """One target host, foraged as a patch. Satisfies the scheduler's branch contract."""

    def __init__(self, patch_id: int, addr, prior_score: float, target: "NASimTarget",
                 brain) -> None:
        self.patch_id = patch_id
        self.addr = addr
        self.prior_score = prior_score
        self.target = target
        self.brain = brain
        # mutable per-episode state
        self.spent = 0.0
        self.cumulative_score = 0.0
        self.ledger = ProgressLedger()
        self.done_scans: set[str] = set()
        self.exploit_i = 0
        self.pe_i = 0

    def work_one(self, rng: np.random.Generator) -> float:
        """One grounded agent action on this host; return the validated-progress delta."""
        self.spent += 1.0
        if self.target._terminated:            # goal already reached this tick: don't step
            return 0.0
        action_idx = self.brain.next_action(self.target, self, rng)
        if action_idx is None:                 # brain has nothing left to try here
            return 0.0

        env = self.target.env
        addr = self.addr
        before_access = env.current_state.get_host(addr).access
        _, _, terminated, truncated, info = env.step(action_idx)
        self.target._terminated = self.target._terminated or bool(terminated)
        success = bool(info.get("success", False))
        after = env.current_state.get_host(addr)
        ref = str(addr)

        delta = 0.0
        if success:
            aname = type(self.target.actions[action_idx]).__name__
            if aname in _SCAN_MILESTONE:
                # small recon signal (also provides early informative events for the
                # keep-exploring estimators before the foothold jump arrives)
                delta += self.ledger.record(_SCAN_MILESTONE[aname], ref, True)
            elif aname == "Exploit" and before_access < 1 <= after.access:
                # FOOTHOLD -- the harvest, scored as a constant (a breadth objective: every
                # host compromised counts equally). This front-loads the per-host return
                # (big jump at access, ~0 after), the diminishing-returns shape MVT assumes;
                # difficulty varies by host, so some branches pay out fast and others stall
                # -- the forage/abandon decision the scheduler must get right.
                if self.ledger.record("foothold_shell", ref, True) > 0:
                    delta += _FOOTHOLD_SCORE
            elif aname == "PrivilegeEscalation" and before_access < 2 <= after.access:
                delta += self.ledger.record("privilege_escalation", ref, True)

        self.cumulative_score += delta
        return delta


class NASimTarget:
    """A sandboxed NASim network presented to the scheduler as a foraging surface."""

    def __init__(self, env, cfg: RealismConfig, rng: np.random.Generator, brain) -> None:
        self.env = env
        self.cfg = cfg
        self.rng = rng
        self._terminated = False

        self.actions = list(env.action_space.actions)
        self.sensitive = set(env.network.sensitive_hosts.keys())

        # Per-host (heterogeneous) value -- the reward for gaining a foothold on it.
        self.host_value: dict = {}
        for addr, host in env.network.hosts.items():
            v = float(getattr(host, "value", 0.0) or 0.0)
            self.host_value[addr] = v if v > 0 else 1.0
        self.total_value = float(sum(self.host_value.values()))

        # Map each host -> {action-type-name: [flat action indices]} for the brain.
        self.actions_by_host: dict = {}
        for i, a in enumerate(self.actions):
            self.actions_by_host.setdefault(a.target, {}).setdefault(
                type(a).__name__, []).append(i)

        # One branch per host, with a recon prior from observable attack surface.
        self.patches: list[NASimBranch] = []
        for j, addr in enumerate(env.network.hosts.keys()):
            self.patches.append(
                NASimBranch(j, addr, self._prior_score(addr), self, brain))

        self.n_hosts = len(self.patches)
        self.target_hosts = max(1, int(round(cfg.target_frac * self.n_hosts)))
        self.target_value = cfg.target_frac * self.total_value

    # --- recon prior: observable attack surface, not ground-truth value ---
    def _prior_score(self, addr) -> float:
        hv = self.env.current_state.get_host(addr)
        surface = 1.0 + float(hv.num_services)
        reach = 1.0 if hv.reachable else 0.25     # reachable hosts look more promising
        noise = float(np.exp(0.3 * self.rng.standard_normal()))
        return surface * reach * noise

    # --- Environment-compatible interface used by the scheduler ---
    @property
    def total_yielded(self) -> float:
        return float(sum(b.cumulative_score for b in self.patches))

    def _n_compromised(self) -> int:
        st = self.env.current_state
        return sum(1 for b in self.patches if st.get_host(b.addr).access >= 1)

    def _value_harvested(self) -> float:
        st = self.env.current_state
        return float(sum(self.host_value[b.addr]
                         for b in self.patches if st.get_host(b.addr).access >= 1))

    def objective_met(self) -> bool:
        # Breadth objective: gain a foothold on a target fraction of hosts.
        if self._terminated:
            return True
        return self._n_compromised() >= self.target_hosts

    def progress_fraction(self) -> float:
        return float(min(1.0, self._n_compromised() / self.target_hosts))


class SubnetBranch:
    """A network segment (subnet) foraged as a patch: harvest its hosts, easy ones first.

    Satisfies the scheduler's branch contract (`patch_id`, `prior_score`, `spent`,
    `work_one`). Its per-tick yield is the value of any host newly compromised this step; as
    the segment's easy hosts fall, the yield decays -- the diminishing-returns signal MVT
    abandons on."""

    def __init__(self, patch_id: int, subnet_id: int, hosts: list, prior_score: float,
                 target: "NASimSubnetTarget", brain) -> None:
        self.patch_id = patch_id
        self.subnet_id = subnet_id
        self.hosts = hosts
        self.prior_score = prior_score
        self.target = target
        self.brain = brain
        self.spent = 0.0
        self.cumulative_score = 0.0
        self.ledger = ProgressLedger()
        self.rr = 0
        self.host_state: dict = {}
        self._n_comp = 0             # hosts compromised in this segment (for value decay)

    def work_one(self, rng: np.random.Generator) -> float:
        self.spent += 1.0
        if self.target._terminated:
            return 0.0
        aidx = self.brain.next_action(self.target, self, rng)
        if aidx is None:
            return 0.0
        env = self.target.env
        addr = self.target.actions[aidx].target
        before = env.current_state.get_host(addr).access
        _, _, terminated, _, info = env.step(aidx)
        self.target._terminated = self.target._terminated or bool(terminated)
        delta = 0.0
        if info.get("success"):
            aname = type(self.target.actions[aidx]).__name__
            ref = str(addr)
            if aname in _SCAN_MILESTONE:
                delta += self.ledger.record(_SCAN_MILESTONE[aname], ref, True)
            elif aname == "Exploit" and before < 1 <= env.current_state.get_host(addr).access:
                if self.ledger.record("foothold_shell", ref, True) > 0:
                    # Value of this segment's k-th foothold decays with k when value_decay<1,
                    # giving the segment a saturating (diminishing-value) yield curve.
                    delta += _FOOTHOLD_SCORE * (self.target.cfg.value_decay ** self._n_comp)
                    self._n_comp += 1
        self.cumulative_score += delta
        return delta


class NASimSubnetTarget:
    """A sandboxed NASim network presented to the scheduler with each SUBNET as a branch."""

    def __init__(self, env, cfg: RealismConfig, rng: np.random.Generator, brain) -> None:
        self.env = env
        self.cfg = cfg
        self.rng = rng
        self._terminated = False
        self.actions = list(env.action_space.actions)

        self.actions_by_host: dict = {}
        for i, a in enumerate(self.actions):
            self.actions_by_host.setdefault(a.target, {}).setdefault(
                type(a).__name__, []).append(i)

        # Group hosts by subnet; one branch per subnet.
        by_subnet: dict = {}
        for addr in env.network.hosts.keys():
            by_subnet.setdefault(addr[0], []).append(addr)

        self.patches: list[SubnetBranch] = []
        for j, (sub, hosts) in enumerate(sorted(by_subnet.items())):
            # Recon prior: segment attack surface (host count x avg services) + noise.
            surface = sum(1 + self.env.current_state.get_host(a).num_services for a in hosts)
            prior = surface * float(np.exp(0.3 * rng.standard_normal()))
            self.patches.append(SubnetBranch(j, sub, hosts, prior, self, brain))

        self.n_hosts = len(env.network.hosts)
        self.target_hosts = max(1, int(round(cfg.target_frac * self.n_hosts)))

        # Total obtainable value (for the "value" objective), given the per-segment decay.
        d = cfg.value_decay
        total_value = 0.0
        for b in self.patches:
            h = len(b.hosts)
            seg = _FOOTHOLD_SCORE * (h if d >= 1.0 else (1.0 - d ** h) / (1.0 - d))
            total_value += seg
        self.total_value = total_value
        self.target_value = cfg.target_frac * total_value

    @property
    def total_yielded(self) -> float:
        return float(sum(b.cumulative_score for b in self.patches))

    def _n_compromised(self) -> int:
        st = self.env.current_state
        return sum(1 for a in self.env.network.hosts if st.get_host(a).access >= 1)

    def objective_met(self) -> bool:
        if self._terminated:
            return True
        if self.cfg.objective == "value":
            return self.total_yielded >= self.target_value
        return self._n_compromised() >= self.target_hosts

    def progress_fraction(self) -> float:
        if self.cfg.objective == "value":
            return float(min(1.0, self.total_yielded / self.target_value)) if self.target_value > 0 else 1.0
        return float(min(1.0, self._n_compromised() / self.target_hosts))


def make_segmented_nasim_target(brain, cfg: RealismConfig | None = None, *,
                                n_subnets: int = 10, hosts_per_subnet: int = 5,
                                num_services: int = 6, heterogeneity: float = 0.8,
                                seed: int = 0) -> NASimSubnetTarget:
    """Build a segmented (subnet-per-branch) NASim network and wrap it -- the faithful
    diminishing-returns foraging testbed in real NASim mechanics."""
    from .segmented_scenario import write_segmented_scenario

    cfg = cfg or RealismConfig()
    path = write_segmented_scenario(n_subnets=n_subnets, hosts_per_subnet=hosts_per_subnet,
                                    n_services=num_services, heterogeneity=heterogeneity,
                                    seed=seed)
    env = nasim.load(path, fully_obs=True, flat_actions=True, flat_obs=True)
    env.reset()
    return NASimSubnetTarget(env, cfg, np.random.default_rng(seed), brain)


def make_flat_nasim_target(brain, cfg: RealismConfig | None = None, *, num_hosts: int = 24,
                           num_services: int = 6, heterogeneity: float = 0.8,
                           seed: int = 0) -> NASimTarget:
    """Build a flat, fully-reachable, heterogeneous NASim network and wrap it.

    This is the Tier-A testbed: real NASim exploit mechanics over a flat frontier whose
    branch heterogeneity we control directly (the paper's independent variable)."""
    from .flat_scenario import write_flat_scenario

    cfg = cfg or RealismConfig()
    path = write_flat_scenario(n_hosts=num_hosts, n_services=num_services,
                               heterogeneity=heterogeneity, seed=seed)
    env = nasim.load(path, fully_obs=True, flat_actions=True, flat_obs=True)
    env.reset()
    return NASimTarget(env, cfg, np.random.default_rng(seed), brain)


def make_nasim_target(brain, cfg: RealismConfig | None = None, *, num_hosts: int = 20,
                      num_services: int = 5, num_os: int = 2, restrictiveness: int = 2,
                      exploit_probs="mixed", seed: int = 0) -> NASimTarget:
    """Generate a sandboxed NASim network and wrap it as a foraging target."""
    cfg = cfg or RealismConfig()
    env = nasim.generate(
        num_hosts=num_hosts, num_services=num_services, num_os=num_os,
        restrictiveness=restrictiveness, exploit_probs=exploit_probs,
        seed=seed, fully_obs=True, flat_actions=True, flat_obs=True,
    )
    env.reset()
    return NASimTarget(env, cfg, np.random.default_rng(seed), brain)
