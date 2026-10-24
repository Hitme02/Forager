"""Per-service branch mechanics for the Tier-B range, with grounded evidence.

A `ServiceModel` is one attackable service on the range and becomes one foraging branch.
The harness drives it through a small, fixed menu of actions (see `agent_harness.py`); each
action returns `Evidence` describing what was *grounded* against the service's real state --
never what an agent claimed. The branch turns that evidence into validated progress via the
grounded `ProgressLedger` (discrete milestones) plus continuous collection value.

Two regimes implement the transfer-boundary contrast the thesis predicts, now with real
agent-harness mechanics rather than an assumed curve:

- `DiminishingService` -- access is cheap, then collection yields a CONCAVE stream of
  artifact value (asymptote = branch richness A_i, time-constant tau_i): the diminishing
  per-branch value MVT needs. Heterogeneous A_i across services (set by the sandbox's sigma
  knob) is what makes abandoning poor services for rich ones pay -- MVT's predicted win.

- `FootholdService` -- a single foothold obtained through a CONSTANT-HAZARD (memoryless)
  access attempt, then a flat uniform bonus: the NASim-style regime. Because expected
  remaining value does not fall with time spent, abandoning buys little and a tuned fixed
  timeout ties MVT. This is the transfer boundary, reproduced in the live harness.

`LiveHttpService` is the real-range analogue (gated): the same milestones, grounded in a
container's actual HTTP responses over localhost, used only by `DockerSandbox`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Evidence:
    """What an action grounded against a service's real state this step.

    `milestone` feeds the discrete `ProgressLedger` (deduped, weighted); `value` is extra
    continuous collection progress (the concave artifact stream). `grounded` must be True
    for either to count -- the harness sets it only when the service's state predicate held.
    """

    milestone: str | None            # ledger milestone name, or None
    target_ref: str                  # which service/artifact the evidence is about
    value: float = 0.0               # continuous collection progress (non-milestone)
    grounded: bool = True            # did the evidence predicate actually hold?


@runtime_checkable
class ServiceModel(Protocol):
    """The view the harness and branch depend on. Both regimes and the live service match."""

    service_id: int
    prior_score: float
    holds_flag: bool
    access_gained: bool
    flag_read: bool
    auto_access: bool            # True => collection establishes access itself (no ramp)

    def probe(self, rng: np.random.Generator) -> Evidence: ...
    def enumerate(self, rng: np.random.Generator) -> Evidence: ...
    def attempt_access(self, rng: np.random.Generator) -> Evidence: ...
    def collect(self, rng: np.random.Generator) -> Evidence: ...
    def exhausted(self) -> bool: ...


class _BaseService(abc.ABC):
    """Shared grounded bookkeeping; subclasses define the regime's access/collection shape."""

    auto_access = False               # overridden True by regimes where access is immediate

    def __init__(self, service_id: int, access_cost: float, holds_flag: bool,
                 burst_p: float, seed: int) -> None:
        self.service_id = service_id
        self.access_cost = max(access_cost, 1e-6)
        self.holds_flag = holds_flag
        self.burst_p = float(np.clip(burst_p, 1e-3, 1.0))
        self._rng = np.random.default_rng(seed)
        # grounded state
        self.probed = False
        self.enumerated = False
        self.access_gained = False
        self.flag_read = False
        self.collected = 0            # number of artifacts actually read

    # recon prior is observable attack surface, not ground-truth richness
    @property
    def prior_score(self) -> float:
        # a cheap, noisy surface signal (service count/version exposure), scaled by a guess
        return self._prior

    def _set_prior(self, surface: float) -> None:
        self._prior = float(surface)

    def probe(self, rng: np.random.Generator) -> Evidence:
        ref = f"svc{self.service_id}"
        if self.probed:
            return Evidence(None, ref, grounded=False)
        self.probed = True
        return Evidence("host_discovered", ref, grounded=True)

    def enumerate(self, rng: np.random.Generator) -> Evidence:
        ref = f"svc{self.service_id}"
        if self.enumerated or not self.probed:
            return Evidence(None, ref, grounded=False)
        self.enumerated = True
        return Evidence("service_enumerated", ref, grounded=True)

    @abc.abstractmethod
    def attempt_access(self, rng: np.random.Generator) -> Evidence: ...

    @abc.abstractmethod
    def collect(self, rng: np.random.Generator) -> Evidence: ...

    @abc.abstractmethod
    def exhausted(self) -> bool: ...


class DiminishingService(_BaseService):
    """Cheap access, then a concave collection stream (the diminishing-value regime).

    After one grounded access step, each `collect` reads the next artifact; the artifacts
    are planted in decreasing value so the marginal collection rate falls as
    g'(c) = (A/tau) e^{-c/tau} -- the diminishing returns MVT abandons on. Richness A is the
    service's heterogeneous `access_cost` re-used as the asymptote scale (the sandbox sets it
    log-normal with mean fixed), so services differ in how much they are worth harvesting.
    """

    def __init__(self, service_id, A, tau, holds_flag, burst_p, seed,
                 prior: float | None = None, prior_noise: float = 0.4,
                 flag_depth: int = 3) -> None:
        # access_cost is unused in this regime (access is cheap); pass A as its scale.
        super().__init__(service_id, A, holds_flag, burst_p, seed)
        self.A = float(A)                               # asymptotic collectable value (richness)
        self.tau = float(tau)                           # return time-constant (payout speed)
        self.flag_depth = flag_depth                    # artifacts to read before the flag
        # recon prior: a noisy observable signal of richness, not ground truth. The sandbox
        # supplies one generated with the same convention as the abstract surface; absent
        # that, fall back to an internally-drawn noisy signal.
        if prior is None:
            prior = self.A * np.exp(prior_noise * self._rng.standard_normal())
        self._set_prior(float(prior))

    def attempt_access(self, rng: np.random.Generator) -> Evidence:
        ref = f"svc{self.service_id}"
        if self.access_gained or not self.enumerated:
            return Evidence(None, ref, grounded=False)
        # Diminishing regime: access is cheap and reliable (the value is in collection).
        self.access_gained = True
        return Evidence("foothold_shell", ref, grounded=True)

    auto_access = True                # diminishing regime: access is cheap -> harvest at once

    def collect(self, rng: np.random.Generator) -> Evidence:
        ref = f"svc{self.service_id}"
        milestone = None
        if not self.access_gained:
            # Access is immediate in this regime: the first collect establishes it and is
            # productive from tick one -- the clean diminishing-returns signal (no dead recon
            # ramp to fool the marginal estimator), exactly as NASim's SubnetBrain skips scans.
            self.access_gained = True
            self.probed = self.enumerated = True
            milestone = "foothold_shell"
        # concave marginal: expected gain for reading the c-th artifact
        c = self.collected
        marginal = (self.A / self.tau) * np.exp(-c / self.tau)
        self.collected += 1
        # burstiness: value arrives as rare Bernoulli jumps (threat-to-concavity case)
        if rng.random() < self.burst_p:
            gained = marginal / self.burst_p
        else:
            gained = 0.0
        # crown-jewel: on a flag service, the deep artifact IS the flag (grounded read);
        # it takes precedence over the first-collect foothold milestone if they coincide.
        if self.holds_flag and not self.flag_read and self.collected >= self.flag_depth:
            self.flag_read = True
            milestone = "objective_flag"
        return Evidence(milestone, f"{ref}:art{c}", value=float(gained), grounded=True)

    def exhausted(self) -> bool:
        # effectively depleted once the concave tail is negligible (~5 time-constants)
        return self.collected >= int(np.ceil(5 * self.tau))

    @property
    def max_value(self) -> float:
        return float(self.A)       # asymptote of the concave collection stream


class FootholdService(_BaseService):
    """A single constant-hazard foothold, then a flat bonus (the transfer-boundary regime).

    Each `attempt_access` is a memoryless Bernoulli trial with success probability
    p = 1/access_cost, so expected remaining effort is constant no matter how long the
    service has been worked: stalling on it defers an equally-valuable reward instead of a
    shrinking one. A tuned fixed timeout therefore ties MVT here -- the NASim-style regime,
    reproduced with live-harness mechanics.
    """

    def __init__(self, service_id, access_cost, holds_flag, burst_p, seed,
                 foothold_value: float = 10.0, bonus_value: float = 2.0) -> None:
        super().__init__(service_id, access_cost, holds_flag, burst_p, seed)
        self.p = 1.0 / self.access_cost
        self.foothold_value = foothold_value
        self.bonus_value = bonus_value
        self._bonus_taken = False
        self._set_prior(np.exp(0.3 * self._rng.standard_normal()) * (1.0 + 1.0 / self.p))

    def attempt_access(self, rng: np.random.Generator) -> Evidence:
        ref = f"svc{self.service_id}"
        if self.access_gained:
            return Evidence(None, ref, grounded=False)
        if rng.random() < self.p:               # constant-hazard success
            self.access_gained = True
            return Evidence("foothold_shell", ref, value=self.foothold_value, grounded=True)
        return Evidence(None, ref, grounded=False)   # miss: flat, no progress

    def collect(self, rng: np.random.Generator) -> Evidence:
        ref = f"svc{self.service_id}"
        if not self.access_gained or self._bonus_taken:
            return Evidence(None, ref, grounded=False)
        self._bonus_taken = True
        milestone = None
        if self.holds_flag and not self.flag_read:
            self.flag_read = True
            milestone = "objective_flag"
        # flat uniform bonus: no diminishing signal to abandon on
        return Evidence(milestone, f"{ref}:bonus", value=self.bonus_value, grounded=True)

    def exhausted(self) -> bool:
        return self.access_gained and self._bonus_taken

    @property
    def max_value(self) -> float:
        return float(self.foothold_value + self.bonus_value)


class LiveHttpService(_BaseService):
    """Real-range service grounded in a container's HTTP responses (gated; DockerSandbox).

    Milestones are scored only from observed responses over localhost: a 200 to a probe
    (host_discovered), a readable version/banner (service_enumerated), a successful login
    with the range's planted default credentials (foothold_shell), an artifact listing
    (collection value), and the crown-jewel file becoming readable (objective_flag). It uses
    only the bounded, non-weaponised action set documented in TIERB.md against a
    deliberately-vulnerable fixture. Construction is lazy and this class is never imported on
    the default path.
    """

    def __init__(self, service_id: int, base_url: str, holds_flag: bool = False) -> None:
        super().__init__(service_id, access_cost=1.0, holds_flag=holds_flag,
                         burst_p=1.0, seed=service_id)
        self.base_url = base_url.rstrip("/")
        self._session = None
        self._token: str | None = None
        self._set_prior(1.0)

    def _http(self):
        if self._session is None:
            import requests  # lazy: live path only
            self._session = requests.Session()
        return self._session

    def probe(self, rng):
        ref = f"svc{self.service_id}"
        if self.probed:
            return Evidence(None, ref, grounded=False)
        try:
            ok = self._http().get(self.base_url, timeout=3).status_code < 500
        except Exception:
            ok = False
        self.probed = ok
        return Evidence("host_discovered", ref, grounded=ok)

    def enumerate(self, rng):
        ref = f"svc{self.service_id}"
        if self.enumerated or not self.probed:
            return Evidence(None, ref, grounded=False)
        try:
            r = self._http().get(f"{self.base_url}/version", timeout=3)
            ok = r.status_code == 200 and bool(r.text.strip())
        except Exception:
            ok = False
        self.enumerated = ok
        return Evidence("service_enumerated", ref, grounded=ok)

    def attempt_access(self, rng):
        ref = f"svc{self.service_id}"
        if self.access_gained or not self.enumerated:
            return Evidence(None, ref, grounded=False)
        # bounded action: a single login with the fixture's PLANTED default credentials
        try:
            r = self._http().post(f"{self.base_url}/login",
                                  json={"user": "admin", "password": "admin"}, timeout=3)
            ok = r.status_code == 200 and "token" in r.text
            if ok:
                self._token = r.json().get("token")
        except Exception:
            ok = False
        self.access_gained = ok
        return Evidence("foothold_shell", ref, value=10.0 if ok else 0.0, grounded=ok)

    def collect(self, rng):
        ref = f"svc{self.service_id}"
        if not self.access_gained:
            return Evidence(None, ref, grounded=False)
        try:
            r = self._http().get(f"{self.base_url}/artifact/{self.collected}",
                                 headers={"X-Session": self._token or ""}, timeout=3)
            ok = r.status_code == 200 and bool(r.content)
            size = len(r.content) / 1024.0 if ok else 0.0
        except Exception:
            ok, size = False, 0.0
        c = self.collected
        self.collected += 1
        milestone = None
        if ok and self.holds_flag and not self.flag_read and "FLAG{" in r.text:
            self.flag_read = True
            milestone = "objective_flag"
        return Evidence(milestone, f"{ref}:art{c}", value=float(size), grounded=ok)

    def exhausted(self) -> bool:
        return self.collected >= 32            # bounded collection depth on the live range

    @property
    def max_value(self) -> float:
        return float("inf")       # unknown a priori on the live range -> use the flag objective

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
