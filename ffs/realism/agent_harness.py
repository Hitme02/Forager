"""Minimal, clean agent harness for Tier B -- independent of any internship/NDA code.

The harness is the only thing that touches the range. It exposes a **bounded, fixed menu**
of standard, non-weaponised actions and enforces sandbox isolation before every one of them.
A per-branch *brain* chooses the next action from the menu; the harness executes it against
the service and returns the grounded `Evidence`. This keeps three concerns clean and
separable, exactly as in the NASim layer:

  brain      -- the *local tactic* on one service (what to try next). Never decides to give
                up: self-abandonment is the scheduler's MVT decision, not the brain's.
  harness    -- executes one bounded action, grounding progress in observed state, and
                refuses to act unless the sandbox proves it is isolated.
  scheduler  -- unchanged; decides across the parallel frontier when to abandon and migrate.

The action menu is deliberately small and standard -- probe, enumerate, attempt access with
the range's planted default credentials, collect a readable artifact. It contains no exploit
synthesis; the live path (see `services.LiveHttpService`) maps each action onto a plain HTTP
request against a deliberately-vulnerable fixture. The `ScriptedBrain` drives the whole
pipeline with no model/API cost; `LLMBrain` is the same interface with a language model
choosing the next action, gated behind a backend the operator supplies.
"""

from __future__ import annotations

import enum
from typing import Protocol

import numpy as np

from .sandbox import Sandbox
from .services import Evidence, ServiceModel


class Action(enum.Enum):
    """The bounded, non-weaponised action menu the harness can execute on a service."""

    PROBE = "probe"                    # is the service reachable? -> host_discovered
    ENUMERATE = "enumerate"            # read a freely-offered banner/version -> service_enumerated
    ATTEMPT_ACCESS = "attempt_access"  # try the range's planted default creds -> foothold_shell
    COLLECT = "collect"                # read an artifact the current access exposes


class Brain(Protocol):
    """Chooses the next `Action` on the committed service, or None if nothing useful remains."""

    def next_action(self, service: ServiceModel) -> Action | None: ...


class ScriptedBrain:
    """Deterministic probe -> enumerate -> access -> collect tactic for one service.

    Mirrors `brains.ScriptedBrain`: it keeps trying to make progress and returns None only
    when the service is genuinely exhausted. On a service whose access never succeeds it
    retries `ATTEMPT_ACCESS` indefinitely, producing flat progress -- and it is then the
    scheduler, via MVT, that decides to abandon. Baking a retry cap in here would pre-empt
    the very decision the paper studies.
    """

    def next_action(self, service: ServiceModel) -> Action | None:
        if not service.probed:
            return Action.PROBE
        if not service.enumerated:
            return Action.ENUMERATE
        if not service.access_gained:
            return Action.ATTEMPT_ACCESS      # retry until the scheduler migrates us off
        if service.exhausted():
            return None                        # harvested: nothing useful left
        return Action.COLLECT


class HarvestBrain:
    """Foraging tactic for the sweep: harvest value from tick one, never self-abandon.

    Mirrors NASim's `SubnetBrain(scan_first=False)`: it skips the recon ramp (which, on a
    fresh branch, would feed the marginal estimator a run of near-zero samples and make MVT
    abandon before any value is collected) and goes straight to the valuable action. On a
    service whose access is immediate (`auto_access`, the diminishing regime) that is COLLECT,
    which establishes access itself and is productive at once. On a constant-hazard service it
    retries ATTEMPT_ACCESS until the foothold lands, then COLLECTs the bonus -- the memoryless
    access whose expected remaining effort never falls is exactly the regime where a tuned
    fixed timeout ties MVT. Giving up is always the scheduler's decision, never the brain's.
    """

    def next_action(self, service: ServiceModel) -> Action | None:
        if service.exhausted():
            return None
        if not service.access_gained and not getattr(service, "auto_access", False):
            return Action.ATTEMPT_ACCESS      # constant-hazard retry until the scheduler moves us
        return Action.COLLECT


class LLMBrain:
    """Same interface, with a language model choosing the next action from the bounded menu.

    `backend(prompt) -> str` is any callable the operator supplies (an API client, a local
    model); the harness never embeds one. The model is constrained to the fixed `Action`
    menu -- it chooses *which* grounded step to take, not free-form commands -- so grounding
    and the bounded action set still hold. Falls back to the scripted tactic if the model's
    reply is not a valid action. Not exercised by the default experiment or the tests.
    """

    def __init__(self, backend, fallback: Brain | None = None) -> None:
        self._backend = backend
        self._fallback = fallback or ScriptedBrain()

    def _observe(self, s: ServiceModel) -> str:
        return (f"service={s.service_id} probed={s.probed} enumerated={s.enumerated} "
                f"access={s.access_gained} collected={getattr(s, 'collected', 0)} "
                f"flag={s.flag_read}. Choose one: {[a.value for a in Action]}.")

    def next_action(self, service: ServiceModel) -> Action | None:
        try:
            reply = self._backend(self._observe(service)).strip().lower()
            for a in Action:
                if a.value in reply:
                    return a
        except Exception:
            pass
        return self._fallback.next_action(service)


class AgentHarness:
    """Executes one bounded action per step, enforcing isolation, grounding every result."""

    _DISPATCH = {
        Action.PROBE: "probe",
        Action.ENUMERATE: "enumerate",
        Action.ATTEMPT_ACCESS: "attempt_access",
        Action.COLLECT: "collect",
    }

    def __init__(self, sandbox: Sandbox, brain: Brain | None = None) -> None:
        self.sandbox = sandbox
        self.brain = brain or ScriptedBrain()

    def step(self, service: ServiceModel, rng: np.random.Generator) -> Evidence:
        """Take one grounded action on `service`; return its evidence (possibly empty).

        Isolation is asserted *before* the action; a sandbox that cannot prove it is isolated
        raises and the action never runs (fail closed)."""
        self.sandbox.assert_isolated()
        action = self.brain.next_action(service)
        if action is None:
            return Evidence(None, f"svc{service.service_id}", grounded=False)
        method = getattr(service, self._DISPATCH[action])
        return method(rng)
