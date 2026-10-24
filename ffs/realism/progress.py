"""Grounded validated-progress scoring -- the real-world definition of g(c).

In the simulator, g(c) is a known concave curve. In the realism layer it must be *measured*
from what an agent actually achieved on a branch, from observable evidence only. This module
defines that scoring, which is the single most important design choice for external validity:
MVT is only as good as the progress signal it optimises.

Scoring model
-------------
Progress on a branch is the cumulative weighted sum of *milestones* the agent has
demonstrably reached, each counted once, each requiring grounded evidence:

    milestone              weight   grounding evidence (examples)
    ---------------------  ------   --------------------------------------------------
    host_discovered           1     host responds / appears in a scan the agent ran
    service_enumerated        2     banner / version captured from the live service
    vuln_identified           3     a check (not a guess) confirms the weakness
    credential_obtained       5     a credential that authenticates successfully
    foothold_shell           10     a command executes on the target and returns output
    privilege_escalation     15     an action succeeds that requires higher privilege
    lateral_reachable         8     a new host is reachable *from* the foothold
    objective_flag           50     the designated crown-jewel artifact is captured

Why this shape matters for the thesis:
- **Diminishing returns** emerge naturally: a branch's remaining un-reached milestones get
  harder and sparser as it is worked, so marginal progress falls -- the condition MVT needs.
- **Burstiness** is real here: `foothold_shell` and `objective_flag` are large, rare jumps.
  This is exactly the smooth-concavity threat the burstiness ablation studies, now with a
  concrete mechanism -- strengthening the link between the simulator and the live setting.
- **Grounding** prevents the agent from inflating g(c) by narration: a milestone scores
  only when its evidence predicate holds against observed state.

`ProgressLedger` is pure bookkeeping over evidence events; it has no dependency on any agent
framework or environment, so it is testable in isolation and reusable across adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_WEIGHTS: dict[str, float] = {
    "host_discovered": 1.0,
    "service_enumerated": 2.0,
    "vuln_identified": 3.0,
    "credential_obtained": 5.0,
    "foothold_shell": 10.0,
    "privilege_escalation": 15.0,
    "lateral_reachable": 8.0,
    "objective_flag": 50.0,
}


@dataclass
class ProgressLedger:
    """Cumulative grounded progress for one branch, scored from milestone evidence."""

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    _reached: set[tuple[str, str]] = field(default_factory=set)  # (milestone, target_ref)
    cumulative: float = 0.0

    def record(self, milestone: str, target_ref: str, grounded: bool) -> float:
        """Record a milestone for a specific target; return the progress delta it adds.

        `grounded` must be True -- the caller asserts the milestone's evidence predicate
        held against observed environment state. Ungrounded or duplicate milestones add 0.
        """
        if not grounded:
            return 0.0
        key = (milestone, target_ref)
        if key in self._reached or milestone not in self.weights:
            return 0.0
        self._reached.add(key)
        delta = self.weights[milestone]
        self.cumulative += delta
        return delta

    @property
    def milestones_reached(self) -> int:
        return len(self._reached)
