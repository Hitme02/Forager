"""The seam between the MVT scheduler and a *real* parallel LLM-agent pentest platform.

The whole point of the Phase-1 abstraction is that the scheduler and the MVT policy do not
know, and must not care, whether a "branch" is a simulated foraging patch or a live LLM
agent attacking a sandboxed host. This module defines the narrow contract that both sides
of that seam satisfy, so the *identical* scheduler + policies that produced the Phase-1
result can drive the realism layer unchanged.

Design principles
-----------------
1. **Agent-framework-agnostic.** The scheduler needs only four things from a branch:
   its id, its recon *prior score*, how much compute it has absorbed, and -- per unit of
   compute -- how much *validated progress* it just produced. Everything about how an LLM
   agent chooses and executes actions lives behind `BranchBackend.work_one`.

2. **Compute is the common currency.** One unit of compute is one agent *action step*
   (one tool call / one reasoning-act cycle), or equivalently a fixed token / wall-clock
   quantum. The scheduler budgets and compares policies in these units, exactly as in the
   simulator. The *switch cost* is the real setup/teardown of a branch: re-establishing
   the agent's reconnaissance context on a new target.

3. **Progress must be GROUNDED, never self-reported.** `g(c)` -- validated progress -- is
   scored from observable environment state (tool output, captured artifacts, actual
   access gained), NOT from the agent's narration. An agent that *claims* it has a shell
   scores nothing until the shell is demonstrated. This is both a validity requirement
   (otherwise MVT optimises a hallucinated signal) and a safety one. See `progress.py`.

4. **Sandboxed only.** A live `TargetEnvironment` must be an isolated range with no egress
   to systems the operator is not authorised to test. The scheduler never relaxes this;
   adapters are responsible for enforcing it.

The existing `simulator.patch.Patch` already satisfies `BranchBackend` structurally, so the
simulator is simply the first backend. A live branch additionally implements the
`LiveBranch` lifecycle (`enter` / `teardown`) that a realism-aware engine calls when a
worker migrates, turning branch setup/teardown into the measured switch cost.
"""

from __future__ import annotations

import abc
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BranchBackend(Protocol):
    """The minimal view of a branch the scheduler and policies depend on.

    `simulator.patch.Patch` conforms to this; so must any live (LLM-agent) branch.
    """

    patch_id: int
    prior_score: float      # recon-derived prior (e.g. attack-surface / exploitability score)
    spent: float            # compute (agent action steps) invested so far

    def work_one(self, rng: np.random.Generator) -> float:
        """Spend one unit of compute on this branch; return the *validated* progress gain.

        For a live branch this is: let the agent take one grounded action on its target,
        observe the environment, and return the scored progress delta (>= 0)."""
        ...


class LiveBranch(abc.ABC):
    """Base class for a real attack branch driven by an LLM agent against a sandbox target.

    Subclasses bind a specific agent framework and a specific `TargetEnvironment` adapter.
    The lifecycle maps directly onto the scheduler's notion of migration:

        enter()      -> establish agent context on this branch's target (part of switch cost)
        work_one()   -> one grounded action step; returns scored validated-progress delta
        teardown()   -> release the target / archive transcript (part of switch cost)
    """

    def __init__(self, patch_id: int, prior_score: float) -> None:
        self.patch_id = patch_id
        self.prior_score = prior_score
        self.spent = 0.0
        self.cumulative_score = 0.0

    @abc.abstractmethod
    def enter(self) -> None:
        """Establish agent + reconnaissance context on the target (bounded setup work)."""

    @abc.abstractmethod
    def work_one(self, rng: np.random.Generator) -> float:
        """One grounded agent action step; return validated-progress delta (>= 0)."""

    @abc.abstractmethod
    def teardown(self) -> None:
        """Release the target and persist the transcript/evidence for this visit."""


@runtime_checkable
class TargetEnvironment(Protocol):
    """A sandboxed attack surface the platform forages over.

    An adapter (CyberBattleSim/NASim, or a Dockerized vulnerable range) implements this to
    enumerate the branches available to the scheduler and to decide when the objective
    (e.g. capture of a designated crown-jewel flag) has been reached.
    """

    def branches(self) -> list[BranchBackend]:
        """Current frontier of attackable branches, each with a recon prior score."""
        ...

    def objective_met(self) -> bool:
        """Has the engagement objective been achieved (grounded, not agent-claimed)?"""
        ...

    @property
    def switch_cost(self) -> float:
        """Compute units to abandon one branch and bring a worker up on another."""
        ...
