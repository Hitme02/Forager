"""Tests for the realism seam: the scheduler must be agent-agnostic."""

from __future__ import annotations

import numpy as np

from ffs.realism.interfaces import BranchBackend
from ffs.realism.progress import ProgressLedger
from ffs.simulator.patch import Patch


def test_patch_is_a_branch_backend():
    # The simulated Patch must structurally satisfy the live-branch contract, so the same
    # scheduler can drive either a simulated patch or an LLM-agent branch.
    p = Patch(0, A=10.0, tau=20.0, prior_score=3.0)
    assert isinstance(p, BranchBackend)


def test_progress_ledger_grounding_and_dedup():
    led = ProgressLedger()
    # Ungrounded milestone scores nothing (no hallucinated progress).
    assert led.record("foothold_shell", "host-1", grounded=False) == 0.0
    # Grounded milestone scores its weight once...
    assert led.record("foothold_shell", "host-1", grounded=True) == 10.0
    # ...and not again (deduplicated per (milestone, target)).
    assert led.record("foothold_shell", "host-1", grounded=True) == 0.0
    # A different target re-earns it.
    assert led.record("foothold_shell", "host-2", grounded=True) == 10.0
    assert led.record("objective_flag", "host-2", grounded=True) == 50.0
    assert led.cumulative == 70.0
    assert led.milestones_reached == 3
