"""Tier-B tests: the same scheduler drives a bounded agent harness over a sandbox range.

All tests run on the in-process `MockSandbox` -- no Docker, no network, no live traffic --
so the whole scheduler -> harness -> grounded-progress pipeline is exercised in CI. The live
`DockerSandbox` path is gated and not tested here (see TIERB.md)."""

from __future__ import annotations

import numpy as np
import pytest

from ffs.realism.agent_harness import Action, AgentHarness, HarvestBrain, LLMBrain, ScriptedBrain
from ffs.realism.cyber_range import CyberRange, RangeConfig, make_mock_range
from ffs.realism.interfaces import BranchBackend
from ffs.realism.sandbox import DockerSandbox, IsolationError, MockSandbox
from ffs.realism.services import DiminishingService, Evidence, FootholdService, ServiceModel
from ffs.scheduler.engine import EngineConfig, Scheduler
from ffs.scheduler.policies import make_policy

_EST = {"base_half_life": 8.0, "max_half_life": 24.0}
MVT = {"type": "mvt", "estimator": "adaptive", "estimator_kw": _EST,
       "min_samples": 6, "hysteresis": 0.3}
GUARDED = {"type": "guarded_mvt", "estimator": "adaptive", "estimator_kw": _EST,
           "min_samples": 6, "hysteresis": 0.3, "budget_cap": 16.0}


def _run(spec, *, regime, sigma, seed, target_frac=0.5, objective="coverage", max_ticks=12000):
    cfg = RangeConfig(switch_cost=5.0, max_ticks=max_ticks, objective=objective,
                      target_frac=target_frac)
    rng = make_mock_range(cfg, n_services=40, flag_services=2, sigma=sigma, regime=regime,
                          seed=seed)
    sch = Scheduler(rng, make_policy(dict(spec)), EngineConfig(n_workers=8, entry_window=5),
                    np.random.default_rng(1000 + seed))
    return rng, sch.run()


# --------------------------------------------------------------------------- #
# Isolation is the safety contract
# --------------------------------------------------------------------------- #
def test_mock_sandbox_is_trivially_isolated():
    MockSandbox(n_services=4, seed=0).assert_isolated()   # no traffic => no raise


def test_docker_sandbox_requires_acknowledgement():
    # The live range must not be constructable without an explicit authorisation ack.
    with pytest.raises(IsolationError):
        DockerSandbox()


def test_harness_asserts_isolation_before_every_action():
    # A sandbox that cannot prove isolation must stop the action (fail closed).
    class Hostile(MockSandbox):
        def assert_isolated(self):
            raise IsolationError("egress detected")

    sb = Hostile(n_services=2, seed=0)
    harness = AgentHarness(sb, HarvestBrain())
    with pytest.raises(IsolationError):
        harness.step(sb.services()[0], np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# Grounding: progress comes only from observed state
# --------------------------------------------------------------------------- #
def test_branch_counts_only_grounded_evidence():
    sb = MockSandbox(n_services=3, regime="diminishing", sigma=0.0, seed=1)
    rng = CyberRange(sb, RangeConfig(), brain=HarvestBrain())
    branch = rng.patches[0]
    gain = branch.work_one(np.random.default_rng(0))
    assert gain > 0.0                                 # first collect is grounded & productive
    assert branch.collection_value > 0.0
    # ungrounded evidence adds nothing
    before = branch.cumulative_score
    branch.ledger.record("host_discovered", "svcX", grounded=False)
    assert branch.cumulative_score == before


def test_services_satisfy_branch_backend():
    for svc in MockSandbox(n_services=2, regime="diminishing", seed=0).services():
        assert isinstance(svc, ServiceModel)
    rng = make_mock_range(RangeConfig(), n_services=4, seed=0)
    assert isinstance(rng.patches[0], BranchBackend)


def test_diminishing_marginal_decreases():
    s = DiminishingService(0, A=100.0, tau=10.0, holds_flag=False, burst_p=1.0, seed=0)
    g = np.random.default_rng(0)
    gains = [s.collect(g).value for _ in range(8)]
    assert all(gains[i] >= gains[i + 1] - 1e-9 for i in range(len(gains) - 1))


def test_foothold_is_constant_hazard():
    # Memoryless access: success probability per attempt does not fall with attempts spent.
    s = FootholdService(0, access_cost=4.0, holds_flag=False, burst_p=1.0, seed=0)
    assert abs(s.p - 0.25) < 1e-9


# --------------------------------------------------------------------------- #
# The same scheduler runs end to end over the range
# --------------------------------------------------------------------------- #
def test_scheduler_solves_diminishing_coverage():
    _, r = _run(MVT, regime="diminishing", sigma=0.9, seed=0)
    assert r.solved and r.ticks > 0


def test_flag_objective_captures_crown_jewel():
    rng, r = _run(MVT, regime="diminishing", sigma=0.6, seed=0, objective="flag")
    assert r.solved
    assert rng.sandbox.flag_captured()


def test_transfer_boundary_direction():
    # The thesis shape: MVT's edge over a fixed timeout is larger at high heterogeneity than
    # at low. We compare MVT's time-to-objective at sigma=1.2 vs sigma=0.0 against a common
    # fixed timeout, over a few seeds (kept small for CI speed).
    seeds = range(6)
    fixed = {"type": "fixed_timeout", "timeout": 25.0}

    def mean_speedup(sigma):
        sp = []
        for s in seeds:
            _, rf = _run(fixed, regime="diminishing", sigma=sigma, seed=s)
            _, rm = _run(MVT, regime="diminishing", sigma=sigma, seed=s)
            if rf.solved and rm.solved:
                sp.append((rf.ticks - rm.ticks) / rf.ticks)
        return float(np.mean(sp))

    assert mean_speedup(1.2) > mean_speedup(0.0)


# --------------------------------------------------------------------------- #
# Brains
# --------------------------------------------------------------------------- #
def test_harvest_brain_never_abandons_itself():
    # On an un-accessible foothold service the brain keeps retrying (returns an action),
    # leaving the give-up decision to the scheduler -- it never self-abandons.
    s = FootholdService(0, access_cost=1e9, holds_flag=False, burst_p=1.0, seed=0)
    brain = HarvestBrain()
    assert brain.next_action(s) == Action.ATTEMPT_ACCESS


# --------------------------------------------------------------------------- #
# GuardedMVT: the boundary-aware policy turns the transfer boundary into a design win
# --------------------------------------------------------------------------- #
def test_plain_mvt_collapses_but_guarded_solves_constant_hazard():
    # Plain MVT abandons memoryless footholds before they land -> it cannot reach the
    # objective in the constant-hazard regime. GuardedMVT falls back to a committed budget
    # where the signal is uninformative, so it still solves.
    seeds = range(5)
    mvt_solved = sum(_run(MVT, regime="constant", sigma=0.6, seed=s)[1].solved for s in seeds)
    grd_solved = sum(_run(GUARDED, regime="constant", sigma=0.6, seed=s)[1].solved for s in seeds)
    assert mvt_solved == 0
    assert grd_solved == len(list(seeds))


def test_guarded_preserves_diminishing_advantage():
    # On the diminishing regime GuardedMVT keeps (here, slightly beats) MVT's edge over a
    # fixed timeout at high heterogeneity.
    seeds = range(6)
    fixed = {"type": "fixed_timeout", "timeout": 40.0}
    sp = []
    for s in seeds:
        _, rf = _run(fixed, regime="diminishing", sigma=1.2, seed=s)
        _, rg = _run(GUARDED, regime="diminishing", sigma=1.2, seed=s)
        if rf.solved and rg.solved:
            sp.append((rf.ticks - rg.ticks) / rf.ticks)
    assert float(np.mean(sp)) > 0.0       # GuardedMVT faster than the fixed timeout


def test_guarded_is_robust_across_both_regimes():
    # The design win: one GuardedMVT configuration solves BOTH regimes (unlike plain MVT,
    # which collapses under constant hazard).
    _, rd = _run(GUARDED, regime="diminishing", sigma=1.2, seed=0)
    _, rc = _run(GUARDED, regime="constant", sigma=1.2, seed=0)
    assert rd.solved and rc.solved


def test_llm_brain_falls_back_to_scripted_on_bad_reply():
    s = MockSandbox(n_services=1, regime="diminishing", seed=0).services()[0]
    brain = LLMBrain(backend=lambda prompt: "garbage not-an-action", fallback=ScriptedBrain())
    assert brain.next_action(s) == Action.PROBE       # fell back to the scripted first step


# --------------------------------------------------------------------------- #
# Live PAGP demo (scripted path -- no model, no Ollama, no network)
# --------------------------------------------------------------------------- #
def test_make_backend_scripted_is_none():
    from ffs.realism.llm_backends import make_backend
    assert make_backend("scripted") is None       # demo degrades to the scripted brain


def test_parse_action_maps_words_and_falls_back():
    from ffs.realism.demo_llm import _parse_action
    s = MockSandbox(n_services=1, regime="diminishing", seed=0).services()[0]
    assert _parse_action("collect", s) == Action.COLLECT
    assert _parse_action("let's attempt_access now", s) == Action.ATTEMPT_ACCESS
    assert _parse_action("i am unsure", s) in set(Action)   # invalid -> grounded fallback


def test_demo_runs_and_captures_flag_with_scripted_brain():
    from ffs.realism.demo_llm import run_demo
    r = run_demo(None, n_services=8, workers=4, seed=11, verbose=False)
    assert r["solved"] and r["ticks"] > 0 and r["decisions"] > 0
