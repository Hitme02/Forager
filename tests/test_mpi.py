"""Tests for the MPI worker-pool helpers (serial path; no MPI runtime needed)."""

from __future__ import annotations

from ffs.mpi.worker_pool import POLICIES, _build_tasks, _run_chunk, _run_episode


def test_build_tasks_covers_all_work():
    levels = [0.0, 0.6, 1.2]
    tasks = _build_tasks(levels, seeds=50, chunk=25)
    # 2 policies x 3 levels x (50/25 = 2 chunks) = 12 tasks, covering every seed once.
    assert len(tasks) == len(POLICIES) * len(levels) * 2
    for pname in POLICIES:
        seeds = sorted(s for t in tasks for s in t["seeds"]
                       if t["policy"] == pname and t["sigma"] == 0.0)
        assert seeds == list(range(50))


def test_run_chunk_returns_times():
    task = {"policy": "mvt", "spec": POLICIES["mvt"], "sigma": 0.9, "seeds": [0, 1, 2]}
    out = _run_chunk(task)
    assert out["policy"] == "mvt" and out["sigma"] == 0.9
    assert len(out["ticks"]) == 3
    assert all((t is None) or (t > 0) for t in out["ticks"])


def test_episode_is_deterministic_per_seed():
    a = _run_episode(POLICIES["fixed_timeout"], 0.6, 7)
    b = _run_episode(POLICIES["fixed_timeout"], 0.6, 7)
    assert a == b          # same seed -> same surface -> same outcome
