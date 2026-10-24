"""Distributed MVT-sweep via an MPI master-worker pool (PAGP Units II & IV).

The heterogeneity sweep runs thousands of independent scheduler episodes -- a natural fit for
distributed-memory parallelism. This implements a classic **manager-worker pool** over MPI:

  * rank 0 (manager) holds the work queue (chunks of episodes) and hands a chunk to any worker
    that reports itself free, collecting results as they come back;
  * ranks 1..N-1 (workers) loop: tell the manager they are ready, receive a chunk, run those
    episodes on the CPU discrete-event engine, and send the results back.

This is **dynamic load balancing** by work-pulling: a worker that finishes a cheap chunk
immediately asks for another, so uneven episode durations even out across ranks without any
static partitioning -- the same give-work-to-where-it-is-most-productive principle the MVT
abandonment rule embodies at the branch level, here realised across processes. It exercises
MPI point-to-point messaging (`send`/`recv` with `ANY_SOURCE`/tags) and decomposition/mapping
(Unit IV), on real OS processes with no shared memory (Unit II).

Run it:
    mpirun -np 8 python -m ffs.mpi.worker_pool              # 1 manager + 7 workers
    mpirun -np 4 python -m ffs.mpi.worker_pool --seeds 400  # heavier sweep
    python -m ffs.mpi.worker_pool                           # no MPI -> serial fallback

For a scaling study, time it at several rank counts:
    for n in 1 2 4 8; do mpirun -np $n python -m ffs.mpi.worker_pool --quiet; done
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ..scheduler.engine import EngineConfig, Scheduler
from ..scheduler.policies import make_policy
from ..simulator.environment import EnvConfig, Environment
from ..simulator.heterogeneity import SurfaceConfig

TAG_REQUEST = 1   # worker -> manager: "I'm free" (payload: results, or None on first call)
TAG_TASK = 2      # manager -> worker: a chunk to run
TAG_STOP = 3      # manager -> worker: no more work, shut down

POLICIES = {
    "fixed_timeout": {"type": "fixed_timeout", "timeout": 20.0},
    "mvt": {"type": "mvt", "estimator": "recent_slope",
            "estimator_kw": {"half_life": 8.0}, "min_samples": 3, "hysteresis": 0.05},
}


def _run_episode(spec: dict, sigma: float, seed: int) -> float | None:
    rng = np.random.default_rng(seed)
    env = Environment(SurfaceConfig(n_branches=40, heterogeneity=sigma),
                      EnvConfig(switch_cost=5.0, target_frac=0.5), rng)
    res = Scheduler(env, make_policy(spec), EngineConfig(n_workers=8), rng).run()
    return float(res.ticks) if res.solved else None


def _run_chunk(task: dict) -> dict:
    """A worker runs one chunk of episodes and returns their times-to-objective."""
    ticks = [_run_episode(task["spec"], task["sigma"], s) for s in task["seeds"]]
    return {"policy": task["policy"], "sigma": task["sigma"], "ticks": ticks}


def _build_tasks(levels, seeds, chunk) -> list[dict]:
    tasks = []
    for pname, spec in POLICIES.items():
        for sigma in levels:
            for i in range(0, seeds, chunk):
                tasks.append({"policy": pname, "spec": spec, "sigma": float(sigma),
                              "seeds": list(range(i, min(i + chunk, seeds)))})
    return tasks


def _aggregate(results: list[dict], levels, verbose=True) -> None:
    import collections
    acc = collections.defaultdict(list)
    for r in results:
        acc[(r["policy"], r["sigma"])].extend(t for t in r["ticks"] if t is not None)
    if not verbose:
        return
    print(f"\n{'sigma':>6} | {'fixed':>9} | {'mvt':>9} | speedup%")
    for sigma in levels:
        fx = np.mean(acc[("fixed_timeout", float(sigma))]) if acc[("fixed_timeout", float(sigma))] else float("nan")
        mv = np.mean(acc[("mvt", float(sigma))]) if acc[("mvt", float(sigma))] else float("nan")
        sp = 100 * (fx - mv) / fx if fx == fx else float("nan")
        print(f"{sigma:>6} | {fx:>9.1f} | {mv:>9.1f} | {sp:+6.1f}")


# --------------------------------------------------------------------------- #
def _manager(comm, size, tasks, levels, quiet):
    results = []
    n_workers = size - 1
    pending = list(tasks)
    finished = 0
    from mpi4py import MPI
    status = MPI.Status()
    while finished < n_workers:
        payload = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_REQUEST, status=status)
        src = status.Get_source()
        if payload is not None:
            results.append(payload)
        if pending:
            comm.send(pending.pop(), dest=src, tag=TAG_TASK)
        else:
            comm.send(None, dest=src, tag=TAG_STOP)
            finished += 1
    _aggregate(results, levels, verbose=not quiet)
    return results


def _worker(comm):
    from mpi4py import MPI
    status = MPI.Status()
    comm.send(None, dest=0, tag=TAG_REQUEST)          # initial "I'm free", no result yet
    while True:
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == TAG_STOP:
            break
        result = _run_chunk(task)
        comm.send(result, dest=0, tag=TAG_REQUEST)    # return result + report free again


def _serial(tasks, levels, quiet):
    results = [_run_chunk(t) for t in tasks]
    _aggregate(results, levels, verbose=not quiet)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="MPI master-worker MVT sweep")
    ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.9, 1.2])
    ap.add_argument("--quiet", action="store_true", help="print only the timing line")
    args = ap.parse_args()

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    except Exception:
        comm, rank, size = None, 0, 1

    tasks = _build_tasks(args.levels, args.seeds, args.chunk)
    n_eps = len(POLICIES) * len(args.levels) * args.seeds

    if size == 1:
        t0 = time.time()
        _serial(tasks, args.levels, args.quiet)
        dt = time.time() - t0
        print(f"[serial] {n_eps} episodes in {dt:.2f}s  ({n_eps/dt:,.0f} eps/s)")
        return

    if rank == 0:
        t0 = time.time()
        _manager(comm, size, tasks, args.levels, args.quiet)
        dt = time.time() - t0
        print(f"[MPI ranks={size} | {size-1} workers] {n_eps} episodes in {dt:.2f}s "
              f"({n_eps/dt:,.0f} eps/s)")
    else:
        _worker(comm)


if __name__ == "__main__":
    main()
