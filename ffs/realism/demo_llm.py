"""Live PAGP demo: a parallel pool of LLM agents foraging a sandbox range under GuardedMVT.

This is the show-the-teacher demo. It foregrounds the PAGP (parallel architecture & GPU)
story and builds the security application on top of it:

  1. TASK PARALLELISM.  W pentest agents forage the range *in parallel*. The GuardedMVT
     scheduler is the parallel load balancer: it watches each worker's return and MIGRATES a
     worker off a stalling branch onto a fresher one (work-stealing by marginal value). The
     demo narrates every ENTER / action / ABANDON+MIGRATE so the parallelism is visible.

  2. GPU DATA PARALLELISM.  Each tick, the W agents must each choose an action. Instead of W
     sequential model calls, the demo serves all W decisions as ONE padded, batched forward
     pass on the GPU (MPS) -- SIMD across the worker pool, the same idea as the batched
     Monte-Carlo in `ffs/gpu/`. It then measures batched-vs-sequential latency so the speed-up
     is concrete.

  3. TUNING AT SCALE.  A closing pointer to the artifacts that make designing this scheduler
     feasible: the GPU batch simulator (~10^5 episodes/s) and the MPI worker pool.

The LLM only ever chooses among the four bounded, non-weaponised actions the harness exposes
(probe / enumerate / attempt_access / collect); an invalid reply falls back to the scripted
tactic, and isolation is asserted before every action. With no model available the demo still
runs (scripted brain), so it never fails in front of an audience.

Run:  python -m ffs.realism.demo_llm            # auto-detect a local model, else scripted
      python -m ffs.realism.demo_llm --backend hf-local --model Qwen/Qwen2.5-0.5B-Instruct
      python -m ffs.realism.demo_llm --backend scripted      # no LLM, always works

The decision logic (GuardedMVT, the frontier opportunity-rate R, the marginal estimators) is
the exact production code from `ffs/scheduler/`; only the tick loop is re-implemented here, in
the open, so each parallel step is legible.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ..scheduler.engine import BranchMemory, Worker
from ..scheduler.estimators import FrontierRateEstimator
from ..scheduler.policies import make_policy
from .agent_harness import Action, HarvestBrain
from .cyber_range import RangeConfig, make_mock_range

GUARDED = {"type": "guarded_mvt", "estimator": "adaptive",
           "estimator_kw": {"base_half_life": 8.0, "max_half_life": 24.0},
           "min_samples": 6, "hysteresis": 0.3, "budget_cap": 16.0}

_WORD2ACTION = {a.value: a for a in Action}


class _FixedBrain:
    """A one-shot brain returning a pre-decided action (so the LLM, not the brain, chooses)."""

    def __init__(self, action: Action | None) -> None:
        self._action = action

    def next_action(self, service):
        return self._action


def _observe(service) -> str:
    """A short, security-framed description of one service's state for the model."""
    return (f"Service svc{service.service_id}. "
            f"reachable={'yes' if service.probed else 'unknown'}, "
            f"enumerated={'yes' if service.enumerated else 'no'}, "
            f"access_gained={'yes' if service.access_gained else 'no'}, "
            f"artifacts_collected={getattr(service, 'collected', 0)}. "
            f"What is the next action?")


def _parse_action(reply: str | None, service) -> Action | None:
    """Map a model reply to one bounded action; fall back to the scripted tactic if invalid."""
    if reply:
        low = reply.lower()
        for word, act in _WORD2ACTION.items():
            if word in low:
                return act
    return HarvestBrain().next_action(service)       # grounded fallback


def run_demo(backend, *, n_services=8, workers=4, switch_cost=4, sigma=1.0,
             regime="diminishing", max_ticks=400, seed=0, verbose=True) -> dict:
    """Run the narrated parallel LLM-agent foraging episode; return a small stats dict."""
    rng = np.random.default_rng(seed)
    cfg = RangeConfig(switch_cost=switch_cost, max_ticks=max_ticks, objective="flag")
    rng_env = make_mock_range(cfg, n_services=n_services, flag_services=1, sigma=sigma,
                              regime=regime, seed=seed, brain=HarvestBrain())
    policy = make_policy(dict(GUARDED))
    policy.reset(rng_env.patches, rng)
    memory = BranchMemory(rng_env.patches)
    frontier = FrontierRateEstimator(switch_cost=switch_cost, entry_window=5,
                                     half_life_entries=15.0)
    pool = [Worker(wid=i) for i in range(workers)]

    def say(msg):
        if verbose:
            print(msg)

    def available():
        return [p for p in rng_env.patches if p.patch_id not in memory.assigned]

    def assign(w):
        choice = policy.select_branch(w, available(), memory)
        if choice is None:
            w.state, w.patch = "idle", None
            return
        w.patch, w.state = choice, "working"
        w.time_on_patch = w.entry_ticks = 0
        w.entry_yield = 0.0
        memory.assigned.add(choice.patch_id)
        policy.on_assign(w)
        say(f"  worker {w.wid}  ENTER   svc{choice.service.service_id}  "
            f"(recon prior {choice.prior_score:5.1f})")

    def decide(prompts):
        """One batched model call for all working workers' decisions (the PAGP step)."""
        if backend is None:
            return [None] * len(prompts)
        try:
            return backend.batch(prompts)
        except Exception:
            return [None] * len(prompts)

    say(f"\n=== Parallel LLM-agent foraging: {workers} agents, {n_services} services, "
        f"GuardedMVT scheduler ===")
    for w in pool:
        assign(w)

    batched_calls = total_decisions = migrations = tick = 0
    solved = rng_env.objective_met()
    while not solved and tick < max_ticks:
        tick += 1
        for w in pool:                       # advance switching / idle workers
            if w.state == "switching":
                w.switch_remaining -= 1
                if w.switch_remaining <= 0:
                    assign(w)
            elif w.state == "idle":
                assign(w)

        working = [w for w in pool if w.state == "working"]
        if working:
            replies = decide([_observe(w.patch.service) for w in working])  # BATCHED
            batched_calls += 1
            total_decisions += len(working)
            for w, reply in zip(working, replies):
                svc = w.patch.service
                was_access, was_flag = svc.access_gained, svc.flag_read
                action = _parse_action(reply, svc)
                rng_env.harness.brain = _FixedBrain(action)      # LLM-chosen action
                gain = w.patch.work_one(rng)                     # grounded execution
                w.time_on_patch += 1
                w.last_gain = gain
                if w.entry_ticks < frontier.entry_window:        # feed the frontier rate R
                    w.entry_yield += gain
                    w.entry_ticks += 1
                    if w.entry_ticks == frontier.entry_window:
                        frontier.record_entry_phase(w.entry_yield / frontier.entry_window)
                memory.on_gain(w.patch.patch_id, gain)
                policy.observe(w, gain)
                act = action.value if action else "idle"
                if svc.flag_read and not was_flag:
                    say(f"  worker {w.wid}  svc{svc.service_id:<2} {act:<14} "
                        f">>> CROWN-JEWEL FLAG CAPTURED <<<")
                elif svc.access_gained and not was_access:
                    say(f"  worker {w.wid}  svc{svc.service_id:<2} {act:<14} foothold gained")
            # periodic heartbeat instead of a line per collect (keeps the trace readable)
            if tick % 10 == 0:
                busy = ", ".join(f"w{w.wid}->svc{w.patch.service.service_id}" for w in working)
                say(f"  [tick {tick:3d}]  harvesting ({rng_env.progress_fraction()*100:3.0f}% "
                    f"to crown jewel)   {busy}")

        R = frontier.rate
        if rng_env.objective_met():
            solved = True
            break
        for w in working:                     # GuardedMVT migration decisions
            if policy.should_abandon(w, R, memory):
                memory.assigned.discard(w.patch.patch_id)
                say(f"  worker {w.wid}  ABANDON svc{w.patch.service.service_id} -> MIGRATE "
                    f"(branch rate below frontier R={R:4.1f})")
                w.state, w.switch_remaining = "switching", switch_cost
                migrations += 1

    say(f"\nResult: flag {'CAPTURED' if solved else 'not captured'} in {tick} ticks  "
        f"| worker migrations: {migrations}  | batched model calls: {batched_calls} "
        f"serving {total_decisions} agent-decisions")
    return {"solved": solved, "ticks": tick, "migrations": migrations,
            "batched_calls": batched_calls, "decisions": total_decisions}


def benchmark_batched_inference(backend, workers=4, reps=5) -> None:
    """Measure serving W agent-decisions as ONE batched GPU pass vs W sequential calls."""
    if backend is None or not hasattr(backend, "batch"):
        print("  (no model loaded -> skipping the batched-inference benchmark)")
        return
    prompts = [_observe(_FakeSvc(i)) for i in range(workers)]
    backend.batch(prompts)                                   # warm up (lazy kernels/caches)
    t0 = time.perf_counter()
    for _ in range(reps):
        backend.batch(prompts)
    t_batched = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        for p in prompts:
            backend(p)
    t_seq = (time.perf_counter() - t0) / reps
    print(f"\n=== PAGP: GPU-batched inference across the {workers}-agent pool ===")
    print(f"  {workers} decisions, batched (1 GPU forward pass): {t_batched*1e3:7.1f} ms")
    print(f"  {workers} decisions, sequential (W passes):        {t_seq*1e3:7.1f} ms")
    if t_batched > 0:
        print(f"  batched speed-up: {t_seq / t_batched:4.2f}x  "
              "(data-parallel across the worker pool -- the same SIMD idea as ffs/gpu/)")


class _FakeSvc:
    def __init__(self, i):
        self.service_id = i
        self.probed = self.enumerated = self.access_gained = self.flag_read = False
        self.collected = 0


def _python_engine_eps_per_s(n_episodes: int = 120) -> float:
    """Throughput of the per-episode discrete-event engine (the reference the GPU replaces)."""
    import time

    from ..scheduler.engine import EngineConfig, Scheduler
    from ..scheduler.policies import make_policy
    from ..simulator.environment import EnvConfig, Environment
    from ..simulator.heterogeneity import SurfaceConfig
    surf = SurfaceConfig(n_branches=40, heterogeneity=0.9)
    env_cfg = EnvConfig(switch_cost=5.0, objective="coverage", target_frac=0.5)
    eng = EngineConfig(n_workers=8)
    pol = {"type": "mvt", "estimator": "recent_slope", "estimator_kw": {"half_life": 8.0}}
    t = time.perf_counter()
    for s in range(n_episodes):
        rng = np.random.default_rng(s)
        Scheduler(Environment(surf, env_cfg, rng), make_policy(dict(pol)), eng, rng).run()
    return n_episodes / (time.perf_counter() - t)


def benchmark_gpu_simulator(n_episodes: int = 50000) -> None:
    """The heavy PAGP artifact: the GPU batch Monte-Carlo that tunes this scheduler.

    Tuning GuardedMVT (the crossover, budget_cap, hysteresis) needs thousands of foraging
    episodes -- embarrassingly parallel. `ffs/gpu/batch_sim.py` runs them as tensor ops, one
    SIMD lane per episode, replacing the per-episode discrete-event engine. This measures the
    real GPU (MPS) throughput against that engine."""
    try:
        import time
        from ..gpu.batch_sim import BatchConfig, pick_device, simulate
    except Exception as exc:
        print(f"\n  (GPU batch simulator unavailable: {exc})")
        return
    pol = {"type": "mvt", "half_life": 8.0, "min_samples": 3, "hysteresis": 0.05}
    cfg = BatchConfig(n_episodes=n_episodes, n_workers=8, sigma=0.9)
    gpu = pick_device()
    simulate(pol, cfg, device=gpu, seed=0)                 # warm up kernels
    t = time.perf_counter(); simulate(pol, cfg, device=gpu, seed=1); tg = time.perf_counter() - t
    gpu_rate = n_episodes / tg
    py_rate = _python_engine_eps_per_s()
    print("\n=== PAGP: GPU batch Monte-Carlo that tunes the scheduler ===")
    print(f"  GPU ({gpu}, SIMD across episodes): {gpu_rate:12,.0f} episodes/s")
    print(f"  per-episode discrete-event engine: {py_rate:12,.0f} episodes/s")
    if py_rate > 0:
        print(f"  GPU speed-up: {gpu_rate/py_rate:5.0f}x  -- a 5-level tuned sweep "
              f"(~0.7M episodes) in ~{0.7e6/gpu_rate:.0f}s vs ~{0.7e6/py_rate/60:.0f} min "
              "(ffs/gpu/batch_sim.py)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Live PAGP demo: parallel LLM agents + GuardedMVT")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "ollama", "hf-local", "hf-api", "scripted"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="model id; for Ollama an HF-style id means 'use the first local model'")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--services", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=11,   # a seed that showcases worker migration
                    help="11 shows the scheduler migrating workers; most seeds also work")
    ap.add_argument("--no-benchmark", action="store_true")
    args = ap.parse_args()

    from .llm_backends import make_backend
    print(f"Loading LLM backend: {args.backend} ({args.model}) ...")
    backend = make_backend(args.backend, args.model)
    if backend is None:
        print("No LLM backend available -> running with the SCRIPTED agent brain "
              "(install `transformers` or set HF_TOKEN for a live model).")
    else:
        print(f"Model ready: {type(backend).__name__} on "
              f"{getattr(backend, 'device', 'remote')}.")

    run_demo(backend, workers=args.workers, n_services=args.services, sigma=args.sigma,
             seed=args.seed)
    if not args.no_benchmark:
        benchmark_batched_inference(backend, workers=args.workers)
        benchmark_gpu_simulator()

    print("\n  Also: distributed MPI worker pool (dynamic load balancing across ranks):")
    print("        mpirun -np 8 python -m ffs.mpi.worker_pool")


if __name__ == "__main__":
    main()
