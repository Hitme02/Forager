"""Live containerised evaluation (camera-ready; reviewer #2, practical validation).

Runs INSIDE the harness container of `docker/eval/compose.yml`, on an internal Docker network
with no route to the host or the internet. Each branch is a real HTTP fixture container; every
action is a real request; the LLM brain (if `OLLAMA_HOST` answers) is an in-network Ollama
server. `DockerSandbox.assert_isolated()` is called before every action and counted.

For each brain (scripted, then the LLM) the fixed timeout is tuned on this range over a grid,
then the tuned timeout, plain MVT, GuardedMVT and GuardedMVT with the dynamic budget are run
on 10 seeds. A seed changes the noisy reconnaissance prior (the scheduler's view of which
service looks rich), not the services themselves. Outputs go to /out (bind-mounted).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from ..scheduler.engine import EngineConfig, Scheduler
from ..scheduler.policies import make_policy
from .agent_harness import ScriptedBrain
from .cyber_range import CyberRange, RangeConfig
from .llm_agent import LLMAgentBrain, OllamaChat
from .run_tierb import GUARDED, MVT_ADAPTIVE
from .sandbox import DockerSandbox

OUT = os.environ.get("LIVE_OUT", "/out")
GRID = [4, 6, 8, 12, 16, 24, 32, 48]
SEEDS = list(range(int(os.environ.get("LIVE_SEEDS", "10"))))
N_WORKERS = int(os.environ.get("LIVE_WORKERS", "2"))
TARGET = float(os.environ.get("LIVE_TARGET", "0.8"))
POLICIES = {"mvt_adaptive": MVT_ADAPTIVE, "guarded_mvt": GUARDED,
            "guarded_mvt_dyn": {**GUARDED, "budget_mode": "dynamic", "budget_kappa": 3.0}}


class CountingDockerSandbox(DockerSandbox):
    checks = 0

    def assert_isolated(self) -> None:
        CountingDockerSandbox.checks += 1
        super().assert_isolated()


def episode(manifest, spec, seed, brain_factory):
    rng = np.random.default_rng(seed)
    man = [dict(m, prior=m["scale"] * float(np.exp(-0.08 + 0.4 * rng.standard_normal())))
           for m in manifest]
    sb = CountingDockerSandbox(acknowledged=True, manifest=man, probe_timeout=1.0)
    c0 = CountingDockerSandbox.checks
    cr = CyberRange(sb, RangeConfig(switch_cost=5.0, max_ticks=1500, objective="coverage",
                                    target_frac=TARGET), brain_factory())
    t0 = time.time()
    r = Scheduler(cr, make_policy(dict(spec)),
                  EngineConfig(n_workers=N_WORKERS, entry_window=5, entry_mode="productive"),
                  np.random.default_rng(1000 + seed)).run()
    sb.teardown()
    return {"seed": seed, "solved": bool(r.solved), "ticks": int(r.ticks),
            "switches": int(r.n_switches), "isolation_checks": CountingDockerSandbox.checks - c0,
            "wall_s": time.time() - t0}


def run_brain(name, factory, manifest):
    rows = []
    best = None
    for t in GRID:
        df = pd.DataFrame([episode(manifest, {"type": "fixed_timeout", "timeout": float(t)}, s,
                                   factory) for s in SEEDS])
        if df.solved.mean() >= 0.95:
            m = df.ticks[df.solved].mean()
            if best is None or m < best[0]:
                best = (m, t, df)
        print(f"  [{name}] tune t={t}: solved {df.solved.mean():.2f} ticks {df.ticks.mean():.1f}",
              flush=True)
    rows += [{"brain": name, "policy": f"fixed* (t={best[1]})", **x}
             for x in best[2].to_dict("records")]
    for pname, spec in POLICIES.items():
        for s in SEEDS:
            rows.append({"brain": name, "policy": pname, **episode(manifest, spec, s, factory)})
        print(f"  [{name}] {pname} done", flush=True)
    return rows


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    with open("/work/manifest.json") as f:
        manifest = json.load(f)
    meta = {"services": len(manifest), "workers": N_WORKERS, "target_frac": TARGET,
            "seeds": len(SEEDS),
            "total_value_kb": sum(m["max_value"] for m in manifest)}
    rows = run_brain("scripted", ScriptedBrain, manifest)
    host = os.environ.get("OLLAMA_HOST")
    model = os.environ.get("LIVE_MODEL", "qwen2.5:7b")
    if host:
        be = OllamaChat(model, host=host, cache_path=os.path.join(OUT, "live_llm_cache.json"))
        be._query("Service state: reachable=unknown. Recent actions, oldest first: none. "
                  "Next action?")                         # warm-up (loads the model)
        be.calls, be.latency_s = 0, 0.0
        rows += run_brain(model, lambda: LLMAgentBrain(be, history=3), manifest)
        be.save()
        meta.update({"llm_model": model, "llm_calls": be.calls, "llm_lookups": be.lookups,
                     "llm_latency_ms": 1000 * be.latency_s / max(be.calls, 1)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "live_perseed.csv"), index=False)
    meta.update({"episodes": len(df), "isolation_checks": int(df.isolation_checks.sum()),
                 "isolation_failures": 0})
    with open(os.path.join(OUT, "live_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    summ = df.groupby(["brain", "policy"], sort=False).agg(
        solved=("solved", "mean"), ticks=("ticks", "mean"), sd=("ticks", "std"),
        switches=("switches", "mean"), wall_s=("wall_s", "mean")).reset_index()
    summ.to_csv(os.path.join(OUT, "live_summary.csv"), index=False)
    print(summ.round(2).to_string(index=False))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
