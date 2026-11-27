"""LLM-driven agent harness evaluation (camera-ready; reviewers #2 and #5).

The Tier-B robustness setting (sigma = 1.2, diminishing-value and constant-hazard regimes, 40
services, 8 workers, 20 seeds) is re-run with the per-branch brain replaced by a real local
LLM served by Ollama (greedy decoding), alongside the scripted control brain
(probe -> enumerate -> access -> collect). For every brain the fixed timeout is re-tuned with
that brain, so each policy is compared against a timeout tuned for the agent actually used.

Outputs (results/cr/llm/):
  llm_perseed.csv     per-seed outcomes for every (brain, regime, policy)
  llm_summary.csv     solve rate, mean ticks and speedup vs the brain's tuned timeout
  llm_decisions.csv   decision quality per (brain, regime): strict / parsed / fallback /
                      agreement-with-scripted / no-progress rates, model calls, latency
  llm_replay.csv      uncached full-episode replay vs memoised decisions (equivalence check)
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from ..experiments.camera_ready import (GRID_FIXED_TB, GUARDED, GUARDED_DYN, TIERB_CAP,
                                        families, run_many, score, tierb_episode, tune)
from ..experiments.stats import censor, speedup_ci, wilcoxon_p
from .agent_harness import ScriptedBrain
from .llm_agent import DecisionStats, LLMAgentBrain, OllamaChat
from ..experiments.camera_ready import load_learned
from .run_tierb import MVT_ADAPTIVE

OUT = "results/cr/llm"
OUTP = ""
MODELS = ["qwen2.5:7b", "llama3.1:8b", "gemma2:9b", "mistral:7b", "qwen2.5:1.5b"]
POLICIES = {"mvt_adaptive": MVT_ADAPTIVE, "guarded_mvt": GUARDED,
            "guarded_mvt_dyn": GUARDED_DYN}
SIGMA = 1.2
HISTORY = 3          # (action -> outcome) steps shown to the model; 1 = last action only


def _cache(model, history=HISTORY):
    tag = "" if history == 1 else f"_h{history}"
    return os.path.join(OUT, f"cache_{model.replace(':', '_')}{tag}.json")


ENTRY_MODE = {"mode": "ticks"}      # set from --entry-mode; "productive" = ramp-aware R
SCRIPTED_TUNED: dict[str, dict] = {}  # regime -> timeout spec tuned for the scripted brain


def _episodes(spec, seeds, regime, factory, stats: DecisionStats | None):
    rows = []
    for s in seeds:
        r = tierb_episode(spec, SIGMA, regime, s, brain_factory=factory,
                          entry_mode=ENTRY_MODE["mode"])
        b = r.pop("_brain", None)
        if stats is not None and b is not None:
            stats.merge(b.stats)
        rows.append({"seed": s, **r})
    return pd.DataFrame(rows)


def evaluate_brain(name, factory, seeds, backend=None):
    per, dec = [], []
    for regime in ("diminishing", "constant"):
        t0 = time.time()
        calls0 = backend.calls if backend else 0
        lat0 = backend.latency_s if backend else 0.0
        label, spec, _ = tune(tierb_episode, families(GRID_FIXED_TB)["fixed"], seeds,
                              sigma=SIGMA, regime=regime, brain_factory=factory,
                              entry_mode=ENTRY_MODE["mode"])
        st = DecisionStats()
        runs = {"fixed*": spec, **POLICIES}
        if name == "scripted":
            SCRIPTED_TUNED[regime] = spec
        elif regime in SCRIPTED_TUNED:
            runs["fixed (scripted-tuned)"] = SCRIPTED_TUNED[regime]
        for pname, sp in runs.items():
            df = _episodes(sp, seeds, regime, factory, st)
            per += [{"brain": name, "regime": regime, "policy": pname,
                     "tuned": label if pname == "fixed*" else "", **x}
                    for x in df.to_dict("records")]
        if backend:
            backend.save()
        d = {"brain": name, "regime": regime, **st.as_dict()}
        if backend:
            n = backend.calls - calls0
            d.update({"model_calls": n, "unique_prompts_total": len(backend.cache),
                      "latency_ms": 1000 * (backend.latency_s - lat0) / max(n, 1)})
        dec.append(d)
        print(f"  {name} {regime}: tuned {label}, {time.time() - t0:.0f}s, {d}")
    return per, dec


def summarise(per: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (b, reg), g in per.groupby(["brain", "regime"], sort=False):
        ref = g[g.policy == "fixed*"].sort_values("seed")
        rt = censor(ref.ticks, ref.solved, TIERB_CAP)
        for pol, gp in g.groupby("policy", sort=False):
            gp = gp.sort_values("seed")
            pt = censor(gp.ticks, gp.solved, TIERB_CAP)
            sr, mt = score(gp)
            sp = lo = hi = np.nan
            if pol != "fixed*" and sr == 1.0 and ref.solved.all():
                sp, lo, hi = speedup_ci(rt, pt)
            rows.append({"brain": b, "regime": reg, "policy": pol,
                         "tuned": gp.tuned.iloc[0], "solve_rate": sr,
                         "ticks": mt if sr > 0 else np.nan, "speedup_pct": sp,
                         "ci_lo": lo, "ci_hi": hi,
                         "p_wilcoxon": wilcoxon_p(rt, pt) if pol != "fixed*" else np.nan})
    return pd.DataFrame(rows)


def replay_check(model, n_episodes=2):
    """Run full GuardedMVT episodes with the memo disabled and compare every reply with the
    memoised one, and the episode outcome with the memoised run."""
    memo = OllamaChat(model, cache_path=_cache(model))
    rows = []
    for regime in ("diminishing", "constant"):
        for s in range(n_episodes):
            live = OllamaChat(model, use_cache=False, record_prompts=True)
            fl = lambda: LLMAgentBrain(live, log_replies=True, history=HISTORY)  # noqa: E731
            r_live = tierb_episode(GUARDED, SIGMA, regime, s, brain_factory=fl)
            b = r_live.pop("_brain")
            # recompute each live prompt's memoised reply
            fm = lambda: LLMAgentBrain(memo, history=HISTORY)  # noqa: E731
            r_memo = tierb_episode(GUARDED, SIGMA, regime, s, brain_factory=fm)
            r_memo.pop("_brain")
            same = sum(int(memo.cache.get(memo._key(p)) == rep)
                       for p, rep in zip(live.prompts, b.stats.replies))
            rows.append({"model": model, "regime": regime, "seed": s,
                         "live_calls": live.calls, "replies_identical_pct":
                         100.0 * same / max(len(b.stats.replies), 1),
                         "ticks_live": r_live["ticks"], "ticks_memo": r_memo["ticks"],
                         "solved_live": r_live["solved"], "solved_memo": r_memo["solved"],
                         "latency_ms": 1000 * live.latency_s / max(live.calls, 1)})
            print("  replay", rows[-1])
    memo.save()
    return rows


def replay_sample(models, n=200, seed=0):
    """Memo check across all models: re-query n memoised prompts per model, drawn at random
    from the 3-step-history cache used by the paper, with the memo disabled, and compare each
    fresh reply with the memoised one (greedy decoding, temperature 0, fixed seed)."""
    import random
    rows = []
    for m in models:
        memo = OllamaChat(m, cache_path=_cache(m))
        keys = sorted(memo.cache)
        pick = random.Random(seed).sample(keys, min(n, len(keys)))
        same = 0
        live = OllamaChat(m, use_cache=False)
        for k in pick:
            _, system, prompt = k.split("\x1f", 2)
            live.system = system
            same += int(live(prompt) == memo.cache[k])
        rows.append({"model": m, "memo_entries": len(keys), "requeried": len(pick),
                     "identical": same, "identical_pct": 100.0 * same / max(len(pick), 1),
                     "latency_ms": 1000 * live.latency_s / max(live.calls, 1)})
        print("  replay-sample", rows[-1], flush=True)
    return rows


def history_ablation(models, seeds=range(5)):
    """Observation-history ablation: last action only (history 1) vs a 3-step trajectory,
    GuardedMVT, both regimes. With history 1 some models loop between no-progress actions."""
    rows = []
    for m in models:
        for h in (1, 3):
            be = OllamaChat(m, cache_path=_cache(m, h))
            for regime in ("diminishing", "constant"):
                st = DecisionStats()
                df = _episodes(GUARDED, seeds, regime,
                               lambda be=be, h=h: LLMAgentBrain(be, history=h), st)
                rows.append({"brain": m, "history": h, "regime": regime,
                             "solve_rate": float(df.solved.mean()),
                             "ticks": float(df.ticks[df.solved].mean()) if df.solved.any()
                             else np.nan, **st.as_dict()})
            be.save()
            print("  history", rows[-2:])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-driven Tier-B harness evaluation")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--replay-model", default="qwen2.5:7b")
    ap.add_argument("--replay-episodes", type=int, default=2)
    ap.add_argument("--skip-scripted", action="store_true")
    ap.add_argument("--entry-mode", choices=["ticks", "productive"], default="ticks")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--replay-sample", type=int, default=0,
                    help="only re-query N memoised prompts per model without the memo, then exit")
    a = ap.parse_args()
    if a.replay_sample:
        pd.DataFrame(replay_sample(a.models, a.replay_sample)).to_csv(
            os.path.join(OUT, "llm_replay_sample.csv"), index=False)
        return
    ENTRY_MODE["mode"] = a.entry_mode
    os.makedirs(OUT, exist_ok=True)
    global OUTP
    OUTP = a.tag
    seeds = list(range(a.seeds))
    per, dec = [], []
    if not a.skip_scripted:
        p, d = evaluate_brain("scripted", ScriptedBrain, seeds)
        per += p
        dec += d
    if os.path.exists(os.path.join("results/cr", "learned_guarded.json")):
        POLICIES["guarded_mvt_learned"] = load_learned()
    for m in a.models:
        backend = OllamaChat(m, cache_path=_cache(m))
        p, d = evaluate_brain(m, lambda be=backend: LLMAgentBrain(be, history=HISTORY),
                              seeds, backend)
        per += p
        dec += d
        pd.DataFrame(per).to_csv(os.path.join(OUT, f"llm_perseed{OUTP}.csv"), index=False)
        pd.DataFrame(dec).to_csv(os.path.join(OUT, f"llm_decisions{OUTP}.csv"), index=False)
    perdf = pd.DataFrame(per)
    perdf.to_csv(os.path.join(OUT, f"llm_perseed{OUTP}.csv"), index=False)
    pd.DataFrame(dec).to_csv(os.path.join(OUT, f"llm_decisions{OUTP}.csv"), index=False)
    summ = summarise(perdf)
    summ.to_csv(os.path.join(OUT, f"llm_summary{OUTP}.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summ.round(2).to_string(index=False))
    if not a.no_ablation:
        pd.DataFrame(history_ablation(a.models)).to_csv(
            os.path.join(OUT, f"llm_history_ablation{OUTP}.csv"), index=False)
    if a.replay_model and a.replay_episodes:
        pd.DataFrame(replay_check(a.replay_model, a.replay_episodes)).to_csv(
            os.path.join(OUT, f"llm_replay{OUTP}.csv"), index=False)


if __name__ == "__main__":
    main()
