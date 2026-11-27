"""Camera-ready experiments (ATIS 2026 reviews): statistics, baselines, sensitivity, thresholds.

Subcommands (all write per-seed data and summaries to results/cr/):

  abstract     abstract-testbed sweep with per-seed output, every baseline (tuned FixedTimeout,
               MaxRetry, Median stopping, ASHA; TopK, UCB, restless index; MVT, GuardedMVT
               static and dynamic), paired bootstrap CIs, Wilcoxon + Holm, held-out tuning
               check, and a fresh-seed replication.
  tierb        Tier-B mock-range sweeps (both regimes) and the cross-regime robustness table
               with minimax-tuned single-configuration baselines, with the same statistics.
  sensitivity  one-at-a-time GuardedMVT threshold sensitivity (Tier-B both regimes, abstract).
  learn        GuardedMVT thresholds learned by random search on training seeds, evaluated on
               disjoint test seeds.
  scale        nine-scenario robustness (regime x heterogeneity x target difficulty) of every
               single-configuration policy against a per-scenario tuned timeout.
  crossover    fine heterogeneity grid locating the MVT / tuned-timeout crossover.

Seeding is identical to `run.py` (abstract: seed s) and `run_tierb.py` (Tier-B: range seed s,
scheduler rng 1000 + s), so the submitted tables are reproduced exactly by the same configs.
Tuning follows the submitted protocol: each tuned baseline is tuned per scenario over a grid,
eligible only if it solves >= 95% of episodes, scored by mean time-to-objective of solved runs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

from ..realism.cyber_range import RangeConfig, make_mock_range
from ..realism.run_tierb import GUARDED, MVT_ADAPTIVE, MVT_SLOPE
from ..scheduler.engine import EngineConfig, Scheduler
from ..simulator.environment import EnvConfig
from ..simulator.heterogeneity import SurfaceConfig
from .run import DEFAULT_POLICIES, run_episode
from .stats import censor, holm, speedup_ci, wilcoxon_p, win_rate

OUT = "results/cr"
LEVELS = [0.0, 0.3, 0.6, 0.9, 1.2]

GUARDED_DYN = {**GUARDED, "budget_mode": "dynamic", "budget_kappa": 3.0}

# Tuning grids for the tuned baselines.
GRID_FIXED_ABS = [5, 8, 12, 16, 20, 25, 30, 40, 60, 80]
GRID_FIXED_TB = [8, 12, 16, 25, 40, 60, 80]
GRID_RETRY = [2, 3, 4, 6, 8, 12, 16, 24, 32]
GRID_GRACE = [2, 4, 8, 16, 32]
GRID_ASHA = [(r, e) for r in (2, 4, 8, 16) for e in (2, 3, 4)]


def families(fixed_grid):
    return {
        "fixed": [(f"t={t}", {"type": "fixed_timeout", "timeout": float(t)}) for t in fixed_grid],
        "max_retry": [(f"k={k}", {"type": "max_retry", "retry_cap": int(k)}) for k in GRID_RETRY],
        "median": [(f"g={g}", {"type": "median_stopping", "grace": int(g)}) for g in GRID_GRACE],
        "asha": [(f"r0={r},eta={e}", {"type": "asha", "r0": float(r), "eta": float(e)})
                 for r, e in GRID_ASHA],
    }


# --------------------------------------------------------------------------- #
# Episode runners
# --------------------------------------------------------------------------- #
def abstract_episode(spec, sigma, seed, *, noise="smooth", burst_p=0.2):
    surf = SurfaceConfig(n_branches=40, mean_A=10.0, heterogeneity=float(sigma), mean_tau=25.0,
                         tau_coupling=0.5, noise_model=noise, burst_p=burst_p)
    env = EnvConfig(switch_cost=5.0, objective="coverage", target_frac=0.5)
    r = run_episode(dict(spec), surf, env, EngineConfig(n_workers=8), int(seed))
    return {"solved": bool(r.solved), "ticks": int(r.ticks), "switches": int(r.n_switches)}


ABSTRACT_CAP = 100_000
TIERB_CAP = 12_000


def tierb_episode(spec, sigma, regime, seed, *, burst_p=1.0, brain_factory=None,
                  sandbox_kw=None, max_ticks=TIERB_CAP, entry_mode="ticks"):
    from ..scheduler.policies import make_policy
    cfg = RangeConfig(switch_cost=5.0, max_ticks=max_ticks, objective="coverage",
                      target_frac=0.5)
    brain = brain_factory() if brain_factory else None
    rng = make_mock_range(cfg, n_services=40, flag_services=2, sigma=float(sigma),
                          regime=regime, burst_p=burst_p, brain=brain, seed=int(seed),
                          **(sandbox_kw or {}))
    sch = Scheduler(rng, make_policy(dict(spec)),
                    EngineConfig(n_workers=8, entry_window=5, entry_mode=entry_mode),
                    np.random.default_rng(1000 + int(seed)))
    r = sch.run()
    out = {"solved": bool(r.solved), "ticks": int(r.ticks), "switches": int(r.n_switches)}
    if brain is not None and hasattr(brain, "stats"):
        out["_brain"] = brain
    return out


def run_many(episode_fn, spec, seeds, **kw) -> pd.DataFrame:
    rows = []
    for s in seeds:
        r = episode_fn(spec, seed=s, **kw)
        r.pop("_brain", None)
        rows.append({"seed": int(s), **r})
    return pd.DataFrame(rows)


def score(df: pd.DataFrame) -> tuple[float, float]:
    sr = float(df["solved"].mean())
    t = df.loc[df["solved"], "ticks"]
    return sr, (float(t.mean()) if len(t) else np.inf)


def tune(episode_fn, grid, seeds, **kw):
    """Best (label, spec, mean_ticks) over the grid by the submitted tuning protocol.

    If no grid point reaches the 95% solve-rate floor (possible only with an unreliable LLM
    brain), the point with the lowest censored mean time is returned instead."""
    best = (None, None, np.inf)
    fallback = (None, None, np.inf)
    for label, spec in grid:
        df = run_many(episode_fn, spec, seeds, **kw)
        sr, mt = score(df)
        if sr >= 0.95 and mt < best[2] - 1e-9:
            best = (label, spec, mt)
        cm = float(censor(df["ticks"], df["solved"], kw.get("max_ticks", TIERB_CAP)).mean())
        if cm < fallback[2]:
            fallback = (label + " (<95% solved)", spec, cm)
    return best if best[0] is not None else fallback


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
def compare(per_seed: pd.DataFrame, group_cols, ref: str, cap: float) -> pd.DataFrame:
    """Paired comparison of every policy against `ref` within each group."""
    rows = []
    for key, g in per_seed.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        piv_t = g.pivot(index="seed", columns="policy", values="ticks")
        piv_s = g.pivot(index="seed", columns="policy", values="solved")
        if ref not in piv_t:
            continue
        rt = censor(piv_t[ref], piv_s[ref], cap)
        for pol in piv_t.columns:
            if pol == ref:
                continue
            pt = censor(piv_t[pol], piv_s[pol], cap)
            sr = float(piv_s[pol].mean())
            solved_mask = piv_s[pol].astype(bool).values
            mean_solved = float(piv_t[pol][solved_mask].mean()) if solved_mask.any() else np.nan
            sd_solved = (float(piv_t[pol][solved_mask].std(ddof=1))
                         if solved_mask.sum() > 1 else np.nan)
            if sr == 1.0 and float(piv_s[ref].mean()) == 1.0:
                sp, lo, hi = speedup_ci(rt, pt)
            else:
                sp = lo = hi = np.nan
            rows.append({**dict(zip(group_cols, key)), "policy": pol, "ref": ref,
                         "solve_rate": sr, "ticks_mean": mean_solved, "ticks_sd": sd_solved,
                         "ref_ticks_mean": float(rt.mean()), "speedup_pct": sp,
                         "ci_lo": lo, "ci_hi": hi, "win_rate": win_rate(rt, pt),
                         "p_wilcoxon": wilcoxon_p(rt, pt), "n": len(rt)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_wilcoxon"].values)
    return out


def _save(df: pd.DataFrame, name: str) -> str:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False)
    print(f"saved {path} ({len(df)} rows)")
    return path


# --------------------------------------------------------------------------- #
# 1. Abstract testbed
# --------------------------------------------------------------------------- #
ABSTRACT_FIXED_POLICIES = {
    "mvt_slope": DEFAULT_POLICIES["mvt_slope"],
    "mvt_bayes": DEFAULT_POLICIES["mvt_bayes"],
    "guarded_mvt": GUARDED,
    "guarded_mvt_dyn": GUARDED_DYN,
    "top_k": DEFAULT_POLICIES["top_k"],
    "ucb": DEFAULT_POLICIES["ucb"],
    "restless_index": DEFAULT_POLICIES["restless_index"],
}


def cmd_abstract(n_seeds=30, fresh=100):
    seeds = list(range(n_seeds))
    tune_seeds = list(range(1000, 1000 + n_seeds))
    fam = families(GRID_FIXED_ABS)
    rows, tuned = [], []
    t0 = time.time()
    for sigma in LEVELS:
        kw = dict(sigma=sigma)
        for fname, grid in fam.items():
            label, spec, _ = tune(abstract_episode, grid, seeds, **kw)
            tuned.append({"sigma": sigma, "family": fname, "tuned_on": "eval", "best": label})
            df = run_many(abstract_episode, spec, seeds, **kw)
            rows += [{"sigma": sigma, "policy": f"{fname}*", **r} for r in df.to_dict("records")]
        # held-out tuning of the headline baseline: tuned on disjoint seeds, scored on `seeds`
        label, spec, _ = tune(abstract_episode, fam["fixed"], tune_seeds, **kw)
        tuned.append({"sigma": sigma, "family": "fixed", "tuned_on": "heldout", "best": label})
        df = run_many(abstract_episode, spec, seeds, **kw)
        rows += [{"sigma": sigma, "policy": "fixed_heldout*", **r} for r in df.to_dict("records")]
        for pname, spec in ABSTRACT_FIXED_POLICIES.items():
            df = run_many(abstract_episode, spec, seeds, **kw)
            rows += [{"sigma": sigma, "policy": pname, **r} for r in df.to_dict("records")]
        print(f"  sigma={sigma} done ({time.time() - t0:.0f}s)")
    per = pd.DataFrame(rows)
    _save(per, "abstract_perseed.csv")
    _save(pd.DataFrame(tuned), "abstract_tuned.csv")
    _save(compare(per, ["sigma"], "fixed*", ABSTRACT_CAP), "abstract_stats.csv")

    # Fresh-seed replication of the headline comparisons (tuned on the original seeds).
    if fresh:
        fseeds = list(range(2000, 2000 + fresh))
        frows = []
        for sigma in LEVELS:
            label = [t["best"] for t in tuned if t["sigma"] == sigma and t["family"] == "fixed"
                     and t["tuned_on"] == "eval"][0]
            spec = dict(fam["fixed"])[label]
            for pname, sp in (("fixed*", spec), ("mvt_slope", DEFAULT_POLICIES["mvt_slope"]),
                              ("guarded_mvt", GUARDED), ("guarded_mvt_dyn", GUARDED_DYN)):
                df = run_many(abstract_episode, sp, fseeds, sigma=sigma)
                frows += [{"sigma": sigma, "policy": pname, **r} for r in df.to_dict("records")]
        fper = pd.DataFrame(frows)
        _save(fper, "abstract_fresh_perseed.csv")
        _save(compare(fper, ["sigma"], "fixed*", ABSTRACT_CAP), "abstract_fresh_stats.csv")


# --------------------------------------------------------------------------- #
# 2. Tier-B mock range
# --------------------------------------------------------------------------- #
TIERB_FIXED_POLICIES = {"mvt_adaptive": MVT_ADAPTIVE, "mvt_slope": MVT_SLOPE,
                        "guarded_mvt": GUARDED, "guarded_mvt_dyn": GUARDED_DYN}


def cmd_tierb(n_seeds=20, extra_policies: dict | None = None):
    seeds = list(range(n_seeds))
    fam = families(GRID_FIXED_TB)
    pols = {**TIERB_FIXED_POLICIES, **(extra_policies or {})}
    rows, tuned = [], []
    t0 = time.time()
    for regime in ("diminishing", "constant"):
        for sigma in LEVELS:
            kw = dict(sigma=sigma, regime=regime)
            label, spec, _ = tune(tierb_episode, fam["fixed"], seeds, **kw)
            tuned.append({"regime": regime, "sigma": sigma, "family": "fixed", "best": label})
            df = run_many(tierb_episode, spec, seeds, **kw)
            rows += [{"regime": regime, "sigma": sigma, "policy": "fixed*", **r}
                     for r in df.to_dict("records")]
            for pname, sp in pols.items():
                df = run_many(tierb_episode, sp, seeds, **kw)
                rows += [{"regime": regime, "sigma": sigma, "policy": pname, **r}
                         for r in df.to_dict("records")]
        print(f"  {regime} sweep done ({time.time() - t0:.0f}s)")
    per = pd.DataFrame(rows)
    _save(per, "tierb_perseed.csv")
    _save(pd.DataFrame(tuned), "tierb_tuned.csv")
    _save(compare(per, ["regime", "sigma"], "fixed*", TIERB_CAP), "tierb_stats.csv")
    robustness(seeds, pols)


def robustness(seeds, pols, sigma=1.2, brain_factory=None, tag=""):
    """Cross-regime table: per-regime tuned timeouts, minimax-tuned single configurations of
    every tuned baseline family, and the fixed single-configuration policies."""
    fam = families(GRID_FIXED_TB)
    regimes = ("diminishing", "constant")
    kw = dict(sigma=sigma, brain_factory=brain_factory)
    # Evaluate every grid point of every family in both regimes once.
    grid_res = {}
    for fname, grid in fam.items():
        for label, spec in grid:
            for reg in regimes:
                grid_res[(fname, label, reg)] = run_many(tierb_episode, spec, seeds,
                                                         regime=reg, **kw)
    best_fixed = {}
    for reg in regimes:
        cands = [(score(grid_res[("fixed", l, reg)]), l) for l, _ in fam["fixed"]]
        cands = [(m, l) for (sr, m), l in cands if sr >= 0.95]
        best_fixed[reg] = min(cands)
    rows = []

    def add(name, dfs):
        rec = {"policy": name, "sigma": sigma}
        for reg in regimes:
            sr, mt = score(dfs[reg])
            rec[f"{reg}_solve_rate"] = sr
            rec[f"{reg}_ticks"] = mt if sr > 0 else np.nan
            rec[f"{reg}_ratio"] = (mt / best_fixed[reg][0]) if sr >= 0.95 else np.inf
        rec["worst_ratio"] = max(rec["diminishing_ratio"], rec["constant_ratio"])
        rows.append(rec)
        return rec

    per = []
    for reg in regimes:
        lbl = best_fixed[reg][1]
        add(f"fixed (tuned {reg}, {lbl})",
            {r: grid_res[("fixed", lbl, r)] for r in regimes})
    for fname, grid in fam.items():
        best = None
        for label, _ in grid:
            ratios = []
            for reg in regimes:
                sr, mt = score(grid_res[(fname, label, reg)])
                ratios.append(mt / best_fixed[reg][0] if sr >= 0.95 else np.inf)
            if best is None or max(ratios) < best[0]:
                best = (max(ratios), label)
        add(f"{fname} (minimax, {best[1]})",
            {r: grid_res[(fname, best[1], r)] for r in regimes})
        for reg in regimes:
            per += [{"regime": reg, "policy": f"{fname}_minimax", **x}
                    for x in grid_res[(fname, best[1], reg)].to_dict("records")]
    for reg in regimes:
        lbl = best_fixed[reg][1]
        per += [{"regime": reg, "policy": "fixed_tuned_here", **x}
                for x in grid_res[("fixed", lbl, reg)].to_dict("records")]
    for pname, sp in pols.items():
        dfs = {reg: run_many(tierb_episode, sp, seeds, regime=reg, **kw) for reg in regimes}
        add(pname, dfs)
        for reg in regimes:
            per += [{"regime": reg, "policy": pname, **x} for x in dfs[reg].to_dict("records")]
    tab = pd.DataFrame(rows)
    _save(tab, f"robustness{tag}.csv")
    perdf = pd.DataFrame(per)
    _save(perdf, f"robustness{tag}_perseed.csv")
    _save(compare(perdf, ["regime"], "fixed_tuned_here", TIERB_CAP), f"robustness{tag}_stats.csv")
    print(tab.to_string(index=False))
    return tab


# --------------------------------------------------------------------------- #
# 3. Sensitivity of GuardedMVT thresholds
# --------------------------------------------------------------------------- #
SENS = {
    "decline_frac": [0.3, 0.4, 0.5, 0.6, 0.7],
    "peak_floor_frac": [0.01, 0.025, 0.05, 0.1, 0.2],
    "budget_cap": [8.0, 12.0, 16.0, 24.0, 32.0],
    "min_samples": [3, 4, 6, 8, 12],
    "hysteresis": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
}


def cmd_sensitivity(n_seeds=20, abs_seeds=30):
    rows = []
    for param, values in SENS.items():
        for v in values:
            spec = {**GUARDED, param: v}
            for reg in ("diminishing", "constant"):
                sr, mt = score(run_many(tierb_episode, spec, range(n_seeds), sigma=1.2, regime=reg))
                rows.append({"backend": "tierb", "regime": reg, "param": param, "value": v,
                             "solve_rate": sr, "ticks": mt})
            sr, mt = score(run_many(abstract_episode, spec, range(abs_seeds), sigma=0.9))
            rows.append({"backend": "abstract", "regime": "diminishing", "param": param,
                         "value": v, "solve_rate": sr, "ticks": mt})
        print(f"  {param} done")
    df = pd.DataFrame(rows)
    _save(df, "sensitivity.csv")
    # Summary: per (backend, regime, param) range of time relative to the default value.
    summ = []
    for (b, reg, p), g in df.groupby(["backend", "regime", "param"]):
        base = g.loc[np.isclose(g["value"].astype(float), float(GUARDED[p])), "ticks"].iloc[0]
        rel = 100.0 * (g["ticks"] / base - 1.0)
        summ.append({"backend": b, "regime": reg, "param": p, "min_rel_pct": rel.min(),
                     "max_rel_pct": rel.max(), "min_solve": g["solve_rate"].min()})
    _save(pd.DataFrame(summ), "sensitivity_summary.csv")


# --------------------------------------------------------------------------- #
# 4. Learned thresholds (random search, train/test split)
# --------------------------------------------------------------------------- #
TRAIN_SEEDS = list(range(100, 120))
TEST_SEEDS = list(range(20))


def _sample_cfg(rng):
    return {**GUARDED,
            "decline_frac": float(rng.uniform(0.2, 0.8)),
            "peak_floor_frac": float(np.exp(rng.uniform(np.log(0.01), np.log(0.3)))),
            "budget_cap": float(rng.integers(4, 41)),
            "min_samples": int(rng.integers(2, 13)),
            "hysteresis": float(rng.uniform(0.0, 0.6))}


def learn_thresholds(n_samples=300, train_sigmas=(0.6, 1.2), seed=0, sandbox_kw=None):
    """Random search minimising the worst-case (over regimes and training sigmas) ratio of
    GuardedMVT's time to a timeout tuned for that exact scenario, on training seeds only."""
    scen = [(reg, s) for reg in ("diminishing", "constant") for s in train_sigmas]
    ref = {}
    for reg, s in scen:
        _, _, m = tune(tierb_episode, families(GRID_FIXED_TB)["fixed"], TRAIN_SEEDS,
                       sigma=s, regime=reg, sandbox_kw=sandbox_kw)
        ref[(reg, s)] = m
    rng = np.random.default_rng(seed)
    cands = [GUARDED] + [_sample_cfg(rng) for _ in range(n_samples)]
    hist = []
    for i, cfg in enumerate(cands):
        worst = 0.0
        for reg, s in scen:
            sr, mt = score(run_many(tierb_episode, cfg, TRAIN_SEEDS, sigma=s, regime=reg,
                                    sandbox_kw=sandbox_kw))
            worst = max(worst, mt / ref[(reg, s)] if sr >= 0.95 else np.inf)
            if not np.isfinite(worst):
                break
        hist.append({"i": i, "worst_ratio": worst,
                     **{k: cfg[k] for k in ("decline_frac", "peak_floor_frac", "budget_cap",
                                            "min_samples", "hysteresis")}})
    h = pd.DataFrame(hist)
    best_i = int(h.loc[h["worst_ratio"].idxmin(), "i"])
    return cands[best_i], h


def cmd_learn(n_samples=300):
    t0 = time.time()
    best, hist = learn_thresholds(n_samples)
    _save(hist, "learn_history.csv")
    with open(os.path.join(OUT, "learned_guarded.json"), "w") as f:
        json.dump(best, f, indent=1)
    print("learned:", {k: best[k] for k in ("decline_frac", "peak_floor_frac", "budget_cap",
                                            "min_samples", "hysteresis")},
          f"({time.time() - t0:.0f}s)")


def load_learned():
    with open(os.path.join(OUT, "learned_guarded.json")) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# 5. Difficulty-scale transfer of the stall budget
# --------------------------------------------------------------------------- #
SCENARIOS = [  # (regime, sigma, sandbox_kw, tag)
    ("diminishing", 1.2, {}, "dim s1.2"),
    ("diminishing", 1.2, {"mean_tau": 12.5}, "dim s1.2 tau/2"),
    ("diminishing", 1.2, {"mean_tau": 50.0}, "dim s1.2 tau*2"),
    ("constant", 0.3, {}, "con s0.3"),
    ("constant", 0.6, {}, "con s0.6"),
    ("constant", 1.2, {}, "con s1.2"),
    ("constant", 1.2, {"mean_access_cost": 5.0}, "con s1.2 cost/2"),
    ("constant", 1.2, {"mean_access_cost": 20.0}, "con s1.2 cost*2"),
    ("constant", 1.2, {"mean_access_cost": 40.0}, "con s1.2 cost*4"),
]


def single_configs():
    """Every single-configuration policy, with the minimax-tuned baseline settings found in
    the robustness study (results/cr/robustness.csv) and the per-regime tuned timeouts."""
    cfg = {
        "fixed t=8 (tuned constant)": {"type": "fixed_timeout", "timeout": 8.0},
        "fixed t=60 (tuned diminishing)": {"type": "fixed_timeout", "timeout": 60.0},
        "fixed t=12 (minimax)": {"type": "fixed_timeout", "timeout": 12.0},
        "max_retry k=8 (minimax)": {"type": "max_retry", "retry_cap": 8},
        "median g=2 (minimax)": {"type": "median_stopping", "grace": 2},
        "asha r0=2,eta=2 (minimax)": {"type": "asha", "r0": 2.0, "eta": 2.0},
        "mvt_adaptive": MVT_ADAPTIVE, "mvt_slope": MVT_SLOPE,
        "guarded_mvt": GUARDED, "guarded_mvt_dyn": GUARDED_DYN,
    }
    if os.path.exists(os.path.join(OUT, "learned_guarded.json")):
        cfg["guarded_mvt_learned"] = load_learned()
    return cfg


def cmd_scale(n_seeds=20):
    """Nine-scenario robustness: regime x heterogeneity x target difficulty (payout time for
    diminishing value, access hazard for constant hazard). Each single configuration is
    compared with a timeout tuned for that exact scenario (an oracle reference)."""
    seeds = list(range(n_seeds))
    grid = families(GRID_FIXED_TB + [120, 160, 240])["fixed"]
    rows = []
    for reg, sigma, skw, tag in SCENARIOS:
        kw = dict(sigma=sigma, regime=reg, sandbox_kw=skw)
        label, spec, _ = tune(tierb_episode, grid, seeds, **kw)
        for pname, sp in {"fixed (tuned per scenario)": spec, **single_configs()}.items():
            df = run_many(tierb_episode, sp, seeds, **kw)
            rows += [{"scenario": tag, "regime": reg, "policy": pname,
                      "tuned_label": label if pname.startswith("fixed (tuned per") else "",
                      **r} for r in df.to_dict("records")]
        print(f"  {tag} done (tuned {label})")
    per = pd.DataFrame(rows)
    _save(per, "scale_perseed.csv")
    summ = []
    for (tag, pol), g in per.groupby(["scenario", "policy"], sort=False):
        sr, mt = score(g)
        summ.append({"scenario": tag, "policy": pol, "solve_rate": sr,
                     "ticks": mt if sr > 0 else np.nan})
    s = pd.DataFrame(summ)
    ref = s[s.policy == "fixed (tuned per scenario)"].set_index("scenario")["ticks"]
    s["ratio"] = [t / ref[sc] if sr == 1.0 else np.inf
                  for sc, t, sr in zip(s.scenario, s.ticks, s.solve_rate)]
    _save(s, "scale_summary.csv")
    agg = s.groupby("policy", sort=False).agg(
        solved_all=("solve_rate", lambda x: int((x == 1.0).sum())),
        min_solve=("solve_rate", "min"), worst_ratio=("ratio", "max"),
        geo_ratio=("ratio", lambda x: float(np.exp(np.mean(np.log(x[np.isfinite(x)]))))))
    _save(agg.reset_index(), "scale_aggregate.csv")
    print(s.pivot_table(index="policy", columns="scenario", values="ticks", sort=False)
          .round(1).to_string())
    print(agg.round(3).to_string())


def cmd_crossover(n_seeds=30, levels=(0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)):
    """Fine heterogeneity grid around the crossover (abstract: MVT recent-slope; Tier-B
    diminishing: GuardedMVT), each against a timeout tuned per level."""
    rows = []
    for sigma in levels:
        _, spec, _ = tune(abstract_episode, families(GRID_FIXED_ABS)["fixed"],
                          range(n_seeds), sigma=sigma)
        ref = run_many(abstract_episode, spec, range(n_seeds), sigma=sigma)
        pol = run_many(abstract_episode, DEFAULT_POLICIES["mvt_slope"], range(n_seeds),
                       sigma=sigma)
        sp, lo, hi = speedup_ci(ref.ticks.values, pol.ticks.values)
        rows.append({"backend": "abstract", "policy": "mvt_slope", "sigma": sigma,
                     "speedup_pct": sp, "ci_lo": lo, "ci_hi": hi})
        _, spec, _ = tune(tierb_episode, families(GRID_FIXED_TB)["fixed"], range(20),
                          sigma=sigma, regime="diminishing")
        ref = run_many(tierb_episode, spec, range(20), sigma=sigma, regime="diminishing")
        pol = run_many(tierb_episode, GUARDED, range(20), sigma=sigma, regime="diminishing")
        sp, lo, hi = speedup_ci(ref.ticks.values, pol.ticks.values)
        rows.append({"backend": "tierb", "policy": "guarded_mvt", "sigma": sigma,
                     "speedup_pct": sp, "ci_lo": lo, "ci_hi": hi})
    df = pd.DataFrame(rows)
    _save(df, "crossover.csv")
    print(df.round(2).to_string(index=False))


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Camera-ready experiments")
    ap.add_argument("cmd", choices=["abstract", "tierb", "sensitivity", "learn", "scale",
                                    "crossover", "all"])
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--samples", type=int, default=300)
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    if a.cmd in ("abstract", "all"):
        cmd_abstract(a.seeds or 30)
    if a.cmd in ("learn", "all"):
        cmd_learn(a.samples)
    if a.cmd in ("tierb", "all"):
        extra = {}
        if os.path.exists(os.path.join(OUT, "learned_guarded.json")):
            extra["guarded_mvt_learned"] = load_learned()
        cmd_tierb(a.seeds or 20, extra)
    if a.cmd in ("sensitivity", "all"):
        cmd_sensitivity(a.seeds or 20)
    if a.cmd in ("scale", "all"):
        cmd_scale(a.seeds or 20)
    if a.cmd in ("crossover", "all"):
        cmd_crossover()


if __name__ == "__main__":
    main()
