"""Paired statistics for the camera-ready analysis.

Every comparison in the paper is *paired*: the same seed hands every policy the same surface,
so the natural unit is the per-seed difference. This module provides

  * `speedup_ci`      -- percentage speedup of a policy over a reference (difference of mean
                         time-to-objective, relative to the reference mean) with a paired
                         percentile-bootstrap 95% confidence interval over seeds;
  * `wilcoxon_p`      -- two-sided Wilcoxon signed-rank test on the paired per-seed times;
  * `holm`            -- Holm-Bonferroni step-down adjustment over a family of p-values;
  * `win_rate`        -- fraction of seeds on which the policy is strictly faster.

Unsolved episodes are censored at the episode cap (`max_ticks`) before pairing, which is
conservative for a policy that fails (it is charged the full budget) and lets the rank test
use every seed. Solve rates are reported separately wherever a policy fails.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def censor(ticks, solved, cap: float) -> np.ndarray:
    t = np.asarray(ticks, dtype=float).copy()
    s = np.asarray(solved, dtype=bool)
    t[~s | ~np.isfinite(t)] = cap
    return t


def speedup_ci(ref, pol, n_boot: int = 10_000, alpha: float = 0.05,
               seed: int = 0) -> tuple[float, float, float]:
    """Return (speedup %, lo, hi): 100 * (mean(ref) - mean(pol)) / mean(ref), paired bootstrap."""
    ref = np.asarray(ref, dtype=float)
    pol = np.asarray(pol, dtype=float)
    assert ref.shape == pol.shape and ref.ndim == 1
    point = 100.0 * (ref.mean() - pol.mean()) / ref.mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ref), size=(n_boot, len(ref)))
    r, p = ref[idx].mean(axis=1), pol[idx].mean(axis=1)
    boot = 100.0 * (r - p) / r
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def wilcoxon_p(ref, pol) -> float:
    """Two-sided Wilcoxon signed-rank p-value on paired samples (1.0 if all differences are 0)."""
    d = np.asarray(ref, dtype=float) - np.asarray(pol, dtype=float)
    if np.allclose(d, 0.0):
        return 1.0
    return float(stats.wilcoxon(ref, pol, zero_method="wilcox",
                                alternative="two-sided").pvalue)


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def win_rate(ref, pol) -> float:
    ref = np.asarray(ref, dtype=float)
    pol = np.asarray(pol, dtype=float)
    return float(np.mean(pol < ref))
