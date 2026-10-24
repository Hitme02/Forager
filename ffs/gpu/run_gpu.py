"""GPU-accelerated heterogeneity sweep + throughput benchmark (PAGP Units III & V).

Runs the whole MVT-vs-fixed heterogeneity sweep as GPU-parallel batch Monte-Carlo, so what
took the sequential Python engine minutes finishes in a second or two, and reports the
speed-up. The fixed timeout is tuned per heterogeneity level (fairness), exactly as in the
CPU experiments; because each candidate is itself a full batch of episodes, tuning is nearly
free here.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from .batch_sim import BatchConfig, pick_device, simulate

TIMEOUTS = [8, 12, 16, 20, 25, 30, 40, 60]


def _mean_tto(policy, cfg, device, seed):
    return simulate(policy, cfg, device=device, seed=seed)


def sweep(levels, n_episodes, n_workers, device, noise="smooth", burst_p=0.2,
          target_yield=120.0, switch_cost=5.0, seed=0) -> pd.DataFrame:
    rows = []
    for sigma in levels:
        cfg = BatchConfig(n_episodes=n_episodes, n_workers=n_workers, sigma=float(sigma),
                          switch_cost=switch_cost, target_yield=target_yield,
                          noise=noise, burst_p=burst_p)
        # Tune the fixed timeout for this level (each candidate is one batched run).
        best_t, best_m = None, np.inf
        for t in TIMEOUTS:
            r = simulate({"type": "fixed_timeout", "timeout": float(t)}, cfg, device=device, seed=seed)
            if r["solve_rate"] >= 0.99 and r["mean_tto"] < best_m:
                best_m, best_t = r["mean_tto"], t
        rm = simulate({"type": "mvt", "half_life": 8.0, "min_samples": 3, "hysteresis": 0.05},
                      cfg, device=device, seed=seed)
        speedup = 100.0 * (best_m - rm["mean_tto"]) / best_m if np.isfinite(best_m) else np.nan
        rows.append({"heterogeneity": float(sigma), "tuned_timeout": best_t,
                     "fixed_ticks": best_m, "mvt_ticks": rm["mean_tto"],
                     "mvt_solve_rate": rm["solve_rate"], "mvt_speedup_pct": speedup})
    return pd.DataFrame(rows)


def benchmark(device, n_workers=8) -> None:
    print("\n--- throughput (episodes/s) ---")
    mvt = {"type": "mvt", "half_life": 8.0, "min_samples": 3, "hysteresis": 0.05}
    for B in (4096, 65536, 262144):
        cfg = BatchConfig(n_episodes=B, n_workers=n_workers, sigma=0.9, target_yield=120.0)
        if str(device) == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        simulate(mvt, cfg, device=device, seed=0)
        if str(device) == "mps":
            torch.mps.synchronize()
        elif str(device) == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"  {str(device):4s} B={B:>7}: {dt:5.2f}s  {B / dt:>11,.0f} eps/s")


def main() -> None:
    ap = argparse.ArgumentParser(description="GPU foraging-scheduler sweep")
    ap.add_argument("--episodes", type=int, default=16384)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--noise", choices=["smooth", "bursty"], default="smooth")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.9, 1.2])
    ap.add_argument("--out", default="results/gpu_sweep.csv")
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}  |  {args.episodes} episodes/level")
    t0 = time.time()
    df = sweep(args.levels, args.episodes, args.workers, device, noise=args.noise)
    dt = time.time() - t0
    df.to_csv(args.out, index=False)
    print(f"\nSwept {len(args.levels)} levels x ({len(TIMEOUTS)} tuning + 1 MVT) x "
          f"{args.episodes} episodes in {dt:.2f}s")
    print(df.to_string(index=False))
    if args.benchmark:
        benchmark(device, args.workers)


if __name__ == "__main__":
    main()
