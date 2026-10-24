"""Correctness test for the Numba CUDA kernel.

Skipped unless a CUDA GPU is present or the CUDA simulator is enabled
(`NUMBA_ENABLE_CUDASIM=1`), so the default test run stays fast and GPU-free. Run locally with:

    NUMBA_ENABLE_CUDASIM=1 python -m pytest tests/test_cuda_kernel.py -q
"""

from __future__ import annotations

import os
import warnings

import pytest

numba = pytest.importorskip("numba")
from numba import cuda

_CUDASIM = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
if not (_CUDASIM or cuda.is_available()):
    pytest.skip("no CUDA GPU and cudasim not enabled", allow_module_level=True)

from ffs.gpu.cuda_kernel import CudaConfig, simulate_cuda


def test_cuda_kernel_reproduces_hypothesis():
    fixed = {"type": "fixed_timeout", "timeout": 20.0}
    mvt = {"type": "mvt", "half_life": 8.0, "min_samples": 3, "hysteresis": 0.05}

    def speedup(sigma):
        cfg = CudaConfig(n_episodes=128, n_workers=8, sigma=sigma,
                         target_yield=120.0, max_ticks=400)
        with warnings.catch_warnings():         # cudasim RNG-init warnings are harmless
            warnings.simplefilter("ignore")
            rf = simulate_cuda(fixed, cfg, seed=1)
            rm = simulate_cuda(mvt, cfg, seed=1)
        assert rf["solve_rate"] > 0.95 and rm["solve_rate"] > 0.95
        return 100.0 * (rf["mean_tto"] - rm["mean_tto"]) / rf["mean_tto"]

    assert speedup(0.0) < 0        # fixed wins when homogeneous
    assert speedup(1.2) > 0        # MVT wins when heterogeneous
