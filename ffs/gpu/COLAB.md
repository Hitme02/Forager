# Running the Numba CUDA kernel on Google Colab (free T4 GPU)

The hand-written CUDA kernel in `cuda_kernel.py` runs on any NVIDIA GPU. Colab's free tier
gives you one (usually a T4), so it's the easiest place to run it for real. It is validated
locally with Numba's CUDA simulator (`NUMBA_ENABLE_CUDASIM=1`), so the same code you test
without a GPU is what runs on the T4.

## 1. Enable the GPU
`Runtime → Change runtime type → Hardware accelerator: GPU → Save`. Verify:

```python
!nvidia-smi -L          # should list a GPU, e.g. "GPU 0: Tesla T4"
```

## 2. Dependencies
Colab ships a working `numba` + CUDA toolkit, so usually nothing is needed. Confirm Numba
sees the GPU:

```python
from numba import cuda
cuda.detect()           # should print the T4 and "1 CUDA devices"
```

If `cuda.detect()` errors (newer Numba split CUDA into its own package):

```python
!pip install -q numba-cuda          # then: Runtime → Restart runtime, and re-run
```

## 3. Get the kernel onto Colab
Either upload the repo, or (simplest) just upload `ffs/gpu/cuda_kernel.py` and run it
standalone — it only needs `numpy` and `numba`. If you uploaded the whole project:

```python
import sys; sys.path.insert(0, "/content/EL")      # path to the repo root
```

## 4. Run it

```python
from ffs.gpu.cuda_kernel import simulate_cuda, CudaConfig

fixed = {"type": "fixed_timeout", "timeout": 20.0}
mvt   = {"type": "mvt", "half_life": 8.0, "min_samples": 3, "hysteresis": 0.05}

for sigma in (0.0, 0.6, 1.2):
    cfg = CudaConfig(n_episodes=65536, n_workers=8, sigma=sigma, target_yield=120.0)
    rf = simulate_cuda(fixed, cfg, seed=0)
    rm = simulate_cuda(mvt,   cfg, seed=0)
    sp = 100 * (rf["mean_tto"] - rm["mean_tto"]) / rf["mean_tto"]
    print(f"sigma={sigma}: fixed={rf['mean_tto']:.1f}  mvt={rm['mean_tto']:.1f}  speedup={sp:+.1f}%")
```

Expected: MVT trails fixed at `sigma=0` and overtakes it by `sigma=1.2` (the central
hypothesis), matching the PyTorch and CPU implementations.

## 5. Benchmark the GPU (thread-per-episode throughput)

```python
import time
from numba import cuda
cfg = CudaConfig(n_episodes=1_048_576, n_workers=8, sigma=0.9, target_yield=120.0)
simulate_cuda(mvt, cfg, seed=0)        # warm up (JIT compile)
cuda.synchronize()
t0 = time.time(); simulate_cuda(mvt, cfg, seed=0); cuda.synchronize()
dt = time.time() - t0
print(f"{cfg.n_episodes:,} episodes in {dt:.2f}s = {cfg.n_episodes/dt:,.0f} episodes/s")
```

On a T4 this should do on the order of 10^6 episodes/second — each of the GPU's thousands of
threads runs one complete episode concurrently. Compare to ~600 episodes/s for the sequential
Python engine: that is the PAGP speed-up the kernel buys.

## Notes
- `n_workers` must be ≤ `MAXW` (32) — local arrays in the kernel are fixed-size.
- The kernel uses `xoroshiro128p` per-thread RNG (`numba.cuda.random`) for reproducible,
  independent streams across episodes.
- To compare the two GPU approaches: `cuda_kernel.py` is *thread-per-episode* (MIMD-ish,
  one thread = one episode); `batch_sim.py` (PyTorch) is *SIMD-across-episodes* (one tensor
  lane = one episode). Both parallelise the same Monte-Carlo and should agree.
```
