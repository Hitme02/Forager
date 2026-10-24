"""Generate a *flat* NASim scenario with tunable branch heterogeneity.

NASim's built-in `generate` produces deeply segmented, multi-subnet topologies where only
one host is reachable at start and progress depends on pivoting through firewalls. That
dependency structure confounds a study of the *scheduler*: we want to test when a worker
should abandon one line of attack for another, not NASim's routing.

This builds a single-subnet network with every host directly reachable from the internet,
so the attack surface is a flat frontier of independent branches -- exactly the foraging
setting -- but realised with *real* NASim mechanics: probabilistic exploits (a Bernoulli
jump in access, i.e. a genuinely bursty progress signal), scans, access levels and action
costs. Heterogeneity is injected two ways, controlled by one knob `heterogeneity`:

- **Exploit difficulty.** Services span easy (high success prob) to hard (low prob). Richer
  hosts are biased toward easy services (quick foothold); poorer hosts carry only hard ones
  (many failed attempts -- flat progress -- before a foothold, if ever). This is what makes
  some branches worth staying on and others worth abandoning.
- **Value.** Host values are drawn log-normal with spread `heterogeneity`, so a few hosts
  are high-value and most are low, mirroring the simulator's richness distribution.

The result is loadable with `nasim.load` and wrapped by `NASimTarget`, so the same MVT
scheduler forages it unchanged.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np


def build_flat_scenario_yaml(n_hosts: int = 24, n_services: int = 6,
                             heterogeneity: float = 0.8, seed: int = 0,
                             prob_min: float = 0.12, prob_max: float = 0.95) -> str:
    """Return NASim scenario YAML text for a flat, fully-reachable heterogeneous network."""
    rng = np.random.default_rng(seed)
    svcs = [f"svc{i}" for i in range(n_services)]
    probs = np.linspace(prob_min, prob_max, n_services)   # svc0 hardest ... last easiest

    # Per-host richness weights (log-normal, mean 1): drive value and service difficulty.
    w = np.exp(heterogeneity * rng.standard_normal(n_hosts))
    w /= w.mean()

    L: list[str] = [f"subnets: [{n_hosts}]",
                    "topology: [[1, 1],",
                    "           [1, 1]]",
                    "sensitive_hosts:"]
    for h in range(n_hosts):
        L.append(f"  (1, {h}): {max(1, int(round(10 * w[h])))}")

    L += ["os:", "  - linux", "services:"] + [f"  - {s}" for s in svcs]
    L += ["processes:", "  - proc0", "exploits:"]
    for i, s in enumerate(svcs):
        L.append(f"  e_{s}:\n    service: {s}\n    os: linux\n"
                 f"    prob: {probs[i]:.2f}\n    cost: 1\n    access: user")
    L += ["privilege_escalation:",
          "  pe_proc0:\n    process: proc0\n    os: linux\n"
          "    prob: 0.99\n    cost: 1\n    access: root",
          "service_scan_cost: 1", "os_scan_cost: 1",
          "subnet_scan_cost: 1", "process_scan_cost: 1",
          "host_configurations:"]
    # Each host's *difficulty* (which single service it runs, hence its exploit success
    # probability) is drawn with spread set by `heterogeneity`: at 0 every host is medium
    # difficulty; as it grows the frontier splits into quick-foothold and stubborn hosts --
    # the spread that makes abandon-vs-persist matter. Difficulty is the foraging IV here,
    # decoupled from host value (which varies only to shape the recon prior).
    for h in range(n_hosts):
        z = rng.standard_normal()
        level = float(np.clip(0.5 + 0.38 * heterogeneity * z, 0.0, 1.0))  # 0 hard .. 1 easy
        idx = int(round(level * (n_services - 1)))
        L.append(f"  (1, {h}):\n    os: linux\n"
                 f"    services: [{svcs[idx]}]\n    processes: [proc0]")

    L += ["firewall:",
          f"  (0, 1): [{', '.join(svcs)}]",       # internet -> subnet: all services allowed
          "  (1, 0): []",
          "step_limit: 1000000"]
    return "\n".join(L) + "\n"


def write_flat_scenario(path: str | None = None, **kw) -> str:
    """Write a flat scenario YAML to `path` (or a temp file) and return the path."""
    text = build_flat_scenario_yaml(**kw)
    if path is None:
        fd, path = tempfile.mkstemp(prefix="ffs_flat_", suffix=".yaml")
        os.close(fd)
    with open(path, "w") as f:
        f.write(text)
    return path
