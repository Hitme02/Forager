"""Generate a *segmented* NASim scenario where each SUBNET is a foraging branch.

Why segments, not hosts. A single NASim host is compromised by one exploit whose per-attempt
success probability is constant -- a constant-hazard process, not diminishing returns. MVT's
premise (marginal intake falls as you work a patch) does not hold for a single such host, so
MVT has no edge there. A realistic attack *branch* is not one exploit; it is a line of
effort over a segment of the network. Working a segment, an operator harvests its
easy/high-value hosts first (large early gains) and is left with its hard hosts (small later
gains) -- genuine diminishing returns. That is the unit MVT is about: abandon a segment when
its marginal yield drops below what a fresh segment offers.

This builds a network of `n_subnets` segments, every one directly reachable from the internet
(so routing is not a confound), each holding `hosts_per_subnet` hosts. Segment *richness* is
drawn with spread `heterogeneity`: a rich segment has many easy, valuable hosts; a poor one
has few, hard hosts. Within a segment, host difficulty varies so that round-robin effort
yields the diminishing-returns curve. Loadable with `nasim.load`.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np


def build_segmented_scenario_yaml(n_subnets: int = 10, hosts_per_subnet: int = 5,
                                  n_services: int = 6, heterogeneity: float = 0.8,
                                  seed: int = 0, prob_min: float = 0.12,
                                  prob_max: float = 0.95) -> str:
    rng = np.random.default_rng(seed)
    svcs = [f"svc{i}" for i in range(n_services)]

    # Per-segment richness (log-normal, mean 1): sets how easy/valuable its hosts are.
    r = np.exp(heterogeneity * rng.standard_normal(n_subnets))
    r /= r.mean()

    L: list[str] = [f"subnets: [{', '.join([str(hosts_per_subnet)] * n_subnets)}]"]

    # Topology: internet (index 0) connects to every subnet; subnets self-connect.
    n = n_subnets + 1
    rows = []
    for i in range(n):
        row = [0] * n
        row[i] = 1
        if i == 0:
            for j in range(1, n):
                row[j] = 1
        else:
            row[0] = 1
        rows.append("[" + ", ".join(str(x) for x in row) + "]")
    L.append("topology: [" + ",\n           ".join(rows) + "]")

    # Sensitive hosts: value per host scales with its segment richness (shapes recon prior).
    L.append("sensitive_hosts:")
    host_specs = {}     # (subnet, host) -> service index (difficulty)
    for s in range(n_subnets):
        for h in range(hosts_per_subnet):
            # rich segment -> hosts biased easy (high service index); poor -> hard
            bias = r[s] / (r[s] + 1.0)
            z = rng.standard_normal()
            level = float(np.clip(0.35 + 0.5 * bias + 0.2 * z, 0.0, 1.0))
            idx = int(round(level * (n_services - 1)))
            host_specs[(s + 1, h)] = idx
            val = max(1, int(round(8 * r[s] * (0.5 + rng.random()))))
            L.append(f"  ({s + 1}, {h}): {val}")

    L += ["os:", "  - linux", "services:"] + [f"  - {s}" for s in svcs]
    L += ["processes:", "  - proc0", "exploits:"]
    probs = np.linspace(prob_min, prob_max, n_services)
    for i, s in enumerate(svcs):
        L.append(f"  e_{s}:\n    service: {s}\n    os: linux\n"
                 f"    prob: {probs[i]:.2f}\n    cost: 1\n    access: user")
    L += ["privilege_escalation:",
          "  pe_proc0:\n    process: proc0\n    os: linux\n"
          "    prob: 0.99\n    cost: 1\n    access: root",
          "service_scan_cost: 1", "os_scan_cost: 1",
          "subnet_scan_cost: 1", "process_scan_cost: 1",
          "host_configurations:"]
    for (sub, h), idx in host_specs.items():
        L.append(f"  ({sub}, {h}):\n    os: linux\n"
                 f"    services: [{svcs[idx]}]\n    processes: [proc0]")

    L.append("firewall:")
    all_svcs = ", ".join(svcs)
    for s in range(1, n_subnets + 1):
        L.append(f"  (0, {s}): [{all_svcs}]")
        L.append(f"  ({s}, 0): []")
    L.append("step_limit: 1000000")
    return "\n".join(L) + "\n"


def write_segmented_scenario(path: str | None = None, **kw) -> str:
    text = build_segmented_scenario_yaml(**kw)
    if path is None:
        fd, path = tempfile.mkstemp(prefix="ffs_seg_", suffix=".yaml")
        os.close(fd)
    with open(path, "w") as f:
        f.write(text)
    return path
