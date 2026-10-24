"""Per-branch agent 'brains': given a target host, choose the next action on it.

A brain encapsulates the *local tactic* a worker uses while it is committed to one branch
(one target host). It does NOT decide when to give up -- that is the scheduler's job (the
MVT rule). This separation is the whole point: the brain keeps trying to make progress on
its host, and the scheduler decides, across the parallel frontier, when a worker should
abandon a stalling host and migrate to a better one.

`ScriptedBrain` is a fixed recon->exploit->escalate tactic, used to validate the full
pipeline cheaply (no API cost). An `LLMBrain` with the same interface will later let a
language model pick the next action from the observation; nothing else in the stack changes.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Brain(Protocol):
    """Chooses the next flat NASim action index targeting `host_addr`, or None if the
    branch is exhausted (nothing useful left to try on this host)."""

    def next_action(self, target: "object", branch: "object",
                    rng: np.random.Generator) -> int | None:
        ...


class ScriptedBrain:
    """A deterministic recon -> exploit (-> escalate) tactic for one host.

    Order of operations on the assigned host:
      1. service_scan  (enumerate services)      -- once
      2. os_scan       (fingerprint OS)          -- once
      3. while not user-access: cycle the host's exploit actions (retry; never self-abandon)
      4. if `stop_at_foothold` (default): once access is gained the host is harvested ->
         return None.  Otherwise continue: process_scan, then cycle PE actions to root.

    `stop_at_foothold=True` encodes a *breadth* objective -- gaining access to many hosts --
    which is the natural coverage goal and gives a front-loaded, diminishing per-host return
    (a big jump at foothold, nothing after), the shape MVT assumes. With it False the brain
    pursues full root (a back-loaded, depth-oriented return).

    Retrying exploits indefinitely (rather than giving up after K tries) is deliberate: on an
    unexploitable or very-low-probability host this yields *flat* progress, and it is then the
    scheduler -- via MVT -- that decides to abandon. Baking a retry cap into the brain would
    pre-empt exactly the decision the paper studies.
    """

    def __init__(self, stop_at_foothold: bool = True) -> None:
        self.stop_at_foothold = stop_at_foothold

    def next_action(self, target, branch, rng) -> int | None:
        env = target.env
        addr = branch.addr
        hv = env.current_state.get_host(addr)
        acts = target.actions_by_host[addr]

        if hv.access >= 2:                      # already root -> nothing left
            return None

        if "ServiceScan" not in branch.done_scans and acts.get("ServiceScan"):
            branch.done_scans.add("ServiceScan")
            return acts["ServiceScan"][0]
        if "OSScan" not in branch.done_scans and acts.get("OSScan"):
            branch.done_scans.add("OSScan")
            return acts["OSScan"][0]

        if hv.access < 1:                       # need a foothold -> cycle exploits
            exploits = acts.get("Exploit", [])
            if not exploits:
                return None                     # no exploit exists for this host
            idx = exploits[branch.exploit_i % len(exploits)]
            branch.exploit_i += 1
            return idx

        if self.stop_at_foothold:               # breadth objective: harvested at foothold
            return None

        # depth variant: enumerate processes once, then cycle PE actions to root
        if "ProcessScan" not in branch.done_scans and acts.get("ProcessScan"):
            branch.done_scans.add("ProcessScan")
            return acts["ProcessScan"][0]
        pes = acts.get("PrivilegeEscalation", [])
        if not pes:
            return None
        idx = pes[branch.pe_i % len(pes)]
        branch.pe_i += 1
        return idx


class SubnetBrain:
    """Within-segment tactic: round-robin effort across the subnet's uncompromised hosts.

    Spreading attempts across the segment's hosts means its *easy* hosts fall first (their
    high-probability exploits succeed in early cycles) and its *hard* hosts linger -- so the
    segment's foothold rate is high early and decays as only stubborn hosts remain. That is
    the diminishing-returns curve MVT needs, produced by real NASim exploit mechanics rather
    than assumed. Each host is service/OS-scanned once, then its exploit is retried on later
    cycles; the brain never abandons the segment (that is the scheduler's MVT decision)."""

    def __init__(self, scan_first: bool = False) -> None:
        # With fully-observable recon (fully_obs), exploits do not require a prior scan, so
        # the default skips scanning: effort goes straight into exploitation and footholds
        # arrive immediately, giving a clean diminishing-returns signal with no ramp-up.
        self.scan_first = scan_first

    def next_action(self, target, branch, rng) -> int | None:
        env = target.env
        uncompromised = [a for a in branch.hosts
                         if env.current_state.get_host(a).access < 1]
        if not uncompromised:
            return None                         # segment fully harvested
        branch.rr %= len(uncompromised)
        addr = uncompromised[branch.rr]
        branch.rr += 1
        hs = branch.host_state.setdefault(addr, {"scanned": False, "os": False, "ei": 0})
        acts = target.actions_by_host[addr]
        if self.scan_first:
            if not hs["scanned"] and acts.get("ServiceScan"):
                hs["scanned"] = True
                return acts["ServiceScan"][0]
            if not hs["os"] and acts.get("OSScan"):
                hs["os"] = True
                return acts["OSScan"][0]
        exploits = acts.get("Exploit", [])
        if not exploits:
            return None
        idx = exploits[hs["ei"] % len(exploits)]
        hs["ei"] += 1
        return idx
