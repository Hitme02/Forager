# Tier B — Live agent harness over a sandboxed cyber range

Tier B is the external-validity layer in which the **exact same** MVT scheduler, policies and
estimators that produced the abstract and NASim results drive a real, tool-executing agent
loop against a sandboxed range, instead of a simulator. It answers one question: *does the
foraging (Marginal Value Theorem) scheduling advantage survive contact with a real agent
harness, with real actions, real grounding, and real switching cost?*

Nothing above the `work_one` seam changes. A branch is now a **service** on the range; one
unit of compute is one **bounded agent action**; validated progress `g(c)` is scored only
from **observed service state** (`ProgressLedger`), never from anything the agent claims.

## What is here

| file | role |
|------|------|
| `sandbox.py` | the isolation contract + `MockSandbox` (default, in-process) and `DockerSandbox` (gated, real range) |
| `services.py` | per-service branch mechanics: `DiminishingService` (concave collection), `FootholdService` (constant-hazard), `LiveHttpService` (real) |
| `agent_harness.py` | the bounded action menu + `HarvestBrain`/`ScriptedBrain`/`LLMBrain`; the single point that enforces isolation before every action |
| `cyber_range.py` | `CyberRange` — presents the sandbox to the unchanged scheduler (same interface as `simulator.Environment`) |
| `run_tierb.py` | the experiment: diminishing sweep, constant sweep, burstiness study; `--live` for the gated flag run |
| `../../docker/` | the deliberately-vulnerable, isolated fixture range (compose + fixture service) |

## Results (`results/tierb.png`, `results/tierb_*.csv`)

1. **Diminishing-value regime — the advantage transfers.** Coverage objective over a
   heterogeneous range whose services yield a *concave* collection stream. MVT's
   time-to-objective speed-up over a per-level-tuned fixed timeout rises monotonically with
   heterogeneity σ: ≈ −18 % at σ = 0, crossing over near σ ≈ 0.6, to **+5–6 % at σ = 1.2** —
   the same crossover the abstract simulator shows, now in the live harness.

2. **Constant-hazard regime — the transfer boundary.** When access is a *memoryless*
   foothold (uniform value, no diminishing signal), the marginal-rate signal is
   uninformative, and plain MVT abandons every service before the hazard pays off: its
   **solve-rate collapses to 0**. This is the same boundary found in NASim — foraging-optimal
   stopping needs diminishing **value**, not merely a hard target.

3. **GuardedMVT — turning the boundary into a design win.** `scheduler.policies.GuardedMVT`
   discovers the regime *online, per branch*: it applies the MVT leaving rule only where the
   observed signal is genuinely diminishing (the smoothed marginal has fallen from a
   productive peak), and elsewhere commits a budget (a raw-tick stall fallback) instead of
   churning. One configuration then serves **both** regimes:
   - diminishing: it **keeps, and slightly beats, MVT's advantage** (+9.7 % at σ = 1.2);
   - constant hazard: it **solves 100 %** (no collapse), at a bounded penalty vs a
     constant-tuned fixed timeout.
   The decisive comparison is *robustness to an unknown regime*: at σ = 1.2, a fixed timeout
   tuned for one regime is poor in the other (the optimal budget differs ~8×: t≈8 vs t≈60),
   and plain MVT collapses under constant hazard — **GuardedMVT is the only policy strong in
   both with a single config** (diminishing 86 ticks / constant 74, vs fixed-tuned-constant
   137/47 and fixed-tuned-diminishing 95/135).

4. **Burstiness — the adaptive estimator earns its keep.** As the collection stream becomes
   bursty (rare large jumps), recent-slope MVT degrades below the fixed baseline; the
   burstiness-**adaptive** estimator (and GuardedMVT on top of it) recover the loss
   (183 → ~109 ticks at `burst_p = 0.1`). The smooth-concavity threat and its fix reproduce live.

## Live PAGP demo (`ffs/realism/demo_llm.py`)

A narrated demo built for showing the course (PAGP, CS372IA): it foregrounds the parallelism
and builds the security application on top.

```bash
python -m ffs.realism.demo_llm                 # auto: Ollama if serving, else scripted
python -m ffs.realism.demo_llm --backend ollama --model llama3.1:8b
python -m ffs.realism.demo_llm --backend scripted   # no LLM, always runs
```

It shows three PAGP layers in one run:

1. **Task parallelism** — *W* pentest agents forage the range in parallel; GuardedMVT is the
   load balancer, **migrating** a worker off a depleting branch onto a fresher one. A real
   local LLM (via Ollama; Llama-3.1-8B in testing) chooses each agent's action from the
   bounded menu; the demo narrates every ENTER / foothold / ABANDON→MIGRATE / flag capture.
2. **GPU data parallelism** — the *W* per-tick decisions are issued as one batched request
   (continuous-batched on the GPU), and the batch Monte-Carlo (`ffs/gpu/batch_sim.py`) runs
   ~10^5 foraging episodes/s on MPS — **~100× the per-episode engine** — which is what makes
   tuning GuardedMVT (the crossover, `budget_cap`, hysteresis) feasible. Both numbers are
   measured live at the end of the demo.
3. **Distributed** — a pointer to the MPI worker pool (`ffs/mpi/worker_pool.py`).

The LLM is optional: with no model (or `--backend scripted`) the demo runs on the scripted
brain and still captures the flag, so it never fails in front of an audience. Ollama is an
external, optional dependency (not in `requirements.txt`); any served model works.

## Safety and authorisation — read before using `--live`

Tier B automates attack actions, so the only defensible target is a **deliberately-vulnerable
range that you are authorised to test and that has no egress** to anything else. The design
enforces this rather than trusting it:

- **Isolation is a hard chokepoint.** `AgentHarness.step` calls `sandbox.assert_isolated()`
  before *every* action and fails closed. There is deliberately no flag to relax it.
  `DockerSandbox.assert_isolated()` verifies no-egress *from inside* the range by attempting an
  outbound connection to public hosts and refusing to run if any succeeds. The compose range
  sits on an `internal` Docker network (no gateway); the harness runs on that same network.
- **Explicit authorisation gate.** `DockerSandbox` cannot be constructed without
  `acknowledged=True`. By passing it you assert the range is isolated and that you are
  authorised to test it. The default experiment never needs it — it uses `MockSandbox`.
- **Bounded, non-weaponised actions.** The entire action menu is `probe`, `enumerate`,
  `attempt_access` (with the fixture's *planted default credential*), `collect`. There is no
  exploit synthesis and no arbitrary command execution; the live path maps each action onto a
  plain HTTP request against a fixture that holds no real data.
- **Grounded progress only.** A milestone scores only when its evidence predicate holds
  against observed state — a validity requirement (MVT must not optimise a hallucinated
  signal) and a safety one.
- **Independent of any internship/NDA code.** This harness is a minimal, clean, self-contained
  research instrument.

## Running it

Default (no Docker, no network — the headline sweep and the figure):

```bash
python -m ffs.realism.run_tierb --seeds 20          # writes results/tierb_*.csv
python -m ffs.experiments.plot_tierb                # writes results/tierb.png
pytest -q tests/test_tierb.py
```

Live, gated flag (crown-jewel) run — only in an isolated range you are authorised to test:

```bash
cd docker
docker compose -p ffs_tierb up --build --abort-on-container-exit harness
docker compose -p ffs_tierb down
```

The harness container drives the fixture services by name on the internal network, captures
the crown-jewel flag, and prints whether the objective was met. The same thing can be run from
the host against localhost-bound fixture instances for development; the egress check still
guards it.
