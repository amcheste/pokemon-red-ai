# Compute Plan

Forward-looking projections of compute requirements for the 3-paper
cascade.  Sits alongside [`compute_ledger.md`](compute_ledger.md) (a
log of what was actually used) and [`analysis_plan.md`](analysis_plan.md)
(the pre-registered protocol that fixes seed counts and step budgets).

This document is **descriptive of plan, not normative for science** —
the numbers in `analysis_plan.md` are locked; the numbers below are
operational estimates that may be revised as we measure throughput on
real hardware.  When this document and `analysis_plan.md` disagree
about a budget figure, `analysis_plan.md` wins.

---

## Summary

| Milestone | Total env-steps | Local M3 Max viable? | Cloud needed? | When to apply for cloud |
|-----------|-----------------|----------------------|---------------|--------------------------|
| **M2 EWRL** (now)        | 90M (3 × 3 × 10M)   | ✅ Yes — ~33-50h  | No  | — |
| **M3 PokeGym release**   | <10M (env smoke runs) | ✅ Trivial      | No  | — |
| **M4 NeurIPS workshop**  | ~1B (15+ runs)      | 🟡 ~45 days local — uncomfortable | Yes (recommended) | NCSA ACCESS application now, target approval before M4 starts |
| **M5 TMLR campaign**     | 4.2B (per `analysis_plan.md` §10) | ❌ ~190 days — infeasible | Yes (mandatory) | NCSA expansion or paid cloud by M5 kickoff |

**Headline:** the M3 Max gets us through EWRL.  Cloud compute applications
should be filed during EWRL pilots so allocations are approved by the time
M4 work starts.

---

## Current setup: Apple M3 Max

Hardware:

- 16 cores total: **12 performance** + 4 efficiency
- 36 GB unified memory
- 40-core GPU (accessible via PyTorch MPS, currently unused — see §Optimisation)
- ~100 W sustained power draw under heavy load

Measured throughput (smoke test, 2026-04-30):

| Configuration | fps (env-steps/s) | Notes |
|---------------|-------------------|-------|
| Single env, pixel obs, CPU PyTorch | **~91** | Baseline measurement (50k steps, RecurrentPPO) |
| 4 parallel envs, projected | **~250-350** | Sub-linear due to PPO update step on main process |
| 8 parallel envs, projected | **~500-700** | Closer to CPU-bound saturation |

The bottleneck is PyBoy emulation, which is single-threaded per
instance.  Adding parallel envs via `SubprocVecEnv` runs N PyBoy
instances on N CPU cores; throughput scales near-linearly until we
saturate either the CPU cores or the PPO update step.

Memory cost is modest: each PyBoy instance uses ~300-500 MB, so 8
envs + the policy network + Python runtime fits comfortably under
8 GB of the 36 GB available.

---

## Compute requirements per milestone

### M2 — EWRL 2026 (current)

**Pre-registered scope** (from `analysis_plan.md`, with deviations
documented in §5.2 of the paper Methods):

- 3 treatments (`pixel`, `symbolic`, `hybrid`)
- 3 seeds per treatment (deviation: plan calls for 5; pilot uses 3)
- 10M env-steps per seed (deviation: plan calls for 100M; pilot uses 10M)
- Single environment per training run (deviation from 8 in plan;
  reverting to 4 with `SubprocVecEnv`)

**Total budget: 90M env-steps.**

Local time estimates with 4 parallel envs per pilot:

| Concurrency | Per-pilot wall-clock | Total wall-clock for all 9 pilots |
|-------------|----------------------|------------------------------------|
| 1 pilot at a time | ~11h | ~100h sequential (~4 days) |
| 2 pilots concurrent | ~11h × 5 batches | ~55h (~2.3 days) |
| **3 pilots concurrent** ⭐ | ~11h × 3 batches | ~33h (~1.4 days) |

3-pilot concurrency uses 12 P-cores (3 × 4 envs), at the edge of
the M3 Max's P-core count but stable with thermal management
(plug-in, lid-open, ambient ~22 °C).

**Decision:** run all 9 pilots locally with `--n-envs 4` and
`scripts/run_pilots.sh --parallel 3`.

### M3 — PokeGym Release

Compute requirements are minimal — env smoke tests, packaging
verification, optional Colab demo runs.  Estimated <10M env-steps
total.  Local M3 Max is overspec; this milestone is mostly
documentation, packaging, and outreach.

### M4 — NeurIPS 2026 workshop

**Planned scope:**

- 5 seeds × 3 treatments × 50M = **750M env-steps** (main results)
- 1 ablation × 5 seeds × 50M = **250M env-steps** (per `analysis_plan.md` §5.3, ablation set is 4; for M4 we ship at least one)
- Total: **~1B env-steps**

Local time estimate at 8 parallel envs (~600 fps):

- 1B / 600 / 3600 = **463 hours = ~19 days continuous**
- Realistically with iteration, debugging, partial reruns: **30-45 days
  total wall-clock** on the M3 Max

This is technically feasible but uncomfortable.  Risks:

- Laptop tied up for weeks; can't take it to conferences / travel
- Single point of failure (laptop crash → restart from last checkpoint)
- Thermal throttling concerns under sustained 30+ day load

**Decision:** apply for cloud compute by EWRL submission so the
allocation is in place when M4 begins.  Target approval timeline:
2-4 weeks for NCSA ACCESS startup allocation.

### M5 — TMLR campaign

**Planned scope** (from `analysis_plan.md` §10):

| Component | Treatments | Seeds | Steps/seed | Total |
|-----------|------------|-------|------------|-------|
| Main experiment | 3 | 5 | 100M | 1.5B |
| Ablations | 4 | 5 | 50M | 1.0B |
| Generalization | 3 | 5 | 30M | 0.45B |
| Optuna HP search | 3 × 50 trials | 1 | 5M | 0.75B |
| Re-run buffer | — | — | — | 0.5B |
| **Total** | | | | **~4.2B** |

Local time estimate at 8 parallel envs:

- 4.2B / 600 / 3600 = **1944 hours = ~81 days continuous**

Cloud is **mandatory**.  At a high-CPU cloud instance running 32-64
parallel envs:

- 4.2B / 4000 fps / 3600 = ~292 hours
- At $2.50/hr (e.g., AWS `c7g.16xlarge`): **~$730** raw compute
- Plus storage, telemetry, retries: budget **~$1000-1500** total

NCSA ACCESS could absorb this fully if approved.  Otherwise, paid
cloud is within personal-research reach.

---

## Cloud compute options

| Option | Cost | Approval / setup | When to use |
|--------|------|------------------|-------------|
| **NCSA ACCESS startup** | Free | 2-4 weeks application; renewable for 1 year, then full allocation review | Best long-term option for M4 / M5.  Apply now (AMC-69 is the Linear issue tracking this). |
| **NCSA ACCESS full allocation** | Free | 2-3 months review | M5 production runs after the startup allocation is exhausted. |
| **Google TRC** (TPU v3-8) | Free | Email application, fast | Skip — would require JAX rewrite of the codebase, not worth the effort for our scale. |
| **RunPod community pods** | $0.40-0.50/hr (RTX 4090) | Instant signup | Good for one-off cloud smoke tests.  Beware: community pods can be evicted with 5 min notice. |
| **Lambda Cloud** | $0.50-2.00/hr | Instant signup | Reliable backup if NCSA isn't approved by M4 start. |
| **AWS / GCP / Azure** | $2-3/hr (high-CPU) | Instant signup | Overkill for our scale.  Use only if other options unavailable.  AWS Graviton (`c7g.16xlarge`) is the best single-node option for CPU-bound PyBoy workloads. |

The cloud GPU advantage is not relevant here — PyBoy is CPU-bound, so
high-CPU instances beat GPU instances dollar-for-dollar.  This rules
out most "AI training cloud" services that price for GPU.

---

## Action triggers

### Now (during EWRL pilots)

- [x] Implement `--n-envs N` parallel envs (this PR)
- [ ] **AMC-69: submit NCSA ACCESS startup application** — uses EWRL
      paper abstract as research justification.  Target: 4 weeks for
      approval.
- [ ] **AMC-70: Oracle IP review** — must be complete before
      arXiv preprint, regardless of compute.

### EWRL submission week (2026-05-25)

- [ ] If NCSA application has progressed: provide them with EWRL
      paper / arXiv preprint as supporting material for the full
      allocation review.
- [ ] If NCSA has not progressed: budget ~$50 for a one-day RunPod
      smoke test of the codebase on Linux+x86, validate
      reproducibility before committing M4 compute there.

### M4 prep (post-EWRL, ~July 2026)

- [ ] Decide cloud target (NCSA / RunPod / Lambda) based on what's
      approved.
- [ ] Run the M4 grid.  Estimated wall-clock 1-2 weeks if on cloud,
      4-6 weeks if local-only.

### M5 prep (post-NeurIPS workshop, ~early 2027)

- [ ] Validate that NCSA full allocation is approved.  If not, request
      expansion of startup allocation or open paid-cloud budget.
- [ ] Run the M5 grid.  Estimated wall-clock ~2 weeks on cloud.

---

## Reproducibility and energy footprint

Per the recommendations of [Henderson et al. 2020, *Towards the
Systematic Reporting of the Energy and Carbon Footprints of Machine
Learning*](https://arxiv.org/abs/2002.05651), every paper in the
cascade will report:

- Total environment steps consumed
- Wall-clock hours per treatment, mean and total
- Hardware: M3 Max P-core count + total RAM, or cloud instance type
- Estimated kWh per run (from instrumented workload, not nameplate
  power)
- Approximate `gCO₂eq` using regional grid intensity (NC for local,
  cloud-region grid mix for cloud runs)

The actual measurements go in [`compute_ledger.md`](compute_ledger.md);
this `compute_plan.md` only documents the projected ranges.

---

## Updates

| Date | What changed |
|------|--------------|
| 2026-04-30 | Initial document.  M3 Max baseline 91 fps measured; 4-env projection 250-350 fps.  EWRL pilots planned for local; NCSA application triggered for M4. |

When projections change (new throughput measurements, scope
adjustments, or cloud allocation status), append a row above and
update the affected sections in place.
