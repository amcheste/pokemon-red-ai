# Testing Roadmap

This document is the durable home for the multi-tier acceptance /
regression testing plan that emerged from the 2026-05-09 QA audit
([PR #45–47](https://github.com/amcheste/pokemon-red-ai) +
[paper PR #1](https://github.com/amcheste/pokemon-rl-paper/pull/1)).

> **Why this file exists.** The audit identified four test-pyramid
> gaps; only Tier 1 was implemented in the audit window
> ([PR #5: see `tests/integration/`](../tests/integration/)).
> Tier 2 and Tier 3 are intentionally deferred — this doc captures
> *what*, *why*, and *when* so a future Claude session can pick them
> up without re-running the audit.

---

## Tier 1 — done

Landed in the acceptance/regression PR.  See
[`tests/integration/`](../tests/integration/) — four test files,
12 tests, all run on every push:

| File | What it locks |
|------|----------------|
| `test_seeding_determinism.py` | The full seeding chain (`seed_everything` + `VecEnv.seed` + SB3) is deterministic on CPU.  Load-bearing for the paper's three-seed design. |
| `test_reward_regression.py` | `EventProgressRewardCalculator` per-step output on a golden 10-step trajectory.  Catches silent reward-weight or milestone changes. |
| `test_eval_schema.py` | Every key in `EVAL_METRIC_SCHEMA` is present and typed correctly in the JSON `evaluate_checkpoint` writes.  Prevents downstream `KeyError` at analysis time. |
| `test_performance_smoke.py` | `predict + step` loop ≥ 50 fps; reward calculator < 100 µs/call.  Catches catastrophic non-PyBoy regressions. |

The integration tests use a fully mocked `DeterministicMockEnv` (see
`tests/integration/conftest.py`) so they need no ROM and run in
under a second.

---

## Tier 2 — deferred until post-EWRL

Useful, but more work and lower urgency than Tier 1.  Pick up
after the EWRL submission lands.

### 2a. Real-ROM acceptance test  (`@pytest.mark.rom`)

**What:** A 2000-step rollout from `save_states/s0_post_intro.state`
that asserts:
- `EVENT_FOLLOWED_OAK_INTO_LAB` fires
- Total reward > 0
- At least one map transition occurs
- The episode terminates cleanly (no PyBoy exception)

**Why:** Tier 1 mocks PyBoy entirely.  Real-ROM tests catch the class
of bug PR #45 caught with event flags (wrong RAM addresses), where the
mock layer would mask the problem.

**How:** Use the existing `@pytest.mark.rom` marker (already declared
in `conftest.py`, currently unused).  Add a `--rom path` pytest option
that skips when not provided.  CI continues to run without a ROM;
contributors run locally with `pytest -m rom --rom path/to/PokemonRed.gb`.

**Effort:** ~2-3 hours.  Main complexity is the pytest option plumbing
and graceful PyBoy teardown on test failure.

### 2b. Cross-repo plan-vs-code spec test

**What:** A pytest test in `pokemon-red-ai` that parses
`pokemon-rl-paper/analysis_plan.md` (via a checked-out sibling
worktree or HTTPS fetch) and asserts every numerical claim it makes
about constants matches the code.  Specifically:

| Plan claim | Code source of truth |
|-----------|----------------------|
| LR = 2.5e-4 | `pokemon_red_ai.training.models.get_model_config('RecurrentPPO')` |
| `n_envs = 4` | `scripts/run_pilots.sh` |
| `seeds = [42, 123, 456]` | `scripts/run_pilots.sh` |
| `total_timesteps = 10_000_000` | `scripts/run_pilots.sh` |
| `max_episode_steps = 15_000` | `scripts/train.py` default |
| `n_episodes = 20` | `scripts/eval.LOCKED_N_EPISODES` |
| `eval_seed = 42` | `scripts/eval.LOCKED_SEED` |
| 15-flag set | `pokemon_red_ai.game.event_flags.BOULDER_PATH_FLAGS` (already locked vs pret/pokered in PR #45) |

**Why:** The audit found 3 drifts (LR, n_envs, flag set).  Manual
audits are not sustainable.  Automated coupling makes future drift
fail CI immediately.

**How:** Two options:
1. CI clones `pokemon-rl-paper` as a sibling repo (needs a deploy key
   or fine-grained PAT since the paper repo is private).
2. Vendor a snapshot of `analysis_plan.md`'s machine-readable
   constants into `pokemon-red-ai` and run a smaller consistency
   check + a separate periodic job that confirms the snapshot is
   in sync.

Option 2 is lower-friction; recommend that.

**Effort:** ~half a day.  Main complexity is the parser for the
plan's numerical claims (regex / a small Markdown→table extractor).

### 2c. Encoder forward-pass golden output

**What:** Lock the output tensor of `PokemonFeaturesExtractor`,
`SymbolicFeaturesExtractor`, and `HybridFeaturesExtractor` on a
fixed input + fixed initialization seed.  Tighter than the existing
parameter-count and FLOP-count tests.

**Why:** PR #36's modality fairness claim is supported by parameter
counts.  A future change that subtly reorders ops, changes
initialization, or swaps an activation could change the output
distribution without changing the param count — and PR #36's
capacity-match assertion would still pass.

**Effort:** ~2 hours.  Mostly mechanical: seed torch, build extractor,
forward a fixed input, hash the output, lock.

---

## Tier 3 — needs pilots first

Cannot be done before the first 9 pilots produce results.  Defer
until post-EWRL when there's a baseline to anchor against.

### 3a. W&B metric baseline lock

**What:** Once the canonical 9 pilots run, snapshot per-treatment
final IQM (reward, event flags triggered, maps visited) and the
shape of the learning curve.  Future code changes that move IQM by
more than 5% require an explicit `compute_ledger.md` entry.

**Why:** Mid-paper-cascade code changes (M3, M4, M5 milestones) will
keep touching the training pipeline.  Without a numerical anchor,
slow drift between camera-ready and the journal version is hard to
detect.

**How:** Add a JSON file in `pokemon-rl-paper` (e.g.
`baselines/m2_ewrl.json`) capturing per-treatment {seed, final IQM,
W&B run URL, git SHA}.  Add a pytest test in `pokemon-red-ai`
(opt-in via `@pytest.mark.paper-baseline`) that re-runs eval against
those checkpoints and asserts the metrics are within tolerance.

**Effort:** half day to set up; the data itself becomes available
once pilots finish.

### 3b. Reward semantics validation against a known trajectory

**What:** Hand-author a 100-step trajectory (sequence of button
inputs) that the reward calculator should score against an
*independently-verified* answer key.  Confirms the reward calculator
encodes the game's semantics, not just internal consistency.

**Why:** Tier 1's golden-output test locks the *current* calculator
against itself.  This test would catch a class of conceptual bug
(e.g. "Brock event flag fires too early" or "exploration counts
revisits") that's invisible from numerical regression alone.

**How:** Record a play-through in PyBoy, dump the per-step state
snapshots, manually annotate the expected reward at each step,
commit both the snapshots and the answer key.  Then a pytest test
replays the snapshots through the calculator and compares.

**Effort:** ~1 day total (most of it is the manual annotation).
Strongly recommend doing this once pilots are running and you have
real trajectories to base it on.

---

## What this doc is NOT

- Not a substitute for `docs/research_playbook.md` (operational
  runbook) or `pokemon-rl-paper/analysis_plan.md` (the pre-registered
  protocol).
- Not authoritative for *which* tests run in CI — that's
  `.github/workflows/test.yml`.
- Not exhaustive — anything not in Tier 1/2/3 is up to the next
  person to design.  These three tiers are the audit's explicit
  recommendation; treat them as a starting point.

## When to update this doc

Update whenever:
- A Tier 2 or Tier 3 item lands (move to "done", link the PR).
- A new gap is identified (add a new tier item, justify the
  deferral).
- The reasoning behind a deferral changes (e.g. EWRL timeline
  slips, so Tier 2 can start earlier).
