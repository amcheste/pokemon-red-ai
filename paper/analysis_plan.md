# Pre-Registered Analysis Plan

**Project title (working):** Symbols or Pixels? A Controlled Study of Observation Representations in Long-Horizon Reinforcement Learning

**Target venue:** TMLR (Transactions on Machine Learning Research)

**Status:** Draft — to be finalized before any main-result experiments are run.

**Last updated:** 2026-04-09

---

## 1. Research question

In a long-horizon, sparse-reward game environment (Pokémon Red, with the Boulder Badge as the primary milestone), how does the choice of observation representation — **symbolic** (RAM-read tile maps + structured game state) vs **pixel** (downsampled screen) vs **hybrid** (both) — affect:

1. Sample efficiency (environment steps to reach a target level of performance)
2. Final performance at a fixed compute budget
3. Generalization to held-out map segments not seen during training

## 2. Hypotheses

- **H1 (primary):** At matched compute and matched architecture, symbolic observations yield significantly better sample efficiency than pixel observations, as measured by environment steps to first Boulder Badge. Predicted effect size: at least 5× fewer env-steps at 50% Brock win-rate.
- **H2:** A hybrid observation (symbolic + pixel streams concatenated at the LSTM input) performs comparably to symbolic alone on in-distribution evaluation but generalizes meaningfully better to held-out map segments.
- **H3:** The gap between symbolic and pixel is driven more by the tile-map component than by the game-state vector (party stats, event flags). We expect a tile-map-only agent to outperform a game-state-only agent in sample efficiency by a factor of at least 2×.

## 3. Primary metric

**Brock win-rate at 100M environment steps**, defined as the fraction of 20 deterministic evaluation episodes (from a fixed `s0_post_intro` save state, seed 42, argmax policy) in which the agent earns the Boulder Badge before episode truncation (15,000 env steps).

## 4. Secondary metrics

1. **Sample efficiency:** environment steps to first Boulder Badge win (mean across seeds)
2. **Event-flag coverage:** fraction of the 18 pre-registered event flags triggered at end of training
3. **Unique maps visited:** max across seeds
4. **Wall-clock time to convergence** — reported only for compute transparency, not used as a performance metric
5. **Mean episodic return** — sanity check that reward signal is being followed

## 5. Experimental design

### 5.1 Treatments (3, main experiment)

1. **`pixel`** — 80×72×1 grayscale screen, 4-frame stack → Nature-DQN CNN → LSTM → PPO
2. **`symbolic`** — 20×18 tile map (RAM `0xC6EF`) + 32-dim event flag vector + 16-dim party stats → MLP → LSTM → PPO
3. **`hybrid`** — `pixel` and `symbolic` feature streams concatenated at the LSTM input

### 5.2 Controls (identical across treatments)

- **Algorithm:** RecurrentPPO (`sb3-contrib`)
- **Reward function:** `EventProgressRewardCalculator` using the 18 pre-registered event flags (see §9)
- **Action space:** `Discrete(7)` — `SELECT` removed from the default 8-action space
- **Max episode steps:** 15,000
- **Parallel envs per seed:** 8
- **Training budget per seed:** 100M env-steps
- **Seeds:** 5 minimum for main result, 8 preferred for camera-ready
- **Hyperparameters:** jointly tuned via Optuna per-method (50 trials each) on a held-out save state (`s3_viridian_pokecenter`) that is not used in final evaluation

### 5.3 Ablations (4, secondary experiment)

Each ablation is a controlled variant of the best-performing main-experiment method:

- **`-lstm`** — replace LSTM with a matched-parameter MLP (tests value of recurrence)
- **`-rnd`** — remove Random Network Distillation intrinsic reward (tests value of novelty bonus)
- **`-framestack`** — 1 frame instead of 4 (tests value of temporal observation)
- **`-curriculum`** — uniform save state sampling instead of Go-Explore-weighted (tests value of curriculum)

### 5.4 Generalization test

Train on Pallet Town → Pewter City only. Evaluate on unseen territory: Route 3 → Mt. Moon → Cerulean City. This is a zero-shot transfer test; no fine-tuning is allowed.

## 6. Statistical analysis

All main-result comparisons use the **`rliable`** library (Agarwal et al. 2021, *Deep Reinforcement Learning at the Edge of the Statistical Precipice*) for analysis:

- **Point estimates:** Interquartile Mean (IQM), not raw mean, to reduce sensitivity to outlier seeds
- **Confidence intervals:** 95% stratified bootstrap CIs with 2,000 resamples
- **Significance test:** Probability of Improvement (PoI) via stratified bootstrap. A method is reported as superior if PoI > 0.75 with the 95% CI excluding 0.5.
- **Sample efficiency curves:** bootstrapped with the `plot_sample_efficiency_curve` helper

Single-seed learning curves will **not** appear in the paper except for qualitative illustrations explicitly labeled as such.

## 7. Stopping rules

- All seeds run to 100M env-steps unconditionally. **No early stopping based on intermediate results** (would invalidate the pre-registration).
- If any seed crashes mid-training (e.g. OOM, PyBoy segfault), re-run from the same seed index with a different RNG initialization and log the replacement in `compute_ledger.md`.
- If >20% of seeds for any treatment fail to train at all (no reward signal increase over random-policy baseline), report that treatment as "training-unstable" and exclude it from aggregate statistics rather than inflating variance.

## 8. Exclusions and deviations

Any deviation from this plan must be:

1. Documented in `paper/compute_ledger.md` with a timestamp and reason
2. Explained in the paper's Methods section under "Deviations from pre-registered plan"

Specifically prohibited mid-experiment changes:
- Changing the primary metric
- Changing the seed count downward
- Modifying the reward function
- Swapping in a different algorithm (e.g. switching from RecurrentPPO to SAC after seeing results)

## 9. Pre-registered event flag set (18 flags)

These are the exact event flags the reward function will reward. They are committed here to prevent post-hoc reward shaping.

1. `EVENT_FOLLOWED_OAK_INTO_LAB`
2. `EVENT_GOT_STARTER`
3. `EVENT_BATTLED_RIVAL_IN_OAKS_LAB`
4. `EVENT_BEAT_RIVAL_IN_OAKS_LAB`
5. `EVENT_GOT_POKEDEX`
6. `EVENT_GOT_OAKS_PARCEL`
7. `EVENT_DELIVERED_OAKS_PARCEL`
8. `EVENT_GOT_POKEBALLS_FROM_OAK`
9. `EVENT_GOT_TOWN_MAP`
10. `EVENT_BEAT_VIRIDIAN_FOREST_TRAINER_0`
11. `EVENT_BEAT_VIRIDIAN_FOREST_TRAINER_1`
12. `EVENT_BEAT_VIRIDIAN_FOREST_TRAINER_2`
13. `EVENT_BEAT_PEWTER_GYM_TRAINER_0`
14. `EVENT_BEAT_PEWTER_GYM_TRAINER_1`
15. `EVENT_BEAT_BROCK`
16. `EVENT_GOT_TM34_BIDE` (confirms Brock victory reward received)
17. `EVENT_BEAT_RIVAL_ROUTE22` (optional stretch milestone)
18. `EVENT_GOT_POKEMON_FROM_FAN_CLUB_CHAIRMAN` (sanity canary — should NOT be reached before Brock)

Exact flag IDs and bit offsets are verified from the `pret/pokered` disassembly and documented in `pokemon_red_ai/game/event_flags.py` (added in PR #1).

## 10. Compute budget (pre-commitment)

| Component | Treatments | Seeds | Steps/seed | Total |
|---|---|---|---|---|
| Main experiment | 3 | 5 | 100M | 1.5B |
| Ablations | 4 | 5 | 50M | 1.0B |
| Generalization test | 3 | 5 | 30M | 0.45B |
| Optuna hyperparameter search | 3 × 50 trials | 1 | 5M | 0.75B |
| Debug + re-run buffer | — | — | — | 0.5B |
| **Total budget** | | | | **~4.2B env-steps** |

Compute hardware and wall-clock costs will be reported in the paper's "Reproducibility and compute" appendix, with full entries in `compute_ledger.md`.

---

*This document is pre-registered before any main-result experiments are run. It may be revised for clarifying language before experiments start, but the hypotheses, metrics, and statistical tests are locked.*
