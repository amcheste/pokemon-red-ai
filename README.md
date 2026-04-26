# Pokémon Red — A Long-Horizon RL Benchmark

**A controlled empirical study of observation representations in
long-horizon, sparse-reward reinforcement learning, using *Pokémon
Red* as the benchmark environment.**

[![Tests](https://img.shields.io/badge/tests-833%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

We compare three observation treatments — **pixel** (raw screen),
**symbolic** (RAM-derived structured state), and **hybrid** (both) —
under matched architecture, reward, and compute. RecurrentPPO agents
are trained against an 18-flag event-progress reward, and results are
reported with [`rliable`](https://github.com/google-research/rliable)
interquartile means and 95% stratified-bootstrap confidence intervals.

This work is being released as a 3-paper cascade:

| # | Venue | Status | Target |
|---|-------|--------|--------|
| A | EWRL 2026 (workshop, non-archival) | Pilot runs scheduled | 2026-05-25 |
| B | NeurIPS 2026 workshop (ARLET / Embodied World Models) | Planned | 2026-09-15 |
| C | TMLR (journal) | Planned | 2027-10-31 |

Pre-registered hypotheses, primary metric, statistical tests, and
stopping rules are in
[`paper/analysis_plan.md`](paper/analysis_plan.md). The paper
itself (LaTeX) lives in [`paper/`](paper/) and is mirrored to Overleaf
via [`scripts/mirror_paper_to_overleaf.sh`](scripts/mirror_paper_to_overleaf.sh).

---

## Repository at a glance

| Path | What's there |
|------|--------------|
| `pokemon_red_ai/environment/` | Gymnasium env wrapping PyBoy + 3 observation treatments |
| `pokemon_red_ai/training/` | RecurrentPPO trainer, callbacks (W&B, alerts, monitoring) |
| `pokemon_red_ai/analysis/` | Treatment-comparison logic (`comparison.py`) |
| `scripts/train.py` | Primary training entry point — used by the pilot launcher |
| `scripts/eval.py` | Deterministic evaluation harness for paper-grade results |
| `scripts/analyze.py` | rliable bootstrap analysis → publication-quality figures |
| `scripts/compare.py` | Streamlit dashboard for side-by-side treatment comparison |
| `scripts/monitor.py` | Streamlit dashboard for live single-run monitoring |
| `scripts/run_pilots.sh` | Launch the canonical 9-pilot grid (3 treatments × 3 seeds) |
| `paper/` | LaTeX source, pre-registered analysis plan, compute ledger |
| `docs/research_playbook.md` | Step-by-step operational guide from setup to submission |
| `tests/` | 833 unit + integration tests (pytest) |

---

## Quick start — reproducing the EWRL pilots

```bash
# 1. Install
git clone https://github.com/amcheste/pokemon-red-ai.git
cd pokemon-red-ai
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Generate save states (one-time; requires a legal Pokémon Red ROM)
python3 scripts/create_save_states.py --rom path/to/PokemonRed.gb

# 3. Smoke test (~5 min — verify the pipeline end-to-end)
python3 scripts/train.py \
    --rom path/to/PokemonRed.gb \
    --save-state states/post_intro.state \
    --observation-type pixel --total-timesteps 50000 --seed 42 \
    --save-dir ./training_output/smoketest

# 4. Run the full pilot grid (10M steps × 3 treatments × 3 seeds)
scripts/run_pilots.sh --rom path/to/PokemonRed.gb --parallel 3

# 5. Generate paper-quality figures
python3 scripts/analyze.py --results-dir ./training_output \
    --output-dir paper/figures --format pdf --reps 10000
```

For unattended overnight runs, configure desktop / Slack / email alerts:

```bash
cp configs/alerts.example.yaml configs/alerts.yaml   # then enable channels
```

The full operational playbook — including compute estimates, parallel
strategy on Apple Silicon, and the path from pilot results to arXiv
submission — is in
[`docs/research_playbook.md`](docs/research_playbook.md).

---

## Observation treatments

The core experimental contrast. All three feed into the same LSTM and
PPO policy / value heads; only the encoder differs.

| Treatment | Observation | Encoder | Feature dim |
|-----------|-------------|---------|-------------|
| `pixel`   | 80×72×1 grayscale Game Boy screen | Nature-DQN-style CNN | 256 |
| `symbolic` | Player position, party stats, 17-flag bit-vector, exploration counters | 2-layer MLP | 256 |
| `hybrid`   | `pixel` ∪ `symbolic` streams | Both, concatenated at the LSTM input | 512 |

Selected via `--observation-type {pixel,symbolic,hybrid}` on
`scripts/train.py`. Implementation is in
[`pokemon_red_ai/training/models.py`](pokemon_red_ai/training/models.py);
observation construction in
[`pokemon_red_ai/environment/observations.py`](pokemon_red_ai/environment/observations.py).

## Reward function

A pre-registered set of 18 event flags between Pallet Town and the
Boulder Badge defines the reward signal. Each flag transition
0 → 1 awards a fixed positive reward exactly once per episode. A
small per-step time penalty, a new-map discovery bonus, and a
party-faint penalty are also active; all other shaping is zero by
default.

The flag list with bit offsets is in
[`pokemon_red_ai/game/event_flags.py`](pokemon_red_ai/game/event_flags.py);
the rationale and locking commitment is in
[§9 of the analysis plan](paper/analysis_plan.md).

## Statistical methodology

Following [Agarwal et al. 2021, *Deep Reinforcement Learning at the
Edge of the Statistical Precipice*](https://arxiv.org/abs/2108.13264):

- **Point estimate:** interquartile mean (IQM) over per-seed scores.
  Robust to outlier seeds in either tail.
- **Uncertainty:** 95% percentile bootstrap with 2,000 resamples.
- **Pairwise comparison:** probability of improvement,
  `Pr[score_A > score_B]` via stratified bootstrap. A treatment is
  reported as superior if this probability exceeds 0.75 with the 95%
  CI excluding 0.5.

Implemented in [`scripts/analyze.py`](scripts/analyze.py)
(post-hoc paper figures) and
[`pokemon_red_ai/analysis/comparison.py`](pokemon_red_ai/analysis/comparison.py)
(reusable backend for the live Streamlit comparison and any notebook
work).

## Live monitoring

| Tool | Use case |
|------|----------|
| Weights & Biases (auto-enabled in `train.py`) | Cloud telemetry; per-treatment run grouping; check from any device |
| `streamlit run scripts/monitor.py` | Single-run live dashboard — reward curves, event flags, maps, level / party / money |
| `streamlit run scripts/compare.py` | Treatment comparison — IQM table, learning-curve overlays with 95% bands, milestone race |
| `pokemon_red_ai.training.alerts` | Desktop / Slack / email alerts on first badge, reward plateau, training crash |

## Use as a library

The training pipeline is fully usable outside the paper context.
Minimal example with a custom training budget:

```python
from pokemon_red_ai.environment import PokemonRedGymEnv
from sb3_contrib import RecurrentPPO

env = PokemonRedGymEnv(
    rom_path="PokemonRed.gb",
    observation_type="hybrid",
    reward_strategy="events",
    max_episode_steps=15_000,
)
model = RecurrentPPO("MultiInputLstmPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

Custom reward strategies, observation types, and callback chains are
documented in [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md).

## Running the tests

```bash
./venv/bin/python3 -m pytest                  # full suite (~17s)
./venv/bin/python3 -m pytest tests/unit/      # unit only
./venv/bin/python3 -m pytest -k comparison    # specific module
```

---

## Citation

```bibtex
@unpublished{chester2026pokegym,
  author = {Alan Chester},
  title  = {Symbols or Pixels? A Controlled Study of Observation
            Representations in Long-Horizon Reinforcement Learning},
  year   = {2026},
  note   = {EWRL 2026 workshop submission, available at
            https://github.com/amcheste/pokemon-red-ai},
}
```

The model card following Mitchell et al. 2019 is in
[`MODEL_CARD.md`](MODEL_CARD.md).

## Acknowledgments

Built on [PyBoy](https://github.com/Baekalfen/PyBoy) (Game Boy
emulation), [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
and
[`sb3-contrib`](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
(RL algorithms), [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
(RL interface), and
[`rliable`](https://github.com/google-research/rliable) (statistics).
Memory addresses verified against the
[`pret/pokered`](https://github.com/pret/pokered) disassembly.

## License & ROM

MIT — see [LICENSE](LICENSE).

You must own a legal copy of the Pokémon Red ROM. This repository does
not distribute, link to, or facilitate acquisition of any copyrighted
game data.
