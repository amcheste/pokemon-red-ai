<div align="center">

<img src="docs/images/banner.png" alt="Pokémon Red RL Toolkit" width="100%" />

# Pokémon Red: Reinforcement Learning Toolkit

**A Gymnasium-compatible environment and training pipeline for
*Pokémon Red*, built on PyBoy, Stable-Baselines3, and `sb3-contrib`.**

[![Tests](https://img.shields.io/badge/tests-passing-0B0B0C)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-0B0B0C)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-1F4D3A)](LICENSE)

</div>

---

This repository provides everything needed to train RL agents to play
*Pokémon Red*: an emulator-backed Gymnasium environment with three
first-class observation treatments (pixel / symbolic / hybrid),
RecurrentPPO training scripts, an event-flag-based reward calculator
covering 15 critical-path milestones (Boulder Badge path, verified
against the [`pret/pokered`](https://github.com/pret/pokered)
disassembly), live Streamlit monitoring
dashboards, configurable alerts (desktop / Slack / email), and an
analysis layer with bootstrap confidence intervals via
[`rliable`](https://github.com/google-research/rliable).

---

## Repository at a glance

| Path | What's there |
|------|--------------|
| `pokemon_red_ai/environment/` | Gymnasium env wrapping PyBoy + 3 observation treatments |
| `pokemon_red_ai/training/` | RecurrentPPO trainer, callbacks (W&B, alerts, monitoring) |
| `pokemon_red_ai/analysis/` | Treatment-comparison logic (`comparison.py`) |
| `scripts/train.py` | Primary training entry point |
| `scripts/eval.py` | Deterministic evaluation harness |
| `scripts/analyze.py` | rliable bootstrap analysis → publication-quality figures |
| `scripts/compare.py` | Streamlit dashboard for side-by-side run comparison |
| `scripts/monitor.py` | Streamlit dashboard for live single-run monitoring |
| `scripts/run_pilots.sh` | Launch a multi-treatment / multi-seed run grid |
| `docs/research_playbook.md` | Step-by-step operational guide for long-running experiments |
| `tests/` | Unit + integration tests (pytest; run `pytest tests/` for the live count) |

---

## Quick start

```bash
# 1. Install
git clone https://github.com/amcheste/pokemon-red-ai.git
cd pokemon-red-ai
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Generate save states (one-time; requires a legal Pokémon Red ROM)
python3 scripts/create_save_states.py --rom path/to/PokemonRed.gb

# 3. Smoke test (~5 min, verify the pipeline end-to-end)
python3 scripts/train.py \
    --rom path/to/PokemonRed.gb \
    --save-state states/s0_post_intro.state \
    --observation-type pixel --total-timesteps 50000 --seed 42 \
    --save-dir ./training_output/smoketest

# 4. Run a multi-treatment / multi-seed grid (3 treatments × 3 seeds)
scripts/run_pilots.sh --rom path/to/PokemonRed.gb --parallel 3

# 5. Generate publication-quality figures
python3 scripts/analyze.py --results-dir ./training_output \
    --output-dir ./figures --format pdf --reps 10000
```

For unattended overnight runs, configure desktop / Slack / email alerts:

```bash
cp configs/alerts.example.yaml configs/alerts.yaml   # then enable channels
```

The full operational playbook (including compute estimates and a
parallel-run strategy on Apple Silicon) is in
[`docs/research_playbook.md`](docs/research_playbook.md).

---

## Observation treatments

Three encoder paths, all feeding into the same LSTM (hidden size 256)
and PPO policy / value heads (`pi=[256,128]`, `vf=[256,128]`).
Selected via `--observation-type` on `scripts/train.py`.

| Treatment | Observation | Encoder | Params | Feature dim |
|-----------|-------------|---------|--------|-------------|
| `pixel`   | 80×72×1 grayscale Game Boy screen | NatureCNN ([Mnih et al. 2015](https://www.nature.com/articles/nature14236)), `features_dim=256` | ~564K | 256 |
| `symbolic` | Player position, party stats, 18-flag bit-vector, exploration counters (29 features) | 3-layer MLP `29 → 640 → 640 → 256` | ~594K | 256 |
| `hybrid`  | `pixel` ∪ `symbolic` streams | NatureCNN(256) + symbolic MLP(256), concatenated | ~1.16M | 512 |

The pixel and symbolic encoders are sized to within 10% on trainable
parameter count to neutralize the encoder-capacity confound when
comparing modalities (Henderson et al. 2018; Engstrom et al. 2020;
Andrychowicz et al. 2021). Strict per-forward FLOP matching across CNN
and MLP architectures distorts encoder design and is reported
transparently rather than enforced. Per-condition learning rates are
selected from a pre-registered log-uniform grid following Eimer et al.
(2023).

Run [`scripts/check_encoder_capacity.py`](scripts/check_encoder_capacity.py)
to print the exact parameter / FLOP table and assert the 10% match
constraint (exits non-zero on violation).

Implementation:
[`pokemon_red_ai/training/models.py`](pokemon_red_ai/training/models.py);
observation construction in
[`pokemon_red_ai/environment/observations.py`](pokemon_red_ai/environment/observations.py).

The package also ships three legacy observation types
(`multi_modal`, `screen_only`, `minimal`) for backward compatibility
with earlier scripts.

## Reward function

The default `events` reward strategy uses a configurable set of 18
event flags between Pallet Town and the Boulder Badge.  Each flag
transition 0 → 1 awards a fixed positive reward exactly once per
episode.  A small per-step time penalty, a new-map discovery bonus,
and a party-faint penalty are also active by default.

Four other reward strategies are available
(`standard` / `exploration` / `progress` / `sparse`); see
[`pokemon_red_ai/environment/rewards.py`](pokemon_red_ai/environment/rewards.py)
for the full menu and configuration knobs.

The flag list with bit offsets is in
[`pokemon_red_ai/game/event_flags.py`](pokemon_red_ai/game/event_flags.py).

## Statistical analysis

Following [Agarwal et al. 2021, *Deep Reinforcement Learning at the
Edge of the Statistical Precipice*](https://arxiv.org/abs/2108.13264),
the analysis tooling reports:

- **Point estimate:** interquartile mean (IQM) over per-seed scores.
  Robust to outlier seeds in either tail.
- **Uncertainty:** 95% percentile bootstrap with 2,000 resamples.
- **Pairwise comparison:** probability of improvement,
  `Pr[score_A > score_B]` via stratified bootstrap.

Implemented in [`scripts/analyze.py`](scripts/analyze.py) (post-hoc
figures) and
[`pokemon_red_ai/analysis/comparison.py`](pokemon_red_ai/analysis/comparison.py)
(reusable backend for the live Streamlit comparison and any notebook
work).

## Live monitoring

| Tool | Use case |
|------|----------|
| Weights & Biases (auto-enabled in `train.py`) | Cloud telemetry; per-treatment run grouping; check from any device |
| `streamlit run scripts/monitor.py` | Single-run live dashboard: reward curves, event flags, maps, level / party / money |
| `streamlit run scripts/compare.py` | Multi-run comparison: IQM table, learning-curve overlays with 95% bands, milestone race |
| `pokemon_red_ai.training.alerts` | Desktop / Slack / email alerts on first badge, reward plateau, training crash |

## Use as a library

The training pipeline is fully usable outside the bundled scripts:

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

MIT, see [LICENSE](LICENSE).

You must own a legal copy of the Pokémon Red ROM. This repository does
not distribute, link to, or facilitate acquisition of any copyrighted
game data.
