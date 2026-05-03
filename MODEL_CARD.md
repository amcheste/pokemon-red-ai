# Model Card: Pokémon Red AI

This is a placeholder model card. It will be populated once the first paper-grade models are trained and released. The structure follows Mitchell et al. 2019, *Model Cards for Model Reporting*.

> **PR #0 placeholder.** No models have been trained against this card yet. Numbers will be filled in starting in PR #5 (first main-experiment seed completes).

---

## Model Details

- **Model name:** *(TBD: will be assigned per main-experiment treatment)*
- **Model versions:** `pixel-v1`, `symbolic-v1`, `hybrid-v1` (planned)
- **Model type:** RecurrentPPO with `{pixel | symbolic | hybrid}` feature extractor
- **Training algorithm:** Proximal Policy Optimization with LSTM, via `sb3-contrib`'s `RecurrentPPO`
- **Development date:** *(TBD: first checkpoints expected ~PR #5)*
- **Developer:** Alan Chester
- **License:** MIT (see [LICENSE](LICENSE))
- **Repository:** https://github.com/amcheste/pokemon-red-ai

## Intended Use

- **Primary use:** Reinforcement-learning research on long-horizon, sparse-reward games. Specifically, as a baseline and reproducible target for studies of observation representation in deep RL.
- **Secondary use:** Educational reference for the Pokémon Red RL community.
- **Out of scope:**
  - Production use of any kind
  - Commercial automation of legitimate Pokémon gameplay
  - Any setting that requires reliability guarantees beyond the published benchmark

## Training Data

- **Environment:** Pokémon Red (1996, Game Boy), emulated via [PyBoy](https://github.com/Baekalfen/PyBoy)
- **ROM requirement:** Users **must provide their own legal copy** of the Pokémon Red ROM. This repository does not distribute, link to, or facilitate acquisition of any copyrighted game data.
- **Save states:** A small number of starting save states (post-intro, post-starter, etc.) are generated locally by the user; the repository ships only the placeholder directory.

## Evaluation Protocol

All evaluation results in the paper use the locked protocol defined in `scripts/eval.py`:

- 20 deterministic episodes
- Fixed starting save state (`s0_post_intro.state`)
- Fixed evaluation seed (42)
- Argmax policy (no exploration noise at eval time)
- Max episode length: 15,000 environment steps

The full pre-registered analysis plan is in [`paper/analysis_plan.md`](paper/analysis_plan.md).

## Metrics

*(To be populated after main-result experiments run.)*

| Metric | Pixel | Symbolic | Hybrid |
|---|---|---|---|
| Brock win-rate @ 100M steps | TBD | TBD | TBD |
| Mean event flags triggered | TBD | TBD | TBD |
| Steps to first Brock win (mean) | TBD | TBD | TBD |
| Unique maps visited (max) | TBD | TBD | TBD |

All numbers will report IQM ± 95% bootstrap CI over 5+ seeds.

## Quantitative Analyses

Sample-efficiency curves, ablation tables, and generalization results will be in [`paper/figures/`](paper/figures/) once experiments complete.

## Ethical Considerations

- **No personal data** is involved at any stage.
- **ROM copyright:** Pokémon Red is © Nintendo / Game Freak / Creatures Inc. Users must own a legal copy. This project does not distribute the ROM and does not endorse piracy.
- **Compute footprint:** Training compute will be reported transparently in the paper (env-steps, wall-clock, hardware, estimated kWh) per the recommendations of Henderson et al. 2020, *Towards the Systematic Reporting of the Energy and Carbon Footprints of Machine Learning*.
- **Dual use:** This work is on a single-player video game. We do not see meaningful dual-use risk.

## Caveats and Recommendations

- Results apply only to a single game (Pokémon Red). Generalization claims about RL representations more broadly require additional environments and are not made by this paper.
- Symbolic observations rely on RAM addresses specific to Pokémon Red and are not transferable to other titles without per-game memory mapping.
- The pre-registered analysis plan exists for a reason. Please do not cherry-pick metrics from these checkpoints in follow-up work without re-running with appropriate statistical rigor.

---

*Last updated: 2026-04-09 (placeholder for PR #0).*
