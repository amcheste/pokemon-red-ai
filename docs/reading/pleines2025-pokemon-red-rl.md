# Reading notes: Pleines et al. 2025 — "Pokémon Red via Reinforcement Learning"

- **arXiv:** [2502.19920](https://arxiv.org/abs/2502.19920) (v2, 2025-03-11)
- **Authors:** Marco Pleines, Daniel Addis, David Rubinstein, Frank Zimmer,
  Mike Preuss, Peter Whidden (the PokemonRedExperiments author)
- **Linear:** AMC-143 (P1 — most directly related prior work)
- **Read:** 2026-07-03, full text (6-page paper)

## Answers to the AMC-143 reading goals

### 1. Exact methodology

- **Algorithm:** PPO (custom implementation, clipped surrogate + clipped
  value loss, no entropy bonus — they found tuning it gave no gain).
  Memory is an *ablation variant*: the body is a plain FC layer by
  default, optionally a **GRU** ("GRU" run). Not SB3, not RecurrentPPO,
  not LSTM. 32 workers, 2048-step horizon, batch 65,536, γ=0.997,
  AdamW, lr 3e-4.
- **Observation space (single fixed design, all experiments):**
  1. Vision: 72×80 grayscale screen, downsampled ×2, **3-frame stack**;
  2. A separate **48×48 binary visited-coordinates crop** centered on
     the player (an exploration memory map fed as an image);
  3. Game-state vector: party HP + levels, plus event-completion flags.
  Both vision modalities go through their own Nature CNN; the state
  vector through one FC layer; all concatenated (~2M params, GRU 4M).
- **Reward:** dense shaping, summed: event reward (+2 per storyline
  event), navigation reward (+0.005 per new overworld coordinate per
  episode), healing reward (proportional to fractional party-HP gain),
  level reward (capped, downscaled past level 22).
- **Scale:** 5 seeds per experiment, ~400M env steps, evaluated with
  30 episodes at 25 checkpoints; mean ± std reported (no IQM /
  bootstrap CIs).
- **Scope:** Pallet Town → Cerulean City (~20% of the game; includes
  Brock = Boulder Badge). Fixed Squirtle starter (ablations try
  Charmander/Bulbasaur/choice). Dynamic episode budget: 10,240 steps
  +2,048 per completed event.

### 2. Pixel-only ablation — CONFIRMED ABSENT ✅

The observation design is **fixed across every experiment**: pixels +
visited-coordinates map + symbolic game-state vector, always jointly.
Ablations cover reward components (nav ×10 / heal / level), starter
choice, "Fast" text-speed setting, and FC-vs-GRU body — **never
observation modality**. There is no pixel-only, no symbolic-only, no
modality comparison of any kind. Quote from §II-C: "this observation
space is intentionally limited for simplicity and to mimic the
information a first time human player would have."

**⇒ Pleines et al. does not preempt our study.** They ask "can DRL
progress in Pokémon Red at all, and how does reward shaping break?" —
we ask "which observation representation drives learning, under
capacity-matched encoders?" Orthogonal questions on the same game.

### 3. Encoder capacity

No capacity matching anywhere; not applicable since there is no
cross-modality comparison. (Their 2M/4M params are for the whole
network.)

### 4. Findings we should cite

- **Reward exploitation:** the heal reward is gamed hard (Bulbasaur
  Leech-Seed/Zubat infinite-battle loop near Mt. Moon; Charmander runs
  heal-farm via mom in Pallet Town, capping Brock success at ~80%).
  Navigation reward ×10 makes agents explore obsessively and avoid
  battles. **Directly motivates our M3 reward-firing analysis** (rule
  out reward-gaming confounds) and our simpler one-shot event-flag
  reward design.
- **Time-bias:** with free starter choice the agent picks Charmander
  because its reward arrives a few actions sooner — discounting bias
  connecting choice to distal outcome (>3,000 steps to Brock vs
  effective horizon ≈333 steps at γ=0.997).
- **Sample-efficiency anchor:** Mt. Moon (post-Brock) milestone rises
  at ~50–100M steps; Cerulean ~50–90% at 400M; humans need ~11k steps,
  agents ~2× that. **Implication for us:** our planned 50M-step runs
  targeting Boulder Badge are in a plausible but tight regime — the
  10M-step LR-sweep pilots will tell, and curriculum save states are
  the fallback lever.
- GRU helps on tasks needing retained context (Bill's quest 48% vs
  19% for Fast) — supports our RecurrentPPO/LSTM choice.

## Differentiation paragraph (draft for related work)

> Pleines et al. (2025) establish a PPO baseline that reaches Cerulean
> City in Pokémon Red and expose reward-shaping exploits, but train a
> single fixed observation design that always combines screen pixels,
> a visited-coordinate map, and a symbolic game-state vector; they do
> not compare observation modalities. In contrast, we hold the
> algorithm, reward, and recurrent policy fixed and vary only the
> observation representation — pixel, symbolic (RAM-derived), or
> hybrid — under capacity-matched encoders, isolating representation
> as the experimental factor. Their reward-exploitation findings
> motivate our one-shot event-flag rewards and per-condition
> reward-firing audit.

## Still to do (Alan)

- [ ] Full read + annotate in Zotero (this note is a working summary)
- [ ] Check their appendix/repo for human replay data we could reuse
      as an evaluation anchor
