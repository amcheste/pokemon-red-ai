# Research Landscape Prompt

Paste the following into a new Claude Opus 4.7 conversation with extended
thinking / research enabled. This is a self-contained prompt — no prior
context needed.

---

## The Prompt

I am planning a reinforcement learning research paper and need your help
surveying the current research landscape to validate and refine my
hypotheses before I commit to expensive experiments.

### My Project

I have a working RL environment for Pokemon Red (1996 Game Boy game)
using the PyBoy emulator. The agent uses RecurrentPPO (PPO with LSTM,
via sb3-contrib) and an event-flag-based reward system that gives
one-shot rewards for story milestones (getting a starter Pokemon,
beating trainers, defeating the first gym leader Brock). The primary
success metric is winning the Boulder Badge.

### My Proposed Paper

**Working title:** "Symbols or Pixels? A Controlled Study of Observation
Representations in Long-Horizon Reinforcement Learning"

**Target venue:** TMLR (Transactions on Machine Learning Research)

**Treatments (same algorithm, same rewards, only observations differ):**

1. **Pixel:** 80x72x1 grayscale screen, 4-frame stack, Nature-DQN CNN
   encoder → LSTM → PPO
2. **Symbolic:** 20x18 tile map read from RAM (0xC6EF) + 32-dim event
   flag vector + 16-dim party stats → MLP encoder → LSTM → PPO
3. **Hybrid:** Both pixel and symbolic streams concatenated at the LSTM
   input

**Hypotheses:**

- H1: Symbolic observations yield at least 5x better sample efficiency
  than pixel observations (env-steps to first Boulder Badge win)
- H2: Hybrid performs comparably to symbolic on in-distribution maps
  but generalizes better to unseen map segments
- H3: The advantage of symbolic comes primarily from the tile map
  component rather than the game-state stats vector

**Statistical approach:** rliable library (Agarwal et al. 2021), IQM
with 95% bootstrap CIs, 5+ seeds per treatment, 100M env-steps per seed

### What I Need From You

Please do deep research on the following and give me a structured report:

**1. Current State of Observation Representation in RL**
- What papers have directly compared pixel vs. symbolic/structured
  observations in the same environment with controlled experiments?
- Are there any systematic studies like what I'm proposing, or is this
  gap genuinely underexplored?
- What do papers like Mnih et al. 2015 (DQN), Hafner et al. (Dreamer),
  and the MuZero line say about observation representations?

**2. Pokemon Red RL Landscape**
- Survey the existing Pokemon Red RL projects (Peter Whidden's
  pokemon-red-experiments, PufferLib's Pokemon Red environment, any
  published or preprint papers using Pokemon Red as a benchmark)
- What observations/rewards/algorithms have they used?
- Has anyone done a controlled observation-representation comparison
  in Pokemon Red specifically?

**3. Long-Horizon Sparse-Reward RL**
- What are the current best practices for long-horizon, sparse-reward
  game environments? (Go-Explore, RND, ICM, reward shaping, curriculum
  learning, etc.)
- How does my event-flag reward shaping approach compare to what others
  have done?

**4. Hypothesis Validation**
- For each of my three hypotheses (H1, H2, H3), tell me:
  - Is there existing evidence that supports or contradicts it?
  - Is the predicted effect size (5x for H1, 2x for H3) reasonable?
  - Is the hypothesis novel enough to be interesting, or has it already
    been answered?
- Are there better or sharper hypotheses I should consider instead?

**5. Positioning and Novelty**
- Where does my paper fit in the literature? What is the clearest
  novelty claim I can make?
- What are the strongest potential reviewer objections and how would I
  preempt them?
- Are there any critical related works I must cite and position against?

**6. Methodology Gaps**
- Is there anything in my experimental design that would weaken the
  paper's contribution? (missing baselines, unfair comparisons,
  statistical concerns, etc.)
- What would make the paper stronger? (additional ablations, different
  environments for generalization, different algorithms to test
  representation-agnosticism, etc.)

Please structure your response with clear sections, cite specific papers
with years, and be direct about whether you think this paper would be
accepted at TMLR as proposed or what changes would improve its chances.
I'd rather hear hard truths now than after running 4 billion env-steps
of experiments.
