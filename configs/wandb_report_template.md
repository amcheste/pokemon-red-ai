# W&B Report Template: Treatment Comparison (AMC-79)

A one-time setup template for the W&B Report you will use to share
treatment comparison results from the EWRL 2026 pilot runs and beyond.
The W&B Reports SDK is unstable across versions, so this is a manual
spec. Paste it into the report once and reuse the workspace forever.

The Streamlit app at ``scripts/compare.py`` mirrors this layout for the
local-first workflow; this report is for sharing with reviewers and
co-authors who don't want to run Python.

## Pre-requisites

* All pilot runs tagged with their treatment (`pixel`, `symbolic`,
  `hybrid`) when launched via ``scripts/train.py``.  The training
  script auto-tags via the W&B run config; confirm by checking
  `Config → observation_type` on any run page.
* Runs grouped via the W&B "Group by" feature on
  `config.observation_type`.

## Workspace setup (one-time)

1. Open the W&B project (``pokemon-red-ai``).
2. **Group by** → select `config.observation_type`.  This groups
   pixel/symbolic/hybrid runs into colored bands.
3. **Sort by** → `summary.episode/reward_max` descending so the best
   seed of each treatment is the line tip.
4. Switch the chart type for `episode/reward_mean` to **line plot
   with mean and standard deviation** (the "shaded area" preset).

## Recommended panels (in report order)

### 1. Headline learning curves

* X axis: `global_step` (or `_step`)
* Y axis: `episode/reward_mean`
* Group by: `config.observation_type`
* Aggregation: **mean ± std**
* Smoothing: rolling average, window = 20 episodes
* Why: this is the headline figure. Answers
  *"is the agent learning?"* in one glance.

### 2. Final-window performance bars

* Chart type: **bar chart**
* X axis: `config.observation_type`
* Y axis: `summary.episode/reward_mean` (last 50 episodes)
* Aggregation: mean ± std across seeds
* Why: numerical comparison without curve clutter.

### 3. Milestone-first-step heatmap

* Chart type: **scatter plot** (manual)
* X axis: event flag name (categorical)
* Y axis: `min(global_step where flag triggered)`  per
  `(observation_type, seed)`
* Group by: `config.observation_type`
* Why: shows which treatment hits each Boulder Path milestone first.

### 4. Map exploration heatmap

* Chart type: **bar chart**
* X axis: `summary.game/maps_visited_max`
* Group by: `config.observation_type`
* Aggregation: mean
* Why: secondary signal beyond raw reward; a treatment that
  explores more is often the treatment that wins long-term.

### 5. Reward component breakdown

* Chart type: **stacked bar**
* Series: `reward/exploration_mean`, `reward/badge_mean`,
  `reward/event_flags_mean`, `reward/time_mean`, `reward/death_mean`
* Group by: `config.observation_type`
* Why: explains *why* one treatment wins,
  e.g. "hybrid earns more from exploration components".

### 6. Per-seed scatter

* Chart type: **scatter plot**
* X axis: `summary.episode/reward_mean`
* Y axis: `summary.game/badges_max`
* Color: `config.observation_type`
* Why: surfaces variance. If pixel has 3 seeds at 200 reward and
  one seed at 0, it's not actually beating symbolic.

## IQM table (manual cell)

W&B doesn't compute IQM natively.  Add a **markdown panel** at the top
of the report with the IQM table from the Streamlit comparison app:

```
streamlit run scripts/compare.py -- --runs-dir ./training_output
# Copy the "Aggregate metrics" dataframe into the report as Markdown.
```

For paper figures, export the matplotlib charts directly from the
Streamlit app (PDF download buttons).

## Naming conventions for auto-grouping

To make this report work without manual tweaking, name runs so the
treatment is the first underscore/dash-separated token:

* ✅ `rppo-pixel-seed42`
* ✅ `pixel_seed42`
* ✅ `symbolic-events-s7`
* ❌ `2026-04-25-runA` (falls back to "unknown" group)

The training script already does this automatically via the
``--wandb-run-name`` default.
