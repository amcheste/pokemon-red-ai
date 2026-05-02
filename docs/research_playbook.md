# Pokemon Red RL: Research Playbook

Everything you (Alan) need to do between right now and EWRL 2026
submission, in order. Keep this open while you work.

> **Status snapshot**
> - **Today:** 2026-04-25
> - **EWRL deadline:** 2026-05-25 (~30 days)
> - **Done:** M1 (100%), Training Observability (75%, pending PR #26 merge)
> - **Critical path:** You run pilots → I write the paper → you review/submit
> - **Open PRs:** #26 (AMC-79, closes Training Observability)
> - **Linear project:** [pokemon-red-ai](https://linear.app/amcheste/project/pokemon-red-ai)

---

## Step 0: Merge PR #26 (5 min, do today)

Closes the Training Observability milestone. Once merged:

```bash
git checkout main
git pull origin main
```

Linear will auto-update AMC-79 to Done. The full observability stack is
now live: W&B custom panels (AMC-76), Streamlit dashboard (AMC-77),
training alerts (AMC-78), treatment comparison (AMC-79).

---

## Step 1: Pre-pilot prep (1-2 hours, do this week)

### 1a. Verify save state exists

The pilots all start from `post_intro.state` (Pallet Town after Oak's
intro). Verify it's there:

```bash
ls -la states/post_intro.state
```

If missing, regenerate:

```bash
./venv/bin/python3 scripts/create_save_states.py --rom <path-to-rom.gb>
```

### 1b. W&B login

```bash
./venv/bin/python3 -m wandb login
```

Confirm the API key is set up. Paste from <https://wandb.ai/authorize>.

### 1c. Configure alerts (recommended)

The training alerts you'll want for unattended runs:

```bash
cp configs/alerts.example.yaml configs/alerts.yaml
# Edit configs/alerts.yaml: at minimum enable desktop, optionally Slack
```

Verify it parses:

```bash
./venv/bin/python3 -c "from pokemon_red_ai.training.alerts import load_alert_config; print(load_alert_config('configs/alerts.yaml'))"
```

If you want Slack, create an incoming webhook at
<https://api.slack.com/messaging/webhooks> and either:
- Paste the URL into `configs/alerts.yaml`, OR
- `export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...`

### 1d. Compute decision

You need ~30 GPU-hours per run × 9 runs = **~270 GPU-hours total** for
the 10M-step pilots (rough estimate, varies 2x with hardware). Options:

| Option | Cost | Notes |
|--------|------|-------|
| Your local GPU | Free, your time | Best if you have RTX 3080/4080+ |
| RunPod (RTX 4090) | ~$0.40/hr × 270 = ~$110 | Spin up community pods on demand |
| Lambda Cloud | ~$0.50/hr × 270 = ~$135 | More reliable than RunPod |
| Google TRC | Free (TPU v3) | Apply at <https://sites.research.google/trc/about/>, requires JAX rewrite, skip for EWRL |

**Recommendation:** Run pilots on your local hardware to validate, push
to a cloud GPU only if local timing is too slow.

### 1e. Smoke-test for 5 minutes

Confirm the full pilot pipeline works end-to-end:

```bash
./venv/bin/python3 scripts/train.py \
  --rom <path-to-rom.gb> \
  --save-state states/post_intro.state \
  --observation-type pixel \
  --algorithm RecurrentPPO \
  --reward-strategy events \
  --total-timesteps 50000 \
  --seed 42 \
  --save-dir ./training_output/smoketest \
  --wandb-run-name smoketest-pixel-s42 \
  --enable-alerts --alerts-config configs/alerts.yaml
```

In another terminal, confirm the Streamlit dashboard works:

```bash
streamlit run scripts/monitor.py -- --runs-dir ./training_output
```

If both produce data and you get at least one alert ping, you're ready.

---

## Step 2: Run the 9 pilots (the long part, ~5-10 days)

3 treatments × 3 seeds = 9 runs of ~10M steps each. Plan: launch them
sequentially (or in parallel if you have multiple GPUs), monitor live,
let alerts ping you on milestones.

### Suggested order

Run pixel first: fastest treatment (smallest observation), gives you a
known-good baseline before symbolic and hybrid burn compute.

```
pixel-s42 → pixel-s123 → pixel-s456
symbolic-s42 → symbolic-s123 → symbolic-s456
hybrid-s42 → hybrid-s123 → hybrid-s456
```

### Launch command (one per run)

Substitute `$TREATMENT` ∈ `{pixel, symbolic, hybrid}` and `$SEED` ∈ `{42, 123, 456}`:

```bash
TREATMENT=pixel SEED=42

./venv/bin/python3 scripts/train.py \
  --rom <path-to-rom.gb> \
  --save-state states/post_intro.state \
  --observation-type "$TREATMENT" \
  --algorithm RecurrentPPO \
  --reward-strategy events \
  --total-timesteps 10000000 \
  --seed "$SEED" \
  --save-dir "./training_output/${TREATMENT}-seed${SEED}" \
  --wandb-project pokemon-red-ai \
  --wandb-run-name "${TREATMENT}-seed${SEED}" \
  --enable-alerts --alerts-config configs/alerts.yaml \
  --alerts-checkpoint-freq 1000000 \
  --save-freq 250000 \
  -v 2>&1 | tee "logs/${TREATMENT}-seed${SEED}.log"
```

> **Tip:** wrap this in a `tmux` or `screen` session so it survives
> SSH disconnects. `tmux new -s pixel-s42` then `Ctrl-B D` to detach.

### Monitor live

In a separate terminal:

```bash
streamlit run scripts/monitor.py -- --runs-dir ./training_output
```

Open <http://localhost:8501>. Auto-refreshes every 30s (configurable in
sidebar). Watch:
- Reward curve trending up
- New maps appearing in the heatmap
- Event flags ticking on
- Player level rising

Also check the W&B run page; it gives you remote access from any browser.

### When to kill a run early

A pilot is **not worth continuing** if after ~2M steps:
- Reward curve still flat near zero (no learning signal)
- Zero new maps discovered (stuck on map 0)
- Zero event flags triggered

Kill it (`Ctrl-C`, saves an `interrupted_model.zip` for debugging) and
start the next one. Note in the W&B run notes why you killed it.

### Tagging in W&B

Once running, tag each W&B run with `pilot`, `ewrl-2026`, and the
treatment name so the comparison report auto-groups them. Either:
- Click "Tags" on the run page and add manually, OR
- Pre-tag via the `--wandb-run-name` (already includes treatment)

### Per-run checklist

Track progress in your own notes file:

```
[ ] pixel-s42      ___ steps done, ___ badges, killed/finished
[ ] pixel-s123     ___ steps done, ___ badges, killed/finished
[ ] pixel-s456     ___ steps done, ___ badges, killed/finished
[ ] symbolic-s42   ___
[ ] symbolic-s123  ___
[ ] symbolic-s456  ___
[ ] hybrid-s42     ___
[ ] hybrid-s123    ___
[ ] hybrid-s456    ___
```

When all 9 are done, mark AMC-63/64/65 as Done in Linear.

---

## Step 3: Analyze results (1-2 days, after all pilots done)

### 3a. Live comparison (Streamlit)

```bash
streamlit run scripts/compare.py -- --runs-dir ./training_output
```

In the sidebar: select all 9 runs. The app auto-groups by treatment
(based on the run name prefix). Look at:
- **Learning curves**: does any treatment dominate?
- **IQM table**: does any 95% CI exclude the others?
- **Final-performance bars**: does the ranking match the curves?
- **Milestone race**: which treatment hits BEAT_BROCK first?

Click PDF download buttons on each figure. These go straight into the
paper.

### 3b. Generate paper figures

The publication-quality figures live behind `scripts/analyze.py`:

```bash
./venv/bin/python3 scripts/analyze.py \
  --results-dir ./training_output \
  --output-dir ./paper/figures \
  --plots aggregate profiles improvement efficiency \
  --reps 10000 \
  --format pdf
```

This produces:
- `paper/figures/aggregate_metrics.pdf`: IQM + 95% CIs
- `paper/figures/performance_profiles.pdf`
- `paper/figures/probability_of_improvement.pdf`
- `paper/figures/sample_efficiency.pdf`

### 3c. W&B report

Open the W&B project, create a new Report, follow
`configs/wandb_report_template.md` to lay out the panels. Share the URL
with co-authors when the paper draft is ready.

---

## Step 4: Write the EWRL paper (AMC-67, ~1 week)

I (Claude) can do most of this once the pilot results are in. Process:

### 4a. I scaffold the LaTeX (1 day, can start immediately)

I can do this **right now** without waiting for pilots:

```
paper/
  main.tex
  references.bib
  sections/
    01_introduction.tex
    02_related_work.tex
    03_environment.tex
    04_methods.tex
    05_results.tex       ← skeleton, fill after pilots
    06_discussion.tex    ← skeleton
  figures/                ← already exists, populated by analyze.py
```

I'll draft sections 1-4 (intro, related work, environment, methods) now
since they don't depend on results, just need pilot numbers later.

**You decide:** want me to set up the LaTeX scaffolding + draft sections
1-4 in parallel with your pilot runs? Say "yes do paper scaffolding" and
I'll open another PR.

### 4b. Pilot results land → I draft results/discussion (1-2 days)

Once pilots finish and `compare.py` shows the IQM table, give me:
1. The IQM table (copy/paste from the dashboard)
2. The 4 PDF figures from `analyze.py`
3. Any qualitative observations ("hybrid agent learned to enter the cave faster than pixel")

I'll produce a draft of section 5 (Results) and section 6 (Discussion)
in another PR.

### 4c. You review and polish (1-2 days)

Read end-to-end, fix anything that sounds like an LLM wrote it (because
one did), tighten transitions, sanity-check all numbers against the W&B
runs.

### 4d. Get a second opinion (1 day)

Ideally before submitting, even a quick read by:
- An AI/ML colleague at Oracle (paid attention to IP separation, see Step 5)
- Or a friend in academia
- Or post the abstract for feedback in /r/MachineLearning

The exit criterion in M2 says "reviewed by at least 1 person before
submission."

### 4e. Submit to EWRL by 2026-05-25

EWRL submission portal will open closer to the deadline. Watch
<https://ewrl.wordpress.com/> for the call. Submission is a single PDF,
~9 pages including references.

### 4f. Post arXiv preprint within 48h

Per the M2 exit criteria. Categories: `cs.LG`, `cs.AI`. License: CC BY 4.0.

```bash
# After arXiv is up, link in Linear AMC-67 and mark Done.
```

---

## Step 5: Research operations (parallel, do whenever)

These don't block pilots/paper but need to be done before the next
papers (M3-M5). Tackle them during pilot wall-clock time.

### AMC-70: Oracle IP compliance review

**Why:** Before posting the arXiv preprint, you need a clear answer that
this work isn't owned by Oracle. **Do this BEFORE step 4f.**

- [ ] Read your Oracle employment agreement IP clauses
- [ ] Confirm: all dev on personal hardware (not Oracle-issued)?
- [ ] Confirm: no Oracle compute used (OCI free tier is your personal account)?
- [ ] Confirm: no overlap with Oracle work products?
- [ ] Document: "CAM Labs LLC owns this research, published under CAM Labs / personal affiliation"
- [ ] If unclear → talk to Oracle Legal before submission

### AMC-69: NCSA ACCESS compute grant (for M4/M5)

**Why:** M4 needs ~2000 GPU-hours and M5 needs ~5000. Self-funding gets
expensive past EWRL.

- [ ] Apply for **Startup Allocation** (fastest, no review board) at <https://access-ci.org/>
- [ ] Use the EWRL paper abstract as the research description
- [ ] PI affiliation: CAM Labs LLC if no academic appointment, else NC State
- [ ] Resource justification: "30 runs × 100M steps × ~1 A100 each ≈ 3000 hours"

### AMC-68: NC State CS advisor (for paper credibility)

**Why:** EWRL is a workshop and CAM Labs alone is fine. NeurIPS / TMLR
benefit from an academic co-author.

- [ ] Browse NC State CS faculty: <https://www.csc.ncsu.edu/people/faculty/>
- [ ] Filter for ML/AI group, RL/game research
- [ ] Email 2-3 promising candidates with EWRL paper draft + your pitch
- [ ] Goal: secure co-author for Papers B and C (M4/M5)
- [ ] **Timeline: start outreach AFTER EWRL submission**

---

## Timeline arithmetic

| Date | Milestone |
|------|-----------|
| 2026-04-25 (today) | PR #26 ready to merge |
| 2026-04-26 | Observability milestone closed |
| 2026-04-26 → 28 | Pre-pilot prep + smoke test |
| 2026-04-28 → 05-08 | 9 pilot runs (10 days, sequential on 1 GPU) |
| 2026-05-08 → 10 | Analysis + figures |
| 2026-05-10 → 17 | Paper drafting (Claude does most) |
| 2026-05-17 → 22 | Polish + 2nd-opinion review |
| 2026-05-22 → 25 | Final tweaks, submit |
| **2026-05-25** | **EWRL deadline** |

**~14 days slack**: comfortable but not generous. If pilots take 14 days
instead of 10 (slow GPU, debugging), slack drops to 10 days. If you
slip a week, slack is gone.

**Risk mitigation:**
- Run pilots in parallel if you have 2+ GPUs (cuts wall clock 50%)
- Drop to 5M-step pilots if 10M is taking too long (note in paper as
  "preliminary, full 10M for camera-ready")
- Skip 2nd-opinion review if behind schedule (acceptable for a workshop)

---

## What "done" looks like

When all of these are checked, M2 is closed and you can start M3:

- [ ] PR #26 merged
- [ ] All 9 pilot runs complete in W&B with `pilot`, `ewrl-2026` tags
- [ ] `paper/figures/*.pdf` populated by `scripts/analyze.py`
- [ ] EWRL paper PDF submitted via the workshop portal
- [ ] arXiv preprint posted, link added to Linear AMC-67
- [ ] AMC-63/64/65/67 marked Done in Linear
- [ ] Oracle IP review documented (AMC-70)

---

## Quick reference: commands

```bash
# Sync to latest main
git checkout main && git pull origin main

# Smoke test
./venv/bin/python3 scripts/train.py --rom <ROM> --save-state states/post_intro.state \
  --total-timesteps 50000 --seed 42 \
  --save-dir ./training_output/smoketest --wandb-run-name smoketest

# Full pilot (substitute TREATMENT, SEED)
./venv/bin/python3 scripts/train.py --rom <ROM> --save-state states/post_intro.state \
  --observation-type pixel --algorithm RecurrentPPO --reward-strategy events \
  --total-timesteps 10000000 --seed 42 \
  --save-dir ./training_output/pixel-seed42 \
  --wandb-run-name pixel-seed42 \
  --enable-alerts --alerts-config configs/alerts.yaml

# Live single-run dashboard
streamlit run scripts/monitor.py -- --runs-dir ./training_output

# Treatment comparison (after multiple runs done)
streamlit run scripts/compare.py -- --runs-dir ./training_output

# Paper figures
./venv/bin/python3 scripts/analyze.py --results-dir ./training_output \
  --output-dir ./paper/figures --plots aggregate profiles improvement efficiency \
  --reps 10000 --format pdf

# Linear status
# https://linear.app/amcheste/project/pokemon-red-ai
```

---

## When you get stuck: ping me

If anything in this playbook breaks or is unclear, just paste the error
or question into a Claude session. I have full context on the codebase
and can debug, refactor, or rewrite sections of this document on demand.

For paper-specific questions (especially LaTeX, citations, framing), I
can also draft alternative phrasings or restructure sections. Just
share the current draft.

Good luck. Ship it. 🚀
