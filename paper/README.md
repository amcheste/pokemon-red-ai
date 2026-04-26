# `paper/` — Research Artifacts

This directory holds everything paper-related.

## Layout

| File / Dir | Purpose |
|---|---|
| `main.tex` | Top-level LaTeX document (workshop-paper format). |
| `references.bib` | Bibliography (alphabetised by key). |
| `sections/` | Per-section LaTeX source — see [`SCAFFOLDING.md`](SCAFFOLDING.md). |
| `Makefile` | `make pdf` / `make quick` / `make clean`. |
| `figures/` | Publication-quality PDFs produced by `scripts/analyze.py`. |
| `analysis_plan.md` | **Pre-registered** hypotheses, primary / secondary metrics, statistical tests, seed counts, and stopping rules. Locked before main experiments begin. |
| `compute_ledger.md` | Append-only log of every significant training run with git SHA, seeds, env-steps, wall-clock, hardware, and result summary. |
| `MODEL_CARD.md` | _(at repository root)_ Model card following Mitchell et al. 2019. |

## Workflow

1. **Before main experiments:** confirm `analysis_plan.md` is committed
   and unchanged since the plan was finalised. Running main-result
   experiments against an uncommitted plan voids the pre-registration.

2. **During experiments:** append every run to `compute_ledger.md` as
   it completes. Don't batch entries — logging at run-end avoids
   losing data on a crash.

3. **After experiments:** generate figures via the analysis pipeline:

   ```bash
   python3 scripts/analyze.py \
       --results-dir ./training_output \
       --output-dir paper/figures \
       --plots aggregate profiles improvement efficiency \
       --reps 10000 --format pdf
   ```

   Then build the PDF:

   ```bash
   cd paper && make pdf
   ```

   See [`SCAFFOLDING.md`](SCAFFOLDING.md) for the full results-section
   workflow (`grep "todopilot" sections/`).

## Mirroring to Overleaf

The `paper/` directory is the canonical source. To mirror to Overleaf
for live preview or co-author editing:

```bash
scripts/mirror_paper_to_overleaf.sh
```

See [`bin/setup-overleaf-project.sh`](../bin/setup-overleaf-project.sh)
for one-shot project provisioning.

## Reproducibility checklist

- [ ] `analysis_plan.md` committed before main experiments
- [ ] All main-result runs logged in `compute_ledger.md`
- [ ] All figures produced via `scripts/analyze.py` (no ad-hoc plotting)
- [ ] W&B project public or accessible to reviewers
- [ ] `scripts/seed_utils.py` used for all RNG seeding
- [ ] `scripts/eval.py` used for all eval results (no ad-hoc eval)
