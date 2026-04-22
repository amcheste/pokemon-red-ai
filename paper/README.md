# `paper/` — Research Artifacts

This directory holds everything paper-related that is **not** source code:

| File / Dir | Purpose |
|---|---|
| `analysis_plan.md` | **Pre-registered** hypotheses, primary/secondary metrics, statistical tests, seed counts, and stopping rules. Locked before main experiments begin. |
| `compute_ledger.md` | Append-only log of every significant training run with git SHA, seeds, env-steps, wall-clock, hardware, and result summary. |
| `figures/` | Final paper figures (PDF/PGF preferred, PNG acceptable for drafts). Regeneratable from `notebooks/`. |
| `notebooks/` | Jupyter notebooks that produce the figures from raw W&B exports. Each notebook should pin its data source (W&B run IDs) at the top. |

## Workflow

1. **Before running experiments:** make sure `analysis_plan.md` is committed and unchanged since the plan was finalized. Running main experiments against an uncommitted plan voids the pre-registration.
2. **During experiments:** append every run to `compute_ledger.md` as it completes. Do not batch entries — logging at run-end avoids losing data on a crash.
3. **After experiments:** figures are produced by notebooks in `notebooks/` pulling data from W&B. The notebooks should never modify the raw data, only read it. Save notebook outputs as `figures/fig_<N>_<shortname>.pdf`.

## Reproducibility checklist

- [ ] `analysis_plan.md` committed before main experiments
- [ ] All main-result runs logged in `compute_ledger.md`
- [ ] All figures produced by deterministic notebooks
- [ ] W&B project is public (or accessible to reviewers)
- [ ] `scripts/seed_utils.py` used for all RNG seeding
- [ ] `scripts/eval.py` used for all eval results (no ad-hoc eval)
