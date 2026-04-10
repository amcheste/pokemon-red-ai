# Compute Ledger

All significant training runs are logged here for reproducibility and for the paper's "Reproducibility and compute" appendix.

**Logging policy:** Every training run that is intended to produce a result referenced in the paper (main experiment, ablation, hyperparameter trial, generalization test) must be logged here. Debug and exploratory runs are optional.

Every entry must include:

- **Date** (UTC)
- **Git SHA** (full commit hash of the code version used)
- **Run name** (must match the W&B run name and the checkpoint directory name)
- **Method** (`pixel` | `symbolic` | `hybrid` | ablation variant)
- **Seed**
- **Env-steps** (completed — not planned)
- **Wall-clock time**
- **Hardware** (e.g., "M3 Pro local (12 cores)", "Vast.ai 16-core + RTX 3060")
- **Result summary** (1 sentence — did it converge, what's the Brock win-rate, etc.)

---

## Entry format

```
### <run name>

- Date: YYYY-MM-DD
- Git SHA: <full hash>
- Method: <pixel | symbolic | hybrid | ablation>
- Seed: <int>
- Env-steps: <int>
- Wall-clock: <hh:mm>
- Hardware: <description>
- W&B URL: <link>
- Checkpoint: <path>
- Result: <1-sentence summary>
- Notes: <any deviations, failures, or observations>
```

---

## Entries

*(No entries yet. First entries will be logged after PR #1 merges and the first event-reward pilot run is completed.)*

---

## Deviations from `analysis_plan.md`

*(None yet. Any deviation from the pre-registered analysis plan must be logged here with a timestamp, the specific change, and the reason.)*
