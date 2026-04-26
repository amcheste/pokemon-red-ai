# Contributing

Thanks for your interest in contributing.  This is an active research
codebase — we welcome bug reports, fixes, and small enhancements that
fit the project's scope.  For larger changes, please open an issue
first to discuss design.

## Scope

The project is the supporting code for a 3-paper research cascade
(EWRL 2026 → NeurIPS 2026 workshop → TMLR).  Contributions that align
with the cascade are most likely to be merged:

- Bug fixes in any layer
- Reproducibility improvements (deterministic seeding, environment hygiene)
- Test coverage on under-tested code paths
- Documentation accuracy fixes
- New observation treatments, reward strategies, or callbacks that
  don't break the existing pre-registered protocol
- Performance improvements that don't change behaviour

Out of scope without prior discussion:

- Changes that alter the pre-registered analysis plan
  ([`paper/analysis_plan.md`](paper/analysis_plan.md))
- Changes to the 18 pre-registered event flags
  ([`pokemon_red_ai/game/event_flags.py`](pokemon_red_ai/game/event_flags.py))
- New reward components that retroactively affect already-completed runs

## Development setup

```bash
git clone https://github.com/amcheste/pokemon-red-ai.git
cd pokemon-red-ai
python3 -m venv venv && source venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Run the test suite:

```bash
./venv/bin/python3 -m pytest                  # full suite (~17s)
./venv/bin/python3 -m pytest tests/unit/      # unit only
./venv/bin/python3 -m pytest -k comparison    # specific module
```

All 833 tests should pass on a clean checkout.  PRs that drop coverage
or break tests will not be merged.

## Branch and PR conventions

- **Never commit to `main` or `master`** — always work on a feature
  branch and open a pull request.
- Branch names: `feature/<short-description>`, `fix/<short-description>`,
  `docs/<short-description>`, `chore/<short-description>`.
- Open the PR against `main`.  PRs are routed to `@amcheste` via
  `CODEOWNERS`; leave the assignee field unset.
- Keep PRs focused — one logical change per PR.  Big mixed-bag PRs
  that combine refactors, features, and docs are hard to review and
  often get split anyway.
- Include a `Test plan` section in the PR body covering what you ran
  and what you didn't.

## Commit style

- Imperative subject line under 72 characters
  (`Add reward breakdowns to monitoring callback` rather than
  `added reward breakdowns to monitoring callback`).
- Body explains *why*, not what — the diff explains what.  Wrap the
  body at 72 characters.
- One logical change per commit.

## Code style

- Python 3.10+
- Follow the existing module structure (see [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md))
- Type hints on every public function (`Optional[str]`, `Dict[str, Any]`, etc.)
- Docstrings on every public class and function.  Args / Returns
  / Raises sections where applicable.
- Tests for every behaviour change — both the happy path and the
  obvious failure modes
- No emojis in code, comments, or documentation unless they're part of
  user-facing CLI output that's intended to be visually distinctive

## Reporting bugs

Use the issue templates under `.github/ISSUE_TEMPLATE/`.  At minimum:

- What you expected to happen
- What actually happened
- Steps to reproduce (smallest possible example)
- Versions: Python, the project, key dependencies (PyBoy, SB3,
  Gymnasium, PyTorch)

For training crashes, please include the relevant section of the
training log and the W&B run URL if applicable.

## Reproducibility expectations

Because this is a research codebase, reproducibility is unusually
important:

- Use `scripts/seed_utils.py` for any RNG seeding in new code
- Use `scripts/eval.py` for evaluation results (no ad-hoc eval loops
  in notebooks or scripts)
- Log every significant training run to
  [`paper/compute_ledger.md`](paper/compute_ledger.md)
- Don't commit secrets (W&B keys, Slack webhooks, ROM files) — the
  `.gitignore` covers the obvious cases but always sanity-check
  `git status` before pushing

## License

By contributing, you agree that your contributions will be licensed
under the [MIT License](LICENSE), the same license as the rest of the
project.
