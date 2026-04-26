# Paper scaffolding — EWRL 2026

LaTeX skeleton + drafted prose for the EWRL submission.  Sections 1-4
are full first drafts that don't depend on pilot results; sections 5-6
are skeletons with `TODO(pilots)` markers everywhere a number, figure,
or interpretation needs to land.

## Layout

```
paper/
├── main.tex                   # Top-level document, includes sections
├── references.bib             # Bibliography (sorted alphabetically)
├── Makefile                   # `make pdf` to build, `make watch` for live
├── analysis_plan.md           # Pre-registered hypotheses (existing)
├── compute_ledger.md          # Per-run compute log (existing)
├── figures/                   # Output dir for scripts/analyze.py
└── sections/
    ├── 01_introduction.tex    # ✅ Full draft
    ├── 02_related_work.tex    # ✅ Full draft
    ├── 03_environment.tex     # ✅ Full draft
    ├── 04_methods.tex         # ✅ Full draft  (1 small TODO: confirm hyperparameters)
    ├── 05_results.tex         # ⬜ Skeleton with TODO(pilots) markers
    └── 06_discussion.tex      # ⬜ Skeleton with TODO(pilots) markers
```

## Building the PDF

You need `pdflatex` + `bibtex` (any TeX distribution).  On macOS:

```bash
brew install --cask mactex-no-gui    # full distribution
# or for a smaller install:
brew install --cask basictex
eval "$(/usr/libexec/path_helper)"
```

Then:

```bash
cd paper
make pdf      # full build (pdflatex → bibtex → pdflatex × 2)
make quick    # single pass — for fast iteration when refs already resolved
make clean    # remove build artefacts (keeps PDF)
```

Verified: builds cleanly to a 12-page PDF on TeX Live 2026 Basic.

## Filling in the TODOs

After the pilots finish, search the LaTeX for `\todopilot{` (the source
macro that renders as `[TODO(pilots): ...]` in red in the PDF):

```bash
cd paper
grep -nR "todopilot" sections/ main.tex
```

Every match is a place pilot data needs to go.  Prioritise by section:

1. **Section 5 (Results)** — every figure and table.  Most matches are
   here.  Generate the figures via:

   ```bash
   ./venv/bin/python3 scripts/analyze.py \
     --results-dir ./training_output \
     --output-dir ./paper/figures \
     --plots aggregate profiles improvement efficiency \
     --reps 10000 --format pdf
   ```

   Then uncomment the `\includegraphics{...}` lines in
   `05_results.tex`.

2. **Section 4 (Methods) — single TODO** — confirm the hyperparameter
   table against `pokemon_red_ai/training/models.py:get_model_config()`
   once you've launched a pilot run.  No pilot results needed for this.

3. **Section 6 (Discussion)** — write the one-paragraph
   interpretations after looking at the IQM table and figures.  The
   structural argument (why pixel might lose, why hybrid might not win,
   threats to validity, future work) is fully drafted; you only need to
   add results-aware connective tissue.

4. **Section 1 (Introduction)** — one TODO at the bottom of the
   contributions list: replace the placeholder bullet with a real
   one-sentence headline result.

5. **Abstract (in `main.tex`)** — one TODO: insert the headline
   sentence.

## Style notes for the user when polishing

- **Avoid hedging that sounds like an LLM wrote it.**  "We would argue
  that..." → "We argue that..."  "It could be the case that..." →
  "We find that..."
- **Drop redundant qualifiers.**  "very", "quite", "somewhat",
  "relatively" — search and remove.
- **Numbers always with units and CIs.**  "IQM was 12.3 (95% CI [10.1,
  14.5])" not "IQM was higher".
- **Past tense for what was done, present tense for what is true.**
  "We trained agents..." but "PPO is a policy-gradient method..."
- **Two-line maximum for any single sentence.**  EWRL reviewers are
  time-constrained.  If a sentence wraps three lines on screen, break
  it.

## When pilot data lands, ping Claude

Once `scripts/compare.py` shows the IQM table and the four PDFs are in
`paper/figures/`, just paste:

1. The IQM table (copy from the Streamlit dashboard).
2. A one-line list of qualitative observations (e.g., "hybrid hit
   Viridian Forest by ep 1500 on average, pixel never did").

I (Claude) will draft the results and discussion paragraphs and open a
follow-up PR.

## EWRL template swap

When the EWRL 2026 LaTeX template is published, swap the
`\documentclass` line in `main.tex` and any venue-specific commands
(usually section formatting and the title block).  The section content
itself is template-agnostic.
