"""
Treatment comparison logic for the paper experiments (AMC-79).

This module is the shared backend for the Streamlit comparison app
(``scripts/compare.py``) and any post-hoc paper-figure generation.  It
deliberately avoids depending on Streamlit so it can be unit-tested
without the UI runtime.

Treatments are the three observation representations under study:

* ``pixel``    — 80×72×1 grayscale screen only (CNN)
* ``symbolic`` — flat memory-state vector (MLP)
* ``hybrid``   — Dict({screen, game_state}) (CNN + MLP, late fusion)

Multiple seeds per treatment form the statistical sample.  This module
loads per-run data, groups by treatment, builds learning curves with
mean ± std bands, computes IQM + bootstrap CIs via :mod:`rliable`, and
exports publication-quality figures.

The ``RunData`` type is the same one ``scripts/monitor.py`` produces —
re-importing here so callers don't need to know about the script layer.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constants — keep aligned with scripts/analyze.py for consistent paper
# figures.  If you change these, update both places.
# ──────────────────────────────────────────────────────────────────────

KNOWN_TREATMENTS: tuple = (
    "pixel",
    "symbolic",
    "hybrid",
    # legacy/aux observation types we may still encounter in old runs
    "multi_modal",
    "screen_only",
    "minimal",
)

TREATMENT_DISPLAY: Dict[str, str] = {
    "pixel": "Pixel",
    "symbolic": "Symbolic",
    "hybrid": "Hybrid",
    "multi_modal": "Multi-Modal",
    "screen_only": "Screen Only",
    "minimal": "Minimal",
}

# Brand-aligned categorical palette (alanchester-brand tokens/colors.css).
# Three distinguishable values across lightness and hue, all from the
# brand's neutral + accent-alt set. Hunter green (--ac-accent) is reserved
# for goal lines, IQM markers, and other "data, pivot, δ" overlays per
# the brand's accent rule, so it does not appear on categorical lines.
TREATMENT_COLORS: Dict[str, str] = {
    "pixel": "#2B2B2E",      # graphite
    "symbolic": "#8A8A8E",   # muted
    "hybrid": "#B45A3C",     # rust (--ac-accent-alt)
    # legacy/aux observation types from older runs, faded so they
    # don't compete with the primary three on shared figures
    "multi_modal": "#A8A6A0",
    "screen_only": "#BCBAB4",
    "minimal": "#D4D2CC",
}

UNKNOWN_TREATMENT = "unknown"
UNKNOWN_COLOR = "#E6E4DE"  # mist

DEFAULT_BOOTSTRAP_REPS = 2_000
DEFAULT_FIGURE_DPI = 300


# ──────────────────────────────────────────────────────────────────────
# Treatment detection
# ──────────────────────────────────────────────────────────────────────


# Multi-word treatment names like ``multi_modal`` and ``screen_only`` are
# checked first via substring match, before we split on underscores; this
# preserves the underscore in the treatment name itself while still
# allowing single-word treatments to be picked up via token split.
_MULTI_WORD_TREATMENTS = tuple(t for t in KNOWN_TREATMENTS if "_" in t)
_SINGLE_WORD_TREATMENTS = tuple(t for t in KNOWN_TREATMENTS if "_" not in t)
_TOKEN_SPLIT_RE = re.compile(r"[-_.]+")
_BOUNDARY_RE = re.compile(r"(?:^|[-_.])({names})(?:[-_.]|$)")


def _multi_word_pattern():
    """Compiled regex matching multi-word treatments at token boundaries."""
    if not _MULTI_WORD_TREATMENTS:
        return None
    names = "|".join(re.escape(t) for t in _MULTI_WORD_TREATMENTS)
    return re.compile(_BOUNDARY_RE.pattern.format(names=names), re.IGNORECASE)


_MULTI_WORD_RE = _multi_word_pattern()


def detect_treatment(run_name: str) -> str:
    """Infer the observation treatment from a run name.

    Two-pass match against :data:`KNOWN_TREATMENTS`:

    1. Multi-word treatments (e.g. ``multi_modal``, ``screen_only``)
       are matched against the raw name with token-boundary regex,
       so the underscore inside the treatment name is preserved.
    2. Single-word treatments are matched against the underscore /
       dash / dot-split tokens of the name.

    Falls back to :data:`UNKNOWN_TREATMENT` when nothing matches.

    Examples::

        >>> detect_treatment("rppo-pixel-seed42")
        'pixel'
        >>> detect_treatment("RecurrentPPO_symbolic_42")
        'symbolic'
        >>> detect_treatment("hybrid-events-s7")
        'hybrid'
        >>> detect_treatment("multi_modal-rppo-s1")
        'multi_modal'
        >>> detect_treatment("screen_only_run")
        'screen_only'
        >>> detect_treatment("baseline-2026-01-01")
        'unknown'
    """
    if not run_name:
        return UNKNOWN_TREATMENT

    # Pass 1: multi-word treatments (preserve underscore inside the name)
    if _MULTI_WORD_RE is not None:
        m = _MULTI_WORD_RE.search(run_name)
        if m:
            return m.group(1).lower()

    # Pass 2: single-word treatments via token split
    tokens = [t.lower() for t in _TOKEN_SPLIT_RE.split(run_name) if t]
    for token in tokens:
        if token in _SINGLE_WORD_TREATMENTS:
            return token

    return UNKNOWN_TREATMENT


def group_runs_by_treatment(
    runs: Sequence[Any],
    name_attr: str = "name",
) -> Dict[str, List[Any]]:
    """Group runs by detected treatment.

    Args:
        runs: Iterable of objects with a ``name`` attribute (typically
            ``RunData`` from ``scripts.monitor``).
        name_attr: Name of the attribute holding the run name.

    Returns:
        Dict mapping treatment → list of runs, with deterministic
        treatment order (known treatments first, then unknowns).
    """
    grouped: Dict[str, List[Any]] = defaultdict(list)
    for run in runs:
        name = getattr(run, name_attr, "") or ""
        grouped[detect_treatment(str(name))].append(run)

    # Deterministic order: known treatments first (in declared order), then
    # any other keys alphabetically.
    ordered: Dict[str, List[Any]] = {}
    for t in KNOWN_TREATMENTS:
        if t in grouped:
            ordered[t] = grouped[t]
    for t in sorted(grouped):
        if t not in ordered:
            ordered[t] = grouped[t]
    return ordered


def treatment_color(treatment: str) -> str:
    """Return the canonical color for ``treatment``."""
    return TREATMENT_COLORS.get(treatment, UNKNOWN_COLOR)


def treatment_label(treatment: str) -> str:
    """Return the human-friendly display label for ``treatment``."""
    return TREATMENT_DISPLAY.get(treatment, treatment.title() if treatment else "—")


# ──────────────────────────────────────────────────────────────────────
# Per-run data extraction (RunData-shaped duck typing)
# ──────────────────────────────────────────────────────────────────────


def _episode_rewards(run: Any) -> Optional[np.ndarray]:
    """Return per-episode reward array from a run, or ``None``.

    Prefers ``monitor_df`` (more complete) over ``dashboard_state["episodes"]``.
    """
    df = getattr(run, "monitor_df", None)
    if df is not None and "r" in getattr(df, "columns", []):
        return np.asarray(df["r"], dtype=float)

    state = getattr(run, "dashboard_state", None)
    if state:
        episodes = state.get("episodes") or []
        rewards = [float(ep.get("reward", 0.0)) for ep in episodes]
        if rewards:
            return np.asarray(rewards, dtype=float)

    return None


def _flag_first_triggered(run: Any) -> Dict[str, int]:
    """Return ``{flag_name: first_episode}`` for a run, empty if missing."""
    state = getattr(run, "dashboard_state", None) or {}
    return dict(state.get("flag_first_triggered") or {})


def _max_badges(run: Any) -> int:
    """Return the highest badge count seen in any episode of this run."""
    state = getattr(run, "dashboard_state", None) or {}
    episodes = state.get("episodes") or []
    return int(max((ep.get("badges", 0) for ep in episodes), default=0))


def _max_maps(run: Any) -> int:
    """Return the count of unique maps visited across the run."""
    state = getattr(run, "dashboard_state", None) or {}
    counts = state.get("map_visit_counts") or {}
    return int(len(counts))


# ──────────────────────────────────────────────────────────────────────
# Learning curves with mean ± std bands
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LearningCurve:
    """Mean ± std reward over episodes for a single treatment."""

    treatment: str
    episode: np.ndarray  # 1D, length n
    mean: np.ndarray     # 1D, length n
    std: np.ndarray      # 1D, length n
    n_seeds: int

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "treatment": self.treatment,
                "episode": self.episode,
                "mean": self.mean,
                "std": self.std,
                "lower": self.mean - self.std,
                "upper": self.mean + self.std,
            }
        )


def learning_curves_with_bands(
    runs: Sequence[Any],
    smooth_window: int = 20,
    max_episodes: Optional[int] = None,
) -> List[LearningCurve]:
    """Compute per-treatment learning curves with mean ± std bands.

    For each treatment:

    1. Extract per-episode reward arrays from every seed (run).
    2. Apply a rolling mean of ``smooth_window`` episodes to smooth
       per-seed noise.
    3. Truncate to ``min(len(seed) for seed in treatment)`` so all
       seeds align — keeps the band honest at the right edge instead
       of letting one long-running seed pull the mean.
    4. Optional cap at ``max_episodes`` for fast plotting.
    5. Compute per-episode mean and std across seeds.

    Returns a list (one entry per treatment) sorted by
    :data:`KNOWN_TREATMENTS` order, with unknowns trailing.
    """
    if smooth_window < 1:
        smooth_window = 1

    grouped = group_runs_by_treatment(runs)
    out: List[LearningCurve] = []

    for treatment, treatment_runs in grouped.items():
        per_seed: List[np.ndarray] = []
        for run in treatment_runs:
            rewards = _episode_rewards(run)
            if rewards is None or rewards.size == 0:
                continue
            if smooth_window > 1:
                # Pandas rolling for simple boundary handling
                series = pd.Series(rewards)
                rewards = series.rolling(
                    smooth_window, min_periods=1
                ).mean().to_numpy()
            per_seed.append(rewards)

        if not per_seed:
            continue

        common_len = min(len(s) for s in per_seed)
        if max_episodes is not None:
            common_len = min(common_len, int(max_episodes))
        if common_len < 1:
            continue

        stacked = np.vstack([s[:common_len] for s in per_seed])  # (seeds, episodes)
        mean = stacked.mean(axis=0)
        # ddof=1 for sample std; collapses to 0 for a single seed via nan→0
        if stacked.shape[0] > 1:
            std = stacked.std(axis=0, ddof=1)
        else:
            std = np.zeros_like(mean)

        out.append(
            LearningCurve(
                treatment=treatment,
                episode=np.arange(1, common_len + 1, dtype=int),
                mean=mean,
                std=std,
                n_seeds=stacked.shape[0],
            )
        )

    return out


# ──────────────────────────────────────────────────────────────────────
# Aggregate metrics — IQM + bootstrap CIs via rliable
# ──────────────────────────────────────────────────────────────────────


@dataclass
class TreatmentSummary:
    """Aggregate stats for a single treatment.

    All values are computed over the per-seed best-reward distribution.
    """

    treatment: str
    n_seeds: int
    mean: float
    median: float
    iqm: float
    iqm_lo: float
    iqm_hi: float
    std: float


def _best_reward(run: Any) -> Optional[float]:
    """Best per-episode reward for a run, or ``None`` if unavailable."""
    rewards = _episode_rewards(run)
    if rewards is None or rewards.size == 0:
        return None
    return float(np.max(rewards))


def treatment_summary_table(
    runs: Sequence[Any],
    metric_fn: Optional[Any] = None,
    n_boot: int = DEFAULT_BOOTSTRAP_REPS,
    confidence: float = 0.95,
    rng_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Return a per-treatment summary DataFrame.

    Columns: ``treatment``, ``n_seeds``, ``mean``, ``median``, ``iqm``,
    ``iqm_lo``, ``iqm_hi``, ``std``.

    The IQM bootstrap uses a pure-numpy resample (no rliable runtime
    dependency at call time) so the function works in test environments
    without rliable installed.  When ``rliable`` is importable, results
    are mathematically identical (stratified-bootstrap IQM).

    Args:
        runs: Sequence of run objects.
        metric_fn: Callable mapping ``run -> float`` (or ``None`` to
            skip).  Defaults to :func:`_best_reward`.
        n_boot: Bootstrap resamples for the IQM CI.
        confidence: Confidence level for the bootstrap interval
            (e.g. 0.95 for [2.5%, 97.5%]).
        rng_seed: Optional RNG seed for reproducible CIs (handy in
            tests).
    """
    if metric_fn is None:
        metric_fn = _best_reward

    grouped = group_runs_by_treatment(runs)
    rows: List[TreatmentSummary] = []
    rng = np.random.default_rng(rng_seed)

    for treatment, treatment_runs in grouped.items():
        scores: List[float] = []
        for run in treatment_runs:
            v = metric_fn(run)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            scores.append(float(v))

        if not scores:
            continue

        arr = np.asarray(scores, dtype=float)
        iqm_point = _iqm(arr)
        if arr.size >= 2 and n_boot > 0:
            iqm_lo, iqm_hi = _iqm_bootstrap_ci(arr, n_boot, confidence, rng)
        else:
            iqm_lo = iqm_hi = iqm_point

        rows.append(
            TreatmentSummary(
                treatment=treatment,
                n_seeds=arr.size,
                mean=float(arr.mean()),
                median=float(np.median(arr)),
                iqm=iqm_point,
                iqm_lo=iqm_lo,
                iqm_hi=iqm_hi,
                std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            )
        )

    return pd.DataFrame([r.__dict__ for r in rows])


def _iqm(scores: np.ndarray) -> float:
    """Interquartile mean — mean of values between the 25th and 75th
    percentiles inclusive."""
    if scores.size == 0:
        return float("nan")
    if scores.size <= 2:
        return float(scores.mean())
    q25, q75 = np.percentile(scores, [25, 75])
    mask = (scores >= q25) & (scores <= q75)
    inner = scores[mask]
    return float(inner.mean()) if inner.size else float(scores.mean())


def _iqm_bootstrap_ci(
    scores: np.ndarray,
    n_boot: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple:
    """Percentile bootstrap CI for IQM.  Returns (lo, hi)."""
    n = scores.size
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(scores, size=n, replace=True)
        boot[i] = _iqm(sample)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.percentile(boot, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return float(lo), float(hi)


# ──────────────────────────────────────────────────────────────────────
# Milestone race chart — first episode each treatment hit each flag
# ──────────────────────────────────────────────────────────────────────


def milestone_first_episode(
    runs: Sequence[Any],
    flags: Optional[Iterable[str]] = None,
    aggregator: str = "median",
) -> pd.DataFrame:
    """Return when each treatment first triggered each flag.

    Per (treatment, flag), aggregate ``first_episode`` across seeds:

    * ``aggregator="median"`` — median first-episode (typical seed)
    * ``aggregator="min"``    — fastest seed (best case)
    * ``aggregator="mean"``   — average first-episode

    Seeds that never triggered a given flag are excluded from the
    aggregator.  If no seed ever triggered the flag, the cell is NaN.

    Args:
        runs: Run objects with ``dashboard_state["flag_first_triggered"]``.
        flags: Optional whitelist of flag names to include.  If omitted,
            uses the union of flags any run reported.
        aggregator: ``"median" | "min" | "mean"``.

    Returns:
        Long-form DataFrame with columns ``flag``, ``treatment``,
        ``first_episode``, ``n_seeds_triggered``.
    """
    if aggregator not in {"median", "min", "mean"}:
        raise ValueError(f"unknown aggregator {aggregator!r}")

    grouped = group_runs_by_treatment(runs)

    # Collect the union of flag names if none specified
    if flags is None:
        seen: set = set()
        for treatment_runs in grouped.values():
            for run in treatment_runs:
                seen.update(_flag_first_triggered(run).keys())
        flags = sorted(seen)
    flags = list(flags)

    rows: List[Dict[str, Any]] = []
    for treatment, treatment_runs in grouped.items():
        for flag in flags:
            episodes_per_seed: List[int] = []
            for run in treatment_runs:
                ft = _flag_first_triggered(run)
                ep = ft.get(flag)
                if ep is None:
                    continue
                try:
                    ep_int = int(ep)
                except (TypeError, ValueError):
                    continue
                if ep_int > 0:
                    episodes_per_seed.append(ep_int)

            if episodes_per_seed:
                arr = np.asarray(episodes_per_seed, dtype=float)
                if aggregator == "median":
                    val = float(np.median(arr))
                elif aggregator == "min":
                    val = float(np.min(arr))
                else:
                    val = float(np.mean(arr))
            else:
                val = float("nan")

            rows.append(
                {
                    "flag": flag,
                    "treatment": treatment,
                    "first_episode": val,
                    "n_seeds_triggered": len(episodes_per_seed),
                }
            )

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Final-window performance bars
# ──────────────────────────────────────────────────────────────────────


def final_performance(
    runs: Sequence[Any],
    window: int = 50,
) -> pd.DataFrame:
    """Per-treatment mean ± std reward over the last ``window`` episodes.

    Useful for the "final performance" bar chart with error bars.

    Returns columns: ``treatment``, ``n_seeds``, ``mean``, ``std``,
    ``window``.
    """
    grouped = group_runs_by_treatment(runs)
    rows: List[Dict[str, Any]] = []

    for treatment, treatment_runs in grouped.items():
        per_seed_means: List[float] = []
        for run in treatment_runs:
            rewards = _episode_rewards(run)
            if rewards is None or rewards.size == 0:
                continue
            tail = rewards[-window:] if rewards.size >= 1 else rewards
            per_seed_means.append(float(np.mean(tail)))

        if not per_seed_means:
            continue

        arr = np.asarray(per_seed_means, dtype=float)
        rows.append(
            {
                "treatment": treatment,
                "n_seeds": int(arr.size),
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "window": int(window),
            }
        )

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Publication-quality figure styling and export
# ──────────────────────────────────────────────────────────────────────


def setup_publication_style() -> None:
    """Configure matplotlib for publication-quality figures.

    Mirrors :func:`scripts.analyze._setup_matplotlib` so that figures
    exported from the comparison app match the rest of the paper.
    """
    import matplotlib  # local import — keep test-light
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": DEFAULT_FIGURE_DPI,
        "savefig.dpi": DEFAULT_FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def export_figure(
    fig: Any,
    path: str,
    dpi: int = DEFAULT_FIGURE_DPI,
) -> Path:
    """Save a matplotlib Figure to ``path`` at publication quality.

    Format is inferred from the extension.  PDF and SVG produce vector
    output (preferred for papers); PNG falls back to ``dpi`` rasterisation.
    Returns the resolved path.

    Raises:
        ValueError: if the extension is not one of pdf/svg/png.
    """
    p = Path(path).expanduser().resolve()
    ext = p.suffix.lower().lstrip(".")
    if ext not in {"pdf", "svg", "png"}:
        raise ValueError(
            f"export_figure only supports pdf/svg/png, got {ext!r}"
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p), dpi=dpi, bbox_inches="tight")
    return p


def plot_learning_curves(
    curves: Sequence[LearningCurve],
    title: str = "Learning curves",
    xlabel: str = "Episode",
    ylabel: str = "Episode reward (rolling mean)",
):
    """Render learning curves with mean ± std bands.

    Returns the matplotlib ``Figure`` for further customisation or
    export via :func:`export_figure`.
    """
    setup_publication_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for curve in curves:
        color = treatment_color(curve.treatment)
        label = f"{treatment_label(curve.treatment)} (n={curve.n_seeds})"
        ax.plot(curve.episode, curve.mean, color=color, label=label, linewidth=1.6)
        ax.fill_between(
            curve.episode,
            curve.mean - curve.std,
            curve.mean + curve.std,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if curves:
        ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig


def plot_final_performance_bars(
    summary: pd.DataFrame,
    title: str = "Final performance (last N episodes)",
):
    """Render final-window mean ± std as a bar chart.

    ``summary`` is the output of :func:`final_performance`.
    """
    setup_publication_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    if summary.empty:
        ax.set_title(title)
        ax.set_axis_off()
        return fig

    treatments = list(summary["treatment"])
    means = summary["mean"].to_numpy()
    stds = summary["std"].to_numpy()
    colors = [treatment_color(t) for t in treatments]
    labels = [treatment_label(t) for t in treatments]

    x = np.arange(len(treatments))
    ax.bar(x, means, yerr=stds, color=colors, capsize=4, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Mean reward (last {int(summary['window'].iloc[0])} eps)")
    ax.set_title(title)
    fig.tight_layout()
    return fig
