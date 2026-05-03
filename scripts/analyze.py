#!/usr/bin/env python3
"""
Publication-quality statistical analysis using rliable.

Implements the pre-registered analysis protocol from ``paper/analysis_plan.md``:

- **Point estimates:** IQM (not raw mean) per Agarwal et al. 2021
- **Confidence intervals:** 95% stratified bootstrap with 2,000 resamples
- **Significance test:** Probability of Improvement > 0.75
- **Plots:** aggregate metrics, performance profiles, probability-of-improvement
  heatmap, sample-efficiency curves

Input data comes from ``scripts/eval.py`` JSON outputs, organized as::

    eval_results/
    +-- pixel/
    |   +-- seed_0.json
    |   +-- seed_1.json
    +-- symbolic/
    |   +-- seed_0.json
    |   +-- seed_1.json
    +-- hybrid/
        +-- seed_0.json
        +-- seed_1.json

Or from a single CSV file (see ``--csv``).

Usage::

    # Analyze from eval JSON directory
    python scripts/analyze.py --results-dir eval_results/

    # Analyze from CSV
    python scripts/analyze.py --csv paper/notebooks/all_results.csv

    # Generate only specific plots
    python scripts/analyze.py --results-dir eval_results/ --plots iqm poi

Run ``python scripts/analyze.py --help`` for all options.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pokemon_red_ai.analysis import (
    TREATMENT_COLORS,
    TREATMENT_DISPLAY,
    UNKNOWN_COLOR,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

# Metrics we extract from eval.py JSON and feed to rliable
METRIC_KEYS = {
    "brock_win_rate": "Brock Win Rate",
    "mean_return": "Mean Return",
    "mean_event_flags_triggered": "Event Flags",
    "unique_maps_visited": "Unique Maps",
}

# Pre-registered analysis constants (analysis_plan.md S6)
BOOTSTRAP_REPS = 2_000
CONFIDENCE_LEVEL = 0.95

# Figure defaults
FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────


def load_eval_jsons(results_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load eval JSON files organized by treatment subdirectories.

    Expected structure::

        results_dir/
        +-- pixel/
        |   +-- seed_0.json    (output of scripts/eval.py)
        |   +-- seed_1.json
        +-- symbolic/
        |   +-- seed_0.json
        +-- hybrid/
            +-- seed_0.json

    Returns:
        Dict mapping treatment name to list of eval metric dicts.
    """
    results: Dict[str, List[Dict[str, Any]]] = {}

    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for treatment_dir in sorted(results_dir.iterdir()):
        if not treatment_dir.is_dir():
            continue

        treatment = treatment_dir.name
        evals = []

        for json_file in sorted(treatment_dir.glob("*.json")):
            with open(json_file) as f:
                data = json.load(f)
            data["_source_file"] = str(json_file)
            evals.append(data)
            logger.debug(f"Loaded {json_file} ({len(data)} keys)")

        if evals:
            results[treatment] = evals
            logger.info(
                f"  {treatment}: {len(evals)} seeds loaded"
            )

    if not results:
        raise ValueError(
            f"No eval JSONs found in {results_dir}. "
            f"Expected subdirectories like pixel/, symbolic/, hybrid/ "
            f"containing *.json files from scripts/eval.py."
        )

    return results


def load_csv(csv_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load results from a single CSV file.

    Expected columns: ``treatment``, ``seed``, ``brock_win_rate``,
    ``mean_return``, ``mean_event_flags_triggered``, ``unique_maps_visited``.

    Returns:
        Dict mapping treatment name to list of metric dicts (one per seed).
    """
    import csv

    results: Dict[str, List[Dict[str, Any]]] = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            treatment = row["treatment"]
            metrics = {}
            for key in METRIC_KEYS:
                raw = row.get(key)
                if raw is not None:
                    try:
                        metrics[key] = float(raw)
                    except ValueError:
                        metrics[key] = None
                else:
                    metrics[key] = None

            # Carry through extra metadata
            metrics["seed"] = row.get("seed")
            metrics["_source_file"] = str(csv_path)

            results.setdefault(treatment, []).append(metrics)

    logger.info(f"Loaded CSV: {csv_path}")
    for t, evals in results.items():
        logger.info(f"  {t}: {len(evals)} seeds")

    return results


def load_sample_efficiency_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    """
    Load sample efficiency data from a CSV.

    Expected columns: ``treatment``, ``seed``, ``timestep``,
    ``brock_win_rate`` (or whichever metric you want curves for).

    Returns:
        Dict with keys:
        - 'frames': 1-D array of timestep values
        - One key per treatment: (n_seeds, n_frames) array of scores
    """
    import csv

    raw: Dict[str, Dict[int, Dict[int, float]]] = {}  # treatment -> seed -> step -> score
    all_steps = set()

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            treatment = row["treatment"]
            seed = int(row["seed"])
            step = int(row["timestep"])
            score = float(row["brock_win_rate"])

            raw.setdefault(treatment, {}).setdefault(seed, {})[step] = score
            all_steps.add(step)

    frames = np.array(sorted(all_steps))
    result: Dict[str, np.ndarray] = {"frames": frames}

    for treatment, seed_data in raw.items():
        seeds = sorted(seed_data.keys())
        matrix = np.zeros((len(seeds), len(frames)))
        for i, seed in enumerate(seeds):
            for j, step in enumerate(frames):
                matrix[i, j] = seed_data[seed].get(step, 0.0)
        result[treatment] = matrix

    return result


# ──────────────────────────────────────────────────────────────────────
# Score matrix construction
# ──────────────────────────────────────────────────────────────────────


def build_score_matrix(
    results: Dict[str, List[Dict[str, Any]]],
    metric: str = "brock_win_rate",
    normalize: bool = False,
    max_score: Optional[float] = None,
    min_score: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Convert loaded results into the ``scores_dict`` format rliable expects.

    Args:
        results: Output of ``load_eval_jsons`` or ``load_csv``.
        metric: Which metric key to extract.
        normalize: If True, min-max normalize scores to [0, 1].
        max_score: Max score for normalization (auto-detected if None).
        min_score: Min score for normalization.

    Returns:
        Dict mapping treatment name to (n_seeds, 1) array.
        rliable's second axis is task index; we have a single task.
    """
    scores_dict: Dict[str, np.ndarray] = {}

    for treatment, evals in results.items():
        values = []
        for e in evals:
            val = e.get(metric)
            if val is None:
                logger.warning(
                    f"Missing metric '{metric}' in {e.get('_source_file', '?')}, "
                    f"using 0.0"
                )
                val = 0.0
            values.append(float(val))

        arr = np.array(values).reshape(-1, 1)  # (n_seeds, 1)

        if normalize:
            if max_score is None:
                max_score = float(arr.max()) if arr.max() > min_score else 1.0
            arr = (arr - min_score) / (max_score - min_score)
            arr = np.clip(arr, 0.0, 1.0)

        scores_dict[treatment] = arr
        logger.debug(
            f"  {treatment}: {len(values)} seeds, "
            f"values={[f'{v:.3f}' for v in values]}"
        )

    return scores_dict


# ──────────────────────────────────────────────────────────────────────
# rliable analysis functions
# ──────────────────────────────────────────────────────────────────────


def compute_aggregate_metrics(
    scores_dict: Dict[str, np.ndarray],
    reps: int = BOOTSTRAP_REPS,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, np.ndarray]]]:
    """
    Compute IQM, mean, median, and optimality gap with bootstrap CIs.

    Returns:
        (point_estimates, interval_estimates) — each a dict mapping
        metric name to {treatment: value} or {treatment: [lo, hi]}.
    """
    from rliable import library as rly
    from rliable import metrics

    aggregate_func = lambda x: np.array([
        metrics.aggregate_iqm(x),
        metrics.aggregate_mean(x),
        metrics.aggregate_median(x),
        metrics.aggregate_optimality_gap(x),
    ])

    aggregate_scores, aggregate_cis = rly.get_interval_estimates(
        scores_dict,
        aggregate_func,
        reps=reps,
    )

    metric_names = ["IQM", "Mean", "Median", "Optimality Gap"]

    point_estimates: Dict[str, Dict[str, float]] = {}
    interval_estimates: Dict[str, Dict[str, np.ndarray]] = {}

    for i, name in enumerate(metric_names):
        point_estimates[name] = {}
        interval_estimates[name] = {}
        for treatment in scores_dict:
            point_estimates[name][treatment] = float(
                aggregate_scores[treatment][i]
            )
            interval_estimates[name][treatment] = aggregate_cis[treatment][:, i]

    return point_estimates, interval_estimates


def compute_probability_of_improvement(
    scores_dict: Dict[str, np.ndarray],
    reps: int = BOOTSTRAP_REPS,
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], np.ndarray]]:
    """
    Compute pairwise Probability of Improvement between all treatments.

    A treatment A is considered superior to B if P(A > B) > 0.75
    and the 95% CI excludes 0.5 (analysis_plan.md S6).

    Uses stratified bootstrap to compute CIs on the POI estimate,
    since ``metrics.probability_of_improvement(scores_x, scores_y)``
    takes two separate arrays and cannot go through
    ``get_interval_estimates``.

    Returns:
        (poi_point, poi_ci): dicts mapping (treatment_a, treatment_b)
        to P(A > B) point estimate and [lo, hi] CI.
    """
    from rliable import metrics

    treatments = sorted(scores_dict.keys())
    poi_point: Dict[Tuple[str, str], float] = {}
    poi_ci: Dict[Tuple[str, str], np.ndarray] = {}

    alpha = (1 - CONFIDENCE_LEVEL) / 2  # 0.025 for 95% CI

    for i, t_a in enumerate(treatments):
        for j, t_b in enumerate(treatments):
            if i >= j:
                continue

            scores_a = scores_dict[t_a]  # (n_seeds_a, 1)
            scores_b = scores_dict[t_b]  # (n_seeds_b, 1)

            # Point estimate
            poi_ab = float(metrics.probability_of_improvement(scores_a, scores_b))
            poi_ba = 1.0 - poi_ab

            # Bootstrap CI by resampling seed indices
            n_a, n_b = scores_a.shape[0], scores_b.shape[0]
            boot_pois = np.zeros(reps)
            rng = np.random.RandomState(42)

            for r in range(reps):
                idx_a = rng.choice(n_a, size=n_a, replace=True)
                idx_b = rng.choice(n_b, size=n_b, replace=True)
                boot_pois[r] = metrics.probability_of_improvement(
                    scores_a[idx_a], scores_b[idx_b],
                )

            ci_lo = float(np.percentile(boot_pois, 100 * alpha))
            ci_hi = float(np.percentile(boot_pois, 100 * (1 - alpha)))

            poi_point[(t_a, t_b)] = poi_ab
            poi_ci[(t_a, t_b)] = np.array([ci_lo, ci_hi])

            poi_point[(t_b, t_a)] = poi_ba
            poi_ci[(t_b, t_a)] = np.array([1.0 - ci_hi, 1.0 - ci_lo])

    return poi_point, poi_ci


def compute_performance_profiles(
    scores_dict: Dict[str, np.ndarray],
    tau_range: Optional[np.ndarray] = None,
    reps: int = BOOTSTRAP_REPS,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Compute performance profiles (fraction of runs above threshold tau).

    Returns:
        (taus, profile_point, profile_ci): threshold array, point estimates,
        and bootstrap CIs per treatment.
    """
    from rliable import library as rly

    if tau_range is None:
        tau_range = np.linspace(0.0, 1.0, 51)

    profiles, profile_cis = rly.create_performance_profile(
        scores_dict,
        tau_list=tau_range,
        reps=reps,
    )

    return tau_range, profiles, profile_cis


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────


def _setup_matplotlib() -> None:
    """Configure matplotlib for publication-quality figures."""
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _get_color(treatment: str) -> str:
    """Get the color for a treatment, with a fallback."""
    return TREATMENT_COLORS.get(treatment, UNKNOWN_COLOR)


def _get_label(treatment: str) -> str:
    """Get the display label for a treatment."""
    return TREATMENT_DISPLAY.get(treatment, treatment.title())


def plot_aggregate_metrics(
    point_estimates: Dict[str, Dict[str, float]],
    interval_estimates: Dict[str, Dict[str, np.ndarray]],
    metric_name: str = "IQM",
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of aggregate metric with bootstrap CIs.

    Produces the main result figure showing IQM across treatments.
    """
    treatments = sorted(point_estimates[metric_name].keys())
    values = [point_estimates[metric_name][t] for t in treatments]
    cis = [interval_estimates[metric_name][t] for t in treatments]

    colors = [_get_color(t) for t in treatments]
    labels = [_get_label(t) for t in treatments]

    # Compute error bars (distance from point estimate to CI bounds)
    errors = np.array([
        [v - ci[0], ci[1] - v] for v, ci in zip(values, cis)
    ]).T

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(treatments))

    bars = ax.bar(
        x, values, yerr=errors,
        color=colors, edgecolor="white", linewidth=0.5,
        capsize=4, error_kw={"linewidth": 1.5},
        width=0.6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric_name)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(
            f"{metric_name} (Brock Win Rate)\n"
            f"95% Bootstrap CI, {BOOTSTRAP_REPS:,} resamples"
        )

    # Annotate values
    for bar, val, ci in zip(bars, values, cis):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ci[1] + 0.02,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, format=FIGURE_FORMAT)
        logger.info(f"Saved: {output_path}")

    return fig


def plot_performance_profiles(
    taus: np.ndarray,
    profiles: Dict[str, np.ndarray],
    profile_cis: Dict[str, np.ndarray],
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Performance profile plot: fraction of runs achieving score >= tau.
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))

    for treatment in sorted(profiles.keys()):
        color = _get_color(treatment)
        label = _get_label(treatment)

        profile = profiles[treatment]
        ci = profile_cis[treatment]

        ax.plot(taus, profile, color=color, label=label, linewidth=2)
        ax.fill_between(
            taus, ci[0], ci[1],
            color=color, alpha=0.15,
        )

    ax.set_xlabel(r"Normalized Score ($\tau$)")
    ax.set_ylabel(r"Fraction of runs with score $\geq \tau$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Performance Profiles")

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, format=FIGURE_FORMAT)
        logger.info(f"Saved: {output_path}")

    return fig


def plot_probability_of_improvement(
    poi_point: Dict[Tuple[str, str], float],
    poi_ci: Dict[Tuple[str, str], np.ndarray],
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of pairwise Probability of Improvement.

    Cells where P(A > B) > 0.75 with CI excluding 0.5 are highlighted
    per the pre-registered significance threshold.
    """
    treatments = sorted({t for pair in poi_point for t in pair})
    n = len(treatments)

    matrix = np.full((n, n), 0.5)
    significant = np.zeros((n, n), dtype=bool)

    for i, t_a in enumerate(treatments):
        for j, t_b in enumerate(treatments):
            if i == j:
                continue
            key = (t_a, t_b)
            if key in poi_point:
                matrix[i, j] = poi_point[key]
                ci = poi_ci[key]
                # Significant if P(A > B) > 0.75 and CI excludes 0.5
                significant[i, j] = (
                    poi_point[key] > 0.75 and ci[0] > 0.5
                )

    fig, ax = plt.subplots(figsize=(4.5, 4))

    im = ax.imshow(
        matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0,
        aspect="equal",
    )

    labels = [_get_label(t) for t in treatments]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=11)
                continue

            val = matrix[i, j]
            sig_marker = "*" if significant[i, j] else ""
            color = "white" if (val > 0.7 or val < 0.3) else "black"

            ax.text(
                j, i, f"{val:.2f}{sig_marker}",
                ha="center", va="center",
                fontsize=10, color=color, fontweight="bold",
            )

    ax.set_xlabel("B (column)")
    ax.set_ylabel("A (row)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("P(A > B)")

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Probability of Improvement\n(* = significant, P > 0.75, CI excl. 0.5)")

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, format=FIGURE_FORMAT)
        logger.info(f"Saved: {output_path}")

    return fig


def plot_sample_efficiency(
    frames: np.ndarray,
    scores_dict: Dict[str, np.ndarray],
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    reps: int = BOOTSTRAP_REPS,
) -> plt.Figure:
    """
    Sample efficiency curves with bootstrap CIs.

    Shows IQM at each checkpoint across training, revealing which
    treatment learns faster.

    Args:
        frames: 1-D array of timestep values.
        scores_dict: Dict mapping treatment to (n_seeds, n_frames) array.
        output_path: Where to save the figure.
        title: Optional title override.
        reps: Bootstrap resamples.
    """
    from rliable import library as rly
    from rliable import metrics

    fig, ax = plt.subplots(figsize=(6, 4))

    for treatment in sorted(scores_dict.keys()):
        color = _get_color(treatment)
        label = _get_label(treatment)
        data = scores_dict[treatment]  # (n_seeds, n_frames)

        # Compute IQM and CI at each frame
        iqm_values = np.zeros(len(frames))
        ci_lo = np.zeros(len(frames))
        ci_hi = np.zeros(len(frames))

        for t_idx in range(len(frames)):
            col = data[:, t_idx:t_idx + 1]  # (n_seeds, 1)
            single_scores = {treatment: col}

            try:
                agg, agg_ci = rly.get_interval_estimates(
                    single_scores,
                    metrics.aggregate_iqm,
                    reps=reps,
                )
                iqm_values[t_idx] = float(agg[treatment])
                ci_lo[t_idx] = float(agg_ci[treatment][0])
                ci_hi[t_idx] = float(agg_ci[treatment][1])
            except Exception:
                iqm_values[t_idx] = np.mean(col)
                ci_lo[t_idx] = np.percentile(col, 2.5)
                ci_hi[t_idx] = np.percentile(col, 97.5)

        ax.plot(frames, iqm_values, color=color, label=label, linewidth=2)
        ax.fill_between(frames, ci_lo, ci_hi, color=color, alpha=0.15)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("IQM (Brock Win Rate)")
    ax.legend()

    if title:
        ax.set_title(title)
    else:
        ax.set_title(
            f"Sample Efficiency\n"
            f"IQM with 95% Bootstrap CI ({reps:,} resamples)"
        )

    # Format x-axis with M suffix
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M")
    )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, format=FIGURE_FORMAT)
        logger.info(f"Saved: {output_path}")

    return fig


# ──────────────────────────────────────────────────────────────────────
# Summary report
# ──────────────────────────────────────────────────────────────────────


def print_summary_report(
    results: Dict[str, List[Dict[str, Any]]],
    point_estimates: Dict[str, Dict[str, float]],
    interval_estimates: Dict[str, Dict[str, np.ndarray]],
    poi_point: Optional[Dict[Tuple[str, str], float]] = None,
    poi_ci: Optional[Dict[Tuple[str, str], np.ndarray]] = None,
) -> str:
    """
    Print and return a human-readable summary of all analysis results.
    """
    lines: List[str] = []
    sep = "=" * 65

    lines.append(sep)
    lines.append("RLIABLE ANALYSIS REPORT")
    lines.append(f"Pre-registered protocol: analysis_plan.md S6")
    lines.append(f"Bootstrap resamples: {BOOTSTRAP_REPS:,}")
    lines.append(sep)

    # Data summary
    lines.append("\nDATA SUMMARY")
    lines.append("-" * 40)
    for treatment, evals in sorted(results.items()):
        label = _get_label(treatment)
        lines.append(f"  {label:12s}: {len(evals)} seeds")

    # Aggregate metrics
    lines.append(f"\nAGGREGATE METRICS (Brock Win Rate)")
    lines.append("-" * 40)
    for metric_name in ["IQM", "Mean", "Median", "Optimality Gap"]:
        if metric_name not in point_estimates:
            continue
        lines.append(f"\n  {metric_name}:")
        for treatment in sorted(point_estimates[metric_name].keys()):
            label = _get_label(treatment)
            val = point_estimates[metric_name][treatment]
            ci = interval_estimates[metric_name][treatment]
            lines.append(
                f"    {label:12s}: {val:.4f}  "
                f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            )

    # Probability of improvement
    if poi_point:
        lines.append(f"\nPROBABILITY OF IMPROVEMENT")
        lines.append("-" * 40)
        for (t_a, t_b), val in sorted(poi_point.items()):
            ci = poi_ci[(t_a, t_b)]
            label_a = _get_label(t_a)
            label_b = _get_label(t_b)
            sig = ""
            if val > 0.75 and ci[0] > 0.5:
                sig = " *SIGNIFICANT*"
            lines.append(
                f"  P({label_a} > {label_b}): "
                f"{val:.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]{sig}"
            )

    lines.append(f"\n{sep}")
    lines.append("* Significant = P > 0.75 with 95% CI excluding 0.5")
    lines.append(sep)

    report = "\n".join(lines)
    print(report)
    return report


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Publication-quality statistical analysis of Pokemon Red AI "
            "experiments using rliable (Agarwal et al. 2021)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input data (mutually exclusive)
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--results-dir", type=Path,
        help="Directory containing treatment subdirectories with eval JSONs.",
    )
    input_group.add_argument(
        "--csv", type=Path,
        help="CSV file with columns: treatment, seed, brock_win_rate, ...",
    )

    # Sample efficiency (optional, separate data source)
    p.add_argument(
        "--efficiency-csv", type=Path, default=None,
        help=(
            "CSV file for sample efficiency curves. Columns: "
            "treatment, seed, timestep, brock_win_rate."
        ),
    )

    # Metric to analyze
    p.add_argument(
        "--metric", type=str, default="brock_win_rate",
        choices=list(METRIC_KEYS.keys()),
        help="Primary metric to analyze.",
    )

    # Which plots to generate
    p.add_argument(
        "--plots", nargs="+", default=["all"],
        choices=["all", "iqm", "profiles", "poi", "efficiency"],
        help="Which plots to generate.",
    )

    # Normalization
    p.add_argument(
        "--normalize", action="store_true",
        help="Min-max normalize scores to [0, 1] for rliable.",
    )
    p.add_argument(
        "--max-score", type=float, default=None,
        help="Maximum score for normalization (auto if omitted).",
    )

    # Output
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("paper/figures"),
        help="Directory for output figures.",
    )
    p.add_argument(
        "--report-file", type=Path, default=None,
        help="Write the text report to a file.",
    )
    p.add_argument(
        "--format", type=str, default="pdf",
        choices=["pdf", "png", "svg"],
        help="Figure output format.",
    )

    # Bootstrap
    p.add_argument(
        "--reps", type=int, default=BOOTSTRAP_REPS,
        help="Number of bootstrap resamples.",
    )

    # Misc
    p.add_argument(
        "--no-show", action="store_true",
        help="Do not display figures (useful in CI/headless).",
    )
    p.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v INFO, -vv DEBUG).",
    )

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Logging
    level = {0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(name)s %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Override module-level constants from CLI
    global BOOTSTRAP_REPS, FIGURE_FORMAT
    BOOTSTRAP_REPS = args.reps
    FIGURE_FORMAT = args.format

    _setup_matplotlib()
    if args.no_show:
        matplotlib.use("Agg")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────
    logger.info("Loading evaluation results...")

    if args.results_dir:
        results = load_eval_jsons(args.results_dir)
    else:
        results = load_csv(args.csv)

    # ── Build score matrix ───────────────────────────────────────────
    logger.info(f"Building score matrix for metric: {args.metric}")

    scores_dict = build_score_matrix(
        results,
        metric=args.metric,
        normalize=args.normalize,
        max_score=args.max_score,
    )

    # Validate minimum seed count
    for treatment, arr in scores_dict.items():
        n_seeds = arr.shape[0]
        if n_seeds < 3:
            logger.warning(
                f"{treatment}: only {n_seeds} seed(s). "
                f"rliable recommends >= 5 seeds for reliable CIs. "
                f"Results may be unstable."
            )

    # ── Compute aggregate metrics ────────────────────────────────────
    plots_to_make = set(args.plots)
    if "all" in plots_to_make:
        plots_to_make = {"iqm", "profiles", "poi", "efficiency"}

    logger.info("Computing aggregate metrics (IQM, Mean, Median)...")
    point_estimates, interval_estimates = compute_aggregate_metrics(
        scores_dict, reps=args.reps,
    )

    # ── Probability of improvement ───────────────────────────────────
    poi_point, poi_ci = None, None
    if len(scores_dict) >= 2:
        logger.info("Computing probability of improvement...")
        poi_point, poi_ci = compute_probability_of_improvement(
            scores_dict, reps=args.reps,
        )

    # ── Summary report ───────────────────────────────────────────────
    report = print_summary_report(
        results, point_estimates, interval_estimates,
        poi_point, poi_ci,
    )

    if args.report_file:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(report)
        logger.info(f"Report written to {args.report_file}")

    # ── Generate plots ───────────────────────────────────────────────
    metric_label = METRIC_KEYS.get(args.metric, args.metric)
    suffix = f".{args.format}"

    if "iqm" in plots_to_make:
        logger.info("Plotting aggregate metrics...")
        plot_aggregate_metrics(
            point_estimates, interval_estimates,
            metric_name="IQM",
            output_path=args.output_dir / f"aggregate_iqm{suffix}",
        )

    if "profiles" in plots_to_make:
        logger.info("Computing and plotting performance profiles...")
        taus, profiles, profile_cis = compute_performance_profiles(
            scores_dict, reps=args.reps,
        )
        plot_performance_profiles(
            taus, profiles, profile_cis,
            output_path=args.output_dir / f"performance_profiles{suffix}",
        )

    if "poi" in plots_to_make and poi_point:
        logger.info("Plotting probability of improvement heatmap...")
        plot_probability_of_improvement(
            poi_point, poi_ci,
            output_path=args.output_dir / f"probability_of_improvement{suffix}",
        )

    if "efficiency" in plots_to_make and args.efficiency_csv:
        logger.info("Loading and plotting sample efficiency curves...")
        efficiency_data = load_sample_efficiency_csv(args.efficiency_csv)
        frames = efficiency_data.pop("frames")
        plot_sample_efficiency(
            frames, efficiency_data,
            output_path=args.output_dir / f"sample_efficiency{suffix}",
            reps=args.reps,
        )

    # Show figures if not headless
    if not args.no_show:
        plt.show()

    logger.info("Analysis complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
