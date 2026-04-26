#!/usr/bin/env python3
"""
Pokemon Red RL — treatment comparison dashboard (AMC-79).

Side-by-side comparison of observation treatments (pixel / symbolic /
hybrid) across seeds.  Reads the same ``dashboard_state.json`` and
``monitor*.csv`` files that ``scripts/train.py`` writes, groups runs by
treatment, and renders learning curves with mean ± std bands, IQM with
bootstrap CIs, milestone-first-episode race, and a final-performance
bar chart.

Usage::

    streamlit run scripts/compare.py -- --runs-dir ./training_output

Run names that contain a known treatment token (``pixel``, ``symbolic``,
``hybrid``, ``multi_modal``, ``screen_only``, ``minimal``) are auto-grouped.
A "Treatment override" sidebar control lets you reassign mis-tagged runs
without renaming directories.

Most logic lives in :mod:`pokemon_red_ai.analysis.comparison` so it can
be unit-tested without launching Streamlit.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Make ``pokemon_red_ai`` and sibling scripts importable when invoked as
# a script.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pokemon_red_ai.analysis import comparison as cmp  # noqa: E402
from pokemon_red_ai.analysis.comparison import (  # noqa: E402
    KNOWN_TREATMENTS,
    LearningCurve,
    detect_treatment,
    final_performance,
    group_runs_by_treatment,
    learning_curves_with_bands,
    milestone_first_episode,
    plot_final_performance_bars,
    plot_learning_curves,
    treatment_label,
    treatment_summary_table,
)
from scripts.monitor import (  # noqa: E402
    RunData,
    discover_runs,
    load_run,
)


# ──────────────────────────────────────────────────────────────────────
# CLI parser
# ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-dir",
        type=str,
        default="./training_output",
        help="Root directory containing per-run save dirs.",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=20,
        help="Rolling-mean window for learning curves (episodes).",
    )
    p.add_argument(
        "--final-window",
        type=int,
        default=50,
        help="Episodes used for final-performance bar chart.",
    )
    p.add_argument(
        "--n-boot",
        type=int,
        default=2_000,
        help="Bootstrap resamples for IQM CI.",
    )
    return p


def _parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


# ──────────────────────────────────────────────────────────────────────
# Streamlit app
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    import streamlit as st  # imported here so tests don't need streamlit

    args = _parse_args()

    st.set_page_config(
        page_title="Pokemon Red RL — Treatment Comparison",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Pokemon Red RL — Treatment Comparison")
    st.caption(
        "Compare observation treatments (pixel / symbolic / hybrid) across "
        "seeds.  Auto-groups runs by name; override below if needed."
    )

    runs_root = Path(args.runs_dir).expanduser().resolve()
    candidates = discover_runs(runs_root)

    if not candidates:
        st.warning(
            f"No training runs found under `{runs_root}`. "
            "Pass `--runs-dir` pointing at a directory containing per-run "
            "save dirs from `scripts/train.py`."
        )
        return

    # ── Sidebar — run selection + treatment override ─────────────────
    with st.sidebar:
        st.header("Runs")
        st.write(f"Root: `{runs_root}`")

        run_names = [c.name for c in candidates]
        selected_names = st.multiselect(
            "Select runs to include",
            options=run_names,
            default=run_names,
        )

        st.subheader("Treatment override")
        st.caption(
            "Reassign runs whose names don't match a known treatment "
            "token (`pixel`, `symbolic`, `hybrid`, ...)."
        )

        overrides: Dict[str, str] = {}
        for name in selected_names:
            detected = detect_treatment(name)
            options = list(KNOWN_TREATMENTS) + ["unknown"]
            try:
                default_idx = options.index(detected)
            except ValueError:
                default_idx = options.index("unknown")
            overrides[name] = st.selectbox(
                f"{name}",
                options=options,
                index=default_idx,
                key=f"override_{name}",
            )

        st.divider()
        st.subheader("Plot config")
        smooth_window = st.slider(
            "Rolling-mean window",
            min_value=1,
            max_value=200,
            value=int(args.smooth_window),
        )
        final_window = st.slider(
            "Final-performance window (episodes)",
            min_value=5,
            max_value=500,
            value=int(args.final_window),
        )
        n_boot = st.slider(
            "Bootstrap resamples (IQM CI)",
            min_value=200,
            max_value=10_000,
            value=int(args.n_boot),
            step=200,
        )
        max_episodes = st.number_input(
            "Truncate to first N episodes (0 = no cap)",
            min_value=0,
            value=0,
            step=100,
        )

    if not selected_names:
        st.info("Pick at least one run in the sidebar.")
        return

    # Load and apply overrides
    selected_runs: List[RunData] = []
    for name in selected_names:
        candidate = next((c for c in candidates if c.name == name), None)
        if candidate is None:
            continue
        run = load_run(candidate)
        # Stash override on the run object so detect_treatment(_run.name) works
        # via group_runs_by_treatment — we wrap in a light shim that swaps
        # the name with a treatment-prefixed alias.  Simpler: just adjust
        # name attribute.
        forced = overrides.get(name)
        if forced and forced != "unknown":
            # Prepend treatment so detect_treatment matches the first token
            run.name = f"{forced}__{name}"
        selected_runs.append(run)

    grouped = group_runs_by_treatment(selected_runs)
    n_groups = sum(1 for k, v in grouped.items() if v)

    # ── Headline: treatment counts ───────────────────────────────────
    st.subheader("Treatment groups")
    if not grouped:
        st.info("No runs grouped — adjust treatment overrides in the sidebar.")
        return

    cols = st.columns(min(n_groups, 4) or 1)
    for i, (treatment, runs) in enumerate(grouped.items()):
        with cols[i % len(cols)]:
            st.metric(
                label=treatment_label(treatment),
                value=f"{len(runs)} seeds",
            )

    # ── Learning curves ──────────────────────────────────────────────
    st.subheader("Learning curves (mean ± std across seeds)")
    cap = int(max_episodes) if max_episodes > 0 else None
    curves: List[LearningCurve] = learning_curves_with_bands(
        selected_runs,
        smooth_window=int(smooth_window),
        max_episodes=cap,
    )
    if curves:
        # Render via the same matplotlib helper used for paper export so
        # the live view matches the publication output.
        fig = plot_learning_curves(curves)
        st.pyplot(fig)
        _figure_download_button(st, fig, basename="learning_curves")
    else:
        st.caption("No reward data available in selected runs.")

    # ── Aggregate metrics — IQM + CIs ────────────────────────────────
    st.subheader("Aggregate metrics (best per-seed reward)")
    summary_df = treatment_summary_table(
        selected_runs,
        n_boot=int(n_boot),
        rng_seed=42,
    )
    if summary_df.empty:
        st.caption("Not enough data for aggregate metrics.")
    else:
        # Display labels alongside the raw treatment column
        display = summary_df.copy()
        display.insert(
            1,
            "label",
            [treatment_label(t) for t in display["treatment"]],
        )
        st.dataframe(
            display.style.format(
                {
                    "mean": "{:.2f}",
                    "median": "{:.2f}",
                    "iqm": "{:.2f}",
                    "iqm_lo": "{:.2f}",
                    "iqm_hi": "{:.2f}",
                    "std": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "IQM = interquartile mean.  IQM CI is a percentile bootstrap "
            f"with {int(n_boot):,} resamples."
        )

    # ── Final performance bars ───────────────────────────────────────
    st.subheader(f"Final performance (last {int(final_window)} episodes)")
    final_df = final_performance(selected_runs, window=int(final_window))
    if final_df.empty:
        st.caption("Not enough data for the final-performance chart.")
    else:
        fig_bars = plot_final_performance_bars(final_df)
        st.pyplot(fig_bars)
        _figure_download_button(st, fig_bars, basename="final_performance")
        st.dataframe(
            final_df.assign(label=[treatment_label(t) for t in final_df["treatment"]]),
            use_container_width=True,
            hide_index=True,
        )

    # ── Milestone race chart ─────────────────────────────────────────
    st.subheader("Milestone first-episode (lower = faster)")
    aggregator = st.radio(
        "Aggregator across seeds",
        options=["median", "min", "mean"],
        index=0,
        horizontal=True,
    )
    milestones_df = milestone_first_episode(
        selected_runs,
        aggregator=aggregator,
    )
    if milestones_df.empty:
        st.caption("No flag data yet — needs `dashboard_state.flag_first_triggered`.")
    else:
        # Pivot to wide form for a clear cross-treatment view
        pivot = milestones_df.pivot(
            index="flag", columns="treatment", values="first_episode"
        )
        st.dataframe(pivot, use_container_width=True)

    # ── Per-run breakdown (collapsible) ──────────────────────────────
    with st.expander("Per-run details"):
        rows = []
        for treatment, runs in grouped.items():
            for run in runs:
                state = getattr(run, "dashboard_state", None) or {}
                rewards = cmp._episode_rewards(run)
                rows.append(
                    {
                        "treatment": treatment,
                        "run": run.name,
                        "episodes": run.episode_count,
                        "steps": run.num_timesteps,
                        "best_reward": run.best_reward,
                        "max_badges": cmp._max_badges(run),
                        "max_maps": cmp._max_maps(run),
                        "n_episode_rewards": (
                            int(rewards.size) if rewards is not None else 0
                        ),
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.caption("No runs loaded.")

    st.caption(
        "Export figures via the download buttons above for paper figures "
        "(PDF/SVG, 300 DPI).  See `configs/wandb_report_template.md` for "
        "a parallel W&B report layout."
    )


# ──────────────────────────────────────────────────────────────────────
# Figure download helper
# ──────────────────────────────────────────────────────────────────────


def _figure_download_button(st_module, fig, basename: str) -> None:
    """Render PDF/SVG/PNG download buttons for a matplotlib figure."""
    cols = st_module.columns(3)
    for i, fmt in enumerate(("pdf", "svg", "png")):
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=300, bbox_inches="tight")
        buf.seek(0)
        with cols[i]:
            st_module.download_button(
                label=f"Download {fmt.upper()}",
                data=buf,
                file_name=f"{basename}.{fmt}",
                mime=f"image/{fmt}" if fmt != "pdf" else "application/pdf",
                use_container_width=True,
                key=f"dl_{basename}_{fmt}",
            )


if __name__ == "__main__":
    main()
