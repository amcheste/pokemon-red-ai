#!/usr/bin/env python3
"""
Pokemon Red RL training monitor dashboard (local, no W&B required).

Reads the ``dashboard_state.json`` snapshots and Monitor CSVs written by
:class:`MonitoringCallback` (see ``pokemon_red_ai/training/callbacks.py``)
and renders a Streamlit dashboard that helps a researcher decide whether
a run is worth continuing.

Usage::

    # Point at one or more training output dirs (the ``--save-dir``
    # passed to ``scripts/train.py``).  Each dir must contain
    # ``dashboard_state.json`` and/or ``monitor.csv``.
    streamlit run scripts/monitor.py -- --runs-dir ./training_output

Features:
- Live reward curves (per-episode, from ``monitor.csv``)
- Map exploration grid (from ``dashboard_state.json``)
- Event flag checklist — 18 pre-registered Boulder Path flags
- Run comparison: pick multiple save dirs to overlay curves
- Optional TensorBoard scalar overlay (``rollout/ep_rew_mean``) if
  ``tensorboard`` is installed

The helper functions (``load_dashboard_state``, ``load_monitor_csv``,
``discover_runs``) are module-level so tests can exercise them without
launching Streamlit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Make ``pokemon_red_ai`` importable when invoked as a script.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pokemon_red_ai.game.event_flags import BOULDER_PATH_FLAGS  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────


@dataclass
class RunData:
    """All monitoring data for a single training run."""

    name: str
    path: Path
    dashboard_state: Optional[Dict[str, Any]] = None
    monitor_df: Optional[pd.DataFrame] = None
    tensorboard_scalars: Optional[Dict[str, pd.DataFrame]] = None

    @property
    def episode_count(self) -> int:
        if self.dashboard_state is not None:
            return int(self.dashboard_state.get("episode_count", 0))
        if self.monitor_df is not None:
            return len(self.monitor_df)
        return 0

    @property
    def num_timesteps(self) -> int:
        if self.dashboard_state is not None:
            return int(self.dashboard_state.get("num_timesteps", 0))
        return 0

    @property
    def best_reward(self) -> Optional[float]:
        if self.dashboard_state is not None:
            return self.dashboard_state.get("best_reward")
        if self.monitor_df is not None and "r" in self.monitor_df:
            return float(self.monitor_df["r"].max())
        return None


def discover_runs(runs_dir: Path) -> List[Path]:
    """Return subdirectories of ``runs_dir`` that look like training runs.

    A directory qualifies if it contains either a ``dashboard_state.json``
    or a ``monitor.csv.monitor.csv`` (SB3's default naming when passing
    a basename without ``.csv``).  The single ``runs_dir`` itself also
    qualifies if it looks like a run — useful when the user points at a
    single ``--save-dir`` directly.
    """
    if not runs_dir.exists():
        return []

    candidates: List[Path] = []
    if _is_run_dir(runs_dir):
        candidates.append(runs_dir)

    for child in sorted(runs_dir.iterdir()):
        if child.is_dir() and _is_run_dir(child):
            candidates.append(child)

    return candidates


def _is_run_dir(path: Path) -> bool:
    if (path / "dashboard_state.json").exists():
        return True
    # SB3 Monitor writes to ``<path>/monitor.monitor.csv`` when the
    # caller passes ``<path>/monitor`` as the filename.
    if any(path.glob("monitor*.csv")):
        return True
    if (path / "tensorboard").exists():
        return True
    return False


def load_dashboard_state(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load the JSON snapshot written by :class:`MonitoringCallback`.

    Returns ``None`` if the file is missing or malformed.
    """
    state_file = run_dir / "dashboard_state.json"
    if not state_file.exists():
        return None
    try:
        with state_file.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def load_monitor_csv(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load SB3 Monitor-wrapper CSV into a DataFrame.

    Adds a cumulative ``episode`` index column for plotting.
    Returns ``None`` if no monitor CSV exists or it is empty.
    """
    csv_paths = sorted(run_dir.glob("monitor*.csv"))
    if not csv_paths:
        return None

    try:
        df = pd.read_csv(csv_paths[0], comment="#")
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None

    if df.empty:
        return None

    df = df.reset_index(drop=True)
    df["episode"] = df.index + 1
    if "l" in df.columns:
        df["cumulative_steps"] = df["l"].cumsum()
    return df


def load_tensorboard_scalars(
    run_dir: Path,
    tags: Optional[List[str]] = None,
) -> Optional[Dict[str, pd.DataFrame]]:
    """Load scalar series from TensorBoard event files, if available.

    ``tags`` filters which scalars to return (default: all).  Returns
    ``None`` if TensorBoard isn't installed or no event files exist.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return None

    tb_dir = run_dir / "tensorboard"
    if not tb_dir.exists():
        return None

    event_files = list(tb_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        return None

    scalars: Dict[str, pd.DataFrame] = {}
    seen_dirs = {event_file.parent for event_file in event_files}
    for event_dir in seen_dirs:
        try:
            acc = EventAccumulator(str(event_dir))
            acc.Reload()
        except Exception:
            continue

        available = acc.Tags().get("scalars", [])
        wanted = [t for t in available if (tags is None or t in tags)]
        for tag in wanted:
            try:
                events = acc.Scalars(tag)
            except KeyError:
                continue
            df = pd.DataFrame(
                {
                    "step": [e.step for e in events],
                    "value": [e.value for e in events],
                    "wall_time": [e.wall_time for e in events],
                }
            )
            scalars[tag] = df

    return scalars or None


def load_run(run_dir: Path) -> RunData:
    """Load all available monitoring data for a run directory."""
    return RunData(
        name=run_dir.name or str(run_dir),
        path=run_dir,
        dashboard_state=load_dashboard_state(run_dir),
        monitor_df=load_monitor_csv(run_dir),
        tensorboard_scalars=load_tensorboard_scalars(run_dir),
    )


# ──────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────


def reward_curve_df(run: RunData) -> Optional[pd.DataFrame]:
    """Return a DataFrame with ``episode``, ``reward``, and a moving average."""
    if run.monitor_df is None or "r" not in run.monitor_df.columns:
        return None

    df = run.monitor_df[["episode", "r"]].copy()
    df = df.rename(columns={"r": "reward"})
    window = max(1, min(20, len(df) // 5 or 1))
    df["reward_avg"] = df["reward"].rolling(window=window, min_periods=1).mean()
    return df


def flag_progress_df(run: RunData) -> pd.DataFrame:
    """Return one row per pre-registered Boulder Path flag.

    Rows always include all 18 flags even if ``dashboard_state`` is
    unavailable — the dashboard should show the full checklist.
    """
    state = run.dashboard_state or {}
    trigger_counts: Dict[str, int] = state.get("flag_trigger_counts", {}) or {}
    first_triggered: Dict[str, int] = state.get("flag_first_triggered", {}) or {}

    rows = []
    for name in BOULDER_PATH_FLAGS:
        count = int(trigger_counts.get(name, 0))
        first = int(first_triggered.get(name, -1))
        rows.append(
            {
                "flag": name,
                "triggered": count > 0,
                "count": count,
                "first_episode": first if first > 0 else None,
            }
        )
    return pd.DataFrame(rows)


def map_heatmap_df(run: RunData) -> Optional[pd.DataFrame]:
    """Return a DataFrame with ``map_id`` and ``visit_count`` columns."""
    state = run.dashboard_state
    if not state:
        return None
    counts = state.get("map_visit_counts") or {}
    if not counts:
        return None
    rows = sorted(((int(k), int(v)) for k, v in counts.items()), key=lambda x: x[0])
    return pd.DataFrame(rows, columns=["map_id", "visit_count"])


def episode_breakdown_df(run: RunData) -> Optional[pd.DataFrame]:
    """Return per-episode rows from dashboard_state (or None)."""
    state = run.dashboard_state
    if not state:
        return None
    episodes = state.get("episodes") or []
    if not episodes:
        return None
    df = pd.DataFrame(episodes)
    if "triggered_flags" in df.columns:
        df["triggered_flags"] = df["triggered_flags"].apply(
            lambda v: ", ".join(v) if isinstance(v, list) else ""
        )
    return df


def run_summary(run: RunData) -> Dict[str, Any]:
    """Return a small dict of headline metrics for the sidebar."""
    summary: Dict[str, Any] = {
        "run": run.name,
        "episodes": run.episode_count,
        "steps": run.num_timesteps,
        "best_reward": run.best_reward,
    }

    flags = flag_progress_df(run)
    summary["flags_triggered"] = int(flags["triggered"].sum())
    summary["flags_total"] = len(flags)

    heatmap = map_heatmap_df(run)
    summary["maps_discovered"] = 0 if heatmap is None else len(heatmap)

    return summary


# ──────────────────────────────────────────────────────────────────────
# CLI parser — used both for ``streamlit run`` and headless testing
# ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-dir",
        type=str,
        default="./training_output",
        help="Root directory of training runs (or a single --save-dir).",
    )
    return p


def _parse_args() -> argparse.Namespace:
    """Parse Streamlit-forwarded CLI args (after ``--``)."""
    parser = build_parser()
    # Streamlit forwards the script's argv verbatim, so args after '--'
    # are available here.
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Streamlit app
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    import streamlit as st  # imported here so tests don't need streamlit

    args = _parse_args()

    st.set_page_config(
        page_title="Pokemon Red RL — Training Monitor",
        layout="wide",
    )
    st.title("Pokemon Red RL — Training Monitor")
    st.caption(
        "Live view of training runs — reward curves, map exploration, "
        "and the 18 Boulder-Path event flags."
    )

    runs_root = Path(args.runs_dir).expanduser().resolve()
    candidates = discover_runs(runs_root)

    if not candidates:
        st.warning(
            f"No training runs found under `{runs_root}`. "
            "Pass `--runs-dir` pointing at a `--save-dir` from "
            "`scripts/train.py`."
        )
        return

    with st.sidebar:
        st.header("Runs")
        st.write(f"Root: `{runs_root}`")

        default_selection = [candidates[-1].name]
        selected_names = st.multiselect(
            "Select runs to display",
            options=[c.name for c in candidates],
            default=default_selection,
        )

        st.caption(
            "Tip: select multiple runs to overlay reward curves "
            "(e.g. different seeds)."
        )

        refresh = st.button("Refresh data")
        if refresh:
            st.rerun()

    selected_runs: List[RunData] = [
        load_run(c) for c in candidates if c.name in selected_names
    ]

    if not selected_runs:
        st.info("Pick at least one run in the sidebar.")
        return

    # ── Headline metrics ─────────────────────────────────────────────
    st.subheader("Headline metrics")
    cols = st.columns(len(selected_runs))
    for col, run in zip(cols, selected_runs):
        summary = run_summary(run)
        with col:
            st.metric(label=run.name, value=f"{summary['episodes']} episodes")
            st.write(f"**Steps:** {summary['steps']:,}")
            st.write(
                f"**Best reward:** "
                f"{'—' if summary['best_reward'] is None else f'{summary["best_reward"]:.1f}'}"
            )
            st.write(
                f"**Flags:** {summary['flags_triggered']}/{summary['flags_total']}"
            )
            st.write(f"**Maps:** {summary['maps_discovered']}")

    # ── Reward curves ────────────────────────────────────────────────
    st.subheader("Episode reward")
    reward_series = {}
    for run in selected_runs:
        df = reward_curve_df(run)
        if df is not None:
            reward_series[run.name] = df.set_index("episode")["reward_avg"]

    if reward_series:
        chart_df = pd.concat(reward_series, axis=1)
        st.line_chart(chart_df, height=320)
    else:
        st.caption("No monitor.csv found for any selected run.")

    # ── Map exploration ──────────────────────────────────────────────
    st.subheader("Map exploration")
    map_cols = st.columns(len(selected_runs))
    for col, run in zip(map_cols, selected_runs):
        with col:
            st.caption(run.name)
            heatmap = map_heatmap_df(run)
            if heatmap is None:
                st.write("No map data yet.")
            else:
                st.bar_chart(
                    heatmap.set_index("map_id")["visit_count"],
                    height=220,
                )
                st.dataframe(heatmap, use_container_width=True, hide_index=True)

    # ── Event flag checklist ─────────────────────────────────────────
    st.subheader("Event flag checklist (Boulder Path)")
    flag_cols = st.columns(len(selected_runs))
    for col, run in zip(flag_cols, selected_runs):
        with col:
            st.caption(run.name)
            flags = flag_progress_df(run)
            triggered_count = int(flags["triggered"].sum())
            total = len(flags)
            st.progress(triggered_count / max(total, 1))
            st.write(f"{triggered_count}/{total} flags triggered")
            st.dataframe(
                flags[["flag", "triggered", "count", "first_episode"]],
                use_container_width=True,
                hide_index=True,
            )

    # ── Per-episode breakdown ────────────────────────────────────────
    st.subheader("Per-episode breakdown")
    for run in selected_runs:
        ep_df = episode_breakdown_df(run)
        if ep_df is None:
            continue
        with st.expander(f"{run.name} — last {len(ep_df)} episodes", expanded=False):
            st.dataframe(ep_df, use_container_width=True, hide_index=True)

    # ── Optional TB overlay ──────────────────────────────────────────
    tb_any = any(r.tensorboard_scalars for r in selected_runs)
    if tb_any:
        st.subheader("TensorBoard scalars (rollout/ep_rew_mean)")
        tb_series = {}
        for run in selected_runs:
            if not run.tensorboard_scalars:
                continue
            df = run.tensorboard_scalars.get("rollout/ep_rew_mean")
            if df is not None and not df.empty:
                tb_series[run.name] = df.set_index("step")["value"]
        if tb_series:
            st.line_chart(pd.concat(tb_series, axis=1), height=260)
        else:
            st.caption("No `rollout/ep_rew_mean` scalar yet.")

    st.caption(
        "Is this run worth continuing? Look for: rising reward curve, "
        "new maps appearing, and flags ticking on in order."
    )


if __name__ == "__main__":
    main()
