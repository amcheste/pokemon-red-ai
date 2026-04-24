"""
Unit tests for scripts/monitor.py — Streamlit training dashboard.

These tests exercise only the pure-data helpers (``load_dashboard_state``,
``load_monitor_csv``, ``discover_runs``, etc.).  The Streamlit UI is
imported lazily inside ``main()``, so the module can be imported without
Streamlit installed.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.monitor import (
    discover_runs,
    episode_breakdown_df,
    flag_progress_df,
    load_dashboard_state,
    load_monitor_csv,
    load_run,
    map_heatmap_df,
    reward_curve_df,
    run_summary,
)
from pokemon_red_ai.game.event_flags import BOULDER_PATH_FLAGS


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def run_dir(tmp_path):
    """Create a minimal run dir with dashboard_state.json and monitor.csv."""
    run = tmp_path / "rppo-events-seed42"
    run.mkdir()

    state = {
        "num_timesteps": 10_000,
        "episode_count": 3,
        "best_reward": 150.5,
        "map_visit_counts": {"1": 3, "2": 2, "40": 1},
        "flag_trigger_counts": {
            "EVENT_GOT_STARTER": 2,
            "EVENT_FOLLOWED_OAK_INTO_LAB": 3,
        },
        "flag_first_triggered": {
            "EVENT_GOT_STARTER": 2,
            "EVENT_FOLLOWED_OAK_INTO_LAB": 1,
        },
        "episodes": [
            {
                "episode": 1,
                "global_step": 1000,
                "reward": 5.0,
                "length": 200,
                "maps_visited": 1,
                "final_map": 1,
                "badges": 0,
                "event_flags_triggered": 1,
                "locations_visited": 20,
                "triggered_flags": ["EVENT_FOLLOWED_OAK_INTO_LAB"],
            },
            {
                "episode": 2,
                "global_step": 5000,
                "reward": 50.0,
                "length": 400,
                "maps_visited": 2,
                "final_map": 2,
                "badges": 0,
                "event_flags_triggered": 2,
                "locations_visited": 60,
                "triggered_flags": [
                    "EVENT_FOLLOWED_OAK_INTO_LAB",
                    "EVENT_GOT_STARTER",
                ],
            },
            {
                "episode": 3,
                "global_step": 10_000,
                "reward": 150.5,
                "length": 600,
                "maps_visited": 3,
                "final_map": 40,
                "badges": 1,
                "event_flags_triggered": 2,
                "locations_visited": 120,
                "triggered_flags": [
                    "EVENT_FOLLOWED_OAK_INTO_LAB",
                    "EVENT_GOT_STARTER",
                ],
            },
        ],
    }
    (run / "dashboard_state.json").write_text(json.dumps(state))

    # Minimal SB3 Monitor CSV (one comment line + header + rows)
    csv_content = (
        '#{"t_start": 1.0, "env_id": null}\n'
        "r,l,t\n"
        "5.0,200,1.5\n"
        "50.0,400,2.5\n"
        "150.5,600,3.5\n"
    )
    (run / "monitor.monitor.csv").write_text(csv_content)

    return run


# ──────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────


class TestDiscoverRuns:
    def test_finds_single_run_passed_directly(self, run_dir):
        runs = discover_runs(run_dir)
        assert run_dir in runs

    def test_finds_child_runs_under_root(self, tmp_path, run_dir):
        # run_dir is a child of tmp_path
        runs = discover_runs(tmp_path)
        assert run_dir in runs

    def test_missing_dir(self, tmp_path):
        assert discover_runs(tmp_path / "nope") == []

    def test_ignores_non_run_dirs(self, tmp_path):
        (tmp_path / "empty").mkdir()
        (tmp_path / "random_file.txt").write_text("hi")
        assert discover_runs(tmp_path) == []


# ──────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────


class TestLoaders:
    def test_load_dashboard_state(self, run_dir):
        state = load_dashboard_state(run_dir)
        assert state is not None
        assert state["num_timesteps"] == 10_000
        assert state["episode_count"] == 3

    def test_load_dashboard_state_missing(self, tmp_path):
        assert load_dashboard_state(tmp_path) is None

    def test_load_dashboard_state_malformed(self, tmp_path):
        (tmp_path / "dashboard_state.json").write_text("{not json")
        assert load_dashboard_state(tmp_path) is None

    def test_load_monitor_csv(self, run_dir):
        df = load_monitor_csv(run_dir)
        assert df is not None
        assert len(df) == 3
        assert "r" in df.columns
        assert list(df["episode"]) == [1, 2, 3]
        assert df["cumulative_steps"].iloc[-1] == 1200

    def test_load_monitor_csv_missing(self, tmp_path):
        assert load_monitor_csv(tmp_path) is None

    def test_load_run_aggregates(self, run_dir):
        run = load_run(run_dir)
        assert run.episode_count == 3
        assert run.num_timesteps == 10_000
        assert run.best_reward == pytest.approx(150.5)


# ──────────────────────────────────────────────────────────────────────
# DataFrame builders
# ──────────────────────────────────────────────────────────────────────


class TestDataFrameBuilders:
    def test_reward_curve_df(self, run_dir):
        run = load_run(run_dir)
        df = reward_curve_df(run)
        assert df is not None
        assert "reward" in df.columns
        assert "reward_avg" in df.columns
        assert len(df) == 3

    def test_reward_curve_df_no_monitor(self, tmp_path):
        run = load_run(tmp_path)
        assert reward_curve_df(run) is None

    def test_flag_progress_df_shows_all_18(self, run_dir):
        run = load_run(run_dir)
        df = flag_progress_df(run)
        assert len(df) == 18
        # Must include the untriggered flags too
        triggered = df[df["triggered"]]["flag"].tolist()
        assert "EVENT_GOT_STARTER" in triggered
        assert "EVENT_BEAT_BROCK" not in triggered

    def test_flag_progress_df_without_state(self, tmp_path):
        run = load_run(tmp_path)
        df = flag_progress_df(run)
        # Still lists all 18 flags even with no data
        assert len(df) == 18
        assert df["triggered"].sum() == 0

    def test_map_heatmap_df(self, run_dir):
        run = load_run(run_dir)
        df = map_heatmap_df(run)
        assert df is not None
        assert len(df) == 3
        # Sorted by map_id
        assert list(df["map_id"]) == [1, 2, 40]

    def test_episode_breakdown_df(self, run_dir):
        run = load_run(run_dir)
        df = episode_breakdown_df(run)
        assert df is not None
        assert len(df) == 3
        assert "triggered_flags" in df.columns
        # triggered_flags list is stringified for display
        assert all(isinstance(v, str) for v in df["triggered_flags"])

    def test_run_summary(self, run_dir):
        run = load_run(run_dir)
        summary = run_summary(run)
        assert summary["episodes"] == 3
        assert summary["steps"] == 10_000
        assert summary["best_reward"] == pytest.approx(150.5)
        assert summary["flags_triggered"] == 2
        assert summary["flags_total"] == len(BOULDER_PATH_FLAGS)
        assert summary["maps_discovered"] == 3


# ──────────────────────────────────────────────────────────────────────
# CLI parser
# ──────────────────────────────────────────────────────────────────────


class TestBuildParser:
    def test_default_runs_dir(self):
        from scripts.monitor import build_parser
        args = build_parser().parse_args([])
        assert args.runs_dir == "./training_output"

    def test_custom_runs_dir(self):
        from scripts.monitor import build_parser
        args = build_parser().parse_args(["--runs-dir", "/tmp/foo"])
        assert args.runs_dir == "/tmp/foo"
