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
    level_curve_df,
    load_dashboard_state,
    load_monitor_csv,
    load_run,
    map_heatmap_df,
    reward_breakdown_df,
    reward_curve_df,
    reward_summary_df,
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
        # AMC-76 enhanced fields
        "level_history": [5, 6, 8],
        "pokemon_count_history": [1, 1, 2],
        "money_history": [0, 500, 1200],
        "reward_component_summary": {
            "exploration": 12.5,
            "badge": 100.0,
            "time": -3.0,
            "level": 25.0,
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
                "player_level": 5,
                "pokemon_count": 1,
                "money": 0,
                "reward_components": {
                    "exploration": 3.0,
                    "time": -1.0,
                },
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
                "player_level": 6,
                "pokemon_count": 1,
                "money": 500,
                "reward_components": {
                    "exploration": 5.0,
                    "level": 10.0,
                    "time": -1.5,
                },
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
                "player_level": 8,
                "pokemon_count": 2,
                "money": 1200,
                "reward_components": {
                    "exploration": 4.5,
                    "badge": 100.0,
                    "level": 15.0,
                    "time": -0.5,
                },
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

    def test_flag_progress_df_shows_all_flags(self, run_dir):
        from pokemon_red_ai.game.event_flags import NUM_BOULDER_FLAGS
        run = load_run(run_dir)
        df = flag_progress_df(run)
        assert len(df) == NUM_BOULDER_FLAGS
        # Must include the untriggered flags too
        triggered = df[df["triggered"]]["flag"].tolist()
        assert "EVENT_GOT_STARTER" in triggered
        assert "EVENT_BEAT_BROCK" not in triggered

    def test_flag_progress_df_without_state(self, tmp_path):
        from pokemon_red_ai.game.event_flags import NUM_BOULDER_FLAGS
        run = load_run(tmp_path)
        df = flag_progress_df(run)
        # Still lists every pre-registered flag even with no data
        assert len(df) == NUM_BOULDER_FLAGS
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
        assert summary["max_level"] == 8
        assert summary["max_pokemon"] == 2


# ──────────────────────────────────────────────────────────────────────
# AMC-77: New helper functions
# ──────────────────────────────────────────────────────────────────────


class TestLevelCurveDF:
    def test_returns_all_columns(self, run_dir):
        run = load_run(run_dir)
        df = level_curve_df(run)
        assert df is not None
        assert list(df.columns) == ["episode", "player_level", "pokemon_count", "money"]
        assert len(df) == 3

    def test_episode_numbers_start_at_1(self, run_dir):
        run = load_run(run_dir)
        df = level_curve_df(run)
        assert list(df["episode"]) == [1, 2, 3]

    def test_values_match_fixture(self, run_dir):
        run = load_run(run_dir)
        df = level_curve_df(run)
        assert list(df["player_level"]) == [5, 6, 8]
        assert list(df["pokemon_count"]) == [1, 1, 2]
        assert list(df["money"]) == [0, 500, 1200]

    def test_returns_none_without_state(self, tmp_path):
        run = load_run(tmp_path)
        assert level_curve_df(run) is None

    def test_returns_none_without_level_history(self, tmp_path):
        """dashboard_state exists but has no level_history."""
        (tmp_path / "dashboard_state.json").write_text(
            json.dumps({"num_timesteps": 100, "episode_count": 1})
        )
        run = load_run(tmp_path)
        assert level_curve_df(run) is None

    def test_fills_zeros_when_pokemon_money_shorter(self, tmp_path):
        """level_history is longer than pokemon/money — should pad with 0."""
        state = {
            "level_history": [5, 6, 7],
            "pokemon_count_history": [1],  # shorter
            "money_history": [],  # empty
        }
        (tmp_path / "dashboard_state.json").write_text(json.dumps(state))
        run = load_run(tmp_path)
        df = level_curve_df(run)
        assert df is not None
        assert len(df) == 3
        assert list(df["pokemon_count"]) == [1, 0, 0]
        assert list(df["money"]) == [0, 0, 0]


class TestRewardBreakdownDF:
    def test_returns_wide_dataframe(self, run_dir):
        run = load_run(run_dir)
        df = reward_breakdown_df(run)
        assert df is not None
        assert df.index.name == "episode"
        # All unique components across episodes
        assert "exploration" in df.columns
        assert "badge" in df.columns
        assert "time" in df.columns
        assert "level" in df.columns

    def test_rows_match_episodes(self, run_dir):
        run = load_run(run_dir)
        df = reward_breakdown_df(run)
        assert list(df.index) == [1, 2, 3]

    def test_fills_missing_components_with_zero(self, run_dir):
        """Episode 1 has no 'badge' component — should be 0.0."""
        run = load_run(run_dir)
        df = reward_breakdown_df(run)
        assert df.loc[1, "badge"] == 0.0

    def test_values_match_fixture(self, run_dir):
        run = load_run(run_dir)
        df = reward_breakdown_df(run)
        assert df.loc[3, "badge"] == pytest.approx(100.0)
        assert df.loc[2, "level"] == pytest.approx(10.0)
        assert df.loc[1, "exploration"] == pytest.approx(3.0)

    def test_returns_none_without_state(self, tmp_path):
        run = load_run(tmp_path)
        assert reward_breakdown_df(run) is None

    def test_returns_none_without_reward_components(self, tmp_path):
        """Episodes exist but none have reward_components."""
        state = {
            "episodes": [
                {"episode": 1, "reward": 5.0},
                {"episode": 2, "reward": 10.0},
            ]
        }
        (tmp_path / "dashboard_state.json").write_text(json.dumps(state))
        run = load_run(tmp_path)
        assert reward_breakdown_df(run) is None


class TestRewardSummaryDF:
    def test_returns_two_columns(self, run_dir):
        run = load_run(run_dir)
        df = reward_summary_df(run)
        assert df is not None
        assert list(df.columns) == ["component", "mean_value"]

    def test_components_sorted_alphabetically(self, run_dir):
        run = load_run(run_dir)
        df = reward_summary_df(run)
        assert list(df["component"]) == sorted(df["component"])

    def test_values_match_fixture(self, run_dir):
        run = load_run(run_dir)
        df = reward_summary_df(run)
        badge_row = df[df["component"] == "badge"]
        assert badge_row["mean_value"].iloc[0] == pytest.approx(100.0)

    def test_returns_none_without_state(self, tmp_path):
        run = load_run(tmp_path)
        assert reward_summary_df(run) is None

    def test_returns_none_with_empty_summary(self, tmp_path):
        state = {"reward_component_summary": {}}
        (tmp_path / "dashboard_state.json").write_text(json.dumps(state))
        run = load_run(tmp_path)
        assert reward_summary_df(run) is None


class TestRefreshIntervalArg:
    def test_default_refresh_interval(self):
        from scripts.monitor import build_parser
        args = build_parser().parse_args([])
        assert args.refresh_interval == 30

    def test_custom_refresh_interval(self):
        from scripts.monitor import build_parser
        args = build_parser().parse_args(["--refresh-interval", "10"])
        assert args.refresh_interval == 10


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
