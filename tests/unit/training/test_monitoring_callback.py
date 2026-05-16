"""
Unit tests for MonitoringCallback.

Covers the monitoring-specific extensions that aren't exercised by
``test_wandb_callback.py``: per-episode recording, map heatmap, event
flag tracking, screen captures, and dashboard JSON snapshots.
"""

import json
import os
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from pokemon_red_ai.training.callbacks import (
    MONITORED_INFO_KEYS,
    MonitoringCallback,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_wandb():
    wandb = MagicMock()
    wandb.log = Mock()
    wandb.Image = Mock(side_effect=lambda arr, caption=None: {"img": arr, "caption": caption})
    wandb.Table = Mock(side_effect=lambda columns: _FakeTable(columns))
    wandb.Artifact = Mock(return_value=Mock())
    wandb.log_artifact = Mock()
    return wandb


class _FakeTable:
    """Minimal stand-in for wandb.Table so we can inspect logged rows."""

    def __init__(self, columns):
        self.columns = list(columns)
        self.data = []

    def add_data(self, *row):
        self.data.append(row)


@pytest.fixture
def monitoring_cb(tmp_path, mock_wandb):
    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        cb = MonitoringCallback(
            save_freq=10_000,
            save_path=str(tmp_path),
            screen_capture_freq=2,
            verbose=0,
        )
    cb._wandb = mock_wandb

    # SB3 exposes ``training_env`` as a property that reads
    # ``self.model.get_env()``, so we inject the fake env via the model.
    fake_env = Mock()
    fake_env.env_method = Mock(
        return_value=[np.full((72, 80, 3), 128, dtype=np.uint8)]
    )

    model = Mock()
    model.ep_info_buffer = []
    model.save = Mock()
    model.get_env = Mock(return_value=fake_env)
    cb.model = model
    cb.num_timesteps = 0
    cb.n_calls = 0
    cb.locals = {}
    cb.globals = {}
    return cb


# ──────────────────────────────────────────────────────────────────────
# Info keyword contract
# ──────────────────────────────────────────────────────────────────────


def test_monitored_info_keys_present():
    """The tuple includes the info keys the callback actually consumes."""
    expected = {
        "maps_visited",
        "badges_earned",
        "event_progress",
        "current_map",
        "unique_maps_list",
    }
    assert expected.issubset(set(MONITORED_INFO_KEYS))


# ──────────────────────────────────────────────────────────────────────
# Per-episode recording via _on_step
# ──────────────────────────────────────────────────────────────────────


def _make_step_info(
    reward=10.0,
    length=100,
    unique_maps=(1, 2),
    triggered_flags=("EVENT_GOT_STARTER",),
    current_map=2,
    badges=1,
    locations=42,
    player_level=5,
    pokemon_count=1,
    money=3000,
    reward_components=None,
):
    info = {
        "episode": {"r": reward, "l": length},
        "unique_maps_list": list(unique_maps),
        "event_progress": {
            "flags_triggered": len(triggered_flags),
            "triggered_names": list(triggered_flags),
        },
        "current_map": current_map,
        "badges_earned": badges,
        "locations_visited": locations,
        "player_level": player_level,
        "pokemon_count": pokemon_count,
        "money": money,
    }
    if reward_components is not None:
        info["reward_components"] = reward_components
    return info


class TestOnStepEpisodeTracking:
    def test_records_completed_episode(self, monitoring_cb):
        monitoring_cb.locals = {
            "dones": [True],
            "infos": [_make_step_info()],
        }
        monitoring_cb.num_timesteps = 500

        monitoring_cb._on_step()

        assert monitoring_cb._episode_count == 1
        row = monitoring_cb._episode_rows[0]
        assert row["reward"] == pytest.approx(10.0)
        assert row["maps_visited"] == 2
        assert row["final_map"] == 2
        assert row["badges"] == 1
        assert row["event_flags_triggered"] == 1
        assert row["triggered_flags"] == ["EVENT_GOT_STARTER"]

    def test_ignores_non_terminal_steps(self, monitoring_cb):
        monitoring_cb.locals = {
            "dones": [False],
            "infos": [_make_step_info()],
        }
        monitoring_cb._on_step()
        assert monitoring_cb._episode_count == 0

    def test_updates_map_counts_across_episodes(self, monitoring_cb):
        monitoring_cb.locals = {
            "dones": [True],
            "infos": [_make_step_info(unique_maps=(1, 2))],
        }
        monitoring_cb._on_step()

        monitoring_cb.locals = {
            "dones": [True],
            "infos": [_make_step_info(unique_maps=(2, 3))],
        }
        monitoring_cb._on_step()

        assert monitoring_cb._map_visit_counts[1] == 1
        assert monitoring_cb._map_visit_counts[2] == 2
        assert monitoring_cb._map_visit_counts[3] == 1

    def test_first_triggered_sticks(self, monitoring_cb):
        monitoring_cb.locals = {
            "dones": [True],
            "infos": [_make_step_info(triggered_flags=("EVENT_GOT_STARTER",))],
        }
        monitoring_cb._on_step()

        # Episode 2 triggers same flag — first_triggered should stay at 1
        monitoring_cb.locals = {
            "dones": [True],
            "infos": [_make_step_info(triggered_flags=("EVENT_GOT_STARTER",))],
        }
        monitoring_cb._on_step()

        assert monitoring_cb._flag_trigger_counts["EVENT_GOT_STARTER"] == 2
        assert monitoring_cb._flag_first_triggered["EVENT_GOT_STARTER"] == 1

    def test_missing_info_does_not_crash(self, monitoring_cb):
        monitoring_cb.locals = {"dones": [True], "infos": [{}]}
        monitoring_cb._on_step()
        assert monitoring_cb._episode_count == 1
        row = monitoring_cb._episode_rows[0]
        assert row["reward"] == 0.0
        assert row["maps_visited"] == 0


# ──────────────────────────────────────────────────────────────────────
# Screen captures
# ──────────────────────────────────────────────────────────────────────


class TestScreenCapture:
    def test_fires_on_schedule(self, monitoring_cb, mock_wandb):
        monitoring_cb.screen_capture_freq = 2

        # Episode 1 — no capture
        monitoring_cb.locals = {"dones": [True], "infos": [_make_step_info()]}
        monitoring_cb._on_step()

        # Episode 2 — capture fires
        monitoring_cb.locals = {"dones": [True], "infos": [_make_step_info()]}
        monitoring_cb._on_step()

        image_logs = [
            call for call in mock_wandb.log.call_args_list
            if "game/screen" in call.args[0]
        ]
        assert len(image_logs) == 1
        mock_wandb.Image.assert_called()

    def test_disable_with_zero_freq(self, tmp_path, mock_wandb):
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cb = MonitoringCallback(
                save_path=str(tmp_path),
                screen_capture_freq=0,
                verbose=0,
            )
        cb._wandb = mock_wandb

        fake_env = Mock()
        fake_env.env_method = Mock()
        cb.model = Mock(
            ep_info_buffer=[],
            save=Mock(),
            get_env=Mock(return_value=fake_env),
        )
        cb.num_timesteps = 0
        cb.n_calls = 0
        cb.locals = {"dones": [True], "infos": [_make_step_info()]}
        cb.globals = {}

        cb._on_step()

        # env_method should never be called when freq=0
        fake_env.env_method.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
# Extras logging (heatmap / flags / episode table)
# ──────────────────────────────────────────────────────────────────────


class TestRolloutEndExtras:
    def _populate(self, cb):
        cb.locals = {"dones": [True], "infos": [_make_step_info()]}
        cb._on_step()

    def test_logs_map_heatmap_table(self, monitoring_cb, mock_wandb):
        self._populate(monitoring_cb)
        monitoring_cb.num_timesteps = 2048
        monitoring_cb._on_rollout_end()

        logged = _merge_all_logs(mock_wandb)
        assert "game/map_heatmap_table" in logged
        table = logged["game/map_heatmap_table"]
        assert set(table.columns) == {"map_id", "visit_count"}
        assert any(row[0] == 1 for row in table.data)
        assert any(row[0] == 2 for row in table.data)

    def test_logs_flag_progress_for_all_flags(self, monitoring_cb, mock_wandb):
        from pokemon_red_ai.game.event_flags import NUM_BOULDER_FLAGS
        self._populate(monitoring_cb)
        monitoring_cb.num_timesteps = 2048
        monitoring_cb._on_rollout_end()

        logged = _merge_all_logs(mock_wandb)
        assert "game/flag_progress" in logged
        table = logged["game/flag_progress"]
        assert len(table.data) == NUM_BOULDER_FLAGS  # All pre-registered flags listed
        assert "game/unique_flags_ever" in logged
        assert logged["game/unique_flags_ever"] == 1

    def test_logs_episode_breakdown(self, monitoring_cb, mock_wandb):
        self._populate(monitoring_cb)
        monitoring_cb.num_timesteps = 2048
        monitoring_cb._on_rollout_end()

        logged = _merge_all_logs(mock_wandb)
        assert "game/episodes" in logged
        table = logged["game/episodes"]
        assert len(table.data) == 1
        # Columns match expected order (including AMC-76 additions)
        expected_cols = {
            "episode", "global_step", "reward", "length",
            "maps_visited", "final_map", "badges",
            "event_flags_triggered", "locations_visited",
            "player_level", "pokemon_count", "money",
        }
        assert expected_cols == set(table.columns)


# ──────────────────────────────────────────────────────────────────────
# Dashboard JSON snapshot
# ──────────────────────────────────────────────────────────────────────


class TestDashboardStateSnapshot:
    def test_writes_snapshot_on_rollout_end(self, monitoring_cb):
        monitoring_cb.locals = {"dones": [True], "infos": [_make_step_info()]}
        monitoring_cb._on_step()
        monitoring_cb.num_timesteps = 2048
        monitoring_cb._on_rollout_end()

        assert os.path.exists(monitoring_cb.dashboard_state_path)
        with open(monitoring_cb.dashboard_state_path) as fh:
            snapshot = json.load(fh)

        assert snapshot["episode_count"] == 1
        assert snapshot["num_timesteps"] == 2048
        assert snapshot["map_visit_counts"]["1"] == 1
        assert snapshot["flag_trigger_counts"]["EVENT_GOT_STARTER"] == 1
        assert snapshot["flag_first_triggered"]["EVENT_GOT_STARTER"] == 1
        assert len(snapshot["episodes"]) == 1

    def test_snapshot_contains_all_flags(self, monitoring_cb):
        from pokemon_red_ai.game.event_flags import NUM_BOULDER_FLAGS
        monitoring_cb.num_timesteps = 100
        monitoring_cb._on_rollout_end()

        with open(monitoring_cb.dashboard_state_path) as fh:
            snapshot = json.load(fh)

        # Even before any episode, snapshot lists every pre-registered flag.
        assert len(snapshot["flag_trigger_counts"]) == NUM_BOULDER_FLAGS
        assert all(v == 0 for v in snapshot["flag_trigger_counts"].values())

    def test_write_failure_is_silent(self, monitoring_cb, tmp_path):
        # Point at a path that cannot be written (directory that doesn't
        # exist and can't be created because the parent is a file)
        monitoring_cb.dashboard_state_path = str(tmp_path / "not_a_dir" / "state.json")
        monitoring_cb.num_timesteps = 100

        # Pre-create a FILE at the parent so replace() fails on write
        (tmp_path / "not_a_dir").write_text("I am not a dir")

        # Should not raise
        monitoring_cb._write_dashboard_state()


# ──────────────────────────────────────────────────────────────────────
# Reward component accumulation (AMC-76)
# ──────────────────────────────────────────────────────────────────────


class TestRewardComponentAccumulation:
    """Tests for per-step reward component tracking."""

    def test_accumulates_across_steps(self, monitoring_cb):
        """Components should sum over the episode, not just last step."""
        # Step 1: non-terminal, has reward components
        monitoring_cb.locals = {
            "dones": [False],
            "infos": [{"reward_components": {"exploration": 5.0, "time": -0.01}}],
        }
        monitoring_cb._on_step()

        # Step 2: non-terminal
        monitoring_cb.locals = {
            "dones": [False],
            "infos": [{"reward_components": {"exploration": 3.0, "time": -0.01}}],
        }
        monitoring_cb._on_step()

        # Step 3: terminal — triggers _record_episode
        monitoring_cb.locals = {
            "dones": [True],
            "infos": [_make_step_info(
                reward_components={"exploration": 2.0, "time": -0.01, "badge": 150.0},
            )],
        }
        monitoring_cb._on_step()

        assert monitoring_cb._episode_count == 1
        rc = monitoring_cb._episode_reward_components[0]
        assert rc["exploration"] == pytest.approx(10.0)  # 5 + 3 + 2
        assert rc["time"] == pytest.approx(-0.03)  # -0.01 * 3
        assert rc["badge"] == pytest.approx(150.0)

    def test_accumulator_resets_after_episode(self, monitoring_cb):
        """Each episode gets its own accumulation, no bleed."""
        # Episode 1
        monitoring_cb.locals = {
            "dones": [True],
            "infos": [_make_step_info(
                reward_components={"exploration": 5.0},
            )],
        }
        monitoring_cb._on_step()

        # Episode 2 (different components)
        monitoring_cb.locals = {
            "dones": [True],
            "infos": [_make_step_info(
                reward_components={"badge": 100.0},
            )],
        }
        monitoring_cb._on_step()

        assert len(monitoring_cb._episode_reward_components) == 2
        assert monitoring_cb._episode_reward_components[0] == {"exploration": 5.0}
        assert monitoring_cb._episode_reward_components[1] == {"badge": 100.0}

    def test_handles_missing_components_gracefully(self, monitoring_cb):
        """Steps without reward_components should not crash."""
        monitoring_cb.locals = {
            "dones": [True],
            "infos": [_make_step_info()],  # no reward_components
        }
        monitoring_cb._on_step()
        assert monitoring_cb._episode_count == 1
        # No components recorded, but episode still logged
        row = monitoring_cb._episode_rows[0]
        assert row["reward_components"] == {}

    def test_multi_env_isolation(self, monitoring_cb):
        """Accumulators for different envs stay separate."""
        # Both envs step, only env 0 terminates
        monitoring_cb.locals = {
            "dones": [True, False],
            "infos": [
                _make_step_info(reward_components={"exploration": 5.0}),
                {"reward_components": {"exploration": 99.0}},
            ],
        }
        monitoring_cb._on_step()

        # Env 0's episode should NOT include env 1's components
        assert monitoring_cb._episode_count == 1
        rc = monitoring_cb._episode_reward_components[0]
        assert rc["exploration"] == pytest.approx(5.0)

        # Env 1's accumulator should still be active
        assert 1 in monitoring_cb._reward_accumulators
        assert monitoring_cb._reward_accumulators[1]["exploration"] == pytest.approx(99.0)


# ──────────────────────────────────────────────────────────────────────
# Game state metrics (AMC-76)
# ──────────────────────────────────────────────────────────────────────


class TestGameStateMetrics:
    """Tests for level, money, and pokemon count tracking."""

    def test_records_player_level(self, monitoring_cb):
        monitoring_cb.locals = {
            "dones": [True],
            "infos": [_make_step_info(player_level=7)],
        }
        monitoring_cb._on_step()

        assert monitoring_cb._level_history == [7]
        row = monitoring_cb._episode_rows[0]
        assert row["player_level"] == 7

    def test_records_pokemon_count(self, monitoring_cb):
        monitoring_cb.locals = {
            "dones": [True],
            "infos": [_make_step_info(pokemon_count=3)],
        }
        monitoring_cb._on_step()

        assert monitoring_cb._pokemon_count_history == [3]
        row = monitoring_cb._episode_rows[0]
        assert row["pokemon_count"] == 3

    def test_records_money(self, monitoring_cb):
        monitoring_cb.locals = {
            "dones": [True],
            "infos": [_make_step_info(money=5000)],
        }
        monitoring_cb._on_step()

        assert monitoring_cb._money_history == [5000]

    def test_logs_level_curves_to_wandb(self, monitoring_cb, mock_wandb):
        # Populate multiple episodes with increasing levels
        for level in [5, 6, 7, 8]:
            monitoring_cb.locals = {
                "dones": [True],
                "infos": [_make_step_info(player_level=level)],
            }
            monitoring_cb._on_step()

        monitoring_cb.num_timesteps = 4096
        monitoring_cb._on_rollout_end()

        logged = _merge_all_logs(mock_wandb)
        assert "game/player_level_mean" in logged
        assert "game/player_level_max" in logged
        assert logged["game/player_level_max"] == 8
        assert logged["game/player_level_mean"] == pytest.approx(6.5)

    def test_logs_reward_breakdown_to_wandb(self, monitoring_cb, mock_wandb):
        monitoring_cb.locals = {
            "dones": [True],
            "infos": [_make_step_info(
                reward_components={"exploration": 10.0, "time": -0.5, "badge": 150.0},
            )],
        }
        monitoring_cb._on_step()

        monitoring_cb.num_timesteps = 2048
        monitoring_cb._on_rollout_end()

        logged = _merge_all_logs(mock_wandb)
        assert "reward/exploration_mean" in logged
        assert "reward/badge_mean" in logged
        assert "reward/time_mean" in logged
        assert "reward/total_components_mean" in logged
        assert logged["reward/badge_mean"] == pytest.approx(150.0)

    def test_missing_game_state_uses_defaults(self, monitoring_cb):
        """Info dict without level/money/count should use 0."""
        monitoring_cb.locals = {
            "dones": [True],
            "infos": [{"episode": {"r": 1.0, "l": 10}}],
        }
        monitoring_cb._on_step()

        row = monitoring_cb._episode_rows[0]
        assert row["player_level"] == 0
        assert row["pokemon_count"] == 0
        assert row["money"] == 0


# ──────────────────────────────────────────────────────────────────────
# W&B define_metric (AMC-76)
# ──────────────────────────────────────────────────────────────────────


class TestWandbMetricDefinition:
    """Tests for W&B panel organisation."""

    def test_define_metric_called_on_init(self, mock_wandb, tmp_path):
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cb = MonitoringCallback(
                save_path=str(tmp_path),
                verbose=0,
            )

        mock_wandb.define_metric.assert_called()
        calls = [str(c) for c in mock_wandb.define_metric.call_args_list]
        # Should have calls for global_step, episode/*, game/*, reward/*
        assert any("global_step" in c for c in calls)
        assert any("episode/*" in c for c in calls)
        assert any("game/*" in c for c in calls)
        assert any("reward/*" in c for c in calls)

    def test_define_metric_failure_is_silent(self, mock_wandb, tmp_path):
        mock_wandb.define_metric.side_effect = AttributeError("old wandb")
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            # Should not raise
            cb = MonitoringCallback(
                save_path=str(tmp_path),
                verbose=0,
            )
        assert cb is not None


# ──────────────────────────────────────────────────────────────────────
# Enhanced dashboard snapshot (AMC-76)
# ──────────────────────────────────────────────────────────────────────


class TestEnhancedDashboardSnapshot:
    """Tests for enhanced dashboard JSON state."""

    def test_snapshot_includes_level_history(self, monitoring_cb):
        monitoring_cb.locals = {
            "dones": [True],
            "infos": [_make_step_info(player_level=8)],
        }
        monitoring_cb._on_step()
        monitoring_cb.num_timesteps = 100
        monitoring_cb._on_rollout_end()

        with open(monitoring_cb.dashboard_state_path) as fh:
            snapshot = json.load(fh)

        assert "level_history" in snapshot
        assert snapshot["level_history"] == [8]

    def test_snapshot_includes_reward_summary(self, monitoring_cb):
        monitoring_cb.locals = {
            "dones": [True],
            "infos": [_make_step_info(
                reward_components={"exploration": 10.0, "badge": 150.0},
            )],
        }
        monitoring_cb._on_step()
        monitoring_cb.num_timesteps = 100
        monitoring_cb._on_rollout_end()

        with open(monitoring_cb.dashboard_state_path) as fh:
            snapshot = json.load(fh)

        assert "reward_component_summary" in snapshot
        assert snapshot["reward_component_summary"]["exploration"] == pytest.approx(10.0)
        assert snapshot["reward_component_summary"]["badge"] == pytest.approx(150.0)

    def test_snapshot_includes_money_history(self, monitoring_cb):
        monitoring_cb.locals = {
            "dones": [True],
            "infos": [_make_step_info(money=5000)],
        }
        monitoring_cb._on_step()
        monitoring_cb.num_timesteps = 100
        monitoring_cb._on_rollout_end()

        with open(monitoring_cb.dashboard_state_path) as fh:
            snapshot = json.load(fh)

        assert "money_history" in snapshot
        assert snapshot["money_history"] == [5000]
        assert "pokemon_count_history" in snapshot


# ──────────────────────────────────────────────────────────────────────
# Info keys contract (AMC-76 additions)
# ──────────────────────────────────────────────────────────────────────


def test_monitored_info_keys_include_new_fields():
    """AMC-76 added pokemon_count and money to MONITORED_INFO_KEYS."""
    assert "pokemon_count" in MONITORED_INFO_KEYS
    assert "money" in MONITORED_INFO_KEYS


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _merge_all_logs(mock_wandb) -> dict:
    merged = {}
    for call in mock_wandb.log.call_args_list:
        merged.update(call.args[0])
    return merged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
