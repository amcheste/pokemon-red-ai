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
):
    return {
        "episode": {"r": reward, "l": length},
        "unique_maps_list": list(unique_maps),
        "event_progress": {
            "flags_triggered": len(triggered_flags),
            "triggered_names": list(triggered_flags),
        },
        "current_map": current_map,
        "badges_earned": badges,
        "locations_visited": locations,
    }


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

    def test_logs_flag_progress_for_all_18(self, monitoring_cb, mock_wandb):
        self._populate(monitoring_cb)
        monitoring_cb.num_timesteps = 2048
        monitoring_cb._on_rollout_end()

        logged = _merge_all_logs(mock_wandb)
        assert "game/flag_progress" in logged
        table = logged["game/flag_progress"]
        assert len(table.data) == 18  # All pre-registered flags listed
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
        # Columns match expected order
        expected_cols = {
            "episode", "global_step", "reward", "length",
            "maps_visited", "final_map", "badges",
            "event_flags_triggered", "locations_visited",
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

    def test_snapshot_contains_all_18_flags(self, monitoring_cb):
        monitoring_cb.num_timesteps = 100
        monitoring_cb._on_rollout_end()

        with open(monitoring_cb.dashboard_state_path) as fh:
            snapshot = json.load(fh)

        # Even before any episode, snapshot lists all 18 flags with count 0
        assert len(snapshot["flag_trigger_counts"]) == 18
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
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _merge_all_logs(mock_wandb) -> dict:
    merged = {}
    for call in mock_wandb.log.call_args_list:
        merged.update(call.args[0])
    return merged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
