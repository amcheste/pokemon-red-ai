"""
Unit tests for scripts/eval.py evaluation harness.

Tests CLI parsing, locked value enforcement, episode rollout logic,
and metric aggregation using mocked environment and model.
"""

import json
import os
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from scripts.eval import (
    build_parser,
    evaluate_checkpoint,
    run_episode,
    load_model,
    detect_algorithm,
    EVAL_METRIC_SCHEMA,
    LOCKED_N_EPISODES,
    LOCKED_SEED,
    PEWTER_CITY_MAP_ID,
    BOULDER_BADGE_BIT,
    _git_sha,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_checkpoint(tmp_path):
    """Create a fake checkpoint file."""
    ckpt = tmp_path / "model.zip"
    ckpt.write_bytes(b"\x00" * 100)
    return ckpt


@pytest.fixture
def tmp_rom(tmp_path):
    """Create a fake ROM file."""
    rom = tmp_path / "PokemonRed.gb"
    rom.write_bytes(b"\x00" * 1024)
    return rom


@pytest.fixture
def tmp_save_state(tmp_path):
    """Create a fake save state file."""
    state = tmp_path / "s0_post_intro.state"
    state.write_bytes(b"\x00" * 500)
    return state


@pytest.fixture
def mock_env():
    """Create a mock gym environment that terminates after a few steps."""
    env = MagicMock()

    # Track step count to terminate after 5 steps
    step_count = [0]

    def mock_reset(**kwargs):
        step_count[0] = 0
        obs = {"screen": np.zeros((72, 80, 1), dtype=np.uint8)}
        info = {"current_map": 40, "badges_earned": 0, "event_progress": {}}
        return obs, info

    def mock_step(action):
        step_count[0] += 1
        obs = {"screen": np.zeros((72, 80, 1), dtype=np.uint8)}
        terminated = step_count[0] >= 5
        info = {
            "current_map": 40,
            "badges_earned": 0,
            "player_level": 5,
            "event_progress": {"flags_triggered": step_count[0]},
        }
        return obs, 1.0, terminated, False, info

    env.reset = Mock(side_effect=mock_reset)
    env.step = Mock(side_effect=mock_step)
    env.close = Mock()
    return env


@pytest.fixture
def mock_model():
    """Create a mock SB3 model."""
    model = MagicMock()
    model.predict = Mock(return_value=(0, None))
    model.policy.parameters = Mock(return_value=[
        Mock(numel=Mock(return_value=1000))
    ])
    return model


# ──────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ──────────────────────────────────────────────────────────────────────


class TestArgumentParser:
    def test_requires_checkpoint_and_rom(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_requires_rom(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--checkpoint", "model.zip"])

    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args([
            "--checkpoint", "model.zip",
            "--rom", "game.gb",
        ])
        assert args.checkpoint == Path("model.zip")
        assert args.rom == Path("game.gb")
        assert args.save_state == Path("save_states/s0_post_intro.state")
        assert args.algorithm == "auto"
        assert args.n_episodes == LOCKED_N_EPISODES
        assert args.seed == LOCKED_SEED
        assert args.max_episode_steps == 15_000
        assert args.allow_override is False
        assert args.output is None

    def test_all_options(self):
        parser = build_parser()
        args = parser.parse_args([
            "--checkpoint", "model.zip",
            "--rom", "game.gb",
            "--save-state", "custom.state",
            "--algorithm", "RecurrentPPO",
            "--observation-type", "minimal",
            "--n-episodes", "5",
            "--seed", "99",
            "--max-episode-steps", "10000",
            "--allow-override",
            "--output", "results.json",
        ])
        assert args.algorithm == "RecurrentPPO"
        assert args.observation_type == "minimal"
        assert args.n_episodes == 5
        assert args.seed == 99
        assert args.allow_override is True
        assert args.output == Path("results.json")

    def test_algorithm_choices(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--checkpoint", "m.zip", "--rom", "g.gb",
                "--algorithm", "DQN",
            ])


# ──────────────────────────────────────────────────────────────────────
# Locked value enforcement
# ──────────────────────────────────────────────────────────────────────


class TestLockedValues:
    @patch("scripts.eval.PokemonRedGymEnv")
    @patch("scripts.eval.load_model")
    def test_rejects_non_locked_episodes(
        self, mock_load, MockEnv, tmp_checkpoint, tmp_rom, tmp_save_state
    ):
        with pytest.raises(ValueError, match="n_episodes=5 deviates"):
            evaluate_checkpoint(
                checkpoint_path=tmp_checkpoint,
                rom_path=tmp_rom,
                save_state=tmp_save_state,
                algorithm="PPO",
                n_episodes=5,
            )

    @patch("scripts.eval.PokemonRedGymEnv")
    @patch("scripts.eval.load_model")
    def test_rejects_non_locked_seed(
        self, mock_load, MockEnv, tmp_checkpoint, tmp_rom, tmp_save_state
    ):
        with pytest.raises(ValueError, match="seed=99 deviates"):
            evaluate_checkpoint(
                checkpoint_path=tmp_checkpoint,
                rom_path=tmp_rom,
                save_state=tmp_save_state,
                algorithm="PPO",
                seed=99,
            )

    @patch("scripts.eval.PokemonRedGymEnv")
    @patch("scripts.eval.load_model")
    @patch("scripts.eval.run_episode")
    def test_allow_override_suppresses_error(
        self, mock_run, mock_load, MockEnv,
        tmp_checkpoint, tmp_rom, tmp_save_state
    ):
        mock_run.return_value = {
            "total_reward": 10.0, "steps": 100,
            "maps_visited": 2, "maps_list": [1, 40],
            "event_flags_triggered": 3, "badges": 0,
            "earned_boulder_badge": False,
            "step_reached_pewter": None, "step_earned_boulder": None,
            "final_map": 1, "player_level": 5,
            "terminated": True, "truncated": False,
        }
        MockEnv.return_value = MagicMock()

        # Should NOT raise
        metrics = evaluate_checkpoint(
            checkpoint_path=tmp_checkpoint,
            rom_path=tmp_rom,
            save_state=tmp_save_state,
            algorithm="PPO",
            n_episodes=3,
            seed=99,
            allow_override=True,
        )
        assert metrics["n_episodes"] == 3


# ──────────────────────────────────────────────────────────────────────
# File validation
# ──────────────────────────────────────────────────────────────────────


class TestFileValidation:
    @patch("scripts.eval.PokemonRedGymEnv")
    @patch("scripts.eval.load_model")
    def test_missing_checkpoint_raises(
        self, mock_load, MockEnv, tmp_rom, tmp_save_state
    ):
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            evaluate_checkpoint(
                checkpoint_path=Path("/nonexistent/model.zip"),
                rom_path=tmp_rom,
                save_state=tmp_save_state,
                algorithm="PPO",
                allow_override=True, n_episodes=1,
            )

    @patch("scripts.eval.PokemonRedGymEnv")
    @patch("scripts.eval.load_model")
    def test_missing_rom_raises(
        self, mock_load, MockEnv, tmp_checkpoint, tmp_save_state
    ):
        with pytest.raises(FileNotFoundError, match="ROM not found"):
            evaluate_checkpoint(
                checkpoint_path=tmp_checkpoint,
                rom_path=Path("/nonexistent/game.gb"),
                save_state=tmp_save_state,
                algorithm="PPO",
                allow_override=True, n_episodes=1,
            )

    @patch("scripts.eval.PokemonRedGymEnv")
    @patch("scripts.eval.load_model")
    def test_missing_save_state_raises(
        self, mock_load, MockEnv, tmp_checkpoint, tmp_rom
    ):
        with pytest.raises(FileNotFoundError, match="Save state not found"):
            evaluate_checkpoint(
                checkpoint_path=tmp_checkpoint,
                rom_path=tmp_rom,
                save_state=Path("/nonexistent/state.state"),
                algorithm="PPO",
                allow_override=True, n_episodes=1,
            )


# ──────────────────────────────────────────────────────────────────────
# Single episode rollout
# ──────────────────────────────────────────────────────────────────────


class TestRunEpisode:
    def test_basic_episode(self, mock_env, mock_model):
        result = run_episode(
            mock_env, mock_model,
            is_recurrent=False, max_steps=100
        )
        assert result["total_reward"] == pytest.approx(5.0)  # 5 steps * 1.0
        assert result["steps"] == 5
        assert result["terminated"] is True
        assert result["truncated"] is False

    def test_tracks_maps(self, mock_env, mock_model):
        result = run_episode(
            mock_env, mock_model,
            is_recurrent=False, max_steps=100
        )
        assert 40 in result["maps_list"]
        assert result["maps_visited"] >= 1

    def test_tracks_event_flags(self, mock_env, mock_model):
        result = run_episode(
            mock_env, mock_model,
            is_recurrent=False, max_steps=100
        )
        # Our mock increments flags each step, so max = 5
        assert result["event_flags_triggered"] == 5

    def test_recurrent_passes_lstm_states(self, mock_env):
        model = MagicMock()
        # Return (action, new_states) tuple
        model.predict = Mock(return_value=(0, "lstm_state_1"))

        result = run_episode(
            mock_env, model,
            is_recurrent=True, max_steps=100
        )

        # Verify predict was called with state and episode_start args
        calls = model.predict.call_args_list
        # First call should have episode_start=True
        assert calls[0].kwargs.get("episode_start", calls[0][1].get("episode_start")) is True
        # Subsequent calls should have episode_start=False
        if len(calls) > 1:
            assert calls[1].kwargs.get("episode_start", calls[1][1].get("episode_start")) is False

    def test_max_steps_truncation(self, mock_model):
        """Episode stops at max_steps even if env doesn't terminate."""
        env = MagicMock()
        obs = {"screen": np.zeros((72, 80, 1), dtype=np.uint8)}
        env.reset.return_value = (obs, {"current_map": 1, "badges_earned": 0})
        # Never terminate
        env.step.return_value = (
            obs, 1.0, False, False,
            {"current_map": 1, "badges_earned": 0, "player_level": 5,
             "event_progress": {"flags_triggered": 0}},
        )

        result = run_episode(env, mock_model, is_recurrent=False, max_steps=10)
        assert result["steps"] == 10

    def test_detects_pewter_city(self, mock_model):
        """Detects when agent reaches Pewter City."""
        env = MagicMock()
        obs = {"screen": np.zeros((72, 80, 1), dtype=np.uint8)}
        step_count = [0]

        def mock_step(action):
            step_count[0] += 1
            # Reach Pewter on step 3
            current_map = PEWTER_CITY_MAP_ID if step_count[0] >= 3 else 1
            terminated = step_count[0] >= 5
            return (
                obs, 1.0, terminated, False,
                {"current_map": current_map, "badges_earned": 0,
                 "player_level": 5,
                 "event_progress": {"flags_triggered": 0}},
            )

        env.reset.return_value = (obs, {"current_map": 1, "badges_earned": 0})
        env.step = Mock(side_effect=mock_step)

        result = run_episode(env, mock_model, is_recurrent=False, max_steps=100)
        assert result["step_reached_pewter"] == 3

    def test_detects_boulder_badge(self, mock_model):
        """Detects when agent earns Boulder Badge."""
        env = MagicMock()
        obs = {"screen": np.zeros((72, 80, 1), dtype=np.uint8)}
        step_count = [0]

        def mock_step(action):
            step_count[0] += 1
            badges = BOULDER_BADGE_BIT if step_count[0] >= 4 else 0
            terminated = step_count[0] >= 5
            return (
                obs, 1.0, terminated, False,
                {"current_map": 3, "badges_earned": badges,
                 "player_level": 10,
                 "event_progress": {"flags_triggered": 0}},
            )

        env.reset.return_value = (obs, {"current_map": 1, "badges_earned": 0})
        env.step = Mock(side_effect=mock_step)

        result = run_episode(env, mock_model, is_recurrent=False, max_steps=100)
        assert result["earned_boulder_badge"] is True
        assert result["step_earned_boulder"] == 4

    def test_no_milestones_when_not_reached(self, mock_env, mock_model):
        result = run_episode(
            mock_env, mock_model,
            is_recurrent=False, max_steps=100
        )
        assert result["step_reached_pewter"] is None
        assert result["step_earned_boulder"] is None
        assert result["earned_boulder_badge"] is False


# ──────────────────────────────────────────────────────────────────────
# Metric aggregation
# ──────────────────────────────────────────────────────────────────────


class TestMetricAggregation:
    @patch("scripts.eval.PokemonRedGymEnv")
    @patch("scripts.eval.load_model")
    @patch("scripts.eval.run_episode")
    def test_aggregates_returns(
        self, mock_run, mock_load, MockEnv,
        tmp_checkpoint, tmp_rom, tmp_save_state
    ):
        mock_run.side_effect = [
            _make_episode_result(total_reward=10.0),
            _make_episode_result(total_reward=20.0),
            _make_episode_result(total_reward=30.0),
        ]
        MockEnv.return_value = MagicMock()

        metrics = evaluate_checkpoint(
            tmp_checkpoint, tmp_rom, tmp_save_state,
            algorithm="PPO", n_episodes=3, allow_override=True,
        )
        assert metrics["mean_return"] == pytest.approx(20.0)
        assert metrics["return_std"] == pytest.approx(np.std([10, 20, 30]))

    @patch("scripts.eval.PokemonRedGymEnv")
    @patch("scripts.eval.load_model")
    @patch("scripts.eval.run_episode")
    def test_aggregates_brock_rate(
        self, mock_run, mock_load, MockEnv,
        tmp_checkpoint, tmp_rom, tmp_save_state
    ):
        mock_run.side_effect = [
            _make_episode_result(earned_boulder_badge=True),
            _make_episode_result(earned_boulder_badge=False),
            _make_episode_result(earned_boulder_badge=True),
            _make_episode_result(earned_boulder_badge=False),
        ]
        MockEnv.return_value = MagicMock()

        metrics = evaluate_checkpoint(
            tmp_checkpoint, tmp_rom, tmp_save_state,
            algorithm="PPO", n_episodes=4, allow_override=True,
        )
        assert metrics["brock_win_rate"] == pytest.approx(0.5)

    @patch("scripts.eval.PokemonRedGymEnv")
    @patch("scripts.eval.load_model")
    @patch("scripts.eval.run_episode")
    def test_steps_to_pewter_none_when_never_reached(
        self, mock_run, mock_load, MockEnv,
        tmp_checkpoint, tmp_rom, tmp_save_state
    ):
        mock_run.side_effect = [
            _make_episode_result(step_reached_pewter=None),
            _make_episode_result(step_reached_pewter=None),
        ]
        MockEnv.return_value = MagicMock()

        metrics = evaluate_checkpoint(
            tmp_checkpoint, tmp_rom, tmp_save_state,
            algorithm="PPO", n_episodes=2, allow_override=True,
        )
        assert metrics["steps_to_pewter"] is None

    @patch("scripts.eval.PokemonRedGymEnv")
    @patch("scripts.eval.load_model")
    @patch("scripts.eval.run_episode")
    def test_steps_to_pewter_averaged(
        self, mock_run, mock_load, MockEnv,
        tmp_checkpoint, tmp_rom, tmp_save_state
    ):
        mock_run.side_effect = [
            _make_episode_result(step_reached_pewter=100),
            _make_episode_result(step_reached_pewter=None),  # didn't reach
            _make_episode_result(step_reached_pewter=200),
        ]
        MockEnv.return_value = MagicMock()

        metrics = evaluate_checkpoint(
            tmp_checkpoint, tmp_rom, tmp_save_state,
            algorithm="PPO", n_episodes=3, allow_override=True,
        )
        # Only episodes that reached Pewter are averaged
        assert metrics["steps_to_pewter"] == 150

    @patch("scripts.eval.PokemonRedGymEnv")
    @patch("scripts.eval.load_model")
    @patch("scripts.eval.run_episode")
    def test_output_contains_schema_keys(
        self, mock_run, mock_load, MockEnv,
        tmp_checkpoint, tmp_rom, tmp_save_state
    ):
        mock_run.return_value = _make_episode_result()
        MockEnv.return_value = MagicMock()

        metrics = evaluate_checkpoint(
            tmp_checkpoint, tmp_rom, tmp_save_state,
            algorithm="PPO", n_episodes=1, allow_override=True,
        )

        for key in EVAL_METRIC_SCHEMA:
            assert key in metrics, f"Missing schema key: {key}"

    @patch("scripts.eval.PokemonRedGymEnv")
    @patch("scripts.eval.load_model")
    @patch("scripts.eval.run_episode")
    def test_env_closed_after_evaluation(
        self, mock_run, mock_load, MockEnv,
        tmp_checkpoint, tmp_rom, tmp_save_state
    ):
        mock_env = MagicMock()
        MockEnv.return_value = mock_env
        mock_run.return_value = _make_episode_result()

        evaluate_checkpoint(
            tmp_checkpoint, tmp_rom, tmp_save_state,
            algorithm="PPO", n_episodes=1, allow_override=True,
        )
        mock_env.close.assert_called_once()

    @patch("scripts.eval.PokemonRedGymEnv")
    @patch("scripts.eval.load_model")
    @patch("scripts.eval.run_episode")
    def test_env_closed_on_error(
        self, mock_run, mock_load, MockEnv,
        tmp_checkpoint, tmp_rom, tmp_save_state
    ):
        mock_env = MagicMock()
        MockEnv.return_value = mock_env
        mock_run.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            evaluate_checkpoint(
                tmp_checkpoint, tmp_rom, tmp_save_state,
                algorithm="PPO", n_episodes=1, allow_override=True,
            )
        mock_env.close.assert_called_once()


# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────


class TestGitSha:
    def test_returns_string(self):
        sha = _git_sha()
        assert isinstance(sha, str)
        assert len(sha) > 0

    def test_sha_looks_like_hex(self):
        sha = _git_sha()
        if sha != "unknown":
            assert all(c in "0123456789abcdef" for c in sha)


# ──────────────────────────────────────────────────────────────────────
# JSON serialization
# ──────────────────────────────────────────────────────────────────────


class TestJsonOutput:
    @patch("scripts.eval.PokemonRedGymEnv")
    @patch("scripts.eval.load_model")
    @patch("scripts.eval.run_episode")
    def test_metrics_json_serializable(
        self, mock_run, mock_load, MockEnv,
        tmp_checkpoint, tmp_rom, tmp_save_state
    ):
        mock_run.return_value = _make_episode_result()
        MockEnv.return_value = MagicMock()

        metrics = evaluate_checkpoint(
            tmp_checkpoint, tmp_rom, tmp_save_state,
            algorithm="PPO", n_episodes=1, allow_override=True,
        )

        # Should not raise
        json_str = json.dumps(metrics, indent=2, default=str)
        parsed = json.loads(json_str)
        assert parsed["n_episodes"] == 1


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _make_episode_result(**overrides) -> dict:
    """Create a default episode result dict with optional overrides."""
    base = {
        "total_reward": 10.0,
        "steps": 100,
        "maps_visited": 2,
        "maps_list": [1, 40],
        "event_flags_triggered": 3,
        "badges": 0,
        "earned_boulder_badge": False,
        "step_reached_pewter": None,
        "step_earned_boulder": None,
        "final_map": 1,
        "player_level": 5,
        "terminated": True,
        "truncated": False,
    }
    base.update(overrides)
    return base


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
