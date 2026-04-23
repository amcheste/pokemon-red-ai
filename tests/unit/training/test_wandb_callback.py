"""
Unit tests for WandbCallback.

Tests the W&B logging callback in isolation using a mock wandb module
so the test suite runs without a real W&B account or network access.
"""

import os
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from pokemon_red_ai.training.callbacks import WandbCallback


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_wandb():
    """Provide a mock wandb module."""
    wandb = MagicMock()
    wandb.log = Mock()
    wandb.Artifact = Mock(return_value=Mock())
    wandb.log_artifact = Mock()
    return wandb


@pytest.fixture
def wandb_callback(tmp_path, mock_wandb):
    """Create a WandbCallback with mocked wandb."""
    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        cb = WandbCallback(
            save_freq=100,
            save_path=str(tmp_path),
            log_freq=1,
            verbose=0,
        )
    # Inject the mock so we can assert on it
    cb._wandb = mock_wandb
    return cb


@pytest.fixture
def attach_model(wandb_callback):
    """Attach a fake SB3 model to the callback."""
    model = Mock()
    model.ep_info_buffer = []
    model.save = Mock()

    # SB3 sets these during learn()
    wandb_callback.model = model
    wandb_callback.num_timesteps = 0
    wandb_callback.n_calls = 0
    wandb_callback.locals = {}
    wandb_callback.globals = {}

    return model


# ──────────────────────────────────────────────────────────────────────
# Initialisation
# ──────────────────────────────────────────────────────────────────────

class TestWandbCallbackInit:
    """Test callback initialisation."""

    def test_creates_model_directory(self, tmp_path, mock_wandb):
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cb = WandbCallback(save_path=str(tmp_path), verbose=0)

        assert os.path.isdir(os.path.join(str(tmp_path), "models"))

    def test_raises_without_wandb(self, tmp_path):
        """Callback raises ImportError when wandb is genuinely missing."""
        # Temporarily make wandb unimportable
        with patch.dict("sys.modules", {"wandb": None}):
            with pytest.raises(ImportError, match="wandb is required"):
                WandbCallback(save_path=str(tmp_path))

    def test_default_attributes(self, wandb_callback):
        assert wandb_callback.best_reward == -float("inf")
        assert wandb_callback._rollout_count == 0
        assert wandb_callback.log_freq == 1


# ──────────────────────────────────────────────────────────────────────
# _on_step
# ──────────────────────────────────────────────────────────────────────

class TestOnStep:
    def test_on_step_returns_true(self, wandb_callback, attach_model):
        assert wandb_callback._on_step() is True


# ──────────────────────────────────────────────────────────────────────
# _on_rollout_end — basic logging
# ──────────────────────────────────────────────────────────────────────

class TestRolloutEndBasic:
    """Test basic metric logging on rollout end."""

    def test_logs_global_step(self, wandb_callback, attach_model, mock_wandb):
        wandb_callback.num_timesteps = 2048
        wandb_callback._on_rollout_end()

        mock_wandb.log.assert_called_once()
        logged = mock_wandb.log.call_args[0][0]
        assert logged["global_step"] == 2048

    def test_logs_reward_stats(self, wandb_callback, attach_model, mock_wandb):
        attach_model.ep_info_buffer = [
            {"r": 10.0, "l": 100},
            {"r": 20.0, "l": 200},
            {"r": 30.0, "l": 300},
        ]

        wandb_callback.num_timesteps = 4096
        wandb_callback._on_rollout_end()

        logged = mock_wandb.log.call_args[0][0]
        assert logged["episode/reward_mean"] == pytest.approx(20.0)
        assert logged["episode/reward_max"] == pytest.approx(30.0)
        assert logged["episode/reward_min"] == pytest.approx(10.0)
        assert logged["episode/length_mean"] == pytest.approx(200.0)

    def test_logs_pokemon_metrics(self, wandb_callback, attach_model, mock_wandb):
        attach_model.ep_info_buffer = [
            {
                "r": 5.0, "l": 50,
                "maps_visited": 3,
                "badges_earned": 1,
                "locations_visited": 42,
                "hp_ratio": 0.8,
            },
        ]

        wandb_callback.num_timesteps = 100
        wandb_callback._on_rollout_end()

        logged = mock_wandb.log.call_args[0][0]
        assert logged["game/maps_visited_mean"] == 3.0
        assert logged["game/maps_visited_max"] == 3
        assert logged["game/badges_max"] == 1
        assert logged["game/locations_mean"] == 42.0
        assert logged["game/hp_ratio_mean"] == pytest.approx(0.8)

    def test_logs_event_flag_progress(self, wandb_callback, attach_model, mock_wandb):
        attach_model.ep_info_buffer = [
            {
                "r": 1.0, "l": 10,
                "event_progress": {"flags_triggered": 7},
            },
            {
                "r": 2.0, "l": 20,
                "event_progress": {"flags_triggered": 12},
            },
        ]

        wandb_callback.num_timesteps = 200
        wandb_callback._on_rollout_end()

        logged = mock_wandb.log.call_args[0][0]
        assert logged["game/event_flags_max"] == 12
        assert logged["game/event_flags_mean"] == pytest.approx(9.5)


# ──────────────────────────────────────────────────────────────────────
# _on_rollout_end — best model tracking
# ──────────────────────────────────────────────────────────────────────

class TestBestModel:

    def test_saves_best_model(self, wandb_callback, attach_model):
        attach_model.ep_info_buffer = [{"r": 50.0, "l": 100}]
        # Use a timestep that does NOT hit save_freq (100) to avoid
        # conflating the best-model save with a checkpoint save.
        wandb_callback.num_timesteps = 50

        wandb_callback._on_rollout_end()

        attach_model.save.assert_called_once()
        assert wandb_callback.best_reward == pytest.approx(50.0)

    def test_updates_best_on_improvement(self, wandb_callback, attach_model):
        # First rollout
        attach_model.ep_info_buffer = [{"r": 10.0, "l": 100}]
        wandb_callback.num_timesteps = 100
        wandb_callback._on_rollout_end()
        assert wandb_callback.best_reward == pytest.approx(10.0)

        # Second rollout — better
        wandb_callback._rollout_count = 0  # reset for log_freq
        attach_model.ep_info_buffer = [{"r": 25.0, "l": 100}]
        wandb_callback.num_timesteps = 200
        wandb_callback._on_rollout_end()
        assert wandb_callback.best_reward == pytest.approx(25.0)

    def test_no_save_when_not_improved(self, wandb_callback, attach_model):
        # First — sets best (use step that avoids checkpoint save_freq)
        attach_model.ep_info_buffer = [{"r": 100.0, "l": 100}]
        wandb_callback.num_timesteps = 50
        wandb_callback._on_rollout_end()

        save_count_after_first = attach_model.save.call_count

        # Second — worse (also avoid checkpoint save_freq)
        wandb_callback._rollout_count = 0
        attach_model.ep_info_buffer = [{"r": 5.0, "l": 100}]
        wandb_callback.num_timesteps = 75
        wandb_callback._on_rollout_end()

        # No additional save (best not beaten, no checkpoint)
        assert attach_model.save.call_count == save_count_after_first


# ──────────────────────────────────────────────────────────────────────
# _on_rollout_end — log_freq throttling
# ──────────────────────────────────────────────────────────────────────

class TestLogFreq:

    def test_log_freq_skips_rollouts(self, tmp_path, mock_wandb):
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cb = WandbCallback(
                save_path=str(tmp_path), log_freq=3, verbose=0
            )
        cb._wandb = mock_wandb

        model = Mock()
        model.ep_info_buffer = [{"r": 1.0, "l": 10}]
        model.save = Mock()
        cb.model = model
        cb.num_timesteps = 100

        # 3 rollouts: only the 3rd should log
        cb._on_rollout_end()  # count=1 → skip
        cb._on_rollout_end()  # count=2 → skip
        cb._on_rollout_end()  # count=3 → log

        assert mock_wandb.log.call_count == 1


# ──────────────────────────────────────────────────────────────────────
# _on_rollout_end — checkpoint artifacts
# ──────────────────────────────────────────────────────────────────────

class TestCheckpointArtifact:

    def test_saves_checkpoint_at_save_freq(self, wandb_callback, attach_model, mock_wandb):
        wandb_callback.save_freq = 100

        attach_model.ep_info_buffer = [{"r": 5.0, "l": 50}]
        wandb_callback.num_timesteps = 100  # hits save_freq

        wandb_callback._on_rollout_end()

        # Model.save should be called (best model + checkpoint)
        assert attach_model.save.call_count >= 1
        # Artifact should be created
        mock_wandb.Artifact.assert_called()
        mock_wandb.log_artifact.assert_called()

    def test_no_checkpoint_between_saves(self, wandb_callback, attach_model, mock_wandb):
        wandb_callback.save_freq = 100

        attach_model.ep_info_buffer = [{"r": 5.0, "l": 50}]
        wandb_callback.num_timesteps = 50  # NOT a save point

        wandb_callback._on_rollout_end()

        # Artifact should NOT be created (only best model save)
        mock_wandb.Artifact.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_ep_buffer(self, wandb_callback, attach_model, mock_wandb):
        """No crash when episode buffer is empty."""
        attach_model.ep_info_buffer = []
        wandb_callback.num_timesteps = 100

        wandb_callback._on_rollout_end()

        logged = mock_wandb.log.call_args[0][0]
        assert "episode/reward_mean" not in logged
        assert "global_step" in logged

    def test_partial_ep_info(self, wandb_callback, attach_model, mock_wandb):
        """Handles episodes missing some keys gracefully."""
        attach_model.ep_info_buffer = [
            {"r": 10.0},  # no 'l'
            {"l": 100},   # no 'r'
        ]

        wandb_callback.num_timesteps = 100
        wandb_callback._on_rollout_end()  # should not raise

        logged = mock_wandb.log.call_args[0][0]
        assert logged["episode/reward_mean"] == pytest.approx(10.0)
        assert logged["episode/length_mean"] == pytest.approx(100.0)

    def test_artifact_upload_failure_doesnt_crash(
        self, wandb_callback, attach_model, mock_wandb
    ):
        """Training continues if artifact upload fails."""
        mock_wandb.log_artifact.side_effect = RuntimeError("upload failed")
        wandb_callback.save_freq = 100

        attach_model.ep_info_buffer = [{"r": 5.0, "l": 50}]
        wandb_callback.num_timesteps = 100

        # Should not raise
        wandb_callback._on_rollout_end()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
