"""
Unit tests for training callbacks.

Tests the callback classes including TrainingCallback, EnhancedTrainingCallback,
EarlyStopping, and PerformanceMonitor.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from pokemon_red_ai.training.callbacks import (
    TrainingCallback,
    EnhancedTrainingCallback,
    EarlyStopping,
    PerformanceMonitor
)


class TestTrainingCallback:
    """Test TrainingCallback class."""

    def test_callback_initialization(self, temp_save_dir):
        """Test callback initialization."""
        callback = TrainingCallback(
            save_freq=1000,
            save_path=str(temp_save_dir),
            verbose=1
        )

        assert callback.save_freq == 1000
        assert callback.save_path == str(temp_save_dir)
        assert callback.best_reward == -float('inf')

    def test_on_step(self, temp_save_dir):
        """Test _on_step method."""
        callback = TrainingCallback(save_path=str(temp_save_dir))

        # Should return True to continue training
        assert callback._on_step() is True

    def test_on_rollout_end_basic(self, temp_save_dir, sample_episode_info):
        """Test _on_rollout_end with episode info."""
        callback = TrainingCallback(save_path=str(temp_save_dir), verbose=0)

        # Mock model with episode buffer
        callback.model = Mock()
        callback.model.ep_info_buffer = sample_episode_info
        callback.model.save = Mock()
        callback.num_timesteps = 1000

        callback._on_rollout_end()

        assert len(callback.episode_rewards) > 0
        assert len(callback.episode_lengths) > 0

    def test_on_rollout_end_saves_best_model(self, temp_save_dir):
        """Test that best model is saved."""
        callback = TrainingCallback(save_path=str(temp_save_dir), verbose=0)

        # Mock model
        callback.model = Mock()
        callback.model.ep_info_buffer = [
            {'r': 500.0, 'l': 1000}  # Very high reward
        ]
        callback.model.save = Mock()
        callback.num_timesteps = 1000

        callback._on_rollout_end()

        # Should save best model
        assert callback