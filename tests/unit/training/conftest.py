"""
Shared fixtures for training package tests.

This conftest.py provides common fixtures used across all training tests.
"""

import pytest
from unittest.mock import Mock
from pathlib import Path


@pytest.fixture
def mock_env():
    """Create a mock Gymnasium environment."""
    env = Mock()

    # Mock observation space
    observation_space = Mock()
    observation_space.spaces = {
        'screen': Mock(shape=(72, 80, 3)),
        'position': Mock(shape=(3,)),
        'stats': Mock(shape=(6,)),
        'exploration': Mock(shape=(2,))
    }
    env.observation_space = observation_space

    # Mock action space
    env.action_space = Mock()
    env.action_space.n = 8

    # Mock methods
    env.reset = Mock(return_value=(Mock(), {}))
    env.step = Mock(return_value=(Mock(), 0.0, False, False, {}))
    env.close = Mock()

    return env


@pytest.fixture
def mock_ppo_model():
    """Create a mock PPO model."""
    model = Mock()
    model.learn = Mock()
    model.save = Mock()
    model.predict = Mock(return_value=(0, None))
    model.ep_info_buffer = []
    return model


@pytest.fixture
def temp_save_dir(tmp_path):
    """Create a temporary save directory.

    Uses pytest's built-in tmp_path fixture which provides a temporary directory
    that is automatically cleaned up after the test.
    """
    save_dir = tmp_path / "training"
    save_dir.mkdir()
    return save_dir


@pytest.fixture
def sample_episode_info():
    """Provide sample episode info for callbacks."""
    return [
        {'r': 100.0, 'l': 500, 'maps_visited': 3, 'badges_earned': 1},
        {'r': 150.0, 'l': 600, 'maps_visited': 4, 'badges_earned': 1},
        {'r': 200.0, 'l': 700, 'maps_visited': 5, 'badges_earned': 2},
    ]


@pytest.fixture
def mock_training_environment():
    """Create a complete mock training environment setup."""
    setup = {
        'env': Mock(),
        'model': Mock(),
        'callback': Mock()
    }

    # Setup environment
    setup['env'].observation_space = Mock()
    setup['env'].action_space = Mock()
    setup['env'].reset = Mock(return_value=(Mock(), {}))
    setup['env'].step = Mock(return_value=(Mock(), 1.0, False, False, {}))
    setup['env'].close = Mock()

    # Setup model
    setup['model'].learn = Mock()
    setup['model'].save = Mock()
    setup['model'].predict = Mock(return_value=(0, None))
    setup['model'].ep_info_buffer = []

    # Setup callback
    setup['callback'].on_step = Mock(return_value=True)
    setup['callback'].on_rollout_end = Mock()

    return setup