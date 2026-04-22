"""
Global test configuration and fixtures for Pokemon Red AI tests.

This file contains fixtures and configurations that are available to all tests
in the project. It's automatically discovered by pytest.
"""

import os
import pytest
import tempfile
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from unittest.mock import Mock, patch

from pokemon_red_ai.game.memory import MEMORY_ADDRESSES


class BenchmarkRunner:
    """Simple benchmark runner for performance tests."""

    def run(self, name: str, func: Callable, iterations: int = 100) -> Dict[str, Any]:
        """
        Run benchmark and return timing statistics.

        Args:
            name: Name of the benchmark
            func: Function to benchmark
            iterations: Number of iterations to run

        Returns:
            Dictionary with timing statistics
        """
        times = []
        for _ in range(iterations):
            start = time.time()
            try:
                func()
            except Exception:
                # Continue timing even if function fails
                pass
            times.append(time.time() - start)

        return {
            'name': name,
            'iterations': iterations,
            'times': times,
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'total': np.sum(times)
        }


@pytest.fixture
def benchmark_runner():
    """Provide benchmark runner for performance tests."""
    return BenchmarkRunner()


@pytest.fixture
def mock_rom_file():
    """Create a temporary ROM file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.gb', delete=False) as f:
        # Create a minimal "ROM" file with the right size (1MB)
        f.write(b'\x00' * 1048576)
        rom_path = f.name

    yield rom_path

    # Clean up
    try:
        os.unlink(rom_path)
    except OSError:
        pass


@pytest.fixture
def mock_memory():
    """Create a basic mock memory object."""
    mock = Mock()
    mock.__getitem__ = Mock(return_value=0)
    return mock


@pytest.fixture
def mock_memory_from_state():
    """Create a mock memory that returns realistic game state."""
    def _create_mock(state_values: Optional[Dict[int, int]] = None):
        if state_values is None:
            # Default realistic game state values
            state_values = {
                MEMORY_ADDRESSES['player_x']: 10,
                MEMORY_ADDRESSES['player_y']: 10,
                MEMORY_ADDRESSES['map_id']: 1,
                MEMORY_ADDRESSES['player_level']: 5,
                MEMORY_ADDRESSES['current_hp_low']: 20,
                MEMORY_ADDRESSES['current_hp_high']: 0,
                MEMORY_ADDRESSES['max_hp_low']: 25,
                MEMORY_ADDRESSES['max_hp_high']: 0,
                MEMORY_ADDRESSES['badges']: 0,
                MEMORY_ADDRESSES['party_count']: 1,
                MEMORY_ADDRESSES['game_state']: 1,
                MEMORY_ADDRESSES['menu_state']: 0,
                MEMORY_ADDRESSES['money_low']: 100,
                MEMORY_ADDRESSES['money_mid']: 0,
                MEMORY_ADDRESSES['money_high']: 0
            }

        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(side_effect=lambda addr: state_values.get(addr, 0))
        return mock_memory

    return _create_mock


@pytest.fixture
def mock_pyboy():
    """Create a mock PyBoy instance for testing."""
    mock = Mock()

    # Mock memory
    mock.memory = Mock()
    mock.memory.__getitem__ = Mock(return_value=0)

    # Mock screen
    mock.screen = Mock()
    mock.screen.image = Mock(return_value=np.zeros((144, 160, 3), dtype=np.uint8))

    # Mock button methods
    mock.button_press = Mock()
    mock.button_release = Mock()
    mock.send_input = Mock()

    # Mock tilemap
    mock.tilemap_background = np.zeros((18, 20), dtype=np.uint8)

    # Mock other methods
    mock.tick = Mock()
    mock.stop = Mock()
    mock.set_emulation_speed = Mock()
    mock.save_state = Mock()
    mock.load_state = Mock()

    return mock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Provide sample configuration data for testing."""
    return {
        'training': {
            'total_timesteps': 1000,
            'algorithm': 'PPO',
            'max_episode_steps': 100
        },
        'rewards': {
            'exploration_reward': 1.0,
            'level_reward_multiplier': 50.0
        },
        'environment': {
            'rom_path': 'test.gb',
            'headless': True
        }
    }


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Configure logging for tests to reduce noise."""
    import logging
    # Reduce log noise during tests
    logging.getLogger('pokemon_red_ai').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)


class MockGameState:
    """Mock game state for testing purposes."""

    def __init__(self, **kwargs):
        self.position = kwargs.get('position', {'x': 10, 'y': 10, 'map': 1})
        self.stats = kwargs.get('stats', {
            'level': 5,
            'current_hp': 20,
            'max_hp': 25,
            'hp_ratio': 0.8,
            'badges': 0,
            'party_count': 1
        })
        self.game_state = kwargs.get('game_state', {
            'game_state': 1,
            'menu_state': 0,
            'map_id': 1
        })
        self.money = kwargs.get('money', 300)
        self.map_name = kwargs.get('map_name', 'pallet_town')
        self.badge_count = kwargs.get('badge_count', 0)
        self.in_game = kwargs.get('in_game', True)
        self.is_alive = kwargs.get('is_alive', True)


@pytest.fixture
def mock_game_state():
    """Provide mock game state factory."""
    return MockGameState


# Test markers
pytest_markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "performance: Performance benchmarks",
    "slow: Tests that take longer to run",
    "gpu: Tests that require GPU",
    "rom: Tests that require a ROM file"
]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    for marker in pytest_markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection."""
    # Add 'unit' marker to all tests in tests/unit/
    for item in items:
        if 'tests/unit/' in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif 'tests/integration/' in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif 'performance' in item.name.lower():
            item.add_marker(pytest.mark.performance)


# Skip tests that require ROM files if no ROM is available
def pytest_runtest_setup(item):
    """Setup for individual test runs."""
    if item.get_closest_marker('rom'):
        # Check if ROM file is available
        rom_paths = ['PokemonRed.gb', 'pokemon_red.gb', 'test.gb']
        if not any(os.path.exists(path) for path in rom_paths):
            pytest.skip("ROM file required but not found")

@pytest.fixture
def sample_memory_state():
    """Provide sample memory state for testing."""
    return {
        'player_x': 5,
        'player_y': 7,
        'map_id': 1,
        'player_level': 15,
        'current_hp': 45,
        'max_hp': 50,
        'badges': 3,
        'party_count': 2,
        'money': 1500,
        'game_state': 1,
        'menu_state': 0
    }

@pytest.fixture
def agent_memory_state():
    """Provide game memory state for testing (alias for sample_memory_state)."""
    return {
        'player_x': 5,
        'player_y': 7,
        'map_id': 1,
        'player_level': 15,
        'current_hp': 45,
        'max_hp': 50,
        'badges': 3,
        'party_count': 2,
        'money': 1500,
        'game_state': 1,
        'menu_state': 0
    }