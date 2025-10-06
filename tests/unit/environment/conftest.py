"""
Test fixtures and configuration for environment package tests.

This conftest.py file contains fixtures specifically for testing the
Pokemon Red RL environment components (gym_env.py, rewards.py, observations.py).
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile


@pytest.fixture
def mock_agent_class():
    """
    Mock PokemonRedAgent class for environment testing.

    This fixture provides a fully-functional mock agent that can be used
    across all environment tests without requiring the actual game ROM.
    """
    with patch('pokemon_red_ai.environment.gym_env.PokemonRedAgent') as MockAgent:
        # Create mock instance
        mock_agent = Mock()

        # Screen methods
        mock_agent.get_screen_array = Mock(
            return_value=np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        )

        # Position methods
        mock_agent.get_player_position = Mock(
            return_value={'x': 10, 'y': 10, 'map': 1}
        )

        # Stats methods
        mock_agent.get_player_stats = Mock(
            return_value={
                'level': 5,
                'current_hp': 25,
                'max_hp': 30,
                'badges': 0,
                'party_count': 1
            }
        )

        # Comprehensive state
        mock_agent.get_comprehensive_state = Mock(
            return_value={
                'position': {'x': 10, 'y': 10, 'map': 1},
                'stats': {
                    'level': 5,
                    'current_hp': 25,
                    'max_hp': 30,
                    'hp_ratio': 0.833,
                    'badges': 0,
                    'party_count': 1
                },
                'game_state': {
                    'game_state': 1,
                    'menu_state': 0,
                    'map_id': 1
                },
                'money': 300,
                'map_name': 'pallet_town',
                'badge_count': 0,
                'in_game': True,
                'is_alive': True
            }
        )

        # Step method
        def step_side_effect(action):
            state = mock_agent.get_comprehensive_state.return_value.copy()
            # Slightly modify position to simulate movement
            state['position']['x'] = (state['position']['x'] + 1) % 256
            return state

        mock_agent.step = Mock(side_effect=step_side_effect)

        # Control methods
        mock_agent.reset_game = Mock(return_value=True)
        mock_agent.wait_frames = Mock()
        mock_agent.cleanup = Mock()

        # Exploration tracking
        mock_agent.get_exploration_progress = Mock(
            return_value={
                'locations_visited': 10,
                'unique_maps': 2,
                'episode_steps': 100
            }
        )

        # Make MockAgent return the mock_agent instance
        MockAgent.return_value = mock_agent

        yield MockAgent


@pytest.fixture
def sample_game_state():
    """
    Provide sample game state for testing.

    Returns a realistic game state dictionary that can be used for
    testing reward calculators and observation processors.
    """
    return {
        'position': {'x': 10, 'y': 15, 'map': 1},
        'stats': {
            'level': 8,
            'current_hp': 35,
            'max_hp': 40,
            'hp_ratio': 0.875,
            'badges': 1,
            'party_count': 2
        },
        'game_state': {
            'game_state': 1,
            'menu_state': 0,
            'map_id': 1
        },
        'money': 500,
        'map_name': 'viridian_city',
        'badge_count': 1,
        'in_game': True,
        'is_alive': True
    }


@pytest.fixture
def sample_episode_states():
    """
    Provide a sequence of game states representing an episode.

    Useful for testing reward strategy evaluation and episode tracking.
    """
    states = []
    for i in range(20):
        states.append({
            'position': {
                'x': 10 + i,
                'y': 10 + (i % 5),
                'map': 1 + (i // 10)
            },
            'stats': {
                'level': 5 + (i // 5),
                'current_hp': max(20 - i, 5),
                'max_hp': 25,
                'hp_ratio': max(20 - i, 5) / 25,
                'badges': i // 10,
                'party_count': min(1 + (i // 8), 6)
            },
            'game_state': {
                'game_state': 1,
                'menu_state': 0,
                'map_id': 1 + (i // 10)
            },
            'money': 300 + (i * 10),
            'map_name': f'map_{1 + (i // 10)}',
            'badge_count': i // 10,
            'in_game': True,
            'is_alive': True
        })
    return states


@pytest.fixture
def mock_screen_array():
    """Provide realistic mock screen array."""
    return np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)


@pytest.fixture
def mock_downsampled_screen():
    """Provide realistic downsampled screen array."""
    return np.random.randint(0, 255, (72, 80, 3), dtype=np.uint8)


@pytest.fixture
def visited_locations_set():
    """Provide sample visited locations set."""
    locations = set()
    for i in range(50):
        x = i % 20
        y = i // 20
        map_id = 1 + (i % 3)
        locations.add((x, y, map_id))
    return locations


@pytest.fixture
def reward_config_factory():
    """
    Factory for creating custom reward configurations.

    Usage:
        config = reward_config_factory(exploration_reward=5.0)
    """
    from pokemon_red_ai.environment.rewards import RewardConfig

    def _create_config(**kwargs):
        return RewardConfig(**kwargs)

    return _create_config


@pytest.fixture
def observation_space_factory():
    """
    Factory for creating observation spaces with different configurations.

    Usage:
        obs_space = observation_space_factory(screen_size=(40, 36))
    """
    from pokemon_red_ai.environment.observations import create_observation_space

    def _create_space(**kwargs):
        return create_observation_space(**kwargs)

    return _create_space


@pytest.fixture
def temp_rom_file():
    """Create a temporary ROM file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.gb', delete=False) as f:
        # Create a minimal "ROM" file (1MB)
        f.write(b'\x00' * 1048576)
        rom_path = f.name

    yield rom_path

    # Cleanup
    try:
        Path(rom_path).unlink()
    except OSError:
        pass


@pytest.fixture
def sample_observation_dict():
    """Provide sample observation dictionary."""
    return {
        'screen': np.random.randint(0, 255, (72, 80, 3), dtype=np.uint8),
        'position': np.array([10, 15, 1], dtype=np.uint8),
        'stats': np.array([5, 80, 1, 2, 50, 1], dtype=np.uint8),
        'exploration': np.array([50, 3], dtype=np.uint16)
    }


@pytest.fixture
def mock_reward_calculator():
    """Provide mock reward calculator."""
    calc = Mock()
    calc.calculate_reward = Mock(return_value=1.0)
    calc.reset = Mock()
    calc.get_reward_breakdown = Mock(return_value={
        'time': -0.01,
        'exploration': 1.0
    })
    calc.visited_locations = set()
    calc.visited_maps = set()
    calc.previous_state = None
    calc.reward_components = {}
    return calc


class MockGymEnv:
    """
    Mock Gymnasium environment for testing vectorized environments.

    Provides a minimal implementation of the gym.Env interface.
    """

    def __init__(self):
        self.observation_space = None
        self.action_space = None
        self.game = Mock()
        self.episode_steps = 0
        self.episode_reward = 0

    def reset(self, seed=None, options=None):
        self.episode_steps = 0
        self.episode_reward = 0
        obs = {
            'screen': np.zeros((72, 80, 3), dtype=np.uint8),
            'position': np.array([0, 0, 0], dtype=np.uint8),
            'stats': np.array([0, 0, 0, 0, 0, 0], dtype=np.uint8),
            'exploration': np.array([0, 0], dtype=np.uint16)
        }
        info = {'episode_steps': 0}
        return obs, info

    def step(self, action):
        self.episode_steps += 1
        obs = {
            'screen': np.zeros((72, 80, 3), dtype=np.uint8),
            'position': np.array([1, 1, 1], dtype=np.uint8),
            'stats': np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8),
            'exploration': np.array([1, 1], dtype=np.uint16)
        }
        reward = 0.0
        terminated = False
        truncated = self.episode_steps >= 100
        info = {'episode_steps': self.episode_steps}
        return obs, reward, terminated, truncated, info

    def close(self):
        self.game.cleanup()

    def render(self, mode='human'):
        return np.zeros((144, 160, 3), dtype=np.uint8)


@pytest.fixture
def mock_gym_env():
    """Provide mock gym environment."""
    return MockGymEnv()


# Test markers for environment tests
def pytest_configure(config):
    """Configure pytest with environment-specific markers."""
    config.addinivalue_line(
        "markers", "environment: Tests for environment package"
    )
    config.addinivalue_line(
        "markers", "rewards: Tests for reward calculation"
    )
    config.addinivalue_line(
        "markers", "observations: Tests for observation processing"
    )
    config.addinivalue_line(
        "markers", "gym_env: Tests for Gymnasium environment wrapper"
    )


# Parameterized fixtures for testing different configurations
@pytest.fixture(params=['standard', 'exploration', 'progress', 'sparse'])
def reward_strategy(request):
    """Parameterized fixture for testing all reward strategies."""
    return request.param


@pytest.fixture(params=['multi_modal', 'minimal', 'screen_only'])
def observation_type(request):
    """Parameterized fixture for testing all observation types."""
    return request.param


@pytest.fixture(params=[True, False])
def headless_mode(request):
    """Parameterized fixture for testing headless and windowed modes."""
    return request.param


@pytest.fixture(params=[(80, 72), (40, 36), (160, 144)])
def screen_size(request):
    """Parameterized fixture for testing different screen sizes."""
    return request.param


# Helper functions for tests
def create_test_observation(screen_size=(80, 72)):
    """Helper to create valid test observation."""
    return {
        'screen': np.random.randint(0, 255, (screen_size[1], screen_size[0], 3), dtype=np.uint8),
        'position': np.array([10, 15, 1], dtype=np.uint8),
        'stats': np.array([5, 80, 1, 2, 50, 1], dtype=np.uint8),
        'exploration': np.array([50, 3], dtype=np.uint16)
    }


def create_test_game_state(
        x=10, y=10, map_id=1,
        level=5, hp=25, max_hp=30,
        badges=0, party_count=1
):
    """Helper to create custom test game state."""
    return {
        'position': {'x': x, 'y': y, 'map': map_id},
        'stats': {
            'level': level,
            'current_hp': hp,
            'max_hp': max_hp,
            'hp_ratio': hp / max(max_hp, 1),
            'badges': badges,
            'party_count': party_count
        },
        'game_state': {
            'game_state': 1,
            'menu_state': 0,
            'map_id': map_id
        },
        'money': 300,
        'map_name': f'map_{map_id}',
        'badge_count': bin(badges).count('1'),
        'in_game': map_id != 0,
        'is_alive': hp > 0
    }


# Add helper functions to pytest namespace
@pytest.fixture(autouse=True)
def add_helper_functions(doctest_namespace):
    """Add helper functions to test namespace."""
    doctest_namespace['create_test_observation'] = create_test_observation
    doctest_namespace['create_test_game_state'] = create_test_game_state


# Performance tracking fixture
@pytest.fixture
def performance_tracker():
    """Track performance metrics across tests."""
    import time

    class PerformanceTracker:
        def __init__(self):
            self.measurements = {}

        def measure(self, name, func, *args, **kwargs):
            """Measure execution time of a function."""
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start

            if name not in self.measurements:
                self.measurements[name] = []
            self.measurements[name].append(elapsed)

            return result, elapsed

        def get_stats(self, name):
            """Get statistics for a measurement."""
            if name not in self.measurements:
                return None

            times = self.measurements[name]
            return {
                'count': len(times),
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'total': np.sum(times)
            }

        def report(self):
            """Print performance report."""
            print("\n=== Performance Report ===")
            for name in sorted(self.measurements.keys()):
                stats = self.get_stats(name)
                print(f"\n{name}:")
                print(f"  Count: {stats['count']}")
                print(f"  Mean:  {stats['mean'] * 1000:.2f}ms")
                print(f"  Std:   {stats['std'] * 1000:.2f}ms")
                print(f"  Min:   {stats['min'] * 1000:.2f}ms")
                print(f"  Max:   {stats['max'] * 1000:.2f}ms")

    return PerformanceTracker()


# Fixture for capturing and analyzing logs
@pytest.fixture
def log_capture():
    """Capture log messages during tests."""
    import logging
    from io import StringIO

    class LogCapture:
        def __init__(self):
            self.stream = StringIO()
            self.handler = logging.StreamHandler(self.stream)
            self.handler.setLevel(logging.DEBUG)
            self.logger = logging.getLogger('pokemon_red_ai')
            self.original_level = self.logger.level

        def start(self):
            """Start capturing logs."""
            self.logger.addHandler(self.handler)
            self.logger.setLevel(logging.DEBUG)

        def stop(self):
            """Stop capturing logs."""
            self.logger.removeHandler(self.handler)
            self.logger.setLevel(self.original_level)

        def get_logs(self):
            """Get captured log messages."""
            return self.stream.getvalue()

        def clear(self):
            """Clear captured logs."""
            self.stream.truncate(0)
            self.stream.seek(0)

        def __enter__(self):
            self.start()
            return self

        def __exit__(self, *args):
            self.stop()

    return LogCapture()


# Fixture for testing with different random seeds
@pytest.fixture
def seeded_rng():
    """Provide seeded random number generator for reproducible tests."""

    def _create_rng(seed=42):
        return np.random.RandomState(seed)

    return _create_rng


# Fixture for memory usage tracking
@pytest.fixture
def memory_tracker():
    """Track memory usage during tests."""
    try:
        import psutil
        import os

        class MemoryTracker:
            def __init__(self):
                self.process = psutil.Process(os.getpid())
                self.measurements = []

            def measure(self):
                """Take a memory measurement."""
                mem_info = self.process.memory_info()
                self.measurements.append({
                    'rss': mem_info.rss / 1024 / 1024,  # MB
                    'vms': mem_info.vms / 1024 / 1024  # MB
                })
                return self.measurements[-1]

            def get_delta(self, start_idx=0):
                """Get memory delta from start."""
                if len(self.measurements) < 2:
                    return {'rss': 0, 'vms': 0}

                start = self.measurements[start_idx]
                end = self.measurements[-1]

                return {
                    'rss': end['rss'] - start['rss'],
                    'vms': end['vms'] - start['vms']
                }

        return MemoryTracker()

    except ImportError:
        # Return mock if psutil not available
        class MockMemoryTracker:
            def measure(self):
                return {'rss': 0, 'vms': 0}

            def get_delta(self, start_idx=0):
                return {'rss': 0, 'vms': 0}

        return MockMemoryTracker()


# Fixture for testing error scenarios
@pytest.fixture
def error_scenarios():
    """Provide common error scenarios for testing."""
    return {
        'memory_error': MemoryError("Out of memory"),
        'value_error': ValueError("Invalid value"),
        'key_error': KeyError("Key not found"),
        'index_error': IndexError("Index out of range"),
        'type_error': TypeError("Invalid type"),
        'runtime_error': RuntimeError("Runtime error"),
        'attribute_error': AttributeError("Attribute not found"),
        'io_error': IOError("IO error"),
        'os_error': OSError("OS error")
    }


# Fixture for validating data types
@pytest.fixture
def type_validator():
    """Provide type validation utilities."""

    class TypeValidator:
        @staticmethod
        def validate_observation(obs, obs_space):
            """Validate observation matches space."""
            if isinstance(obs_space, dict):
                return all(
                    key in obs and TypeValidator.validate_observation(obs[key], space)
                    for key, space in obs_space.items()
                )
            elif hasattr(obs_space, 'contains'):
                return obs_space.contains(obs)
            return True

        @staticmethod
        def validate_numpy_array(arr, expected_shape=None, expected_dtype=None):
            """Validate numpy array properties."""
            if not isinstance(arr, np.ndarray):
                return False

            if expected_shape and arr.shape != expected_shape:
                return False

            if expected_dtype and arr.dtype != expected_dtype:
                return False

            return True

        @staticmethod
        def validate_dict_structure(d, required_keys):
            """Validate dictionary has required keys."""
            return all(key in d for key in required_keys)

    return TypeValidator()


# Cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup resources after each test."""
    yield

    # Force garbage collection
    import gc
    gc.collect()


# Fixture to suppress warnings during tests
@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress common warnings during tests."""
    import warnings

    # Suppress specific warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
    warnings.filterwarnings('ignore', message='.*gym.*')

    yield

    # Reset warnings
    warnings.resetwarnings()


# Test data generators
class TestDataGenerator:
    """Generate test data for various scenarios."""

    @staticmethod
    def generate_random_screens(count=10, shape=(144, 160, 3)):
        """Generate random screen arrays."""
        return [
            np.random.randint(0, 255, shape, dtype=np.uint8)
            for _ in range(count)
        ]

    @staticmethod
    def generate_episode_trajectory(length=100):
        """Generate a complete episode trajectory."""
        trajectory = []
        for i in range(length):
            trajectory.append({
                'state': create_test_game_state(
                    x=i % 20,
                    y=i // 20,
                    level=5 + (i // 20)
                ),
                'action': i % 8,
                'reward': np.random.randn(),
                'next_state': create_test_game_state(
                    x=(i + 1) % 20,
                    y=(i + 1) // 20,
                    level=5 + ((i + 1) // 20)
                ),
                'terminated': i == length - 1,
                'truncated': False
            })
        return trajectory

    @staticmethod
    def generate_varied_game_states(count=50):
        """Generate varied game states for testing."""
        states = []
        for i in range(count):
            states.append(create_test_game_state(
                x=np.random.randint(0, 256),
                y=np.random.randint(0, 256),
                map_id=np.random.randint(1, 10),
                level=np.random.randint(1, 100),
                hp=np.random.randint(0, 100),
                max_hp=np.random.randint(50, 100),
                badges=np.random.randint(0, 8),
                party_count=np.random.randint(1, 6)
            ))
        return states


@pytest.fixture
def test_data_generator():
    """Provide test data generator."""
    return TestDataGenerator()


# Add test categories to markers
def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their location."""
    for item in items:
        # Add environment marker to all tests in this directory
        if 'environment' in str(item.fspath):
            item.add_marker(pytest.mark.environment)

        # Add specific markers based on file name
        if 'rewards' in str(item.fspath):
            item.add_marker(pytest.mark.rewards)
        elif 'observations' in str(item.fspath):
            item.add_marker(pytest.mark.observations)
        elif 'gym_env' in str(item.fspath):
            item.add_marker(pytest.mark.gym_env)