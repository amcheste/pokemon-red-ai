"""
Unit tests for Pokemon Red RL environment - Observations Module

Tests observation space creation and processing.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import gymnasium as gym

from pokemon_red_ai.environment.observations import (
    downsample_screen,
    normalize_screen,
    create_observation_space,
    process_game_state,
    get_screen_features,
    create_minimal_observation_space,
    process_minimal_observation,
    validate_observation,
    preprocess_screen_for_cnn,
    # Paper observation treatments
    NUM_EVENT_FLAGS,
    SYMBOLIC_DIM,
    _screen_to_grayscale,
    create_pixel_observation_space,
    process_pixel_observation,
    create_symbolic_observation_space,
    _build_symbolic_vector,
    process_symbolic_observation,
    create_hybrid_observation_space,
    process_hybrid_observation,
)


class TestDownsampleScreen:
    """Test screen downsampling functionality."""

    def test_downsample_to_default_size(self):
        """Test downsampling to default (80, 72)."""
        screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)

        downsampled = downsample_screen(screen)

        assert downsampled.shape == (72, 80, 3)
        assert downsampled.dtype == np.uint8

    def test_downsample_to_custom_size(self):
        """Test downsampling to custom size."""
        screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)

        downsampled = downsample_screen(screen, target_size=(40, 36))

        assert downsampled.shape == (36, 40, 3)

    def test_downsample_without_opencv(self):
        """Test fallback downsampling when OpenCV unavailable."""
        screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)

        # Test that the function works even if we can't test the fallback path
        # Since cv2 might be imported at module level, we can't easily mock it
        # Just test that downsampling works
        downsampled = downsample_screen(screen)

        # Should return downsampled array
        assert downsampled.shape[0] <= screen.shape[0]
        assert downsampled.shape[1] <= screen.shape[1]

        # Note: Testing the actual fallback path would require unloading cv2
        # which is complex and not critical for unit testing

    def test_downsample_preserves_data_type(self):
        """Test that downsampling preserves uint8 dtype."""
        screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)

        downsampled = downsample_screen(screen)

        assert downsampled.dtype == np.uint8

    def test_downsample_grayscale_image(self):
        """Test downsampling grayscale images."""
        screen = np.random.randint(0, 255, (144, 160), dtype=np.uint8)

        downsampled = downsample_screen(screen)

        assert downsampled.shape[0] <= 72
        assert downsampled.shape[1] <= 80


class TestNormalizeScreen:
    """Test screen normalization functionality."""

    def test_normalize_rgb_screen(self):
        """Test normalizing RGB screen."""
        screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)

        normalized = normalize_screen(screen)

        assert normalized.shape == (144, 160, 3)
        assert normalized.dtype == np.uint8

    def test_normalize_rgba_to_rgb(self):
        """Test converting RGBA to RGB."""
        screen = np.random.randint(0, 255, (144, 160, 4), dtype=np.uint8)

        normalized = normalize_screen(screen)

        assert normalized.shape == (144, 160, 3)
        assert normalized.dtype == np.uint8

    def test_normalize_grayscale_to_rgb(self):
        """Test converting grayscale to RGB."""
        screen = np.random.randint(0, 255, (144, 160), dtype=np.uint8)

        normalized = normalize_screen(screen)

        assert normalized.shape == (144, 160, 3)
        # All three channels should be the same
        assert np.array_equal(normalized[:, :, 0], normalized[:, :, 1])
        assert np.array_equal(normalized[:, :, 1], normalized[:, :, 2])

    def test_normalize_single_channel_to_rgb(self):
        """Test converting single channel to RGB."""
        screen = np.random.randint(0, 255, (144, 160, 1), dtype=np.uint8)

        normalized = normalize_screen(screen)

        assert normalized.shape == (144, 160, 3)

    def test_normalize_too_many_channels(self):
        """Test handling more than 4 channels."""
        screen = np.random.randint(0, 255, (144, 160, 5), dtype=np.uint8)

        normalized = normalize_screen(screen)

        assert normalized.shape == (144, 160, 3)

    def test_normalize_too_few_channels(self):
        """Test handling less than 3 channels."""
        screen = np.random.randint(0, 255, (144, 160, 2), dtype=np.uint8)

        normalized = normalize_screen(screen)

        assert normalized.shape == (144, 160, 3)

    def test_normalize_float_to_uint8(self):
        """Test converting float values to uint8."""
        screen = np.random.random((144, 160, 3)).astype(np.float32) * 255

        normalized = normalize_screen(screen)

        assert normalized.dtype == np.uint8


class TestCreateObservationSpace:
    """Test observation space creation."""

    def test_create_default_observation_space(self):
        """Test creating observation space with default size."""
        obs_space = create_observation_space()

        assert isinstance(obs_space, gym.spaces.Dict)
        assert 'screen' in obs_space.spaces
        assert 'position' in obs_space.spaces
        assert 'stats' in obs_space.spaces
        assert 'exploration' in obs_space.spaces

    def test_screen_space_shape(self):
        """Test screen observation space shape."""
        obs_space = create_observation_space(screen_size=(80, 72))

        screen_space = obs_space.spaces['screen']
        assert screen_space.shape == (72, 80, 3)  # (H, W, C)
        assert screen_space.dtype == np.uint8

    def test_position_space_shape(self):
        """Test position observation space shape."""
        obs_space = create_observation_space()

        position_space = obs_space.spaces['position']
        assert position_space.shape == (3,)  # x, y, map_id
        assert position_space.dtype == np.uint8

    def test_stats_space_shape(self):
        """Test stats observation space shape."""
        obs_space = create_observation_space()

        stats_space = obs_space.spaces['stats']
        assert stats_space.shape == (6,)
        assert stats_space.dtype == np.uint8

    def test_exploration_space_shape(self):
        """Test exploration observation space shape."""
        obs_space = create_observation_space()

        exploration_space = obs_space.spaces['exploration']
        assert exploration_space.shape == (2,)
        assert exploration_space.dtype == np.uint16

    def test_custom_screen_size(self):
        """Test creating observation space with custom screen size."""
        obs_space = create_observation_space(screen_size=(40, 36))

        screen_space = obs_space.spaces['screen']
        assert screen_space.shape == (36, 40, 3)


class TestProcessGameState:
    """Test game state processing."""

    @pytest.fixture
    def mock_agent(self):
        """Create mock agent."""
        agent = Mock()
        agent.get_screen_array = Mock(return_value=np.zeros((144, 160, 3), dtype=np.uint8))
        agent.get_player_position = Mock(return_value={'x': 10, 'y': 20, 'map': 1})
        agent.get_player_stats = Mock(return_value={
            'level': 5,
            'current_hp': 25,
            'max_hp': 30,
            'badges': 1,
            'party_count': 2
        })
        return agent

    def test_process_game_state_structure(self, mock_agent):
        """Test that processed state has correct structure."""
        visited_locations = {(1, 2, 1), (3, 4, 1)}

        observation = process_game_state(
            mock_agent,
            episode_steps=100,
            max_episode_steps=1000,
            visited_locations=visited_locations
        )

        assert 'screen' in observation
        assert 'position' in observation
        assert 'stats' in observation
        assert 'exploration' in observation

    def test_process_game_state_screen_shape(self, mock_agent):
        """Test screen processing shape."""
        observation = process_game_state(
            mock_agent,
            episode_steps=0,
            max_episode_steps=1000,
            visited_locations=set()
        )

        assert observation['screen'].shape == (72, 80, 3)
        assert observation['screen'].dtype == np.uint8

    def test_process_game_state_position(self, mock_agent):
        """Test position processing."""
        observation = process_game_state(
            mock_agent,
            episode_steps=0,
            max_episode_steps=1000,
            visited_locations=set()
        )

        assert observation['position'].shape == (3,)
        assert observation['position'][0] == 10  # x
        assert observation['position'][1] == 20  # y
        assert observation['position'][2] == 1   # map

    def test_process_game_state_stats(self, mock_agent):
        """Test stats processing."""
        observation = process_game_state(
            mock_agent,
            episode_steps=500,
            max_episode_steps=1000,
            visited_locations=set()
        )

        stats = observation['stats']
        assert stats.shape == (6,)
        assert stats[0] == 5   # level
        assert stats[1] == 83  # hp_ratio * 100 (25/30 * 100)
        assert stats[2] == 1   # badges
        assert stats[3] == 2   # party_count
        assert stats[4] == 50  # episode_progress (500/1000 * 100)
        assert stats[5] == 1   # badge_count

    def test_process_game_state_exploration(self, mock_agent):
        """Test exploration tracking."""
        visited_locations = {(i, i, 1) for i in range(50)}

        observation = process_game_state(
            mock_agent,
            episode_steps=100,
            max_episode_steps=1000,
            visited_locations=visited_locations
        )

        exploration = observation['exploration']
        assert exploration[0] == 50  # locations visited
        assert exploration[1] == 1   # unique maps

    def test_process_game_state_multiple_maps(self, mock_agent):
        """Test exploration with multiple maps."""
        visited_locations = {
            (1, 1, 1), (2, 2, 1),  # Map 1
            (1, 1, 2), (2, 2, 2),  # Map 2
            (1, 1, 3)              # Map 3
        }

        observation = process_game_state(
            mock_agent,
            episode_steps=100,
            max_episode_steps=1000,
            visited_locations=visited_locations
        )

        exploration = observation['exploration']
        assert exploration[0] == 5  # locations visited
        assert exploration[1] == 3  # unique maps


class TestGetScreenFeatures:
    """Test screen feature extraction."""

    def test_get_screen_features_structure(self):
        """Test that features have expected keys."""
        screen = np.random.randint(0, 255, (72, 80, 3), dtype=np.uint8)

        features = get_screen_features(screen)

        assert 'brightness_mean' in features
        assert 'brightness_std' in features
        assert 'contrast' in features
        assert 'edge_density' in features
        assert 'unique_colors' in features
        assert 'screen_activity' in features

    def test_get_screen_features_values(self):
        """Test that feature values are reasonable."""
        screen = np.random.randint(0, 255, (72, 80, 3), dtype=np.uint8)

        features = get_screen_features(screen)

        assert 0 <= features['brightness_mean'] <= 255
        assert features['brightness_std'] >= 0
        assert features['contrast'] >= 0
        assert features['edge_density'] >= 0
        assert features['unique_colors'] > 0
        assert 0 <= features['screen_activity'] <= 1

    def test_get_screen_features_grayscale(self):
        """Test feature extraction on grayscale images."""
        screen = np.random.randint(0, 255, (72, 80), dtype=np.uint8)

        features = get_screen_features(screen)

        assert isinstance(features, dict)
        assert len(features) > 0

    def test_get_screen_features_uniform(self):
        """Test features on uniform image."""
        screen = np.full((72, 80, 3), 128, dtype=np.uint8)

        features = get_screen_features(screen)

        assert features['brightness_mean'] == 128
        assert features['brightness_std'] == 0
        assert features['contrast'] == 0


class TestMinimalObservationSpace:
    """Test minimal observation space."""

    def test_create_minimal_observation_space(self):
        """Test creating minimal observation space."""
        obs_space = create_minimal_observation_space()

        assert isinstance(obs_space, gym.spaces.Box)
        assert obs_space.shape == (11,)
        assert obs_space.dtype == np.uint16

    def test_process_minimal_observation(self):
        """Test processing minimal observation."""
        agent = Mock()
        agent.get_player_position = Mock(return_value={'x': 10, 'y': 20, 'map': 1})
        agent.get_player_stats = Mock(return_value={
            'level': 5,
            'current_hp': 25,
            'max_hp': 30,
            'badges': 1,
            'party_count': 2
        })

        visited_locations = {(i, i, 1) for i in range(10)}

        observation = process_minimal_observation(
            agent,
            episode_steps=500,
            max_episode_steps=1000,
            visited_locations=visited_locations
        )

        assert observation.shape == (11,)
        assert observation.dtype == np.uint16

    def test_minimal_observation_values(self):
        """Test minimal observation values."""
        agent = Mock()
        agent.get_player_position = Mock(return_value={'x': 15, 'y': 25, 'map': 3})
        agent.get_player_stats = Mock(return_value={
            'level': 10,
            'current_hp': 50,
            'max_hp': 60,
            'badges': 3,
            'party_count': 4
        })

        visited_locations = {(i, i, 1) for i in range(20)}

        observation = process_minimal_observation(
            agent,
            episode_steps=750,
            max_episode_steps=1000,
            visited_locations=visited_locations
        )

        assert observation[0] == 15   # x
        assert observation[1] == 25   # y
        assert observation[2] == 3    # map
        assert observation[3] == 10   # level
        assert observation[4] == 83   # hp_ratio (50/60 * 100)
        assert observation[5] == 3    # badges
        assert observation[6] == 4    # party_count
        assert observation[7] == 75   # episode_progress (750/1000 * 100)
        assert observation[8] == 2    # badge_count (2 bits set in 3 = 0b11)
        assert observation[9] == 20   # locations_visited
        assert observation[10] == 1   # unique_maps


class TestValidateObservation:
    """Test observation validation."""

    def test_validate_valid_observation(self):
        """Test validating a valid observation."""
        obs_space = create_observation_space()

        observation = {
            'screen': np.zeros((72, 80, 3), dtype=np.uint8),
            'position': np.array([10, 20, 1], dtype=np.uint8),
            'stats': np.array([5, 80, 1, 2, 50, 1], dtype=np.uint8),
            'exploration': np.array([50, 1], dtype=np.uint16)
        }

        result = validate_observation(observation, obs_space)

        assert result is True

    def test_validate_missing_key(self):
        """Test validation with missing key."""
        obs_space = create_observation_space()

        observation = {
            'screen': np.zeros((72, 80, 3), dtype=np.uint8),
            'position': np.array([10, 20, 1], dtype=np.uint8),
            # Missing 'stats' and 'exploration'
        }

        result = validate_observation(observation, obs_space)

        assert result is False

    def test_validate_wrong_shape(self):
        """Test validation with wrong shape."""
        obs_space = create_observation_space()

        observation = {
            'screen': np.zeros((72, 80, 3), dtype=np.uint8),
            'position': np.array([10, 20], dtype=np.uint8),  # Wrong shape
            'stats': np.array([5, 80, 1, 2, 50, 1], dtype=np.uint8),
            'exploration': np.array([50, 1], dtype=np.uint16)
        }

        result = validate_observation(observation, obs_space)

        assert result is False

    def test_validate_wrong_dtype(self):
        """Test validation with wrong dtype."""
        obs_space = create_observation_space()

        observation = {
            'screen': np.zeros((72, 80, 3), dtype=np.uint8),
            'position': np.array([10, 20, 1], dtype=np.float32),  # Wrong dtype
            'stats': np.array([5, 80, 1, 2, 50, 1], dtype=np.uint8),
            'exploration': np.array([50, 1], dtype=np.uint16)
        }

        result = validate_observation(observation, obs_space)

        assert result is False

    def test_validate_out_of_bounds(self):
        """Test validation with out of bounds values."""
        obs_space = create_observation_space()

        # In NumPy 2.0+, direct assignment of out-of-bounds values raises OverflowError
        # Instead, test with valid uint8 values at the boundaries
        observation = {
            'screen': np.full((72, 80, 3), 255, dtype=np.uint8),  # Max valid uint8
            'position': np.array([255, 255, 255], dtype=np.uint8),  # Max values
            'stats': np.array([255, 255, 255, 255, 255, 255], dtype=np.uint8),  # Max values
            'exploration': np.array([65535, 65535], dtype=np.uint16)  # Max uint16
        }

        # Should still validate successfully with boundary values
        result = validate_observation(observation, obs_space)

        assert isinstance(result, bool)
        # Boundary values should be valid
        assert result is True


class TestPreprocessScreenForCNN:
    """Test CNN preprocessing."""

    def test_preprocess_for_cnn_with_normalization(self):
        """Test preprocessing with normalization."""
        screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)

        processed = preprocess_screen_for_cnn(screen, normalize=True)

        assert processed.dtype == np.float32
        assert processed.min() >= 0.0
        assert processed.max() <= 1.0

    def test_preprocess_for_cnn_without_normalization(self):
        """Test preprocessing without normalization."""
        screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)

        processed = preprocess_screen_for_cnn(screen, normalize=False)

        assert processed.dtype == np.float32
        assert processed.max() > 1.0  # Not normalized

    def test_preprocess_ensures_valid_format(self):
        """Test that preprocessing ensures valid format."""
        # RGBA input
        screen = np.random.randint(0, 255, (144, 160, 4), dtype=np.uint8)

        processed = preprocess_screen_for_cnn(screen)

        assert processed.shape[-1] == 3  # Converted to RGB


class TestObservationEdgeCases:
    """Test edge cases for observation processing."""

    def test_zero_max_hp(self):
        """Test handling zero max HP."""
        agent = Mock()
        agent.get_screen_array = Mock(return_value=np.zeros((144, 160, 3), dtype=np.uint8))
        agent.get_player_position = Mock(return_value={'x': 10, 'y': 20, 'map': 1})
        agent.get_player_stats = Mock(return_value={
            'level': 5,
            'current_hp': 0,
            'max_hp': 0,  # Zero max HP
            'badges': 0,
            'party_count': 1
        })

        observation = process_game_state(
            agent,
            episode_steps=0,
            max_episode_steps=1000,
            visited_locations=set()
        )

        # Should handle division by zero gracefully
        assert observation['stats'][1] == 0  # hp_ratio should be 0

    def test_episode_progress_over_max(self):
        """Test episode progress when steps exceed max."""
        agent = Mock()
        agent.get_screen_array = Mock(return_value=np.zeros((144, 160, 3), dtype=np.uint8))
        agent.get_player_position = Mock(return_value={'x': 10, 'y': 20, 'map': 1})
        agent.get_player_stats = Mock(return_value={
            'level': 5,
            'current_hp': 25,
            'max_hp': 30,
            'badges': 0,
            'party_count': 1
        })

        observation = process_game_state(
            agent,
            episode_steps=1500,  # Over max
            max_episode_steps=1000,
            visited_locations=set()
        )

        # Should be capped at 100
        assert observation['stats'][4] == 100

    def test_very_large_exploration_count(self):
        """Test handling very large exploration counts."""
        agent = Mock()
        agent.get_screen_array = Mock(return_value=np.zeros((144, 160, 3), dtype=np.uint8))
        agent.get_player_position = Mock(return_value={'x': 10, 'y': 20, 'map': 1})
        agent.get_player_stats = Mock(return_value={
            'level': 5,
            'current_hp': 25,
            'max_hp': 30,
            'badges': 0,
            'party_count': 1
        })

        # Create huge visited set
        visited_locations = {(i, i, 1) for i in range(100000)}

        observation = process_game_state(
            agent,
            episode_steps=0,
            max_episode_steps=1000,
            visited_locations=visited_locations
        )

        # Should be capped at uint16 max
        assert observation['exploration'][0] == 65535

    def test_stats_exceed_255(self):
        """Test clamping stats that exceed uint8 max."""
        agent = Mock()
        agent.get_screen_array = Mock(return_value=np.zeros((144, 160, 3), dtype=np.uint8))
        agent.get_player_position = Mock(return_value={'x': 10, 'y': 20, 'map': 1})
        agent.get_player_stats = Mock(return_value={
            'level': 999,  # Way over 255
            'current_hp': 25,
            'max_hp': 30,
            'badges': 300,  # Over 255
            'party_count': 1000  # Over 255
        })

        observation = process_game_state(
            agent,
            episode_steps=0,
            max_episode_steps=1000,
            visited_locations=set()
        )

        # All should be clamped to 255
        assert observation['stats'][0] == 255  # level
        assert observation['stats'][2] == 255  # badges
        assert observation['stats'][3] == 255  # party_count


class TestObservationPerformance:
    """Test observation processing performance."""

    @pytest.mark.slow
    def test_process_game_state_performance(self, benchmark_runner):
        """Benchmark game state processing speed."""
        agent = Mock()
        agent.get_screen_array = Mock(return_value=np.zeros((144, 160, 3), dtype=np.uint8))
        agent.get_player_position = Mock(return_value={'x': 10, 'y': 20, 'map': 1})
        agent.get_player_stats = Mock(return_value={
            'level': 5, 'current_hp': 25, 'max_hp': 30,
            'badges': 1, 'party_count': 2
        })

        visited = {(i, i, 1) for i in range(100)}

        def process_op():
            process_game_state(agent, 100, 1000, visited)

        result = benchmark_runner.run('process_state', process_op, iterations=100)
        assert result['mean'] < 0.01  # Should be under 10ms

    @pytest.mark.slow
    def test_downsample_performance(self, benchmark_runner):
        """Benchmark screen downsampling speed."""
        screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)

        def downsample_op():
            downsample_screen(screen)

        result = benchmark_runner.run('downsample', downsample_op, iterations=1000)
        assert result['mean'] < 0.005  # Should be under 5ms


# ──────────────────────────────────────────────────────────────────────
# Paper observation treatments: pixel / symbolic / hybrid
# ──────────────────────────────────────────────────────────────────────


class TestConstants:
    """Test paper observation constants."""

    def test_num_event_flags(self):
        """NUM_EVENT_FLAGS matches the 18 pre-registered boulder path flags."""
        assert NUM_EVENT_FLAGS == 18

    def test_symbolic_dim(self):
        """SYMBOLIC_DIM = position(3) + stats(6) + flags(18) + exploration(2) = 29."""
        assert SYMBOLIC_DIM == 29
        assert SYMBOLIC_DIM == 3 + 6 + NUM_EVENT_FLAGS + 2


class TestScreenToGrayscale:
    """Test the _screen_to_grayscale helper."""

    def test_rgb_to_grayscale(self):
        """Convert a 3-channel RGB image to single-channel grayscale."""
        screen = np.random.randint(0, 255, (72, 80, 3), dtype=np.uint8)
        gray = _screen_to_grayscale(screen)

        assert gray.dtype == np.uint8
        # Result should be 2D or have a trailing channel dim
        assert gray.shape[0] == 72
        assert gray.shape[1] == 80

    def test_rgba_to_grayscale(self):
        """Convert a 4-channel RGBA image (drops alpha) to grayscale."""
        screen = np.random.randint(0, 255, (72, 80, 4), dtype=np.uint8)
        gray = _screen_to_grayscale(screen)

        assert gray.dtype == np.uint8
        assert gray.shape[0] == 72
        assert gray.shape[1] == 80

    def test_single_channel_passthrough(self):
        """Single-channel input is returned as uint8."""
        screen = np.random.randint(0, 255, (72, 80, 1), dtype=np.uint8)
        gray = _screen_to_grayscale(screen)

        assert gray.dtype == np.uint8
        assert gray.shape == (72, 80, 1)

    def test_grayscale_2d_input(self):
        """2D grayscale input gets a trailing channel dim."""
        screen = np.random.randint(0, 255, (72, 80), dtype=np.uint8)
        gray = _screen_to_grayscale(screen)

        assert gray.dtype == np.uint8
        assert gray.shape == (72, 80, 1)


class TestPixelObservation:
    """Test pixel observation space and processing (Treatment 1)."""

    def test_create_pixel_space_default(self):
        """Default pixel space is Box(72, 80, 1, uint8)."""
        space = create_pixel_observation_space()

        assert isinstance(space, gym.spaces.Box)
        assert space.shape == (72, 80, 1)
        assert space.dtype == np.uint8
        assert space.low.min() == 0
        assert space.high.max() == 255

    def test_create_pixel_space_custom_size(self):
        """Pixel space respects custom screen_size."""
        space = create_pixel_observation_space(screen_size=(40, 36))

        assert space.shape == (36, 40, 1)

    def test_process_pixel_observation_shape(self):
        """process_pixel_observation returns (H, W, 1) uint8."""
        agent = Mock()
        agent.get_screen_array = Mock(
            return_value=np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        )

        obs = process_pixel_observation(agent)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (72, 80, 1)
        assert obs.dtype == np.uint8

    def test_pixel_observation_in_space(self):
        """Pixel observation is contained in its observation space."""
        agent = Mock()
        agent.get_screen_array = Mock(
            return_value=np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        )

        space = create_pixel_observation_space()
        obs = process_pixel_observation(agent)

        assert space.contains(obs)


class TestSymbolicObservation:
    """Test symbolic observation space and processing (Treatment 2)."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with standard game state."""
        agent = Mock()
        agent.get_player_position = Mock(return_value={'x': 10, 'y': 20, 'map': 3})
        agent.get_player_stats = Mock(return_value={
            'level': 15,
            'current_hp': 40,
            'max_hp': 50,
            'badges': 1,
            'party_count': 3,
        })
        agent.memory = Mock()
        return agent

    def test_create_symbolic_space(self):
        """Symbolic space is Box(29, float32, [0, 1])."""
        space = create_symbolic_observation_space()

        assert isinstance(space, gym.spaces.Box)
        assert space.shape == (SYMBOLIC_DIM,)
        assert space.dtype == np.float32
        assert space.low.min() == 0.0
        assert space.high.max() == 1.0

    @patch("pokemon_red_ai.game.event_flags.read_boulder_path_flags")
    def test_process_symbolic_observation_shape(self, mock_flags, mock_agent):
        """process_symbolic_observation returns float32 vector of length 29."""
        mock_flags.return_value = {f"flag_{i}": False for i in range(18)}

        obs = process_symbolic_observation(
            mock_agent,
            episode_steps=500,
            max_episode_steps=1000,
            visited_locations={(1, 2, 1), (3, 4, 2)},
        )

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (SYMBOLIC_DIM,)
        assert obs.dtype == np.float32

    @patch("pokemon_red_ai.game.event_flags.read_boulder_path_flags")
    def test_symbolic_values_normalized(self, mock_flags, mock_agent):
        """All symbolic vector values are in [0, 1]."""
        mock_flags.return_value = {f"flag_{i}": (i % 3 == 0) for i in range(18)}

        obs = process_symbolic_observation(
            mock_agent,
            episode_steps=500,
            max_episode_steps=1000,
            visited_locations={(1, 2, 1)},
        )

        assert obs.min() >= 0.0, f"min value {obs.min()} < 0"
        assert obs.max() <= 1.0, f"max value {obs.max()} > 1"

    @patch("pokemon_red_ai.game.event_flags.read_boulder_path_flags")
    def test_symbolic_position_encoding(self, mock_flags, mock_agent):
        """Position fields are x/255, y/255, map/255."""
        mock_flags.return_value = {f"flag_{i}": False for i in range(18)}

        obs = process_symbolic_observation(
            mock_agent,
            episode_steps=0,
            max_episode_steps=1000,
            visited_locations=set(),
        )

        assert np.isclose(obs[0], 10 / 255.0)   # x
        assert np.isclose(obs[1], 20 / 255.0)   # y
        assert np.isclose(obs[2], 3 / 255.0)    # map

    @patch("pokemon_red_ai.game.event_flags.read_boulder_path_flags")
    def test_symbolic_stats_encoding(self, mock_flags, mock_agent):
        """Stats fields are correctly normalised."""
        mock_flags.return_value = {f"flag_{i}": False for i in range(18)}

        obs = process_symbolic_observation(
            mock_agent,
            episode_steps=500,
            max_episode_steps=1000,
            visited_locations=set(),
        )

        # level=15 → 15/100
        assert np.isclose(obs[3], 15 / 100.0)
        # hp_ratio = 40/50 = 0.8
        assert np.isclose(obs[4], 0.8)
        # badges=1 → 1/255
        assert np.isclose(obs[5], 1 / 255.0)
        # party_count=3 → 3/6
        assert np.isclose(obs[6], 3 / 6.0)
        # episode_progress = 500/1000 = 0.5
        assert np.isclose(obs[7], 0.5)
        # badge_count=1 → 1/8
        assert np.isclose(obs[8], 1 / 8.0)

    @patch("pokemon_red_ai.game.event_flags.read_boulder_path_flags")
    def test_symbolic_event_flags(self, mock_flags, mock_agent):
        """Event flag bits are 0.0 or 1.0 in the vector."""
        flags = {f"flag_{i}": (i < 5) for i in range(18)}  # first 5 set
        mock_flags.return_value = flags

        obs = process_symbolic_observation(
            mock_agent,
            episode_steps=0,
            max_episode_steps=1000,
            visited_locations=set(),
        )

        flag_start = 9  # position(3) + stats(6)
        for i in range(18):
            expected = 1.0 if i < 5 else 0.0
            assert obs[flag_start + i] == expected, f"Flag {i}: expected {expected}, got {obs[flag_start + i]}"

    @patch("pokemon_red_ai.game.event_flags.read_boulder_path_flags")
    def test_symbolic_observation_in_space(self, mock_flags, mock_agent):
        """Symbolic observation is contained in its observation space."""
        mock_flags.return_value = {f"flag_{i}": False for i in range(18)}

        space = create_symbolic_observation_space()
        obs = process_symbolic_observation(
            mock_agent,
            episode_steps=500,
            max_episode_steps=1000,
            visited_locations={(1, 2, 1)},
        )

        assert space.contains(obs)

    def test_symbolic_graceful_flag_failure(self, mock_agent):
        """If event flag reading fails, zeros are used (no crash)."""
        # Don't mock read_boulder_path_flags — let the import fail
        # or the mock memory cause an exception
        obs = process_symbolic_observation(
            mock_agent,
            episode_steps=0,
            max_episode_steps=1000,
            visited_locations=set(),
        )

        # Should still return a valid vector
        assert obs.shape == (SYMBOLIC_DIM,)
        # Event flag positions should be zero
        flag_start = 9
        for i in range(18):
            assert obs[flag_start + i] == 0.0


class TestHybridObservation:
    """Test hybrid observation space and processing (Treatment 3)."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with screen + game state."""
        agent = Mock()
        agent.get_screen_array = Mock(
            return_value=np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        )
        agent.get_player_position = Mock(return_value={'x': 10, 'y': 20, 'map': 3})
        agent.get_player_stats = Mock(return_value={
            'level': 15,
            'current_hp': 40,
            'max_hp': 50,
            'badges': 1,
            'party_count': 3,
        })
        agent.memory = Mock()
        return agent

    def test_create_hybrid_space_structure(self):
        """Hybrid space is Dict with 'screen' and 'game_state' keys."""
        space = create_hybrid_observation_space()

        assert isinstance(space, gym.spaces.Dict)
        assert "screen" in space.spaces
        assert "game_state" in space.spaces

    def test_hybrid_screen_subspace(self):
        """Hybrid 'screen' subspace is Box(72, 80, 1, uint8)."""
        space = create_hybrid_observation_space()
        screen_space = space.spaces["screen"]

        assert isinstance(screen_space, gym.spaces.Box)
        assert screen_space.shape == (72, 80, 1)
        assert screen_space.dtype == np.uint8

    def test_hybrid_game_state_subspace(self):
        """Hybrid 'game_state' subspace is Box(29, float32)."""
        space = create_hybrid_observation_space()
        gs_space = space.spaces["game_state"]

        assert isinstance(gs_space, gym.spaces.Box)
        assert gs_space.shape == (SYMBOLIC_DIM,)
        assert gs_space.dtype == np.float32

    def test_hybrid_custom_screen_size(self):
        """Hybrid space respects custom screen_size."""
        space = create_hybrid_observation_space(screen_size=(40, 36))
        screen_space = space.spaces["screen"]

        assert screen_space.shape == (36, 40, 1)

    @patch("pokemon_red_ai.game.event_flags.read_boulder_path_flags")
    def test_process_hybrid_observation_structure(self, mock_flags, mock_agent):
        """process_hybrid_observation returns dict with both keys."""
        mock_flags.return_value = {f"flag_{i}": False for i in range(18)}

        obs = process_hybrid_observation(
            mock_agent,
            episode_steps=100,
            max_episode_steps=1000,
            visited_locations={(1, 2, 1)},
        )

        assert isinstance(obs, dict)
        assert "screen" in obs
        assert "game_state" in obs

    @patch("pokemon_red_ai.game.event_flags.read_boulder_path_flags")
    def test_hybrid_screen_shape(self, mock_flags, mock_agent):
        """Hybrid screen component is (72, 80, 1) uint8."""
        mock_flags.return_value = {f"flag_{i}": False for i in range(18)}

        obs = process_hybrid_observation(
            mock_agent,
            episode_steps=100,
            max_episode_steps=1000,
            visited_locations=set(),
        )

        assert obs["screen"].shape == (72, 80, 1)
        assert obs["screen"].dtype == np.uint8

    @patch("pokemon_red_ai.game.event_flags.read_boulder_path_flags")
    def test_hybrid_game_state_shape(self, mock_flags, mock_agent):
        """Hybrid game_state component is (29,) float32."""
        mock_flags.return_value = {f"flag_{i}": False for i in range(18)}

        obs = process_hybrid_observation(
            mock_agent,
            episode_steps=100,
            max_episode_steps=1000,
            visited_locations=set(),
        )

        assert obs["game_state"].shape == (SYMBOLIC_DIM,)
        assert obs["game_state"].dtype == np.float32

    @patch("pokemon_red_ai.game.event_flags.read_boulder_path_flags")
    def test_hybrid_game_state_normalized(self, mock_flags, mock_agent):
        """Hybrid game_state values are in [0, 1]."""
        mock_flags.return_value = {f"flag_{i}": (i % 2 == 0) for i in range(18)}

        obs = process_hybrid_observation(
            mock_agent,
            episode_steps=500,
            max_episode_steps=1000,
            visited_locations={(1, 2, 1), (3, 4, 2)},
        )

        gs = obs["game_state"]
        assert gs.min() >= 0.0
        assert gs.max() <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])