"""
Unit tests for model creation and configuration.

Tests model configuration functions, model creation factories,
and the custom features extractor.
"""

import pytest
import torch
from unittest.mock import Mock, patch

from pokemon_red_ai.training.models import (
    get_model_config,
    create_ppo_model,
    create_a2c_model,
    create_dqn_model,
    create_model,
    get_policy_kwargs_for_observation_type,
    PokemonFeaturesExtractor
)


class TestModelConfig:
    """Test model configuration functions."""

    def test_get_model_config_ppo(self):
        """Test getting PPO model config."""
        config = get_model_config('PPO')

        assert 'learning_rate' in config
        assert 'n_steps' in config
        assert 'batch_size' in config
        assert 'n_epochs' in config
        assert config['device'] == 'auto'

    def test_get_model_config_a2c(self):
        """Test getting A2C model config."""
        config = get_model_config('A2C')

        assert 'learning_rate' in config
        assert 'n_steps' in config
        assert config['device'] == 'auto'

    def test_get_model_config_dqn(self):
        """Test getting DQN model config."""
        config = get_model_config('DQN')

        assert 'learning_rate' in config
        assert 'buffer_size' in config
        assert 'batch_size' in config
        assert 'target_update_interval' in config

    def test_get_model_config_unknown(self):
        """Test getting config for unknown algorithm."""
        # Should default to PPO
        config = get_model_config('UNKNOWN')
        assert 'n_steps' in config  # PPO-specific parameter

    def test_config_values_reasonable(self):
        """Test that config values are reasonable."""
        config = get_model_config('PPO')

        assert config['learning_rate'] > 0
        assert config['learning_rate'] < 1
        assert config['batch_size'] > 0
        assert config['gamma'] > 0
        assert config['gamma'] <= 1


class TestPPOModelCreation:
    """Test PPO model creation."""

    @patch('pokemon_red_ai.training.models.PPO')
    def test_create_ppo_model(self, mock_ppo, mock_env):
        """Test PPO model creation."""
        mock_model = Mock()
        mock_ppo.return_value = mock_model

        model = create_ppo_model(mock_env, tensorboard_log="./logs")

        assert model == mock_model
        mock_ppo.assert_called_once()

    @patch('pokemon_red_ai.training.models.PPO')
    def test_create_ppo_with_custom_params(self, mock_ppo, mock_env):
        """Test PPO creation with custom parameters."""
        mock_model = Mock()
        mock_ppo.return_value = mock_model

        create_ppo_model(
            mock_env,
            learning_rate=0.0001,
            batch_size=128,
            n_epochs=20
        )

        # Verify custom parameters were passed
        call_kwargs = mock_ppo.call_args[1]
        assert call_kwargs['learning_rate'] == 0.0001
        assert call_kwargs['batch_size'] == 128
        assert call_kwargs['n_epochs'] == 20

    @patch('pokemon_red_ai.training.models.PPO')
    def test_create_ppo_with_policy_kwargs(self, mock_ppo, mock_env):
        """Test PPO creation with custom policy kwargs."""
        mock_model = Mock()
        mock_ppo.return_value = mock_model

        custom_policy_kwargs = {
            'net_arch': dict(pi=[128, 64], vf=[128, 64])
        }

        create_ppo_model(
            mock_env,
            policy_kwargs=custom_policy_kwargs
        )

        # Verify policy kwargs were merged
        call_kwargs = mock_ppo.call_args[1]
        assert 'policy_kwargs' in call_kwargs


class TestA2CModelCreation:
    """Test A2C model creation."""

    @patch('pokemon_red_ai.training.models.A2C')
    def test_create_a2c_model(self, mock_a2c, mock_env):
        """Test A2C model creation."""
        mock_model = Mock()
        mock_a2c.return_value = mock_model

        model = create_a2c_model(mock_env, tensorboard_log="./logs")

        assert model == mock_model
        mock_a2c.assert_called_once()

    @patch('pokemon_red_ai.training.models.A2C')
    def test_create_a2c_with_custom_params(self, mock_a2c, mock_env):
        """Test A2C creation with custom parameters."""
        mock_model = Mock()
        mock_a2c.return_value = mock_model

        create_a2c_model(
            mock_env,
            learning_rate=0.001,
            n_steps=10
        )

        call_kwargs = mock_a2c.call_args[1]
        assert call_kwargs['learning_rate'] == 0.001
        assert call_kwargs['n_steps'] == 10


class TestDQNModelCreation:
    """Test DQN model creation."""

    def test_create_dqn_model_multi_modal_fails(self, mock_env):
        """Test DQN fails with multi-modal observations."""
        # DQN doesn't support Dict observation spaces
        with pytest.raises(ValueError, match="single observation space"):
            create_dqn_model(mock_env)

    @patch('pokemon_red_ai.training.models.DQN')
    def test_create_dqn_with_simple_obs_space(self, mock_dqn):
        """Test DQN with simple observation space."""
        # Create env with simple Box observation space (no 'spaces' attribute)
        simple_env = Mock()
        simple_env.observation_space = Mock(spec=['shape'])  # Only has 'shape', not 'spaces'
        simple_env.observation_space.shape = (84, 84, 3)

        mock_model = Mock()
        mock_dqn.return_value = mock_model

        model = create_dqn_model(simple_env)

        assert model == mock_model
        mock_dqn.assert_called_once()


class TestModelFactory:
    """Test create_model factory function."""

    @patch('pokemon_red_ai.training.models.PPO')
    def test_create_model_ppo(self, mock_ppo, mock_env):
        """Test create_model factory with PPO."""
        mock_model = Mock()
        mock_ppo.return_value = mock_model

        model = create_model('PPO', mock_env)

        assert model == mock_model

    @patch('pokemon_red_ai.training.models.A2C')
    def test_create_model_a2c(self, mock_a2c, mock_env):
        """Test create_model factory with A2C."""
        mock_model = Mock()
        mock_a2c.return_value = mock_model

        model = create_model('A2C', mock_env)

        assert model == mock_model

    def test_create_model_unknown_algorithm(self, mock_env):
        """Test create_model with unknown algorithm."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            create_model('UNKNOWN', mock_env)

    @patch('pokemon_red_ai.training.models.PPO')
    def test_create_model_with_kwargs(self, mock_ppo, mock_env):
        """Test create_model passes through kwargs."""
        mock_model = Mock()
        mock_ppo.return_value = mock_model

        create_model('PPO', mock_env, learning_rate=0.0005)

        call_kwargs = mock_ppo.call_args[1]
        assert call_kwargs['learning_rate'] == 0.0005


class TestPolicyKwargs:
    """Test policy kwargs generation."""

    def test_policy_kwargs_multi_modal(self):
        """Test policy kwargs for multi-modal observations."""
        kwargs = get_policy_kwargs_for_observation_type("multi_modal")

        assert 'features_extractor_class' in kwargs
        assert 'net_arch' in kwargs
        assert isinstance(kwargs['net_arch'], dict)
        assert 'pi' in kwargs['net_arch']
        assert 'vf' in kwargs['net_arch']

    def test_policy_kwargs_screen_only(self):
        """Test policy kwargs for screen-only observations."""
        kwargs = get_policy_kwargs_for_observation_type("screen_only")

        assert 'net_arch' in kwargs
        assert isinstance(kwargs['net_arch'], list)

    def test_policy_kwargs_minimal(self):
        """Test policy kwargs for minimal observations."""
        kwargs = get_policy_kwargs_for_observation_type("minimal")

        assert 'net_arch' in kwargs
        assert isinstance(kwargs['net_arch'], dict)

    def test_policy_kwargs_unknown(self):
        """Test policy kwargs for unknown observation type."""
        kwargs = get_policy_kwargs_for_observation_type("unknown")

        # Should return empty dict as fallback
        assert isinstance(kwargs, dict)

    def test_policy_kwargs_have_activation_fn(self):
        """Test policy kwargs include activation function."""
        kwargs = get_policy_kwargs_for_observation_type("multi_modal")

        assert 'activation_fn' in kwargs


class TestPokemonFeaturesExtractor:
    """Test PokemonFeaturesExtractor class."""

    def test_features_extractor_initialization(self, mock_env):
        """Test features extractor initialization."""
        observation_space = mock_env.observation_space

        extractor = PokemonFeaturesExtractor(
            observation_space,
            features_dim=256
        )

        assert extractor._features_dim == 256
        assert hasattr(extractor, 'screen_cnn')
        assert hasattr(extractor, 'position_mlp')
        assert hasattr(extractor, 'stats_mlp')
        assert hasattr(extractor, 'exploration_mlp')
        assert hasattr(extractor, 'fusion')

    def test_features_extractor_custom_features_dim(self, mock_env):
        """Test features extractor with custom features dimension."""
        observation_space = mock_env.observation_space

        extractor = PokemonFeaturesExtractor(
            observation_space,
            features_dim=128
        )

        assert extractor._features_dim == 128

    def test_features_extractor_has_cnn_layers(self, mock_env):
        """Test that CNN has proper layers."""
        observation_space = mock_env.observation_space

        extractor = PokemonFeaturesExtractor(observation_space, features_dim=256)

        # Check that CNN is Sequential
        assert hasattr(extractor.screen_cnn, '__iter__')

    def test_features_extractor_has_mlp_layers(self, mock_env):
        """Test that MLPs have proper layers."""
        observation_space = mock_env.observation_space

        extractor = PokemonFeaturesExtractor(observation_space, features_dim=256)

        # Check that MLPs are Sequential
        assert hasattr(extractor.position_mlp, '__iter__')
        assert hasattr(extractor.stats_mlp, '__iter__')
        assert hasattr(extractor.exploration_mlp, '__iter__')

    def test_features_extractor_with_invalid_obs_space(self):
        """Test features extractor with invalid observation space."""
        invalid_obs_space = Mock()
        invalid_obs_space.spaces = {}  # Missing required keys

        with pytest.raises(KeyError):
            PokemonFeaturesExtractor(invalid_obs_space, features_dim=256)


class TestParameterized:
    """Parameterized tests for models."""

    @pytest.mark.parametrize("algorithm", ["PPO", "A2C"])
    def test_model_creation_algorithms(self, mock_env, algorithm):
        """Test model creation with different algorithms."""
        with patch(f'pokemon_red_ai.training.models.{algorithm}') as MockAlgo:
            mock_model = Mock()
            MockAlgo.return_value = mock_model

            model = create_model(algorithm, mock_env)

            assert model == mock_model

    @pytest.mark.parametrize("obs_type,expected_key", [
        ("multi_modal", "features_extractor_class"),
        ("screen_only", "net_arch"),
        ("minimal", "net_arch"),
    ])
    def test_policy_kwargs_for_different_obs_types(self, obs_type, expected_key):
        """Test policy kwargs for different observation types."""
        kwargs = get_policy_kwargs_for_observation_type(obs_type)

        assert expected_key in kwargs

    @pytest.mark.parametrize("features_dim", [64, 128, 256, 512])
    def test_features_extractor_different_dims(self, mock_env, features_dim):
        """Test features extractor with different dimensions."""
        observation_space = mock_env.observation_space

        extractor = PokemonFeaturesExtractor(
            observation_space,
            features_dim=features_dim
        )

        assert extractor._features_dim == features_dim


class TestModelErrorHandling:
    """Test error handling in model creation."""

    def test_create_model_with_invalid_env(self):
        """Test creating model with invalid environment."""
        with pytest.raises(Exception):
            create_ppo_model(None)

    def test_create_model_with_none_env(self):
        """Test creating model with None environment."""
        with pytest.raises(Exception):
            create_model('PPO', None)

    @patch('pokemon_red_ai.training.models.PPO')
    def test_create_model_handles_sb3_errors(self, mock_ppo, mock_env):
        """Test model creation handles Stable-Baselines3 errors."""
        mock_ppo.side_effect = ValueError("Invalid parameters")

        with pytest.raises(ValueError):
            create_ppo_model(mock_env)


class TestModelIntegration:
    """Integration tests for model creation."""

    @patch('pokemon_red_ai.training.models.PPO')
    def test_model_creation_pipeline(self, mock_ppo, mock_env):
        """Test complete model creation pipeline."""
        mock_model = Mock()
        mock_ppo.return_value = mock_model

        # Get config
        config = get_model_config('PPO')

        # Get policy kwargs
        policy_kwargs = get_policy_kwargs_for_observation_type("multi_modal")

        # Create model
        model = create_ppo_model(
            mock_env,
            policy_kwargs=policy_kwargs,
            **config
        )

        assert model == mock_model
        mock_ppo.assert_called_once()


class TestModelConfiguration:
    """Test model configuration details."""

    def test_ppo_config_completeness(self):
        """Test PPO config has all necessary parameters."""
        config = get_model_config('PPO')

        required_params = [
            'learning_rate', 'n_steps', 'batch_size', 'n_epochs',
            'gamma', 'gae_lambda', 'clip_range', 'ent_coef',
            'vf_coef', 'max_grad_norm', 'device'
        ]

        for param in required_params:
            assert param in config

    def test_a2c_config_completeness(self):
        """Test A2C config has all necessary parameters."""
        config = get_model_config('A2C')

        required_params = [
            'learning_rate', 'n_steps', 'gamma', 'gae_lambda',
            'ent_coef', 'vf_coef', 'max_grad_norm', 'device'
        ]

        for param in required_params:
            assert param in config

    def test_dqn_config_completeness(self):
        """Test DQN config has all necessary parameters."""
        config = get_model_config('DQN')

        required_params = [
            'learning_rate', 'buffer_size', 'batch_size',
            'gamma', 'exploration_fraction', 'device'
        ]

        for param in required_params:
            assert param in config

    def test_config_returns_copy(self):
        """Test that config returns a copy, not reference."""
        config1 = get_model_config('PPO')
        config2 = get_model_config('PPO')

        # Modify one
        config1['learning_rate'] = 999

        # Other should be unchanged
        assert config2['learning_rate'] != 999


class TestFeaturesExtractorDetails:
    """Detailed tests for features extractor."""

    def test_features_extractor_output_dimension(self, mock_env):
        """Test features extractor output dimension."""
        observation_space = mock_env.observation_space
        features_dim = 256

        extractor = PokemonFeaturesExtractor(observation_space, features_dim)

        # The features_dim property should match
        assert extractor._features_dim == features_dim

    def test_features_extractor_cnn_for_small_screen(self):
        """Test CNN works with small screen size (72x80)."""
        obs_space = Mock()
        obs_space.spaces = {
            'screen': Mock(shape=(72, 80, 3)),
            'position': Mock(shape=(3,)),
            'stats': Mock(shape=(6,)),
            'exploration': Mock(shape=(2,))
        }

        # Should not raise error with small screen
        extractor = PokemonFeaturesExtractor(obs_space, features_dim=128)

        assert extractor is not None

    def test_features_extractor_mlp_dimensions(self, mock_env):
        """Test MLP layer dimensions are correct."""
        observation_space = mock_env.observation_space

        extractor = PokemonFeaturesExtractor(observation_space, features_dim=256)

        # Check that MLPs exist and have layers
        assert len(list(extractor.position_mlp.children())) > 0
        assert len(list(extractor.stats_mlp.children())) > 0
        assert len(list(extractor.exploration_mlp.children())) > 0


class TestModelHyperparameters:
    """Test model hyperparameter ranges."""

    def test_learning_rate_reasonable(self):
        """Test learning rates are in reasonable range."""
        for algo in ['PPO', 'A2C', 'DQN']:
            config = get_model_config(algo)
            lr = config['learning_rate']

            assert lr > 1e-6
            assert lr < 1.0

    def test_gamma_in_valid_range(self):
        """Test discount factor is valid."""
        for algo in ['PPO', 'A2C', 'DQN']:
            config = get_model_config(algo)
            gamma = config['gamma']

            assert gamma > 0
            assert gamma <= 1

    def test_batch_size_positive(self):
        """Test batch sizes are positive."""
        for algo in ['PPO', 'A2C', 'DQN']:
            config = get_model_config(algo)
            if 'batch_size' in config:
                assert config['batch_size'] > 0

    def test_entropy_coefficient_range(self):
        """Test entropy coefficient is reasonable."""
        for algo in ['PPO', 'A2C']:
            config = get_model_config(algo)
            ent_coef = config['ent_coef']

            assert ent_coef >= 0
            assert ent_coef < 1


class TestEdgeCases:
    """Test edge cases in model creation."""

    @patch('pokemon_red_ai.training.models.PPO')
    def test_create_model_with_empty_kwargs(self, mock_ppo, mock_env):
        """Test creating model with empty kwargs dict."""
        mock_model = Mock()
        mock_ppo.return_value = mock_model

        model = create_ppo_model(mock_env, **{})

        assert model == mock_model

    def test_policy_kwargs_empty_string(self):
        """Test policy kwargs with empty string."""
        kwargs = get_policy_kwargs_for_observation_type("")

        assert isinstance(kwargs, dict)

    def test_features_extractor_very_small_features_dim(self, mock_env):
        """Test features extractor with very small features dim."""
        observation_space = mock_env.observation_space

        # Should still work with small dimension
        extractor = PokemonFeaturesExtractor(observation_space, features_dim=16)

        assert extractor._features_dim == 16

    def test_features_extractor_very_large_features_dim(self, mock_env):
        """Test features extractor with very large features dim."""
        observation_space = mock_env.observation_space

        # Should still work with large dimension
        extractor = PokemonFeaturesExtractor(observation_space, features_dim=2048)

        assert extractor._features_dim == 2048


class TestModelDocumentation:
    """Test that models have proper documentation."""

    def test_get_model_config_has_docstring(self):
        """Test get_model_config has docstring."""
        assert get_model_config.__doc__ is not None

    def test_create_ppo_model_has_docstring(self):
        """Test create_ppo_model has docstring."""
        assert create_ppo_model.__doc__ is not None

    def test_features_extractor_has_docstring(self):
        """Test PokemonFeaturesExtractor has docstring."""
        assert PokemonFeaturesExtractor.__doc__ is not None


class TestBackwardCompatibility:
    """Test backward compatibility."""

    def test_config_format_unchanged(self):
        """Test config format hasn't changed unexpectedly."""
        config = get_model_config('PPO')

        # These keys should always exist
        essential_keys = ['learning_rate', 'device', 'verbose']
        for key in essential_keys:
            assert key in config

    def test_policy_kwargs_format_unchanged(self):
        """Test policy kwargs format is consistent."""
        kwargs = get_policy_kwargs_for_observation_type("multi_modal")

        # Should always have these for multi_modal
        assert 'features_extractor_class' in kwargs
        assert 'net_arch' in kwargs


if __name__ == '__main__':
    pytest.main([__file__, '-v'])