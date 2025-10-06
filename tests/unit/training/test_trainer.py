"""
Unit tests for PokemonTrainer class.

Tests the main trainer orchestration including initialization,
training, testing, and statistics management.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

from pokemon_red_ai.training.trainer import PokemonTrainer
from pokemon_red_ai.environment import RewardConfig


class TestTrainerInitialization:
    """Test trainer initialization."""

    def test_trainer_initialization(self, mock_rom_file, temp_save_dir):
        """Test basic trainer initialization."""
        trainer = PokemonTrainer(
            rom_path=str(mock_rom_file),
            save_dir=str(temp_save_dir)
        )

        assert trainer.rom_path == str(mock_rom_file)
        assert trainer.save_dir == str(temp_save_dir)
        assert trainer.reward_strategy == "standard"
        assert trainer.env is None
        assert trainer.model is None

    def test_trainer_custom_reward_strategy(self, mock_rom_file, temp_save_dir):
        """Test trainer with custom reward strategy."""
        trainer = PokemonTrainer(
            rom_path=str(mock_rom_file),
            save_dir=str(temp_save_dir),
            reward_strategy="exploration"
        )

        assert trainer.reward_strategy == "exploration"

    def test_trainer_custom_reward_config(self, mock_rom_file, temp_save_dir):
        """Test trainer with custom reward config."""
        reward_config = RewardConfig(exploration_reward=5.0)

        trainer = PokemonTrainer(
            rom_path=str(mock_rom_file),
            save_dir=str(temp_save_dir),
            reward_config=reward_config
        )

        assert trainer.reward_config == reward_config

    def test_trainer_custom_observation_type(self, mock_rom_file, temp_save_dir):
        """Test trainer with custom observation type."""
        trainer = PokemonTrainer(
            rom_path=str(mock_rom_file),
            save_dir=str(temp_save_dir),
            observation_type="minimal"
        )

        assert trainer.observation_type == "minimal"


class TestEnvironmentCreation:
    """Test environment creation methods."""

    def test_create_environment(self, mock_rom_file, temp_save_dir):
        """Test environment creation."""
        with patch('pokemon_red_ai.training.trainer.PokemonRedGymEnv') as MockEnv:
            mock_env = Mock()
            MockEnv.return_value = mock_env

            trainer = PokemonTrainer(
                rom_path=str(mock_rom_file),
                save_dir=str(temp_save_dir)
            )

            env = trainer.create_environment()

            assert env == mock_env
            MockEnv.assert_called_once()

    def test_create_environment_headless(self, mock_rom_file, temp_save_dir):
        """Test creating headless environment."""
        with patch('pokemon_red_ai.training.trainer.PokemonRedGymEnv') as MockEnv:
            trainer = PokemonTrainer(
                rom_path=str(mock_rom_file),
                save_dir=str(temp_save_dir)
            )

            trainer.create_environment(headless=True)

            # Verify headless parameter was passed
            call_kwargs = MockEnv.call_args[1]
            assert call_kwargs['headless'] is True

    def test_create_environment_custom_max_steps(self, mock_rom_file, temp_save_dir):
        """Test creating environment with custom max steps."""
        with patch('pokemon_red_ai.training.trainer.PokemonRedGymEnv') as MockEnv:
            trainer = PokemonTrainer(
                rom_path=str(mock_rom_file),
                save_dir=str(temp_save_dir)
            )

            trainer.create_environment(max_episode_steps=10000)

            call_kwargs = MockEnv.call_args[1]
            assert call_kwargs['max_episode_steps'] == 10000


class TestModelCreation:
    """Test model creation methods."""

    def test_create_model(self, mock_rom_file, temp_save_dir, mock_env):
        """Test model creation."""
        with patch('pokemon_red_ai.training.trainer.create_ppo_model') as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model

            trainer = PokemonTrainer(
                rom_path=str(mock_rom_file),
                save_dir=str(temp_save_dir)
            )

            model = trainer.create_model(mock_env)

            assert model == mock_model
            mock_create.assert_called_once()

    def test_create_model_custom_hyperparameters(self, mock_rom_file, temp_save_dir, mock_env):
        """Test model creation with custom hyperparameters."""
        with patch('pokemon_red_ai.training.trainer.create_ppo_model') as mock_create:
            trainer = PokemonTrainer(
                rom_path=str(mock_rom_file),
                save_dir=str(temp_save_dir)
            )

            trainer.create_model(
                mock_env,
                learning_rate=0.0001,
                batch_size=128
            )

            # Verify custom parameters were passed
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs['learning_rate'] == 0.0001
            assert call_kwargs['batch_size'] == 128


class TestTraining:
    """Test training functionality."""

    def test_train_basic(self, mock_rom_file, temp_save_dir):
        """Test basic training workflow."""
        with patch('pokemon_red_ai.training.trainer.PokemonRedGymEnv') as MockEnv, \
                patch('pokemon_red_ai.training.trainer.Monitor') as MockMonitor, \
                patch('pokemon_red_ai.training.trainer.create_ppo_model') as MockModel, \
                patch('pokemon_red_ai.training.trainer.TrainingCallback') as MockCallback:
            mock_env = Mock()
            mock_model = Mock()
            mock_callback = Mock()

            MockEnv.return_value = mock_env
            MockMonitor.return_value = mock_env
            MockModel.return_value = mock_model
            MockCallback.return_value = mock_callback

            trainer = PokemonTrainer(
                rom_path=str(mock_rom_file),
                save_dir=str(temp_save_dir)
            )

            trainer.train(total_timesteps=1000, show_plots=False)

            # Verify training was called
            mock_model.learn.assert_called_once()
            mock_model.save.assert_called()
            mock_env.close.assert_called()

    def test_train_with_plots(self, mock_rom_file, temp_save_dir):
        """Test training with plots enabled."""
        with patch('pokemon_red_ai.training.trainer.PokemonRedGymEnv') as MockEnv, \
                patch('pokemon_red_ai.training.trainer.Monitor') as MockMonitor, \
                patch('pokemon_red_ai.training.trainer.create_ppo_model') as MockModel, \
                patch('pokemon_red_ai.training.trainer.EnhancedTrainingCallback') as MockCallback:
            mock_env = Mock()
            mock_model = Mock()
            mock_callback = Mock()
            mock_callback.cleanup = Mock()

            MockEnv.return_value = mock_env
            MockMonitor.return_value = mock_env
            MockModel.return_value = mock_model
            MockCallback.return_value = mock_callback

            trainer = PokemonTrainer(
                rom_path=str(mock_rom_file),
                save_dir=str(temp_save_dir)
            )

            trainer.train(total_timesteps=1000, show_plots=True)

            MockCallback.assert_called_once()
            mock_callback.cleanup.assert_called()

    def test_train_keyboard_interrupt(self, mock_rom_file, temp_save_dir):
        """Test training handles keyboard interrupt gracefully."""
        with patch('pokemon_red_ai.training.trainer.PokemonRedGymEnv') as MockEnv, \
                patch('pokemon_red_ai.training.trainer.Monitor') as MockMonitor, \
                patch('pokemon_red_ai.training.trainer.create_ppo_model') as MockModel:
            mock_env = Mock()
            mock_model = Mock()
            mock_model.learn = Mock(side_effect=KeyboardInterrupt())

            MockEnv.return_value = mock_env
            MockMonitor.return_value = mock_env
            MockModel.return_value = mock_model

            trainer = PokemonTrainer(
                rom_path=str(mock_rom_file),
                save_dir=str(temp_save_dir)
            )

            # Should not raise exception
            trainer.train(total_timesteps=1000, show_plots=False)

            # Should still save model
            mock_model.save.assert_called()

    def test_train_custom_hyperparameters(self, mock_rom_file, temp_save_dir):
        """Test training with custom hyperparameters."""
        with patch('pokemon_red_ai.training.trainer.PokemonRedGymEnv') as MockEnv, \
                patch('pokemon_red_ai.training.trainer.Monitor') as MockMonitor, \
                patch('pokemon_red_ai.training.trainer.create_ppo_model') as MockModel:
            mock_env = Mock()
            mock_model = Mock()

            MockEnv.return_value = mock_env
            MockMonitor.return_value = mock_env
            MockModel.return_value = mock_model

            trainer = PokemonTrainer(
                rom_path=str(mock_rom_file),
                save_dir=str(temp_save_dir)
            )

            trainer.train(
                total_timesteps=1000,
                learning_rate=0.0001,
                batch_size=128,
                show_plots=False
            )

            # Verify custom parameters were passed to model creation
            call_kwargs = MockModel.call_args[1]
            assert call_kwargs['learning_rate'] == 0.0001
            assert call_kwargs['batch_size'] == 128


class TestTesting:
    """Test model testing functionality."""

    def test_test_model(self, mock_rom_file, temp_save_dir):
        """Test model testing functionality."""
        with patch('pokemon_red_ai.training.trainer.PokemonRedGymEnv') as MockEnv, \
                patch('pokemon_red_ai.training.trainer.PPO') as MockPPO:
            mock_env = Mock()
            mock_model = Mock()
            mock_model.predict = Mock(return_value=(0, None))

            # Mock environment step
            mock_env.reset = Mock(return_value=(Mock(), {}))
            mock_env.step = Mock(return_value=(
                Mock(),
                10.0,
                True,
                False,
                {'maps_visited': 3, 'badges_earned': 1, 'locations_visited': 50}
            ))

            MockEnv.return_value = mock_env
            MockPPO.load = Mock(return_value=mock_model)

            trainer = PokemonTrainer(
                rom_path=str(mock_rom_file),
                save_dir=str(temp_save_dir)
            )

            # Create a temporary model file
            model_path = temp_save_dir / "test_model.zip"
            model_path.touch()

            results = trainer.test(
                model_path=str(model_path),
                episodes=2,
                render=False
            )

            assert 'episodes_tested' in results
            assert results['episodes_tested'] == 2
            assert 'avg_reward' in results
            assert 'avg_steps' in results

    def test_test_model_with_render(self, mock_rom_file, temp_save_dir):
        """Test model testing with rendering enabled."""
        with patch('pokemon_red_ai.training.trainer.PokemonRedGymEnv') as MockEnv, \
                patch('pokemon_red_ai.training.trainer.PPO') as MockPPO:
            mock_env = Mock()
            mock_model = Mock()
            mock_model.predict = Mock(return_value=(0, None))
            mock_env.reset = Mock(return_value=(Mock(), {}))
            mock_env.step = Mock(return_value=(Mock(), 10.0, True, False, {}))
            mock_env.render = Mock()

            MockEnv.return_value = mock_env
            MockPPO.load = Mock(return_value=mock_model)

            trainer = PokemonTrainer(
                rom_path=str(mock_rom_file),
                save_dir=str(temp_save_dir)
            )

            model_path = temp_save_dir / "model.zip"
            model_path.touch()

            trainer.test(
                model_path=str(model_path),
                episodes=1,
                render=True
            )

            # Verify rendering was used
            assert MockEnv.call_args[1]['headless'] is False


class TestStatistics:
    """Test statistics management."""

    def test_save_training_stats(self, mock_rom_file, temp_save_dir):
        """Test saving training statistics."""
        trainer = PokemonTrainer(
            rom_path=str(mock_rom_file),
            save_dir=str(temp_save_dir)
        )

        trainer.training_stats['total_timesteps'] = 10000
        trainer.training_stats['best_reward'] = 150.0

        trainer.save_training_stats()

        stats_file = Path(temp_save_dir) / 'training_stats.json'
        assert stats_file.exists()

        with open(stats_file) as f:
            loaded_stats = json.load(f)

        assert loaded_stats['total_timesteps'] == 10000
        assert loaded_stats['best_reward'] == 150.0

    def test_load_training_stats(self, mock_rom_file, temp_save_dir):
        """Test loading training statistics."""
        trainer = PokemonTrainer(
            rom_path=str(mock_rom_file),
            save_dir=str(temp_save_dir)
        )

        # Create stats file
        stats = {
            'total_timesteps': 5000,
            'best_reward': 100.0,
            'training_start_time': '2025-01-01T12:00:00'
        }

        stats_file = Path(temp_save_dir) / 'training_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f)

        loaded_stats = trainer.load_training_stats()

        assert loaded_stats['total_timesteps'] == 5000
        assert loaded_stats['best_reward'] == 100.0

    def test_get_training_summary(self, mock_rom_file, temp_save_dir):
        """Test getting training summary."""
        trainer = PokemonTrainer(
            rom_path=str(mock_rom_file),
            save_dir=str(temp_save_dir)
        )

        trainer.training_stats['total_timesteps'] = 10000

        summary = trainer.get_training_summary()

        assert 'trainer_config' in summary
        assert 'training_stats' in summary
        assert summary['trainer_config']['rom_path'] == str(mock_rom_file)

    def test_stats_serialization_datetime(self, mock_rom_file, temp_save_dir):
        """Test datetime serialization in stats."""
        trainer = PokemonTrainer(
            rom_path=str(mock_rom_file),
            save_dir=str(temp_save_dir)
        )

        trainer.training_stats['training_start_time'] = datetime.now()

        # Should serialize datetime without error
        trainer.save_training_stats()

        stats_file = temp_save_dir / 'training_stats.json'
        assert stats_file.exists()

        # Should be able to load back
        with open(stats_file) as f:
            loaded = json.load(f)

        assert 'training_start_time' in loaded
        assert isinstance(loaded['training_start_time'], str)


class TestCleanup:
    """Test cleanup functionality."""

    def test_cleanup_save_files(self, mock_rom_file, temp_save_dir):
        """Test cleanup of ROM save files."""
        trainer = PokemonTrainer(
            rom_path=str(mock_rom_file),
            save_dir=str(temp_save_dir)
        )

        # Create fake save files
        save_files = [
            Path(str(mock_rom_file) + '.ram'),
            Path(str(mock_rom_file) + '.sav')
        ]
        for f in save_files:
            f.touch()

        trainer.cleanup_save_files()

        # Check files are removed
        for f in save_files:
            assert not f.exists()

    def test_cleanup_on_exception(self, mock_rom_file, temp_save_dir):
        """Test trainer cleans up resources on exception."""
        with patch('pokemon_red_ai.training.trainer.PokemonRedGymEnv') as MockEnv, \
                patch('pokemon_red_ai.training.trainer.Monitor') as MockMonitor, \
                patch('pokemon_red_ai.training.trainer.create_ppo_model') as MockModel:
            mock_env = Mock()
            mock_model = Mock()
            mock_model.learn = Mock(side_effect=RuntimeError("Training error"))

            MockEnv.return_value = mock_env
            MockMonitor.return_value = mock_env
            MockModel.return_value = mock_model

            trainer = PokemonTrainer(
                rom_path=str(mock_rom_file),
                save_dir=str(temp_save_dir)
            )

            with pytest.raises(RuntimeError):
                trainer.train(total_timesteps=100)

            # Environment should still be closed
            mock_env.close.assert_called()


class TestErrorHandling:
    """Test error handling."""

    def test_train_with_invalid_rom(self, temp_save_dir):
        """Test training with invalid ROM path."""
        with patch('pokemon_red_ai.training.trainer.PokemonRedGymEnv') as MockEnv:
            MockEnv.side_effect = FileNotFoundError("ROM not found")

            trainer = PokemonTrainer(
                rom_path="nonexistent.gb",
                save_dir=str(temp_save_dir)
            )

            with pytest.raises(Exception):
                trainer.train(total_timesteps=100)

    def test_test_with_invalid_model(self, mock_rom_file, temp_save_dir):
        """Test testing with invalid model path."""
        trainer = PokemonTrainer(
            rom_path=str(mock_rom_file),
            save_dir=str(temp_save_dir)
        )

        with pytest.raises(FileNotFoundError):
            trainer.test(
                model_path="nonexistent_model.zip",
                episodes=1
            )


class TestParameterized:
    """Parameterized tests for trainer."""

    @pytest.mark.parametrize("reward_strategy", [
        "standard", "exploration", "progress", "sparse"
    ])
    def test_trainer_with_reward_strategies(self, mock_rom_file, temp_save_dir, reward_strategy):
        """Test trainer with different reward strategies."""
        trainer = PokemonTrainer(
            rom_path=str(mock_rom_file),
            save_dir=str(temp_save_dir),
            reward_strategy=reward_strategy
        )

        assert trainer.reward_strategy == reward_strategy

    @pytest.mark.parametrize("observation_type", [
        "multi_modal", "minimal", "screen_only"
    ])
    def test_trainer_with_observation_types(self, mock_rom_file, temp_save_dir, observation_type):
        """Test trainer with different observation types."""
        trainer = PokemonTrainer(
            rom_path=str(mock_rom_file),
            save_dir=str(temp_save_dir),
            observation_type=observation_type
        )

        assert trainer.observation_type == observation_type


if __name__ == '__main__':
    pytest.main([__file__, '-v'])