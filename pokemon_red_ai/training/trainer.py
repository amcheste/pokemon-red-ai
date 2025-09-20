"""
Pokemon Red RL training system.

This module provides the main trainer class that orchestrates the entire
training process using the modular game interface and environment components.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ..environment import PokemonRedGymEnv, RewardConfig
from ..utils import create_directories
from .callbacks import TrainingCallback, EnhancedTrainingCallback
from .models import create_ppo_model, get_model_config

logger = logging.getLogger(__name__)


class PokemonTrainer:
    """
    Complete Pokemon Red training system using modular components.

    This trainer orchestrates the entire RL training pipeline including:
    - Environment creation and management
    - Model initialization and training
    - Progress monitoring and visualization
    - Model saving and evaluation
    """

    def __init__(self,
                 rom_path: str,
                 save_dir: str = './pokemon_training/',
                 reward_strategy: str = "standard",
                 reward_config: Optional[RewardConfig] = None,
                 observation_type: str = "multi_modal"):
        """
        Initialize Pokemon Red trainer.

        Args:
            rom_path: Path to Pokemon Red ROM file
            save_dir: Directory to save training artifacts
            reward_strategy: Reward calculation strategy
            reward_config: Custom reward configuration
            observation_type: Type of observations for the agent
        """
        self.rom_path = rom_path
        self.save_dir = save_dir
        self.reward_strategy = reward_strategy
        self.reward_config = reward_config
        self.observation_type = observation_type

        # Create save directory structure
        create_directories(save_dir)

        # Training components (initialized during training)
        self.env: Optional[gym.Env] = None
        self.model: Optional[PPO] = None

        # Training statistics
        self.training_stats = {
            'episodes_completed': 0,
            'total_timesteps': 0,
            'best_reward': -float('inf'),
            'best_exploration': 0,
            'training_start_time': None,
            'training_end_time': None,
            'total_training_time': None,
            'model_saves': 0,
            'successful_resets': 0,
            'failed_resets': 0
        }

        logger.info(f"PokemonTrainer initialized")
        logger.info(f"  ROM: {rom_path}")
        logger.info(f"  Save directory: {save_dir}")
        logger.info(f"  Reward strategy: {reward_strategy}")
        logger.info(f"  Observation type: {observation_type}")

    def cleanup_save_files(self) -> None:
        """Remove ROM save files to ensure clean training starts."""
        save_files = [
            self.rom_path + '.ram',
            self.rom_path + '.sav',
            self.rom_path + '.rtc'
        ]

        for save_file in save_files:
            try:
                if os.path.exists(save_file):
                    os.remove(save_file)
                    logger.debug(f"Removed save file: {save_file}")
            except Exception as e:
                logger.warning(f"Could not remove {save_file}: {e}")

    def create_environment(self, headless: bool = True, max_episode_steps: int = 5000) -> PokemonRedGymEnv:
        """
        Create and configure Pokemon Red environment.

        Args:
            headless: If True, runs without display window
            max_episode_steps: Maximum steps per episode

        Returns:
            Configured PokemonRedGymEnv instance
        """
        logger.info("Creating Pokemon Red environment...")

        env = PokemonRedGymEnv(
            rom_path=self.rom_path,
            headless=headless,
            max_episode_steps=max_episode_steps,
            reward_strategy=self.reward_strategy,
            reward_config=self.reward_config,
            observation_type=self.observation_type
        )

        logger.info("Environment created successfully")
        return env

    def create_model(self, env: gym.Env, algorithm: str = 'PPO', **model_kwargs) -> PPO:
        """
        Create RL model with specified configuration.

        Args:
            env: Training environment
            algorithm: RL algorithm to use
            **model_kwargs: Additional model arguments

        Returns:
            Configured RL model
        """
        logger.info(f"Creating {algorithm} model...")

        if algorithm == 'PPO':
            # Get default config and merge with provided kwargs
            config = get_model_config('PPO')
            config.update(model_kwargs)

            model = create_ppo_model(
                env=env,
                tensorboard_log=os.path.join(self.save_dir, 'tensorboard/'),
                **config
            )
        else:
            raise ValueError(f"Algorithm {algorithm} not supported")

        logger.info(f"{algorithm} model created successfully")
        return model

    def train(self,
              total_timesteps: int = 100000,
              algorithm: str = 'PPO',
              show_game: bool = False,
              show_plots: bool = False,
              max_episode_steps: int = 5000,
              save_freq: int = 10000,
              **model_kwargs) -> None:
        """
        Train Pokemon Red AI agent.

        Args:
            total_timesteps: Total training timesteps
            algorithm: RL algorithm to use
            show_game: Whether to show game window
            show_plots: Whether to show live training plots
            max_episode_steps: Maximum steps per episode
            save_freq: Frequency of model saves
            **model_kwargs: Additional model configuration
        """
        logger.info("=" * 50)
        logger.info("STARTING POKEMON RED RL TRAINING")
        logger.info("=" * 50)
        logger.info(f"Algorithm: {algorithm}")
        logger.info(f"Total timesteps: {total_timesteps:,}")
        logger.info(f"Save directory: {self.save_dir}")
        logger.info(f"Max episode steps: {max_episode_steps}")
        logger.info(f"Save frequency: {save_freq}")

        if show_game:
            logger.info("Game window will be visible")
        if show_plots:
            logger.info("Live training plots enabled")

        # Clean up any existing save files
        self.cleanup_save_files()

        # Create environment
        self.env = self.create_environment(headless=not show_game, max_episode_steps=max_episode_steps)
        self.env = Monitor(self.env, os.path.join(self.save_dir, 'monitor'))

        # Create model
        self.model = self.create_model(self.env, algorithm, **model_kwargs)

        # Create appropriate callback
        if show_plots:
            callback = EnhancedTrainingCallback(
                save_freq=save_freq,
                save_path=self.save_dir,
                show_plots=True,
                verbose=1
            )
        else:
            callback = TrainingCallback(
                save_freq=save_freq,
                save_path=self.save_dir,
                verbose=1
            )

        # Record training start
        self.training_stats['training_start_time'] = datetime.now()

        try:
            logger.info("Training started! Press Ctrl+C to stop gracefully.")

            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                tb_log_name=f"pokemon_red_{algorithm.lower()}",
                progress_bar=True
            )

            # Save final model
            final_model_path = os.path.join(self.save_dir, 'models', 'final_model')
            self.model.save(final_model_path)
            self.training_stats['model_saves'] += 1

            logger.info("=" * 50)
            logger.info("TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 50)
            logger.info(f"Final model saved to: {final_model_path}")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            # Save current model
            interrupt_model_path = os.path.join(self.save_dir, 'models', 'interrupted_model')
            self.model.save(interrupt_model_path)
            self.training_stats['model_saves'] += 1
            logger.info(f"Current model saved to: {interrupt_model_path}")

        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise

        finally:
            # Record training end
            self.training_stats['training_end_time'] = datetime.now()
            if self.training_stats['training_start_time']:
                self.training_stats['total_training_time'] = (
                        self.training_stats['training_end_time'] -
                        self.training_stats['training_start_time']
                ).total_seconds()

            # Update final statistics
            self.training_stats['total_timesteps'] = total_timesteps

            # Clean up resources
            logger.info("Cleaning up training resources...")
            if hasattr(callback, 'cleanup'):
                callback.cleanup()
            if self.env:
                self.env.close()

            # Clean up save files for next run
            self.cleanup_save_files()

            # Save training statistics
            self.save_training_stats()

            logger.info("Training cleanup completed")

    def test(self,
             model_path: str,
             episodes: int = 10,
             render: bool = True,
             max_episode_steps: int = 5000) -> Dict[str, Any]:
        """
        Test trained model and return evaluation metrics.

        Args:
            model_path: Path to trained model
            episodes: Number of test episodes
            render: Whether to show game window
            max_episode_steps: Maximum steps per episode

        Returns:
            Dictionary with evaluation results
        """
        logger.info("=" * 50)
        logger.info("TESTING POKEMON RED RL MODEL")
        logger.info("=" * 50)
        logger.info(f"Model: {model_path}")
        logger.info(f"Episodes: {episodes}")
        logger.info(f"Render: {render}")

        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = PPO.load(model_path)
        logger.info("Model loaded successfully")

        # Create environment
        self.env = self.create_environment(headless=not render, max_episode_steps=max_episode_steps)

        # Test episodes
        episode_results = []
        total_reward = 0
        total_steps = 0

        for episode in range(episodes):
            logger.info(f"Starting test episode {episode + 1}/{episodes}")

            obs, info = self.env.reset()
            episode_reward = 0
            episode_steps = 0

            while True:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)

                episode_reward += reward
                episode_steps += 1

                if render and episode_steps % 100 == 0:
                    self.env.render()

                if terminated or truncated:
                    break

            # Record episode results
            episode_result = {
                'episode': episode + 1,
                'reward': episode_reward,
                'steps': episode_steps,
                'maps_visited': info.get('maps_visited', 0),
                'badges_earned': info.get('badges_earned', 0),
                'locations_visited': info.get('locations_visited', 0),
                'player_level': info.get('player_level', 1),
                'final_map': info.get('current_map', 0)
            }

            episode_results.append(episode_result)
            total_reward += episode_reward
            total_steps += episode_steps

            logger.info(f"Episode {episode + 1} completed:")
            logger.info(f"  Reward: {episode_reward:.2f}")
            logger.info(f"  Steps: {episode_steps}")
            logger.info(f"  Maps visited: {info.get('maps_visited', 0)}")
            logger.info(f"  Badges: {info.get('badges_earned', 0)}")
            logger.info(f"  Level: {info.get('player_level', 1)}")

        # Calculate summary statistics
        rewards = [result['reward'] for result in episode_results]
        steps = [result['steps'] for result in episode_results]
        maps_visited = [result['maps_visited'] for result in episode_results]
        badges = [result['badges_earned'] for result in episode_results]

        evaluation_results = {
            'episodes_tested': episodes,
            'total_reward': total_reward,
            'total_steps': total_steps,
            'avg_reward': total_reward / episodes,
            'avg_steps': total_steps / episodes,
            'avg_maps_visited': sum(maps_visited) / episodes,
            'max_badges': max(badges) if badges else 0,
            'avg_badges': sum(badges) / episodes if badges else 0,
            'episode_results': episode_results
        }

        # Log summary
        logger.info("=" * 50)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Average reward: {evaluation_results['avg_reward']:.2f}")
        logger.info(f"Average steps: {evaluation_results['avg_steps']:.0f}")
        logger.info(f"Average maps visited: {evaluation_results['avg_maps_visited']:.1f}")
        logger.info(f"Maximum badges earned: {evaluation_results['max_badges']}")
        logger.info(f"Average badges: {evaluation_results['avg_badges']:.1f}")

        # Clean up
        self.env.close()

        return evaluation_results

    def save_training_stats(self) -> None:
        """Save training statistics to JSON file."""
        stats_path = os.path.join(self.save_dir, 'training_stats.json')

        # Prepare stats for JSON serialization
        stats_copy = self.training_stats.copy()
        for key, value in stats_copy.items():
            if isinstance(value, datetime):
                stats_copy[key] = value.isoformat()

        try:
            with open(stats_path, 'w') as f:
                json.dump(stats_copy, f, indent=2)
            logger.info(f"Training statistics saved to: {stats_path}")
        except Exception as e:
            logger.error(f"Failed to save training statistics: {e}")

    def load_training_stats(self) -> Dict[str, Any]:
        """Load training statistics from JSON file."""
        stats_path = os.path.join(self.save_dir, 'training_stats.json')

        if not os.path.exists(stats_path):
            logger.warning("No training statistics file found")
            return {}

        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)

            # Convert datetime strings back to datetime objects
            datetime_fields = ['training_start_time', 'training_end_time']
            for field in datetime_fields:
                if field in stats and stats[field]:
                    stats[field] = datetime.fromisoformat(stats[field])

            logger.info(f"Training statistics loaded from: {stats_path}")
            return stats

        except Exception as e:
            logger.error(f"Failed to load training statistics: {e}")
            return {}

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        return {
            'trainer_config': {
                'rom_path': self.rom_path,
                'save_dir': self.save_dir,
                'reward_strategy': self.reward_strategy,
                'observation_type': self.observation_type
            },
            'training_stats': self.training_stats.copy(),
            'model_info': {
                'algorithm': 'PPO' if self.model else None,
                'model_loaded': self.model is not None
            },
            'environment_info': {
                'env_created': self.env is not None
            }
        }