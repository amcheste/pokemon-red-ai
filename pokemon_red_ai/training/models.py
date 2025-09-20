"""
Model creation and configuration utilities for Pokemon Red RL.

This module provides utilities for creating and configuring RL models
with appropriate hyperparameters for Pokemon Red training.
"""

import logging
from typing import Dict, Any, Optional
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_model_config(algorithm: str = 'PPO') -> Dict[str, Any]:
    """
    Get default hyperparameters for different RL algorithms optimized for Pokemon Red.

    Args:
        algorithm: RL algorithm name ('PPO', 'A2C', 'DQN')

    Returns:
        Dictionary of hyperparameters
    """
    configs = {
        'PPO': {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 1,
            'device': 'auto'
        },
        'A2C': {
            'learning_rate': 7e-4,
            'n_steps': 5,
            'gamma': 0.99,
            'gae_lambda': 1.0,
            'ent_coef': 0.01,
            'vf_coef': 0.25,
            'max_grad_norm': 0.5,
            'verbose': 1,
            'device': 'auto'
        },
        'DQN': {
            'learning_rate': 1e-4,
            'buffer_size': 100000,
            'learning_starts': 10000,
            'batch_size': 32,
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'gradient_steps': 1,
            'target_update_interval': 1000,
            'exploration_fraction': 0.1,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.05,
            'verbose': 1,
            'device': 'auto'
        }
    }

    if algorithm not in configs:
        logger.warning(f"Unknown algorithm {algorithm}, using PPO config")
        algorithm = 'PPO'

    return configs[algorithm].copy()


def create_ppo_model(env: gym.Env,
                     tensorboard_log: Optional[str] = None,
                     policy_kwargs: Optional[Dict[str, Any]] = None,
                     **kwargs) -> PPO:
    """
    Create PPO model optimized for Pokemon Red.

    Args:
        env: Training environment
        tensorboard_log: Path for tensorboard logs
        policy_kwargs: Additional policy arguments
        **kwargs: Additional PPO arguments

    Returns:
        Configured PPO model
    """
    # Get default config
    config = get_model_config('PPO')
    config.update(kwargs)

    # Default policy kwargs for multi-modal observations
    default_policy_kwargs = {
        'features_extractor_class': PokemonFeaturesExtractor,
        'features_extractor_kwargs': {'features_dim': 256},
        'net_arch': dict(pi=[256, 128], vf=[256, 128]),  # Fixed: use dict instead of list
        'activation_fn': nn.ReLU
    }

    if policy_kwargs:
        default_policy_kwargs.update(policy_kwargs)

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=default_policy_kwargs,
        tensorboard_log=tensorboard_log,
        **config
    )

    logger.info("PPO model created successfully")
    logger.info(f"  Learning rate: {config['learning_rate']}")
    logger.info(f"  Batch size: {config['batch_size']}")
    logger.info(f"  Device: {config['device']}")

    return model


def create_a2c_model(env: gym.Env,
                     tensorboard_log: Optional[str] = None,
                     policy_kwargs: Optional[Dict[str, Any]] = None,
                     **kwargs) -> A2C:
    """
    Create A2C model optimized for Pokemon Red.

    Args:
        env: Training environment
        tensorboard_log: Path for tensorboard logs
        policy_kwargs: Additional policy arguments
        **kwargs: Additional A2C arguments

    Returns:
        Configured A2C model
    """
    config = get_model_config('A2C')
    config.update(kwargs)

    default_policy_kwargs = {
        'features_extractor_class': PokemonFeaturesExtractor,
        'features_extractor_kwargs': {'features_dim': 256},
        'net_arch': dict(pi=[128, 64], vf=[128, 64]),  # Fixed: use dict
        'activation_fn': nn.ReLU
    }

    if policy_kwargs:
        default_policy_kwargs.update(policy_kwargs)

    model = A2C(
        "MultiInputPolicy",
        env,
        policy_kwargs=default_policy_kwargs,
        tensorboard_log=tensorboard_log,
        **config
    )

    logger.info("A2C model created successfully")
    return model


def create_dqn_model(env: gym.Env,
                     tensorboard_log: Optional[str] = None,
                     policy_kwargs: Optional[Dict[str, Any]] = None,
                     **kwargs) -> DQN:
    """
    Create DQN model optimized for Pokemon Red.

    Note: DQN only works with single observation spaces, not multi-modal.
    Use only with screen-only or minimal observation types.

    Args:
        env: Training environment
        tensorboard_log: Path for tensorboard logs
        policy_kwargs: Additional policy arguments
        **kwargs: Additional DQN arguments

    Returns:
        Configured DQN model
    """
    config = get_model_config('DQN')
    config.update(kwargs)

    # DQN policy selection based on observation space
    if hasattr(env.observation_space, 'spaces'):
        logger.warning("DQN doesn't support multi-modal observations. Use PPO or A2C instead.")
        raise ValueError("DQN requires single observation space")

    # Determine policy type
    if len(env.observation_space.shape) == 3:  # Image observation
        policy = "CnnPolicy"
    else:  # Vector observation
        policy = "MlpPolicy"

    default_policy_kwargs = {
        'net_arch': [256, 128, 64],
        'activation_fn': nn.ReLU
    }

    if policy_kwargs:
        default_policy_kwargs.update(policy_kwargs)

    model = DQN(
        policy,
        env,
        policy_kwargs=default_policy_kwargs,
        tensorboard_log=tensorboard_log,
        **config
    )

    logger.info("DQN model created successfully")
    return model


class PokemonFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor for Pokemon Red multi-modal observations.

    Combines screen (CNN), position/stats (MLP), and exploration (MLP) features
    into a unified representation for the policy network.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        """
        Initialize features extractor.

        Args:
            observation_space: Multi-modal observation space
            features_dim: Dimension of output features
        """
        super().__init__(observation_space, features_dim)

        # Get dimensions from observation space
        screen_shape = observation_space.spaces['screen'].shape
        position_dim = observation_space.spaces['position'].shape[0]
        stats_dim = observation_space.spaces['stats'].shape[0]
        exploration_dim = observation_space.spaces['exploration'].shape[0]

        # CNN for screen processing - Updated for smaller screen size (72x80)
        self.screen_cnn = nn.Sequential(
            # First conv layer: 8x8 kernel is too big for 72x80 screen, use 4x4
            nn.Conv2d(screen_shape[2], 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Second conv layer: smaller kernel for smaller input
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Third conv layer: even smaller
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output dimension
        with torch.no_grad():
            sample_input = torch.zeros((1, *screen_shape)).permute(0, 3, 1, 2).float()
            cnn_output_dim = self.screen_cnn(sample_input).shape[1]

        # MLPs for other modalities
        self.position_mlp = nn.Sequential(
            nn.Linear(position_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.stats_mlp = nn.Sequential(
            nn.Linear(stats_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.exploration_mlp = nn.Sequential(
            nn.Linear(exploration_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # Fusion layer
        fusion_input_dim = cnn_output_dim + 32 + 32 + 16  # CNN + position + stats + exploration
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

        logger.info(f"PokemonFeaturesExtractor initialized:")
        logger.info(f"  Screen CNN output: {cnn_output_dim}")
        logger.info(f"  Fusion input: {fusion_input_dim}")
        logger.info(f"  Features output: {features_dim}")

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through features extractor.

        Args:
            observations: Dictionary of observation tensors

        Returns:
            Extracted features tensor
        """
        # Process screen with CNN (convert from HWC to CHW format)
        screen = observations['screen'].float() / 255.0
        if len(screen.shape) == 4:  # Batch of images
            screen = screen.permute(0, 3, 1, 2)  # BHWC -> BCHW
        else:  # Single image
            screen = screen.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW

        screen_features = self.screen_cnn(screen)

        # Process other modalities
        position_features = self.position_mlp(observations['position'].float())
        stats_features = self.stats_mlp(observations['stats'].float())
        exploration_features = self.exploration_mlp(observations['exploration'].float())

        # Concatenate all features
        combined_features = torch.cat([
            screen_features,
            position_features,
            stats_features,
            exploration_features
        ], dim=1)

        # Apply fusion layer
        output_features = self.fusion(combined_features)

        return output_features


def create_model(algorithm: str,
                 env: gym.Env,
                 tensorboard_log: Optional[str] = None,
                 **kwargs) -> Any:
    """
    Factory function to create RL models.

    Args:
        algorithm: Algorithm name ('PPO', 'A2C', 'DQN')
        env: Training environment
        tensorboard_log: Path for tensorboard logs
        **kwargs: Additional model arguments

    Returns:
        Configured RL model
    """
    creators = {
        'PPO': create_ppo_model,
        'A2C': create_a2c_model,
        'DQN': create_dqn_model
    }

    if algorithm not in creators:
        raise ValueError(f"Unknown algorithm: {algorithm}. Supported: {list(creators.keys())}")

    return creators[algorithm](env, tensorboard_log, **kwargs)


def get_policy_kwargs_for_observation_type(observation_type: str) -> Dict[str, Any]:
    """
    Get appropriate policy kwargs based on observation type.

    Args:
        observation_type: Type of observation ('multi_modal', 'minimal', 'screen_only')

    Returns:
        Policy kwargs dictionary
    """
    if observation_type == "multi_modal":
        return {
            'features_extractor_class': PokemonFeaturesExtractor,
            'features_extractor_kwargs': {'features_dim': 256},
            'net_arch': dict(pi=[256, 128], vf=[256, 128]),  # Fixed: use dict
            'activation_fn': nn.ReLU
        }
    elif observation_type == "screen_only":
        return {
            'net_arch': [256, 128, 64],
            'activation_fn': nn.ReLU
        }
    elif observation_type == "minimal":
        return {
            'net_arch': dict(pi=[128, 64], vf=[128, 64]),  # Fixed: use dict
            'activation_fn': nn.ReLU
        }
    else:
        logger.warning(f"Unknown observation type: {observation_type}, using default")
        return {}


def optimize_hyperparameters(env: gym.Env,
                            algorithm: str = 'PPO',
                            n_trials: int = 50) -> Dict[str, Any]:
    """
    Optimize hyperparameters using Optuna (optional feature).

    Args:
        env: Training environment
        algorithm: Algorithm to optimize
        n_trials: Number of optimization trials

    Returns:
        Best hyperparameters found
    """
    try:
        import optuna
        from stable_baselines3.common.evaluation import evaluate_policy

        def objective(trial):
            if algorithm == 'PPO':
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                    'n_steps': trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
                    'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                    'n_epochs': trial.suggest_int('n_epochs', 5, 20),
                    'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                    'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99),
                    'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
                    'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True),
                }
            else:
                raise ValueError(f"Hyperparameter optimization not implemented for {algorithm}")

            # Create and train model
            model = create_model(algorithm, env, **params)
            model.learn(total_timesteps=10000)  # Short training for optimization

            # Evaluate performance
            mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)

            return mean_reward

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Hyperparameter optimization completed")
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best reward: {study.best_value:.2f}")

        return study.best_params

    except ImportError:
        logger.error("Optuna not available for hyperparameter optimization")
        logger.info("Install with: pip install optuna")
        return get_model_config(algorithm)