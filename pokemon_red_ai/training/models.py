"""
Model creation and configuration utilities for Pokemon Red RL.

This module provides utilities for creating and configuring RL models
with appropriate hyperparameters for Pokemon Red training.

Supported algorithms:
  - PPO, A2C, DQN  (stable-baselines3)
  - RecurrentPPO    (sb3-contrib — PPO with LSTM for partial observability)
"""

import logging
from typing import Dict, Any, Optional, Union
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from sb3_contrib import RecurrentPPO
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_model_config(algorithm: str = 'PPO') -> Dict[str, Any]:
    """
    Get default hyperparameters for different RL algorithms optimized for Pokemon Red.

    Args:
        algorithm: RL algorithm name ('PPO', 'A2C', 'DQN', 'RecurrentPPO')

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
        'RecurrentPPO': {
            'learning_rate': 2.5e-4,
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
                     observation_type: str = "multi_modal",
                     **kwargs) -> PPO:
    """
    Create PPO model optimized for Pokemon Red.

    Args:
        env: Training environment
        tensorboard_log: Path for tensorboard logs
        policy_kwargs: Additional policy arguments
        observation_type: Observation type — determines policy string
            and default policy kwargs.
        **kwargs: Additional PPO arguments

    Returns:
        Configured PPO model
    """
    # Get default config
    config = get_model_config('PPO')
    config.update(kwargs)

    # Select policy string and kwargs based on observation type
    policy_name = get_policy_type_for_observation(observation_type, "PPO")
    default_policy_kwargs = get_policy_kwargs_for_observation_type(observation_type)

    if policy_kwargs:
        default_policy_kwargs.update(policy_kwargs)

    model = PPO(
        policy_name,
        env,
        policy_kwargs=default_policy_kwargs,
        tensorboard_log=tensorboard_log,
        **config
    )

    logger.info("PPO model created successfully")
    logger.info(f"  Policy: {policy_name}")
    logger.info(f"  Learning rate: {config['learning_rate']}")
    logger.info(f"  Batch size: {config['batch_size']}")
    logger.info(f"  Device: {config['device']}")

    return model


def create_recurrent_ppo_model(
    env: gym.Env,
    tensorboard_log: Optional[str] = None,
    policy_kwargs: Optional[Dict[str, Any]] = None,
    n_lstm_layers: int = 1,
    lstm_hidden_size: int = 256,
    observation_type: str = "multi_modal",
    **kwargs,
) -> RecurrentPPO:
    """
    Create RecurrentPPO (PPO + LSTM) model for Pokemon Red.

    RecurrentPPO adds an LSTM layer between the feature extractor and the
    policy/value heads, giving the agent short-term memory.  This is
    critical for Pokemon Red because the agent needs to remember what
    happened during multi-frame menu navigation, battle sequences, and
    dialogue — information that a single frame cannot convey.

    Args:
        env: Training environment.
        tensorboard_log: Path for tensorboard logs.
        policy_kwargs: Additional policy arguments.  Merged on top of
            defaults.
        n_lstm_layers: Number of stacked LSTM layers (1 is usually enough).
        lstm_hidden_size: Hidden size of each LSTM layer.
        observation_type: Observation type — determines policy string
            and default policy kwargs.
        **kwargs: Additional RecurrentPPO arguments (overrides defaults
            from ``get_model_config('RecurrentPPO')``).

    Returns:
        Configured RecurrentPPO model.
    """
    # Get default config
    config = get_model_config('RecurrentPPO')
    config.update(kwargs)

    # Select policy string and kwargs based on observation type
    policy_name = get_policy_type_for_observation(observation_type, "RecurrentPPO")
    default_policy_kwargs: Dict[str, Any] = get_policy_kwargs_for_observation_type(observation_type)

    # Add LSTM-specific parameters
    default_policy_kwargs['n_lstm_layers'] = n_lstm_layers
    default_policy_kwargs['lstm_hidden_size'] = lstm_hidden_size

    if policy_kwargs:
        default_policy_kwargs.update(policy_kwargs)

    model = RecurrentPPO(
        policy_name,
        env,
        policy_kwargs=default_policy_kwargs,
        tensorboard_log=tensorboard_log,
        **config,
    )

    logger.info("RecurrentPPO model created successfully")
    logger.info(f"  Policy: {policy_name}")
    logger.info(f"  Learning rate: {config['learning_rate']}")
    logger.info(f"  Batch size: {config['batch_size']}")
    logger.info(f"  LSTM layers: {n_lstm_layers}")
    logger.info(f"  LSTM hidden size: {lstm_hidden_size}")
    logger.info(f"  Device: {config['device']}")

    return model


def create_a2c_model(env: gym.Env,
                     tensorboard_log: Optional[str] = None,
                     policy_kwargs: Optional[Dict[str, Any]] = None,
                     observation_type: str = "multi_modal",
                     **kwargs) -> A2C:
    """
    Create A2C model optimized for Pokemon Red.

    Args:
        env: Training environment
        tensorboard_log: Path for tensorboard logs
        policy_kwargs: Additional policy arguments
        observation_type: Observation type — determines policy string
            and default policy kwargs.
        **kwargs: Additional A2C arguments

    Returns:
        Configured A2C model
    """
    config = get_model_config('A2C')
    config.update(kwargs)

    policy_name = get_policy_type_for_observation(observation_type, "PPO")  # A2C uses same policy names as PPO
    default_policy_kwargs = get_policy_kwargs_for_observation_type(observation_type)

    if policy_kwargs:
        default_policy_kwargs.update(policy_kwargs)

    model = A2C(
        policy_name,
        env,
        policy_kwargs=default_policy_kwargs,
        tensorboard_log=tensorboard_log,
        **config
    )

    logger.info("A2C model created successfully")
    logger.info(f"  Policy: {policy_name}")
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


# ──────────────────────────────────────────────────────────────────────
# Paper observation treatments — capacity-matched encoders
#
# To avoid confounding modality with encoder capacity (Henderson et al.
# 2018; Engstrom et al. 2020), the pixel and symbolic feature extractors
# are sized to within 10% on trainable parameter count and emit identical
# 256-dimensional features; the hybrid extractor concatenates one
# capacity-matched copy of each to a 512-dim joint representation
# (Andrychowicz et al. 2021). Strict FLOP matching across CNN and MLP
# architectures distorts encoder design, so per-forward FLOPs are reported
# transparently (see scripts/check_encoder_capacity.py) rather than
# enforced. Per-condition learning rates are selected from a pre-registered
# log-uniform grid following Eimer et al. (2023), so no treatment is
# disadvantaged by an inappropriate hyperparameter.
#
# Pixel encoder:    NatureCNN (Mnih et al. 2015) with features_dim=256.
#                   On 80x72x1 input: ~564K params.
# Symbolic encoder: 3-layer MLP 29 -> 640 -> 640 -> 256 with ReLU.
#                   ~594K params (5.3% over pixel target — within 10%).
# Hybrid encoder:   NatureCNN(256) + SymbolicFeaturesExtractor(256),
#                   concatenated to 512-dim. ~1.16M params (2x single-modality).
# ──────────────────────────────────────────────────────────────────────


# Symbolic input dimensionality is locked here for the capacity match.
# Must equal observations.SYMBOLIC_DIM. The capacity check script asserts
# both sources agree.
PAPER_SYMBOLIC_DIM = 29

# Hidden width chosen to match NatureCNN(features_dim=256) parameter
# count to within 10%. See scripts/check_encoder_capacity.py.
SYMBOLIC_HIDDEN_DIM = 640


class SymbolicFeaturesExtractor(BaseFeaturesExtractor):
    """
    Capacity-matched MLP encoder for the symbolic observation treatment.

    Architecture: input_dim -> 640 -> 640 -> features_dim, ReLU between
    layers and after the final projection (matching NatureCNN's trailing
    activation). The width is sized so the parameter count lands within
    10% of NatureCNN(features_dim=256) on the project's 80x72x1 input.

    The width is *deliberately* larger than typical SAC/TD3 MLP defaults
    (256) — this is a fairness intervention to neutralize the
    encoder-capacity confound when comparing modalities. See the module
    docstring above for the rationale.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        hidden_dim: int = SYMBOLIC_HIDDEN_DIM,
    ):
        super().__init__(observation_space, features_dim)
        input_dim = int(observation_space.shape[0])
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations.float())


class HybridFeaturesExtractor(BaseFeaturesExtractor):
    """
    Capacity-matched two-stream encoder for the hybrid observation treatment.

    Concatenates a 256-dim NatureCNN feature for the screen and a 256-dim
    SymbolicFeaturesExtractor feature for the game-state vector, producing
    a 512-dim joint representation. Each half is the same encoder used in
    the corresponding single-modality treatment, so by construction
    hybrid_params ≈ pixel_params + symbolic_params (option (c) in the
    capacity-matching design notes).
    """

    PIXEL_FEATURES_DIM = 256
    SYMBOLIC_FEATURES_DIM = 256

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 512,
    ):
        # features_dim must equal pixel + symbolic halves
        assert features_dim == (
            self.PIXEL_FEATURES_DIM + self.SYMBOLIC_FEATURES_DIM
        ), (
            "HybridFeaturesExtractor features_dim must equal the sum of the "
            f"pixel ({self.PIXEL_FEATURES_DIM}) and symbolic "
            f"({self.SYMBOLIC_FEATURES_DIM}) halves; got {features_dim}."
        )
        super().__init__(observation_space, features_dim)

        if "screen" not in observation_space.spaces or "game_state" not in observation_space.spaces:
            raise ValueError(
                "HybridFeaturesExtractor requires a Dict observation space "
                "with 'screen' and 'game_state' keys; got "
                f"{list(observation_space.spaces.keys())}"
            )

        self.pixel_extractor = NatureCNN(
            observation_space.spaces["screen"],
            features_dim=self.PIXEL_FEATURES_DIM,
        )
        self.symbolic_extractor = SymbolicFeaturesExtractor(
            observation_space.spaces["game_state"],
            features_dim=self.SYMBOLIC_FEATURES_DIM,
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        pixel_features = self.pixel_extractor(observations["screen"])
        symbolic_features = self.symbolic_extractor(observations["game_state"])
        return torch.cat([pixel_features, symbolic_features], dim=1)


def create_model(algorithm: str,
                 env: gym.Env,
                 tensorboard_log: Optional[str] = None,
                 observation_type: str = "multi_modal",
                 **kwargs) -> Union[PPO, RecurrentPPO, A2C, DQN]:
    """
    Factory function to create RL models.

    Args:
        algorithm: Algorithm name ('PPO', 'RecurrentPPO', 'A2C', 'DQN')
        env: Training environment
        tensorboard_log: Path for tensorboard logs
        observation_type: Observation type — forwarded to the model
            creator so it can select the correct policy string and kwargs.
        **kwargs: Additional model arguments

    Returns:
        Configured RL model
    """
    creators = {
        'PPO': create_ppo_model,
        'RecurrentPPO': create_recurrent_ppo_model,
        'A2C': create_a2c_model,
        'DQN': create_dqn_model
    }

    if algorithm not in creators:
        raise ValueError(f"Unknown algorithm: {algorithm}. Supported: {list(creators.keys())}")

    # DQN handles its own policy selection from the observation space shape
    if algorithm == 'DQN':
        return creators[algorithm](env, tensorboard_log, **kwargs)

    return creators[algorithm](env, tensorboard_log, observation_type=observation_type, **kwargs)


def get_policy_kwargs_for_observation_type(observation_type: str) -> Dict[str, Any]:
    """
    Get appropriate policy kwargs based on observation type.

    Args:
        observation_type: Type of observation ('multi_modal', 'minimal',
            'screen_only', 'pixel', 'symbolic', 'hybrid')

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
    # ── Paper observation treatments (capacity-matched) ─────────────
    # See models.py module docstring above the encoder definitions for
    # the capacity-matching protocol.
    elif observation_type == "pixel":
        return {
            'features_extractor_class': NatureCNN,
            'features_extractor_kwargs': {'features_dim': 256},
            'net_arch': dict(pi=[256, 128], vf=[256, 128]),
            'activation_fn': nn.ReLU,
        }
    elif observation_type == "symbolic":
        return {
            'features_extractor_class': SymbolicFeaturesExtractor,
            'features_extractor_kwargs': {'features_dim': 256},
            'net_arch': dict(pi=[256, 128], vf=[256, 128]),
            'activation_fn': nn.ReLU,
        }
    elif observation_type == "hybrid":
        return {
            'features_extractor_class': HybridFeaturesExtractor,
            'features_extractor_kwargs': {'features_dim': 512},
            'net_arch': dict(pi=[256, 128], vf=[256, 128]),
            'activation_fn': nn.ReLU,
        }
    else:
        logger.warning(f"Unknown observation type: {observation_type}, using default")
        return {}


def get_policy_type_for_observation(
    observation_type: str,
    algorithm: str = "PPO",
) -> str:
    """
    Return the correct SB3 policy string for a given observation type
    and algorithm combination.

    Args:
        observation_type: One of 'pixel', 'symbolic', 'hybrid',
            'multi_modal', 'screen_only', 'minimal'.
        algorithm: 'PPO' or 'RecurrentPPO'.

    Returns:
        Policy class name string (e.g. 'CnnPolicy', 'MlpLstmPolicy').
    """
    # Map observation type → (PPO policy, RecurrentPPO policy)
    policy_map = {
        # Paper treatments
        "pixel":       ("CnnPolicy",        "CnnLstmPolicy"),
        "symbolic":    ("MlpPolicy",        "MlpLstmPolicy"),
        "hybrid":      ("MultiInputPolicy", "MultiInputLstmPolicy"),
        # Legacy types
        "multi_modal": ("MultiInputPolicy", "MultiInputLstmPolicy"),
        "screen_only": ("CnnPolicy",        "CnnLstmPolicy"),
        "minimal":     ("MlpPolicy",        "MlpLstmPolicy"),
    }

    if observation_type not in policy_map:
        raise ValueError(
            f"Unknown observation type: {observation_type}. "
            f"Valid types: {list(policy_map.keys())}"
        )

    ppo_policy, rppo_policy = policy_map[observation_type]
    return rppo_policy if algorithm == "RecurrentPPO" else ppo_policy


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