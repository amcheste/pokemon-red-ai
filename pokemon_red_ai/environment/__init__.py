"""
Pokemon Red RL Environment Module

This module provides the Gymnasium environment implementation for Pokemon Red,
including observation processing, reward calculation, and environment management.
"""

from .gym_env import PokemonRedGymEnv, PokemonRedVecEnv
from .rewards import (
    BaseRewardCalculator,
    StandardRewardCalculator,
    ExplorationFocusedCalculator,
    ProgressFocusedCalculator,
    SparseRewardCalculator,
    RewardConfig,
    create_reward_calculator,
    evaluate_reward_strategy
)
from .observations import (
    create_observation_space,
    process_game_state,
    downsample_screen,
    normalize_screen,
    validate_observation,
    get_screen_features,
    create_minimal_observation_space,
    process_minimal_observation,
    preprocess_screen_for_cnn,
    # Paper observation treatments
    NUM_EVENT_FLAGS,
    SYMBOLIC_DIM,
    create_pixel_observation_space,
    process_pixel_observation,
    create_symbolic_observation_space,
    process_symbolic_observation,
    create_hybrid_observation_space,
    process_hybrid_observation,
)

__all__ = [
    # Main environment classes
    "PokemonRedGymEnv",
    "PokemonRedVecEnv",

    # Reward system
    "BaseRewardCalculator",
    "StandardRewardCalculator",
    "ExplorationFocusedCalculator",
    "ProgressFocusedCalculator",
    "SparseRewardCalculator",
    "RewardConfig",
    "create_reward_calculator",
    "evaluate_reward_strategy",

    # Observation processing
    "create_observation_space",
    "process_game_state",
    "downsample_screen",
    "normalize_screen",
    "validate_observation",
    "get_screen_features",
    "create_minimal_observation_space",
    "process_minimal_observation",
    "preprocess_screen_for_cnn",

    # Paper observation treatments
    "NUM_EVENT_FLAGS",
    "SYMBOLIC_DIM",
    "create_pixel_observation_space",
    "process_pixel_observation",
    "create_symbolic_observation_space",
    "process_symbolic_observation",
    "create_hybrid_observation_space",
    "process_hybrid_observation",
]