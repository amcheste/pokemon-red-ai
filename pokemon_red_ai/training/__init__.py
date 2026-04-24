"""
Pokemon Red RL Training Module

This module provides the training infrastructure for Pokemon Red RL agents,
including the main trainer class, training callbacks, and model utilities.
"""

# Main trainer class
from .trainer import PokemonTrainer

# Training callbacks
from .callbacks import (
    TrainingCallback,
    EnhancedTrainingCallback,
    EarlyStopping,
    PerformanceMonitor,
    WandbCallback,
    MonitoringCallback,
    MONITORED_INFO_KEYS,
)

# Model creation utilities
from .models import (
    create_ppo_model,
    create_recurrent_ppo_model,
    create_a2c_model,
    create_dqn_model,
    create_model,
    get_model_config,
    get_policy_kwargs_for_observation_type,
    get_policy_type_for_observation,
    PokemonFeaturesExtractor,
    optimize_hyperparameters
)

__all__ = [
    # Main trainer
    "PokemonTrainer",

    # Callbacks
    "TrainingCallback",
    "EnhancedTrainingCallback",
    "EarlyStopping",
    "PerformanceMonitor",
    "WandbCallback",
    "MonitoringCallback",
    "MONITORED_INFO_KEYS",

    # Model creation
    "create_ppo_model",
    "create_recurrent_ppo_model",
    "create_a2c_model",
    "create_dqn_model",
    "create_model",
    "get_model_config",
    "get_policy_kwargs_for_observation_type",
    "get_policy_type_for_observation",
    "PokemonFeaturesExtractor",
    "optimize_hyperparameters",
]