"""
Configuration management utilities for Pokemon Red RL.

This module provides configuration loading, validation, and management
functionality with support for YAML files and environment variables.
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not available - YAML config loading disabled")


@dataclass
class TrainingConfig:
    """Configuration for RL training parameters."""
    # Basic training settings
    total_timesteps: int = 100000
    algorithm: str = 'PPO'
    max_episode_steps: int = 5000
    save_freq: int = 10000

    # Environment settings
    headless: bool = True
    show_game: bool = False
    show_plots: bool = False
    reward_strategy: str = "standard"
    observation_type: str = "multi_modal"
    screen_size: tuple = (80, 72)

    # Model hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Callback settings
    early_stopping_patience: int = 10
    performance_monitoring: bool = True
    tensorboard_log: bool = True


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    # Base rewards
    time_penalty: float = -0.01
    exploration_reward: float = 1.0
    new_map_reward: float = 20.0

    # Progress rewards
    level_reward_multiplier: float = 50.0
    badge_reward_multiplier: float = 200.0
    pokemon_reward_multiplier: float = 100.0

    # Health penalties
    low_health_threshold: float = 0.5
    health_penalty_multiplier: float = 10.0
    death_penalty: float = -100.0

    # Advanced rewards
    money_reward_multiplier: float = 0.01
    battle_victory_reward: float = 10.0
    item_acquisition_reward: float = 5.0
    story_progress_reward: float = 50.0


@dataclass
class EnvironmentConfig:
    """Configuration for environment settings."""
    # ROM settings
    rom_path: str = ""
    headless: bool = True
    speed_multiplier: int = 0  # 0 = unlimited speed
    save_state_enabled: bool = False

    # Episode settings
    max_episode_steps: int = 5000
    reset_on_death: bool = False
    auto_save_frequency: int = 0  # 0 = disabled

    # Observation settings
    observation_type: str = "multi_modal"
    screen_size: tuple = (80, 72)
    normalize_observations: bool = True
    include_previous_action: bool = False


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    level: str = "INFO"
    format: str = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
    file_logging: bool = True
    log_dir: str = "./logs"
    max_log_files: int = 10

    # Component-specific logging levels
    game_log_level: str = "INFO"
    training_log_level: str = "INFO"
    environment_log_level: str = "INFO"


@dataclass
class PokemonAIConfig:
    """Master configuration for Pokemon Red AI."""
    # Sub-configurations
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Global settings
    project_name: str = "pokemon_red_ai"
    save_dir: str = "./pokemon_training"
    random_seed: Optional[int] = None

    # Hardware settings
    device: str = "auto"  # auto, cpu, cuda
    num_workers: int = 1
    use_multiprocessing: bool = False


def load_config(config_path: Union[str, Path]) -> PokemonAIConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Loaded configuration object
    """
    config_path = Path(config_path)

    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if not YAML_AVAILABLE:
        logger.error("PyYAML not available - cannot load YAML configuration")
        raise ImportError("PyYAML required for configuration loading. Install with: pip install pyyaml")

    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Convert nested dictionaries to dataclass instances
        config = dict_to_config(config_dict)

        logger.info(f"Configuration loaded from: {config_path}")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def save_config(config: PokemonAIConfig, config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object to save
        config_path: Path to save YAML configuration file
    """
    if not YAML_AVAILABLE:
        logger.error("PyYAML not available - cannot save YAML configuration")
        raise ImportError("PyYAML required for configuration saving. Install with: pip install pyyaml")

    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        config_dict = config_to_dict(config)

        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to: {config_path}")

    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise


def dict_to_config(config_dict: Dict[str, Any]) -> PokemonAIConfig:
    """
    Convert dictionary to configuration dataclass.

    Args:
        config_dict: Dictionary with configuration values

    Returns:
        Configuration dataclass instance
    """
    # Extract sub-configurations
    training_dict = config_dict.get('training', {})
    rewards_dict = config_dict.get('rewards', {})
    environment_dict = config_dict.get('environment', {})
    logging_dict = config_dict.get('logging', {})

    # Create sub-configuration objects
    training_config = TrainingConfig(**training_dict)
    rewards_config = RewardConfig(**rewards_dict)
    environment_config = EnvironmentConfig(**environment_dict)
    logging_config = LoggingConfig(**logging_dict)

    # Create main configuration
    main_config_dict = {k: v for k, v in config_dict.items()
                       if k not in ['training', 'rewards', 'environment', 'logging']}

    config = PokemonAIConfig(
        training=training_config,
        rewards=rewards_config,
        environment=environment_config,
        logging=logging_config,
        **main_config_dict
    )

    return config


def config_to_dict(config: PokemonAIConfig) -> Dict[str, Any]:
    """
    Convert configuration dataclass to dictionary.

    Args:
        config: Configuration dataclass instance

    Returns:
        Dictionary representation
    """
    return asdict(config)


def create_default_config(config_path: Union[str, Path]) -> PokemonAIConfig:
    """
    Create and save default configuration file.

    Args:
        config_path: Path where to save the default configuration

    Returns:
        Default configuration object
    """
    config = PokemonAIConfig()
    save_config(config, config_path)
    logger.info(f"Default configuration created: {config_path}")
    return config


def validate_config(config: PokemonAIConfig) -> bool:
    """
    Validate configuration values.

    Args:
        config: Configuration to validate

    Returns:
        True if configuration is valid
    """
    errors = []

    # Validate training config
    if config.training.total_timesteps <= 0:
        errors.append("total_timesteps must be positive")

    if config.training.algorithm not in ['PPO', 'A2C', 'DQN']:
        errors.append(f"Unsupported algorithm: {config.training.algorithm}")

    if config.training.max_episode_steps <= 0:
        errors.append("max_episode_steps must be positive")

    # Validate environment config
    if config.environment.rom_path and not Path(config.environment.rom_path).exists():
        errors.append(f"ROM file not found: {config.environment.rom_path}")

    if config.environment.observation_type not in ['multi_modal', 'screen_only', 'minimal']:
        errors.append(f"Invalid observation_type: {config.environment.observation_type}")

    # Validate reward config
    if config.rewards.exploration_reward < 0:
        errors.append("exploration_reward should be non-negative")

    # Validate logging config
    valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if config.logging.level not in valid_log_levels:
        errors.append(f"Invalid log level: {config.logging.level}")

    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False

    logger.info("Configuration validation passed")
    return True


def load_config_with_overrides(config_path: Union[str, Path],
                              overrides: Optional[Dict[str, Any]] = None) -> PokemonAIConfig:
    """
    Load configuration with command-line or programmatic overrides.

    Args:
        config_path: Path to base configuration file
        overrides: Dictionary of override values

    Returns:
        Configuration with overrides applied
    """
    # Load base configuration
    if Path(config_path).exists():
        config = load_config(config_path)
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = PokemonAIConfig()

    # Apply overrides
    if overrides:
        config = apply_overrides(config, overrides)
        logger.info(f"Applied {len(overrides)} configuration overrides")

    # Apply environment variable overrides
    env_overrides = get_env_overrides()
    if env_overrides:
        config = apply_overrides(config, env_overrides)
        logger.info(f"Applied {len(env_overrides)} environment variable overrides")

    return config


def apply_overrides(config: PokemonAIConfig, overrides: Dict[str, Any]) ->PokemonAIConfig:
    """
    Apply override values to configuration.

    Args:
        config: Base configuration
        overrides: Override values (supports nested keys like 'training.learning_rate')

    Returns:
        Configuration with overrides applied
    """
    config_dict = config_to_dict(config)

    for key, value in overrides.items():
        # Support nested keys like 'training.learning_rate'
        keys = key.split('.')
        current = config_dict

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value
        logger.debug(f"Override applied: {key} = {value}")

    return dict_to_config(config_dict)


def get_env_overrides() -> Dict[str, Any]:
    """
    Get configuration overrides from environment variables.

    Environment variables should be prefixed with 'POKEMON_AI_' and use
    double underscores for nested values (e.g., POKEMON_AI_TRAINING__LEARNING_RATE).

    Returns:
        Dictionary of environment variable overrides
    """
    overrides = {}
    prefix = 'POKEMON_AI_'

    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Remove prefix and convert to config key
            config_key = key[len(prefix):].lower()
            config_key = config_key.replace('__', '.')

            # Try to parse value as appropriate type
            parsed_value = parse_env_value(value)
            overrides[config_key] = parsed_value

            logger.debug(f"Environment override: {config_key} = {parsed_value}")

    return overrides


def parse_env_value(value: str) -> Any:
    """
    Parse environment variable value to appropriate Python type.

    Args:
        value: String value from environment variable

    Returns:
        Parsed value with appropriate type
    """
    # Handle boolean values
    if value.lower() in ('true', '1', 'yes', 'on'):
        return True
    elif value.lower() in ('false', '0', 'no', 'off'):
        return False

    # Handle numeric values
    try:
        if '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        pass

    # Handle None
    if value.lower() in ('none', 'null', ''):
        return None

    # Return as string
    return value


def get_config_template() -> str:
    """
    Get YAML configuration file template.

    Returns:
        YAML template string
    """
    template = """# Pokemon Red AI Configuration File
# This file contains all configuration options for training and environment setup

# Training Configuration
training:
  total_timesteps: 100000      # Total training timesteps
  algorithm: "PPO"             # RL algorithm (PPO, A2C, DQN)
  max_episode_steps: 5000      # Maximum steps per episode
  save_freq: 10000             # Model save frequency
  
  # Display settings
  headless: true               # Run without game window
  show_game: false             # Show game window during training
  show_plots: false            # Show live training plots
  
  # Environment settings
  reward_strategy: "standard"  # Reward strategy (standard, exploration, progress, sparse)
  observation_type: "multi_modal"  # Observation type (multi_modal, screen_only, minimal)
  screen_size: [80, 72]        # Screen downsampling size [width, height]
  
  # Model hyperparameters
  learning_rate: 0.0003        # Learning rate
  batch_size: 64               # Training batch size
  n_epochs: 10                 # Training epochs per update
  gamma: 0.99                  # Discount factor
  gae_lambda: 0.95             # GAE lambda parameter
  clip_range: 0.2              # PPO clip range
  ent_coef: 0.01               # Entropy coefficient
  vf_coef: 0.5                 # Value function coefficient
  max_grad_norm: 0.5           # Maximum gradient norm
  
  # Advanced settings
  early_stopping_patience: 10  # Early stopping patience
  performance_monitoring: true # Enable performance monitoring
  tensorboard_log: true        # Enable tensorboard logging

# Reward Configuration
rewards:
  # Base rewards
  time_penalty: -0.01          # Penalty per step (encourages efficiency)
  exploration_reward: 1.0      # Reward for visiting new locations
  new_map_reward: 20.0         # Reward for discovering new maps
  
  # Progress rewards
  level_reward_multiplier: 50.0      # Reward per level gained
  badge_reward_multiplier: 200.0     # Reward per badge earned
  pokemon_reward_multiplier: 100.0   # Reward per Pokemon caught
  
  # Health management
  low_health_threshold: 0.5    # HP ratio threshold for penalties
  health_penalty_multiplier: 10.0    # Penalty multiplier for low health
  death_penalty: -100.0        # Penalty for Pokemon fainting
  
  # Advanced rewards
  money_reward_multiplier: 0.01       # Reward per money unit gained
  battle_victory_reward: 10.0         # Reward for winning battles
  item_acquisition_reward: 5.0        # Reward for acquiring items
  story_progress_reward: 50.0         # Reward for story milestones

# Environment Configuration
environment:
  # ROM settings
  rom_path: ""                 # Path to Pokemon Red ROM file
  headless: true               # Run emulator headless
  speed_multiplier: 0          # Game speed (0 = unlimited)
  save_state_enabled: false    # Enable save states
  
  # Episode settings
  max_episode_steps: 5000      # Maximum steps per episode
  reset_on_death: false        # Reset episode when Pokemon faints
  auto_save_frequency: 0       # Auto-save frequency (0 = disabled)
  
  # Observation settings
  observation_type: "multi_modal"     # Type of observations
  screen_size: [80, 72]        # Screen size for processing
  normalize_observations: true # Normalize observation values
  include_previous_action: false       # Include previous action in state

# Logging Configuration
logging:
  level: "INFO"                # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  format: "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
  file_logging: true           # Enable file logging
  log_dir: "./logs"            # Log directory
  max_log_files: 10            # Maximum number of log files to keep
  
  # Component-specific log levels
  game_log_level: "INFO"       # Game interface log level
  training_log_level: "INFO"   # Training log level
  environment_log_level: "INFO"        # Environment log level

# Global Settings
project_name: "pokemon_red_ai"  # Project name
save_dir: "./pokemon_training"  # Directory for saving models and logs
random_seed: null               # Random seed for reproducibility (null = random)

# Hardware Settings
device: "auto"                  # Device for training (auto, cpu, cuda)
num_workers: 1                  # Number of parallel workers
use_multiprocessing: false     # Enable multiprocessing
"""
    return template