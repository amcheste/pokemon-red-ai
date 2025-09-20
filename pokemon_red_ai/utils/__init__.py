"""
Pokemon Red RL Utilities Module

This module provides utility functions and classes for configuration management,
file operations, and various helper functions used throughout the package.
"""

# Configuration management
from .config import (
    # Configuration dataclasses
    TrainingConfig,
    RewardConfig,
    EnvironmentConfig,
    LoggingConfig,
    PokemonAIConfig,

    # Configuration functions
    load_config,
    save_config,
    create_default_config,
    validate_config,
    load_config_with_overrides,
    apply_overrides,
    get_env_overrides,
    get_config_template,
    dict_to_config,
    config_to_dict,
    parse_env_value
)

# File utilities
from .file_utils import (
    # Directory management
    create_directories,
    get_disk_usage,
    monitor_directory_size,
    cleanup_temp_files,
    get_project_info,

    # ROM file operations
    cleanup_rom_save_files,
    find_rom_files,
    validate_rom_file,

    # Model and backup management
    backup_model,
    cleanup_old_backups,

    # Data persistence
    save_training_metadata,
    load_training_metadata,
    safe_pickle_save,
    safe_pickle_load,

    # Archive operations
    compress_directory,
    extract_archive
)

__all__ = [
    # Configuration classes
    "TrainingConfig",
    "RewardConfig",
    "EnvironmentConfig",
    "LoggingConfig",
    "PokemonAIConfig",

    # Configuration functions
    "load_config",
    "save_config",
    "create_default_config",
    "validate_config",
    "load_config_with_overrides",
    "apply_overrides",
    "get_env_overrides",
    "get_config_template",
    "dict_to_config",
    "config_to_dict",
    "parse_env_value",

    # Directory utilities
    "create_directories",
    "get_disk_usage",
    "monitor_directory_size",
    "cleanup_temp_files",
    "get_project_info",

    # ROM utilities
    "cleanup_rom_save_files",
    "find_rom_files",
    "validate_rom_file",

    # Model utilities
    "backup_model",
    "cleanup_old_backups",

    # Data utilities
    "save_training_metadata",
    "load_training_metadata",
    "safe_pickle_save",
    "safe_pickle_load",

    # Archive utilities
    "compress_directory",
    "extract_archive",
]