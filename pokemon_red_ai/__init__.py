"""
Pokemon Red AI Package

A comprehensive toolkit for training AI agents to play Pokemon Red using
reinforcement learning techniques with PyBoy emulation.

This package provides:
- Game interface components for Pokemon Red
- Gymnasium environment implementations
- Flexible reward calculation systems
- Training infrastructure and callbacks
- Monitoring and visualization tools

Usage:
    Basic training:
        from pokemon_red_ai import PokemonTrainer
        trainer = PokemonTrainer("PokemonRed.gb")
        trainer.train(timesteps=100000)

    Custom environment:
        from pokemon_red_ai import PokemonRedGymEnv
        env = PokemonRedGymEnv("PokemonRed.gb", reward_strategy="exploration")

    Direct game interface:
        from pokemon_red_ai import PokemonRedAgent
        agent = PokemonRedAgent("PokemonRed.gb")
        agent.run_opening_sequence()
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"
__url__ = "https://github.com/yourusername/pokemon-red-ai"

# Core game interface
from .game import PokemonRedAgent

# Environment components
from .environment import (
    PokemonRedGymEnv,
    PokemonRedVecEnv,
    RewardConfig,
    create_reward_calculator
)

# Training components (import with error handling in case modules don't exist yet)
try:
    from .training import PokemonTrainer
except ImportError:
    # Training module might not exist yet during development
    PokemonTrainer = None

# Utility functions
from .utils import load_config

# Public API - main classes and functions users should use
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",

    # Core classes
    "PokemonRedAgent",
    "PokemonRedGymEnv",
    "PokemonRedVecEnv",
    "PokemonTrainer",

    # Configuration
    "RewardConfig",
    "load_config",

    # Factory functions
    "create_reward_calculator",
]

# Convenience functions for common use cases
def create_environment(rom_path: str, **kwargs) -> PokemonRedGymEnv:
    """
    Create a Pokemon Red RL environment with sensible defaults.

    Args:
        rom_path: Path to Pokemon Red ROM file
        **kwargs: Additional arguments for PokemonRedGymEnv

    Returns:
        Configured PokemonRedGymEnv instance
    """
    return PokemonRedGymEnv(rom_path, **kwargs)


def create_trainer(rom_path: str, save_dir: str = "./pokemon_training/") -> "PokemonTrainer":
    """
    Create a Pokemon Red trainer with sensible defaults.

    Args:
        rom_path: Path to Pokemon Red ROM file
        save_dir: Directory to save training artifacts

    Returns:
        Configured PokemonTrainer instance
    """
    if PokemonTrainer is None:
        raise ImportError("Training module not available. Please ensure all dependencies are installed.")

    return PokemonTrainer(rom_path, save_dir)


def quick_train(rom_path: str, timesteps: int = 100000, **kwargs):
    """
    Quick training function for simple use cases.

    Args:
        rom_path: Path to Pokemon Red ROM file
        timesteps: Number of training timesteps
        **kwargs: Additional arguments for trainer
    """
    trainer = create_trainer(rom_path)
    trainer.train(total_timesteps=timesteps, **kwargs)


# Package metadata for tools that need it
PACKAGE_DATA = {
    'name': 'pokemon-red-ai',
    'version': __version__,
    'description': 'AI toolkit for Pokemon Red with reinforcement learning',
    'author': __author__,
    'license': __license__,
    'url': __url__,
    'python_requires': '>=3.8',
    'dependencies': [
        'gymnasium>=0.29.0',
        'stable-baselines3>=2.0.0',
        'pyboy>=2.0.0',
        'numpy>=1.21.0',
        'opencv-python>=4.5.0',
        'matplotlib>=3.5.0',
        'pyyaml>=6.0',
    ]
}


# Import error handling and helpful messages
def _check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []

    try:
        import gymnasium
    except ImportError:
        missing_deps.append("gymnasium")

    try:
        import stable_baselines3
    except ImportError:
        missing_deps.append("stable-baselines3")

    try:
        import pyboy
    except ImportError:
        missing_deps.append("pyboy")

    if missing_deps:
        deps_str = ", ".join(missing_deps)
        print(f"Warning: Missing dependencies: {deps_str}")
        print("Install with: pip install " + " ".join(missing_deps))


# Run dependency check on import
_check_dependencies()


# Helpful package information
def get_package_info():
    """Get package information dictionary."""
    return PACKAGE_DATA.copy()


def print_package_info():
    """Print package information."""
    info = get_package_info()
    print(f"Pokemon Red AI v{info['version']}")
    print(f"Author: {info['author']}")
    print(f"License: {info['license']}")
    print(f"URL: {info['url']}")
    print(f"Python: {info['python_requires']}")


# Setup logging for the package
import logging

def setup_logging(level=logging.INFO):
    """
    Setup logging for the pokemon_red_ai package.

    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Set specific loggers
    logging.getLogger('pokemon_red_ai').setLevel(level)

    # Reduce noise from other libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


# Initialize package logging
setup_logging()