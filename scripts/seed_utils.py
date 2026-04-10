"""
Multi-library seeding utility for reproducible RL experiments.

Seeds every source of randomness that could affect a training run:

- Python built-in `random`
- NumPy
- PyTorch (CPU and CUDA if available)
- `PYTHONHASHSEED` (affects dict iteration order in some edge cases)
- stable-baselines3 (via `set_random_seed`)

Usage:

    from scripts.seed_utils import seed_everything

    # Single-env training
    seed_everything(42)

    # Per-env seeding in a vectorized setup
    for rank in range(n_envs):
        seed_everything(base_seed=42, seed_offset=rank)

`deterministic_torch=True` forces fully reproducible PyTorch but substantially
hurts throughput. We keep it off for normal training and only enable it for
paper-grade final runs.
"""

import logging
import os
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def seed_everything(
    seed: int,
    seed_offset: int = 0,
    deterministic_torch: bool = False,
) -> int:
    """
    Seed all sources of randomness for reproducibility.

    Args:
        seed: Base seed.
        seed_offset: Added to the base seed. Use to give each parallel env its
            own deterministic seed while keeping a single CLI argument.
        deterministic_torch: If True, force PyTorch into fully deterministic
            mode. Substantially slower; only use for final paper runs that
            need to be bit-reproducible.

    Returns:
        The effective seed used (`seed + seed_offset`).
    """
    effective_seed = seed + seed_offset

    os.environ["PYTHONHASHSEED"] = str(effective_seed)
    random.seed(effective_seed)
    np.random.seed(effective_seed)

    try:
        import torch

        torch.manual_seed(effective_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(effective_seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.debug("torch not installed; skipping torch seeding")

    try:
        from stable_baselines3.common.utils import set_random_seed

        set_random_seed(effective_seed)
    except ImportError:
        logger.debug("stable_baselines3 not installed; skipping sb3 seeding")

    logger.info(
        "Seeded all RNGs with effective_seed=%d (base=%d, offset=%d)",
        effective_seed,
        seed,
        seed_offset,
    )
    return effective_seed


def get_seed_from_env(var_name: str = "POKEMON_AI_SEED", default: Optional[int] = None) -> Optional[int]:
    """
    Read a seed from an environment variable. Useful for CI and cron runs.

    Returns None if the variable is not set and no default is provided.
    """
    raw = os.environ.get(var_name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Environment variable %s is not a valid int: %r", var_name, raw)
        return default
