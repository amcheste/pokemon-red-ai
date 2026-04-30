"""
Observation processing utilities for Pokemon Red RL environment.

This module handles screen processing, observation space definition,
and state representation for the RL game.
"""

import numpy as np
import logging
from typing import Dict, Tuple
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)


def downsample_screen(screen: np.ndarray, target_size: Tuple[int, int] = (80, 72)) -> np.ndarray:
    """
    Downsample screen from 144x160 to target size for efficiency.

    Args:
        screen: Original screen array
        target_size: Target (width, height) for downsampling

    Returns:
        Downsampled screen array
    """
    try:
        import cv2
        height, width = target_size[1], target_size[0]  # cv2 uses (width, height)
        return cv2.resize(screen, (width, height), interpolation=cv2.INTER_AREA)
    except ImportError:
        logger.warning("OpenCV not available, using simple downsampling")
        # Fallback: simple downsampling by taking every nth pixel
        h_factor = screen.shape[0] // target_size[1]
        w_factor = screen.shape[1] // target_size[0]
        return screen[::h_factor, ::w_factor]


def normalize_screen(screen: np.ndarray) -> np.ndarray:
    """
    Normalize screen array to proper format for RL.

    Args:
        screen: Raw screen array from PyBoy

    Returns:
        Normalized screen array (H, W, 3) with uint8 values
    """
    # Handle different input formats
    if len(screen.shape) == 2:
        # Grayscale to RGB
        screen = np.stack([screen] * 3, axis=-1)
    elif len(screen.shape) == 3:
        if screen.shape[2] == 4:
            # RGBA to RGB (remove alpha channel)
            screen = screen[:, :, :3]
        elif screen.shape[2] == 1:
            # Single channel to RGB
            screen = np.repeat(screen, 3, axis=2)
        elif screen.shape[2] > 3:
            # Take first 3 channels if more than RGB
            screen = screen[:, :, :3]
        elif screen.shape[2] < 3:
            # Pad with zeros if less than 3 channels
            pad_width = ((0, 0), (0, 0), (0, 3 - screen.shape[2]))
            screen = np.pad(screen, pad_width, mode='constant')

    # Ensure uint8 type
    if screen.dtype != np.uint8:
        screen = screen.astype(np.uint8)

    return screen


def create_observation_space(screen_size: Tuple[int, int] = (80, 72)) -> spaces.Dict:
    """
    Create the observation space for Pokemon Red environment.

    Args:
        screen_size: (width, height) of the downsampled screen

    Returns:
        Gymnasium Dict space defining the observation structure
    """
    width, height = screen_size

    return spaces.Dict({
        'screen': spaces.Box(
            low=0, high=255,
            shape=(height, width, 3),  # (H, W, C) format
            dtype=np.uint8
        ),
        'position': spaces.Box(
            low=0, high=255,
            shape=(3,),  # x, y, map_id
            dtype=np.uint8
        ),
        'stats': spaces.Box(
            low=0, high=255,
            shape=(6,),  # level, hp_ratio*100, badges, party_count, episode_progress, badge_count
            dtype=np.uint8
        ),
        'exploration': spaces.Box(
            low=0, high=65535,
            shape=(2,),  # locations_visited, unique_maps
            dtype=np.uint16
        )
    })


def process_game_state(agent, episode_steps: int, max_episode_steps: int,
                       visited_locations: set, screen_size: Tuple[int, int] = (80, 72)) -> Dict[str, np.ndarray]:
    """
    Process raw game state into RL observation format.

    Args:
        agent: PokemonRedAgent instance
        episode_steps: Current episode step count
        max_episode_steps: Maximum steps per episode
        visited_locations: Set of visited (x, y, map) tuples
        screen_size: Target screen size

    Returns:
        Processed observation dictionary
    """
    # Get raw screen and process it
    raw_screen = agent.get_screen_array()
    screen_small = downsample_screen(raw_screen, screen_size)
    screen_normalized = normalize_screen(screen_small)

    # Get game state
    position = agent.get_player_position()
    stats = agent.get_player_stats()

    # Calculate derived values
    hp_ratio = int((stats['current_hp'] / max(stats['max_hp'], 1)) * 100)
    episode_progress = min(int((episode_steps / max_episode_steps) * 100), 100)

    # Count unique maps from visited locations
    unique_maps = set()
    for x, y, map_id in visited_locations:
        if map_id != 0:  # Exclude non-game states
            unique_maps.add(map_id)

    # Badge count (count bits set in badge bitfield)
    badge_count = bin(stats['badges']).count('1')

    observation = {
        'screen': screen_normalized,
        'position': np.array([
            position['x'],
            position['y'],
            position['map']
        ], dtype=np.uint8),
        'stats': np.array([
            min(stats['level'], 255),
            min(hp_ratio, 255),
            min(stats['badges'], 255),
            min(stats['party_count'], 255),
            min(episode_progress, 255),
            min(badge_count, 255)
        ], dtype=np.uint8),
        'exploration': np.array([
            min(len(visited_locations), 65535),
            min(len(unique_maps), 65535)
        ], dtype=np.uint16)
    }

    return observation


def get_screen_features(screen: np.ndarray) -> Dict[str, float]:
    """
    Extract statistical features from screen for analysis.

    Args:
        screen: Screen array

    Returns:
        Dictionary of screen features
    """
    if len(screen.shape) == 3:
        # Convert to grayscale for analysis
        gray = np.mean(screen, axis=2)
    else:
        gray = screen

    features = {
        'brightness_mean': float(np.mean(gray)),
        'brightness_std': float(np.std(gray)),
        'contrast': float(np.std(gray)) / max(float(np.mean(gray)), 1.0),
        'edge_density': float(np.mean(np.abs(np.gradient(gray.astype(float))))),
        'unique_colors': len(np.unique(gray.flatten())),
        'screen_activity': float(np.sum(gray > gray.mean())) / gray.size
    }

    return features


def create_minimal_observation_space() -> spaces.Box:
    """
    Create a minimal observation space using only essential features.
    Useful for faster training with reduced input dimensionality.

    Returns:
        Box space with essential game features
    """
    # Essential features: position (3) + stats (6) + exploration (2) = 11 features
    return spaces.Box(
        low=0, high=65535,
        shape=(11,),
        dtype=np.uint16
    )


def process_minimal_observation(agent, episode_steps: int, max_episode_steps: int,
                                visited_locations: set) -> np.ndarray:
    """
    Process game state into minimal feature vector.

    Args:
        agent: PokemonRedAgent instance
        episode_steps: Current episode step count
        max_episode_steps: Maximum steps per episode
        visited_locations: Set of visited locations

    Returns:
        Minimal observation array
    """
    position = agent.get_player_position()
    stats = agent.get_player_stats()

    # Calculate derived values
    hp_ratio = int((stats['current_hp'] / max(stats['max_hp'], 1)) * 100)
    episode_progress = min(int((episode_steps / max_episode_steps) * 100), 100)
    badge_count = bin(stats['badges']).count('1')

    # Count unique maps
    unique_maps = set()
    for x, y, map_id in visited_locations:
        if map_id != 0:
            unique_maps.add(map_id)

    observation = np.array([
        position['x'],
        position['y'],
        position['map'],
        stats['level'],
        hp_ratio,
        stats['badges'],
        stats['party_count'],
        episode_progress,
        badge_count,
        min(len(visited_locations), 65535),
        min(len(unique_maps), 65535)
    ], dtype=np.uint16)

    return observation


def validate_observation(
    observation,
    observation_space,
) -> bool:
    """
    Validate that observation matches the defined observation space.

    Handles both ``gymnasium.spaces.Dict`` (e.g., ``hybrid``,
    ``multi_modal``) and single-Box spaces (e.g., ``pixel``,
    ``symbolic``, ``minimal``, ``screen_only``).

    Args:
        observation: Observation to validate.  ``Dict[str, np.ndarray]``
            for Dict spaces, ``np.ndarray`` for Box spaces.
        observation_space: Expected observation space.

    Returns:
        True if observation is valid.
    """
    try:
        # Dict-shaped spaces — validate each component individually.
        if isinstance(observation_space, spaces.Dict):
            for key in observation_space.spaces.keys():
                if key not in observation:
                    logger.error(f"Missing observation key: {key}")
                    return False
            for key, space in observation_space.spaces.items():
                obs_component = observation[key]
                if obs_component.shape != space.shape:
                    logger.error(
                        f"Shape mismatch for {key}: "
                        f"{obs_component.shape} != {space.shape}"
                    )
                    return False
                if obs_component.dtype != space.dtype:
                    logger.error(
                        f"Dtype mismatch for {key}: "
                        f"{obs_component.dtype} != {space.dtype}"
                    )
                    return False
                if not space.contains(obs_component):
                    logger.error(f"Observation {key} outside bounds")
                    return False
            return True

        # Single Box space (pixel / symbolic / minimal / screen_only).
        if isinstance(observation_space, spaces.Box):
            if observation.shape != observation_space.shape:
                logger.error(
                    f"Shape mismatch: "
                    f"{observation.shape} != {observation_space.shape}"
                )
                return False
            if observation.dtype != observation_space.dtype:
                logger.error(
                    f"Dtype mismatch: "
                    f"{observation.dtype} != {observation_space.dtype}"
                )
                return False
            if not observation_space.contains(observation):
                logger.error("Observation outside bounds")
                return False
            return True

        logger.error(
            f"Unsupported observation space type: "
            f"{type(observation_space).__name__}"
        )
        return False

    except Exception as e:
        logger.error(f"Observation validation failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────
# Paper observation treatments (pixel / symbolic / hybrid)
# ──────────────────────────────────────────────────────────────────────

# Number of pre-registered event flags tracked in event_flags.py
NUM_EVENT_FLAGS = 18
# Symbolic vector layout: position(3) + stats(6) + event_flags(18) + exploration(2) = 29
SYMBOLIC_DIM = 3 + 6 + NUM_EVENT_FLAGS + 2


def _screen_to_grayscale(screen: np.ndarray) -> np.ndarray:
    """Convert a screen array to single-channel grayscale uint8."""
    if len(screen.shape) == 3 and screen.shape[2] >= 3:
        # Luminance weights: 0.299R + 0.587G + 0.114B
        gray = np.dot(screen[:, :, :3], [0.299, 0.587, 0.114])
        return gray.astype(np.uint8)
    elif len(screen.shape) == 3 and screen.shape[2] == 1:
        return screen.astype(np.uint8)
    elif len(screen.shape) == 2:
        return screen.astype(np.uint8)[..., np.newaxis]
    # Fallback
    return screen.astype(np.uint8)


def create_pixel_observation_space(
    screen_size: Tuple[int, int] = (80, 72),
) -> spaces.Box:
    """
    Observation space for the **pixel** treatment.

    Single-channel grayscale screen (H, W, 1).  Intended for use with
    SB3's ``CnnPolicy`` / ``CnnLstmPolicy`` which expect image-shaped
    ``Box`` spaces.

    This is Treatment 1 in the paper's 3-treatment experimental design.
    """
    width, height = screen_size
    return spaces.Box(
        low=0, high=255,
        shape=(height, width, 1),
        dtype=np.uint8,
    )


def process_pixel_observation(
    agent,
    screen_size: Tuple[int, int] = (80, 72),
) -> np.ndarray:
    """
    Build the **pixel** observation: 80x72x1 grayscale screen.

    Args:
        agent: PokemonRedAgent instance.
        screen_size: Target (width, height) for downsampling.

    Returns:
        ``np.ndarray`` of shape ``(H, W, 1)`` with dtype ``uint8``.
    """
    raw = agent.get_screen_array()
    small = downsample_screen(raw, screen_size)
    gray = _screen_to_grayscale(small)
    if len(gray.shape) == 2:
        gray = gray[..., np.newaxis]
    return gray


def create_symbolic_observation_space() -> spaces.Box:
    """
    Observation space for the **symbolic** treatment.

    A flat ``float32`` vector containing only structured game-state
    features read from memory — no pixel data at all.  Intended for
    use with SB3's ``MlpPolicy`` / ``MlpLstmPolicy``.

    Layout (29 features total):
        - position: x, y, map_id  (3)
        - stats: level, hp_ratio, badges_bitfield, party_count,
                 episode_progress, badge_count  (6)
        - event_flags: 18 binary flags  (18)
        - exploration: locations_visited, unique_maps  (2)

    This is Treatment 2 in the paper's 3-treatment experimental design.
    """
    return spaces.Box(
        low=0.0, high=1.0,
        shape=(SYMBOLIC_DIM,),
        dtype=np.float32,
    )


def _build_symbolic_vector(
    agent,
    episode_steps: int,
    max_episode_steps: int,
    visited_locations: set,
) -> np.ndarray:
    """
    Build the normalised symbolic feature vector (shared by symbolic
    and hybrid treatments).

    All values are normalised to [0, 1] for stable MLP training.
    """
    position = agent.get_player_position()
    stats = agent.get_player_stats()

    hp_ratio = stats['current_hp'] / max(stats['max_hp'], 1)
    episode_progress = episode_steps / max(max_episode_steps, 1)
    badge_count = bin(stats['badges']).count('1')

    unique_maps = set()
    for x, y, map_id in visited_locations:
        if map_id != 0:
            unique_maps.add(map_id)

    # Event flags (18 binary features)
    event_flag_values = np.zeros(NUM_EVENT_FLAGS, dtype=np.float32)
    try:
        from ..game.event_flags import read_boulder_path_flags
        flags = read_boulder_path_flags(agent.memory)
        for i, val in enumerate(flags.values()):
            if i < NUM_EVENT_FLAGS:
                event_flag_values[i] = 1.0 if val else 0.0
    except Exception:
        pass  # Graceful fallback — zeros if flags unavailable

    vec = np.concatenate([
        # Position (3) — normalised to [0, 1]
        np.array([
            position['x'] / 255.0,
            position['y'] / 255.0,
            position['map'] / 255.0,
        ], dtype=np.float32),
        # Stats (6)
        np.array([
            min(stats['level'], 100) / 100.0,
            float(np.clip(hp_ratio, 0, 1)),
            stats['badges'] / 255.0,
            stats['party_count'] / 6.0,
            float(np.clip(episode_progress, 0, 1)),
            badge_count / 8.0,
        ], dtype=np.float32),
        # Event flags (18)
        event_flag_values,
        # Exploration (2)
        np.array([
            min(len(visited_locations), 10000) / 10000.0,
            min(len(unique_maps), 50) / 50.0,
        ], dtype=np.float32),
    ])

    return vec


def process_symbolic_observation(
    agent,
    episode_steps: int,
    max_episode_steps: int,
    visited_locations: set,
) -> np.ndarray:
    """
    Build the **symbolic** observation: flat float32 vector.

    Args:
        agent: PokemonRedAgent instance.
        episode_steps: Current step in the episode.
        max_episode_steps: Max steps before truncation.
        visited_locations: Set of (x, y, map) tuples visited so far.

    Returns:
        ``np.ndarray`` of shape ``(SYMBOLIC_DIM,)`` with dtype ``float32``.
    """
    return _build_symbolic_vector(
        agent, episode_steps, max_episode_steps, visited_locations
    )


def create_hybrid_observation_space(
    screen_size: Tuple[int, int] = (80, 72),
) -> spaces.Dict:
    """
    Observation space for the **hybrid** treatment.

    A ``Dict`` space with two keys:

    - ``"screen"``: grayscale image ``(H, W, 1)`` processed by a CNN.
    - ``"game_state"``: flat ``float32`` vector processed by an MLP.

    The two feature streams are concatenated after extraction by
    SB3's ``MultiInputPolicy`` / ``MultiInputLstmPolicy``.

    This is Treatment 3 in the paper's 3-treatment experimental design.
    """
    width, height = screen_size
    return spaces.Dict({
        "screen": spaces.Box(
            low=0, high=255,
            shape=(height, width, 1),
            dtype=np.uint8,
        ),
        "game_state": spaces.Box(
            low=0.0, high=1.0,
            shape=(SYMBOLIC_DIM,),
            dtype=np.float32,
        ),
    })


def process_hybrid_observation(
    agent,
    episode_steps: int,
    max_episode_steps: int,
    visited_locations: set,
    screen_size: Tuple[int, int] = (80, 72),
) -> Dict[str, np.ndarray]:
    """
    Build the **hybrid** observation: grayscale screen + symbolic vector.

    Args:
        agent: PokemonRedAgent instance.
        episode_steps: Current step in the episode.
        max_episode_steps: Max steps before truncation.
        visited_locations: Set of (x, y, map) tuples visited so far.
        screen_size: Target (width, height) for downsampling.

    Returns:
        Dict with ``"screen"`` and ``"game_state"`` arrays.
    """
    screen = process_pixel_observation(agent, screen_size)
    game_state = _build_symbolic_vector(
        agent, episode_steps, max_episode_steps, visited_locations
    )
    return {
        "screen": screen,
        "game_state": game_state,
    }


def preprocess_screen_for_cnn(screen: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Preprocess screen specifically for CNN training.

    Args:
        screen: Raw screen array
        normalize: Whether to normalize pixel values to [0, 1]

    Returns:
        Preprocessed screen ready for CNN
    """
    # Ensure proper format
    screen = normalize_screen(screen)

    # Convert to float32 for CNN training
    screen = screen.astype(np.float32)

    # Normalize to [0, 1] if requested
    if normalize:
        screen = screen / 255.0

    # Move channel dimension to front for PyTorch compatibility (C, H, W)
    # Uncomment if using PyTorch:
    # screen = np.transpose(screen, (2, 0, 1))

    return screen