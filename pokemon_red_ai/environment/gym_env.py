"""
Pokemon Red Gymnasium environment implementation.

This module provides the main RL environment that integrates the game game,
observation processing, and reward calculation into a standard Gymnasium interface.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, Set
import gymnasium as gym

from ..game.agent import PokemonRedAgent
from .observations import (
    create_observation_space,
    process_game_state,
    validate_observation
)
from .rewards import (
    create_reward_calculator,
    RewardConfig,
    BaseRewardCalculator
)

logger = logging.getLogger(__name__)


class PokemonRedGymEnv(gym.Env):
    """
    OpenAI Gymnasium environment wrapper for Pokemon Red.

    This environment provides a standard RL interface for training agents
    to play Pokemon Red using the modular game interface components.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self,
                 rom_path: str,
                 headless: bool = True,
                 max_episode_steps: int = 15000,
                 reward_strategy: str = "exploration",
                 reward_config: Optional[RewardConfig] = None,
                 screen_size: Tuple[int, int] = (80, 72),
                 observation_type: str = "multi_modal"):
        """
        Initialize Pokemon Red Gymnasium environment.

        Args:
            rom_path: Path to Pokemon Red ROM file
            headless: If True, runs without display window
            max_episode_steps: Maximum steps per episode (increased default)
            reward_strategy: Reward calculation strategy (default to exploration)
            reward_config: Custom reward configuration
            screen_size: Target screen size (width, height)
            observation_type: Type of observation ('multi_modal', 'minimal', 'screen_only')
        """
        super().__init__()

        self.rom_path = rom_path
        self.headless = headless
        self.max_episode_steps = max_episode_steps
        self.screen_size = screen_size
        self.observation_type = observation_type

        # Initialize game game
        self.game = PokemonRedAgent(
            rom_path,
            show_window=not headless,
            speed_multiplier=0  # Unlimited speed for training
        )

        # Use improved reward configuration if none provided
        if reward_config is None and reward_strategy == "exploration":
            reward_config = RewardConfig(
                time_penalty=-0.001,
                exploration_reward=5.0,
                new_map_reward=100.0,
                level_reward_multiplier=25.0,
                badge_reward_multiplier=150.0,
                pokemon_reward_multiplier=75.0,
                low_health_threshold=0.3,
                health_penalty_multiplier=5.0,
                death_penalty=-50.0,
                money_reward_multiplier=0.005,
                battle_victory_reward=15.0,
                item_acquisition_reward=8.0
            )

        # Initialize reward calculator
        self.reward_calculator = create_reward_calculator(
            strategy=reward_strategy,
            config=reward_config
        )

        # Define action space (8 Game Boy buttons)
        self.action_space = gym.spaces.Discrete(8)
        self.action_names = ['A', 'B', 'SELECT', 'START', 'RIGHT', 'LEFT', 'UP', 'DOWN']

        # Define observation space based on type
        if observation_type == "multi_modal":
            self.observation_space = create_observation_space(screen_size)
        elif observation_type == "minimal":
            from .observations import create_minimal_observation_space
            self.observation_space = create_minimal_observation_space()
        elif observation_type == "screen_only":
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=(screen_size[1], screen_size[0], 3),  # (H, W, C)
                dtype=np.uint8
            )
        else:
            raise ValueError(f"Unknown observation type: {observation_type}")

        # Episode tracking
        self.episode_steps = 0
        self.episode_reward = 0
        self.visited_locations: Set[Tuple[int, int, int]] = set()
        self.episode_info = {}

        # Enhanced tracking for better monitoring
        self.episode_count = 0
        self.total_exploration_rewards = 0
        self.total_progress_rewards = 0
        self.maps_discovered_this_episode = 0
        self.last_significant_progress = 0

        # Performance tracking
        self.total_episodes = 0
        self.successful_resets = 0

        logger.info(f"PokemonRedGymEnv initialized: {observation_type} observations, "
                    f"{reward_strategy} rewards, max_steps={max_episode_steps}")

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation in the specified format."""
        if self.observation_type == "multi_modal":
            return process_game_state(
                self.game,
                self.episode_steps,
                self.max_episode_steps,
                self.visited_locations,
                self.screen_size
            )
        elif self.observation_type == "minimal":
            from .observations import process_minimal_observation
            return process_minimal_observation(
                self.game,
                self.episode_steps,
                self.max_episode_steps,
                self.visited_locations
            )
        elif self.observation_type == "screen_only":
            from .observations import downsample_screen, normalize_screen
            screen = self.game.get_screen_array()
            screen = downsample_screen(screen, self.screen_size)
            return normalize_screen(screen)

    def _calculate_reward(self) -> float:
        """Calculate reward for current state."""
        # Get comprehensive game state
        game_state = self.game.get_comprehensive_state()

        # Calculate reward using the configured strategy
        reward = self.reward_calculator.calculate_reward(game_state)

        # Track reward components for monitoring
        components = self.reward_calculator.get_reward_breakdown()
        if 'exploration' in components:
            self.total_exploration_rewards += components['exploration']
        if any(key in components for key in ['level', 'badge', 'pokemon']):
            progress_reward = sum(components.get(key, 0) for key in ['level', 'badge', 'pokemon'])
            self.total_progress_rewards += progress_reward
            if progress_reward > 0:
                self.last_significant_progress = self.episode_steps

        return reward

    def _check_done(self) -> Tuple[bool, bool]:
        """
        Check if episode should end.

        Returns:
            (terminated, truncated) - terminated for natural end, truncated for timeout
        """
        # Check for maximum steps (truncation)
        if self.episode_steps >= self.max_episode_steps:
            logger.debug(f"Episode truncated: Max steps reached ({self.max_episode_steps})")
            return False, True

        # More lenient termination conditions
        game_state = self.game.get_comprehensive_state()

        # Only terminate if Pokemon has been unconscious for too long
        if (game_state['stats']['current_hp'] == 0 and
                game_state['stats']['max_hp'] > 0):
            if hasattr(self, '_unconscious_steps'):
                self._unconscious_steps += 1
                # Longer grace period (1000 steps instead of 500)
                if self._unconscious_steps > 1000:
                    logger.debug("Episode terminated: Pokemon unconscious too long")
                    return True, False
            else:
                self._unconscious_steps = 1
        else:
            self._unconscious_steps = 0

        # Early termination if completely stuck (no exploration for very long time)
        if (self.episode_steps > 1000 and
            self.episode_steps - self.last_significant_progress > 2000 and
            len(self.visited_locations) < 10):
            logger.debug("Episode terminated: Agent appears completely stuck")
            return True, False

        return False, False

    def _get_info(self) -> Dict[str, Any]:
        """Get additional episode information."""
        game_state = self.game.get_comprehensive_state()
        reward_breakdown = self.reward_calculator.get_reward_breakdown()
        exploration_progress = self.game.get_exploration_progress()

        # Count unique maps from visited locations
        unique_maps = set()
        for x, y, map_id in self.visited_locations:
            if map_id != 0:
                unique_maps.add(map_id)

        # Enhanced info with additional metrics
        info = {
            # Episode metrics
            'episode_steps': self.episode_steps,
            'episode_reward': self.episode_reward,
            'episode_count': self.episode_count,

            # Game state
            'current_map': game_state['position']['map'],
            'map_name': game_state['map_name'],
            'player_level': game_state['stats']['level'],
            'badges_earned': game_state['badge_count'],
            'pokemon_count': game_state['stats']['party_count'],
            'hp_ratio': game_state['stats']['hp_ratio'],
            'money': game_state['money'],
            'in_game': game_state['in_game'],
            'is_alive': game_state['is_alive'],

            # Exploration metrics
            'locations_visited': len(self.visited_locations),
            'maps_visited': len(unique_maps),
            'unique_maps_list': list(unique_maps),

            # Enhanced tracking metrics
            'exploration_efficiency': len(self.visited_locations) / max(self.episode_steps, 1),
            'maps_discovered_this_episode': len(unique_maps),
            'total_exploration_rewards': self.total_exploration_rewards,
            'total_progress_rewards': self.total_progress_rewards,
            'steps_since_progress': self.episode_steps - self.last_significant_progress,

            # Reward breakdown
            'reward_components': reward_breakdown,

            # Performance metrics
            'total_episodes': self.total_episodes,
            'successful_resets': self.successful_resets
        }

        return info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Execute action
        action_name = self.action_names[action]
        game_state = self.game.step(action_name)

        # Update episode tracking
        self.episode_steps += 1

        # Track visited locations
        position = game_state['position']
        location_key = (position['x'], position['y'], position['map'])
        self.visited_locations.add(location_key)

        # Get new observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward()
        self.episode_reward += reward

        # Check if episode should end
        terminated, truncated = self._check_done()

        # Get episode info
        info = self._get_info()

        # Enhanced periodic logging with exploration focus
        if self.episode_steps % 200 == 0:  # More frequent logging
            unique_maps = len(set(loc[2] for loc in self.visited_locations if loc[2] != 0))
            logger.debug(f"Step {self.episode_steps}: Map {position['map']}, "
                         f"Pos({position['x']}, {position['y']}), "
                         f"Reward: {reward:.2f}, Total: {self.episode_reward:.1f}, "
                         f"Maps: {unique_maps}, Locations: {len(self.visited_locations)}")

        # Validate observation
        if not validate_observation(observation, self.observation_space):
            logger.error("Invalid observation generated")
            # Return a valid default observation to prevent crashes
            observation = self._get_default_observation()

        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[
        Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment to starting state."""
        logger.info(f"Environment reset called (episode {self.total_episodes + 1}, "
                    f"previous steps: {self.episode_steps})")

        # Handle seed for reproducibility
        super().reset(seed=seed)

        # Enhanced episode tracking reset
        self.episode_steps = 0
        self.episode_reward = 0
        self.visited_locations.clear()
        self._unconscious_steps = 0
        self.total_episodes += 1
        self.episode_count += 1

        # Reset enhanced tracking
        self.total_exploration_rewards = 0
        self.total_progress_rewards = 0
        self.maps_discovered_this_episode = 0
        self.last_significant_progress = 0

        # Reset reward calculator
        self.reward_calculator.reset()

        # Reset game using the proven working method
        logger.info("Starting game reset...")
        try:
            success = self.game.reset_game()
            if success:
                self.successful_resets += 1
                logger.info("Game reset successful!")
            else:
                logger.warning("Game reset may have failed - continuing anyway")
        except Exception as e:
            logger.error(f"Game reset failed: {e}")
            # Try to continue anyway
            success = False

        # Wait for game to stabilize
        logger.debug("Waiting for game to stabilize...")
        self.game.wait_frames(60)

        # Get initial observation
        try:
            observation = self._get_observation()
        except Exception as e:
            logger.error(f"Failed to get initial observation: {e}")
            observation = self._get_default_observation()

        # Initialize reward calculator with first state
        try:
            self._calculate_reward()
        except Exception as e:
            logger.warning(f"Failed to initialize reward calculator: {e}")

        # Get initial info
        info = self._get_info()
        info['reset_successful'] = success

        logger.info(f"Reset complete! Episode {self.total_episodes}, "
                    f"Success rate: {self.successful_resets}/{self.total_episodes}")

        return observation, info

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == 'rgb_array':
            return self.game.get_screen_array()
        elif mode == 'human':
            # Enhanced display with exploration info
            game_state = self.game.get_comprehensive_state()
            position = game_state['position']
            stats = game_state['stats']
            unique_maps = len(set(loc[2] for loc in self.visited_locations if loc[2] != 0))

            print(f"Step {self.episode_steps}: "
                  f"Map {position['map']} ({game_state['map_name']}) "
                  f"Pos({position['x']}, {position['y']}) "
                  f"Level:{stats['level']} HP:{stats['current_hp']}/{stats['max_hp']} "
                  f"Badges:{game_state['badge_count']} "
                  f"Maps:{unique_maps} Locations:{len(self.visited_locations)} "
                  f"Reward:{self.episode_reward:.1f}")
        else:
            super().render()

    def close(self):
        """Clean up environment."""
        logger.info("Closing Pokemon Red environment...")
        try:
            self.game.cleanup()
            logger.info("Environment closed successfully")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")

    def _get_default_observation(self) -> Dict[str, np.ndarray]:
        """Get a valid default observation in case of errors."""
        if self.observation_type == "multi_modal":
            return {
                'screen': np.zeros((self.screen_size[1], self.screen_size[0], 3), dtype=np.uint8),
                'position': np.zeros(3, dtype=np.uint8),
                'stats': np.zeros(6, dtype=np.uint8),
                'exploration': np.zeros(2, dtype=np.uint16)
            }
        elif self.observation_type == "minimal":
            return np.zeros(11, dtype=np.uint16)
        elif self.observation_type == "screen_only":
            return np.zeros((self.screen_size[1], self.screen_size[0], 3), dtype=np.uint8)

    def get_action_meanings(self) -> list:
        """Get human-readable action meanings."""
        return self.action_names.copy()

    def seed(self, seed: Optional[int] = None) -> list:
        """Seed the environment's random number generator."""
        # Pokemon Red is deterministic, but we can seed numpy for consistency
        if seed is not None:
            np.random.seed(seed)
        return [seed]

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class PokemonRedVecEnv:
    """
    Vectorized environment wrapper for running multiple Pokemon Red environments.

    Note: This is a basic implementation. For production use, consider using
    stable-baselines3's VecEnv implementations with proper multiprocessing.
    """

    def __init__(self, rom_paths: list, **env_kwargs):
        """
        Initialize vectorized environment.

        Args:
            rom_paths: List of ROM file paths (can be the same ROM multiple times)
            **env_kwargs: Arguments to pass to each environment
        """
        self.num_envs = len(rom_paths)
        self.envs = []

        for i, rom_path in enumerate(rom_paths):
            # Make each environment headless except potentially the first
            kwargs = env_kwargs.copy()
            kwargs['headless'] = i > 0 or kwargs.get('headless', True)

            env = PokemonRedGymEnv(rom_path, **kwargs)
            self.envs.append(env)

        # Use first environment's spaces as template
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.num_envs = len(self.envs)

    def reset(self):
        """Reset all environments."""
        observations = []
        infos = []

        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)

        return observations, infos

    def step(self, actions):
        """Step all environments."""
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []

        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)

        return observations, rewards, terminateds, truncateds, infos

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()

    def render(self, mode='human'):
        """Render first environment."""
        return self.envs[0].render(mode)

    def get_attr(self, attr_name):
        """Get attribute from all environments."""
        return [getattr(env, attr_name) for env in self.envs]

    def set_attr(self, attr_name, values):
        """Set attribute on all environments."""
        for env, value in zip(self.envs, values):
            setattr(env, attr_name, value)