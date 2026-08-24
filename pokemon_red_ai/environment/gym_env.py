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
    normalize_screen,
    process_game_state,
    validate_observation,
    # Paper observation treatments
    create_pixel_observation_space,
    process_pixel_observation,
    create_symbolic_observation_space,
    process_symbolic_observation,
    create_hybrid_observation_space,
    process_hybrid_observation,
    SYMBOLIC_DIM,
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
                 observation_type: str = "multi_modal",
                 save_state_path: Optional[str] = None,
                 dynamic_episode_budget: bool = False,
                 dynamic_budget_initial: int = 10_240,
                 dynamic_budget_per_event: int = 2_048):
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
            save_state_path: Optional path to a PyBoy ``.state`` file.
                If provided, ``reset()`` loads this state instead of
                replaying the intro sequence (much faster).
            dynamic_episode_budget: If True, replace the fixed
                ``max_episode_steps`` truncation with the Pleines et al.
                (2025) dynamic budget: episodes start with
                ``dynamic_budget_initial`` steps and gain
                ``dynamic_budget_per_event`` steps for each event flag
                set since reset.  Desynchronizes parallel-env resets and
                mitigates catastrophic forgetting (paper §II-E).
            dynamic_budget_initial: Starting step budget (paper: 10,240).
            dynamic_budget_per_event: Extra steps per completed event
                (paper: 2,048).
        """
        super().__init__()

        self.rom_path = rom_path
        self.headless = headless
        self.max_episode_steps = max_episode_steps
        self.screen_size = screen_size
        self.observation_type = observation_type
        self.save_state_path = save_state_path
        self.dynamic_episode_budget = dynamic_episode_budget
        self.dynamic_budget_initial = dynamic_budget_initial
        self.dynamic_budget_per_event = dynamic_budget_per_event
        self._episode_event_baseline = 0

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

        # Define action space (7 useful Game Boy buttons — SELECT removed
        # per analysis_plan.md §5.2; it has no effect in Pokemon Red)
        self.action_space = gym.spaces.Discrete(7)
        self.action_names = ['A', 'B', 'START', 'RIGHT', 'LEFT', 'UP', 'DOWN']

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
        # ── Paper observation treatments ────────────────────────────
        elif observation_type == "pixel":
            self.observation_space = create_pixel_observation_space(screen_size)
        elif observation_type == "symbolic":
            self.observation_space = create_symbolic_observation_space()
        elif observation_type == "hybrid":
            self.observation_space = create_hybrid_observation_space(screen_size)
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
        # ── Paper observation treatments ────────────────────────────
        elif self.observation_type == "pixel":
            return process_pixel_observation(self.game, self.screen_size)
        elif self.observation_type == "symbolic":
            return process_symbolic_observation(
                self.game, self.episode_steps,
                self.max_episode_steps, self.visited_locations
            )
        elif self.observation_type == "hybrid":
            return process_hybrid_observation(
                self.game, self.episode_steps,
                self.max_episode_steps, self.visited_locations,
                self.screen_size
            )

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
        game_state = self.game.get_comprehensive_state()

        # Check for maximum steps (truncation).  With the dynamic
        # budget (Pleines et al. 2025 §II-E) the cap grows with each
        # event flag set since reset instead of being fixed.
        if self.dynamic_episode_budget:
            events_completed = max(
                0,
                game_state.get('event_flag_count', 0)
                - self._episode_event_baseline,
            )
            step_budget = (self.dynamic_budget_initial
                           + self.dynamic_budget_per_event * events_completed)
            if self.episode_steps >= step_budget:
                logger.debug(
                    f"Episode truncated: dynamic budget exhausted "
                    f"({step_budget} steps, {events_completed} events)")
                return False, True
        elif self.episode_steps >= self.max_episode_steps:
            logger.debug(f"Episode truncated: Max steps reached ({self.max_episode_steps})")
            return False, True

        # More lenient termination conditions

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
            'successful_resets': self.successful_resets,

            # Event flag progress (when using EventProgressRewardCalculator)
            'battle_state': game_state.get('battle_state', 0),
        }

        # Event flag progress when the reward calculator supports it.
        # Always present (empty dict otherwise): Monitor(info_keywords=
        # MONITORED_INFO_KEYS) raises KeyError at episode end for any
        # missing key, which would crash training under every reward
        # strategy without get_event_progress (pleines, standard, ...).
        if hasattr(self.reward_calculator, 'get_event_progress'):
            info['event_progress'] = self.reward_calculator.get_event_progress()
        else:
            info['event_progress'] = {}

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

        # Determine save state: options override, then instance default
        save_state = None
        if options and 'save_state' in options:
            save_state = options['save_state']
        elif self.save_state_path:
            save_state = self.save_state_path

        # Reset game — use save state when available (much faster than
        # replaying the intro sequence every episode)
        logger.info("Starting game reset...")
        try:
            if save_state:
                success = self.game.load_save_state(save_state)
                if success:
                    self.successful_resets += 1
                    logger.info(f"Loaded save state: {save_state}")
                else:
                    logger.warning("Save state load failed, falling back to full reset")
                    success = self.game.reset_game()
            else:
                success = self.game.reset_game()
            if success:
                self.successful_resets += 1
                logger.info("Game reset successful!")
            else:
                logger.warning("Game reset may have failed - continuing anyway")
        except Exception as e:
            logger.error(f"Game reset failed: {e}")
            success = False

        # Wait for game to stabilize
        logger.debug("Waiting for game to stabilize...")
        self.game.wait_frames(60)

        # Baseline for the dynamic episode budget: only flags set AFTER
        # reset extend the episode (a save state starts with some set).
        if self.dynamic_episode_budget:
            try:
                self._episode_event_baseline = (
                    self.game.get_comprehensive_state()
                    .get('event_flag_count', 0)
                )
            except Exception as e:
                logger.warning(f"Failed to read event-flag baseline: {e}")
                self._episode_event_baseline = 0

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

    def get_screen_rgb(self) -> Optional[np.ndarray]:
        """Return the current game screen as an (H, W, 3) uint8 array.

        Screen-capture entry point for monitoring callbacks.  Two hard
        requirements, both learned from the 2026-07-23 grid outage
        (AMC-254):

        * **Zero-arg**, so it can be invoked via ``VecEnv.env_method``
          through wrapper ``__getattr__`` forwarding.  Calling
          ``render("rgb_array")`` that way hits gymnasium's
          ``Wrapper.render()``, which takes no positional args — the
          TypeError is raised *inside* the SubprocVecEnv worker and
          kills it, taking the whole training run down.
        * **Never raises**: an exception inside ``env_method`` also
          kills the worker, so any failure returns None instead.

        The raw PyBoy screen is RGBA — ``pyboy.screen.image`` is a PIL
        image in mode ``RGBA``, so ``get_screen_array`` hands back
        ``(144, 160, 4)``.  ``normalize_screen`` drops the alpha channel
        so this actually returns the RGB it advertises, matching what
        every observation path already does with the same array.
        """
        try:
            return normalize_screen(self.game.get_screen_array())
        except Exception as e:
            logger.debug(f"get_screen_rgb failed: {e}")
            return None

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
        # ── Paper observation treatments ────────────────────────────
        elif self.observation_type == "pixel":
            return np.zeros((self.screen_size[1], self.screen_size[0], 1), dtype=np.uint8)
        elif self.observation_type == "symbolic":
            return np.zeros(SYMBOLIC_DIM, dtype=np.float32)
        elif self.observation_type == "hybrid":
            return {
                "screen": np.zeros((self.screen_size[1], self.screen_size[0], 1), dtype=np.uint8),
                "game_state": np.zeros(SYMBOLIC_DIM, dtype=np.float32),
            }

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
    .. deprecated:: 0.2.0
       This class loops sub-envs **sequentially** in the main process —
       it does not parallelise across cores.  All paper training and
       evaluation uses ``stable_baselines3.common.vec_env.SubprocVecEnv``
       directly (see ``scripts/train.py:_make_vec_env``).  Use that
       instead.  This class will be removed in a future release.

    Vectorized environment wrapper for running multiple Pokemon Red environments.

    Note: serial implementation.  For production use, use
    stable-baselines3's ``SubprocVecEnv``.
    """

    def __init__(self, rom_paths: list, **env_kwargs):
        """
        Initialize vectorized environment.

        Args:
            rom_paths: List of ROM file paths (can be the same ROM multiple times)
            **env_kwargs: Arguments to pass to each environment
        """
        import warnings as _warnings
        _warnings.warn(
            "PokemonRedVecEnv is a serial wrapper kept for back-compat; "
            "it does not parallelise across cores and will be removed in "
            "a future release.  Use stable_baselines3.common.vec_env."
            "SubprocVecEnv (or DummyVecEnv) instead — see scripts/train.py.",
            DeprecationWarning,
            stacklevel=2,
        )
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