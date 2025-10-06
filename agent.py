"""
Complete Pokemon Red Reinforcement Learning Training System

This module provides a comprehensive RL training setup for Pokemon Red including:
- Game interface and control
- Gymnasium environment wrapper
- Training infrastructure
- Progress visualization
- Save/load functionality

Usage:
    python pokemon_rl_trainer.py --train --rom PokemonRed.gb --timesteps 100000
    python pokemon_rl_trainer.py --train --rom PokemonRed.gb --timesteps 100000 --monitor-mode
    python pokemon_rl_trainer.py --test --model best_model.zip
"""

import time
import numpy as np
import json
import argparse
import os
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, Set
import threading

# Core dependencies - Updated for Gymnasium
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# RL training dependencies
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Game emulation
from pyboy import PyBoy


class PokemonRedAgent:
    """
    Core Pokemon Red game interface and control system

    Handles game initialization, input control, and state reading
    """

    def __init__(self, rom_path: str, show_window: bool = False):
        """
        Initialize Pokemon Red game

        Args:
            rom_path: Path to Pokemon Red ROM file
            show_window: Whether to display game window
        """
        # Store initialization parameters for reset_game
        self.rom_path = rom_path
        self.show_window = show_window

        # Initialize PyBoy emulator
        window_type = "SDL2" if show_window else "null"
        self.pyboy = PyBoy(rom_path, window=window_type)

        # Direct property access for PyBoy 2.x
        self.memory = self.pyboy.memory
        self.screen = self.pyboy.screen

        # Button mappings for PyBoy 2.x
        self.buttons = {
            'A': 'a', 'B': 'b', 'SELECT': 'select', 'START': 'start',
            'RIGHT': 'right', 'LEFT': 'left', 'UP': 'up', 'DOWN': 'down'
        }

        # Pokemon Red memory addresses (from ROM hacking documentation)
        self.memory_addresses = {
            'player_x': 0xD362,  # Player X coordinate
            'player_y': 0xD361,  # Player Y coordinate
            'map_id': 0xD35E,  # Current map ID
            'player_level': 0xD18C,  # Pokemon level
            'current_hp_low': 0xD16C,  # Current HP (low byte)
            'current_hp_high': 0xD16D,  # Current HP (high byte)
            'max_hp_low': 0xD16E,  # Max HP (low byte)
            'max_hp_high': 0xD16F,  # Max HP (high byte)
            'badges': 0xD356,  # Badge bitfield
            'party_count': 0xD163,  # Number of Pokemon in party
            'game_state': 0xFF80,  # Game state flag
            'menu_state': 0xCC26,  # Menu state
        }

        # Tracking for rewards and progress
        self.visited_locations = set()
        self.previous_stats = {}
        self.episode_steps = 0

    def press_button(self, button: str, hold_frames: int = 10, release_frames: int = 5):
        """
        Press and release a Game Boy button

        Args:
            button: Button name ('A', 'B', etc.)
            hold_frames: Frames to hold button (10 frames ≈ 167ms at 60fps)
            release_frames: Frames to wait after release (5 frames ≈ 83ms)
        """
        try:
            # Try PyBoy 2.x method first
            self.pyboy.button_press(self.buttons[button])
            self.wait_frames(hold_frames)
            self.pyboy.button_release(self.buttons[button])
            self.wait_frames(release_frames)
        except (AttributeError, KeyError):
            # Fallback to send_input method
            button_map = {'A': 0, 'B': 1, 'SELECT': 2, 'START': 3,
                          'RIGHT': 4, 'LEFT': 5, 'UP': 6, 'DOWN': 7}
            if button in button_map:
                self.pyboy.send_input(button_map[button])
                self.wait_frames(hold_frames)
                self.pyboy.send_input(button_map[button], False)
                self.wait_frames(release_frames)

    def wait_frames(self, frames: int):
        """Wait for specified number of frames (60 frames = 1 second)"""
        for _ in range(frames):
            self.pyboy.tick()

    def read_memory_value(self, address: int, is_16bit: bool = False) -> int:
        """
        Safely read value from Game Boy memory

        Args:
            address: Memory address to read
            is_16bit: Whether to read as 16-bit value (little-endian)

        Returns:
            Memory value or 0 if read fails
        """
        try:
            if is_16bit:
                low = self.memory[address]
                high = self.memory[address + 1]
                return low | (high << 8)
            else:
                return self.memory[address]
        except:
            return 0

    def get_screen_array(self) -> np.ndarray:
        """Get current screen as numpy array"""
        screen_image = self.screen.image
        return np.array(screen_image)

    def get_player_position(self) -> Dict[str, int]:
        """Get current player position and map"""
        return {
            'x': self.read_memory_value(self.memory_addresses['player_x']),
            'y': self.read_memory_value(self.memory_addresses['player_y']),
            'map': self.read_memory_value(self.memory_addresses['map_id'])
        }

    def get_player_stats(self) -> Dict[str, int]:
        """Get current player/Pokemon statistics"""
        return {
            'level': self.read_memory_value(self.memory_addresses['player_level']),
            'current_hp': self.read_memory_value(self.memory_addresses['current_hp_low'], is_16bit=True),
            'max_hp': self.read_memory_value(self.memory_addresses['max_hp_low'], is_16bit=True),
            'badges': self.read_memory_value(self.memory_addresses['badges']),
            'party_count': self.read_memory_value(self.memory_addresses['party_count'])
        }

    def get_game_state_type(self) -> str:
        """Determine current game state (title, menu, in-game, etc.)"""
        try:
            map_id = self.read_memory_value(self.memory_addresses['map_id'])
            if map_id != 0:
                return "in_game"

            game_state = self.read_memory_value(self.memory_addresses['game_state'])
            menu_state = self.read_memory_value(self.memory_addresses['menu_state'])

            if game_state == 0 and menu_state == 0:
                return "title_screen"
            elif menu_state in [1, 2, 3]:
                return "main_menu"
            else:
                return "unknown"
        except:
            return "unknown"

    def complete_opening_sequence(self) -> bool:
        def complete_opening_sequence(self) -> bool:
            """
            Complete the entire Pokemon Red opening sequence automatically

            Uses the proven robust method from the working code

            Returns:
                True if successful, False otherwise
            """
            try:
                print("Starting Pokemon Red opening sequence...")

                # Step 1: Navigate through intro screens
                if not self.skip_intro_sequence():
                    print("Failed to navigate intro screens automatically")

                    # Wait for manual intervention (1 minute timeout)
                    for i in range(60):
                        current_screen = self.get_current_screen_type()
                        if current_screen == "in_game":
                            break
                        if i % 10 == 0:
                            print(f"Waiting for manual intervention... {i}/60 seconds")
                        self.wait_frames(60)
                    else:
                        return False

                # Step 2: Skip Professor Oak's intro dialogue
                print("Skipping Professor Oak introduction...")
                self.skip_professor_oak_intro()

                # Step 3: Handle naming screens
                print("Handling player naming (accepting default)...")
                self.handle_naming_screen("RED", is_player=True)

                print("Handling rival naming (accepting default)...")
                self.handle_naming_screen("BLUE", is_player=False)

                # Step 4: Complete remaining intro dialogue
                print("Completing intro dialogue...")
                self.complete_intro_dialogue()

                # Step 5: Verify game control is working
                print("Testing movement controls...")
                self.take_initial_steps()

                # Display final status
                position = self.get_player_position()
                stats = self.get_player_stats()
                print(f"Opening sequence completed successfully!")
                print(f"Player position: {position}")
                print(f"Player stats: {stats}")

                # Verify we're in game
                if self.get_current_screen_type() == "in_game":
                    print("Successfully in game!")
                    return True
                else:
                    print("Opening sequence may have failed - not detected as in game")
                    return False

            except Exception as e:
                print(f"Error in opening sequence: {e}")
                return False

    def reset_game(self):
        """Reset the game to starting state"""
        try:
            # Clean up current PyBoy instance
            self.pyboy.stop()
        except Exception as e:
            print(f"Warning during PyBoy cleanup: {e}")

        # Create new PyBoy instance
        try:
            window_type = "SDL2" if self.show_window else "null"
            self.pyboy = PyBoy(self.rom_path, window=window_type)
            self.memory = self.pyboy.memory
            self.screen = self.pyboy.screen
        except Exception as e:
            print(f"Error recreating PyBoy instance: {e}")
            return False

        self.visited_locations.clear()
        self.previous_stats.clear()
        self.episode_steps = 0
        return self.complete_opening_sequence()

    def cleanup(self):
        """Clean up PyBoy resources"""
        self.pyboy.stop()


class PokemonRedGymEnv(gym.Env):
    """
    OpenAI Gymnasium environment wrapper for Pokemon Red

    Provides standard RL interface for training algorithms
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, rom_path: str, headless: bool = True, max_episode_steps: int = 5000):
        """
        Initialize Pokemon Red Gymnasium environment

        Args:
            rom_path: Path to Pokemon Red ROM
            headless: Whether to run without display
            max_episode_steps: Maximum steps per episode
        """
        super().__init__()

        self.rom_path = rom_path
        self.headless = headless
        self.max_episode_steps = max_episode_steps

        # Initialize game game
        self.game = PokemonRedAgent(rom_path, show_window=not headless)

        # Define action space (8 Game Boy buttons)
        self.action_space = spaces.Discrete(8)
        self.action_names = ['A', 'B', 'SELECT', 'START', 'RIGHT', 'LEFT', 'UP', 'DOWN']

        # Define observation space (multi-modal: screen + game state)
        self.observation_space = spaces.Dict({
            'screen': spaces.Box(
                low=0, high=255,
                shape=(72, 80, 3),  # Downsampled for efficiency
                dtype=np.uint8
            ),
            'position': spaces.Box(
                low=0, high=255,
                shape=(3,),  # x, y, map_id
                dtype=np.uint8
            ),
            'stats': spaces.Box(
                low=0, high=255,
                shape=(5,),  # level, hp_ratio*100, badges, party_count, episode_progress
                dtype=np.uint8
            )
        })

        # Episode tracking
        self.episode_steps = 0
        self.episode_reward = 0
        self.visited_maps = set()
        self.visited_locations = set()
        self.previous_position = None
        self.previous_stats = None

        # Reward tracking
        self.reward_components = defaultdict(float)

    def _downsample_screen(self, screen: np.ndarray) -> np.ndarray:
        """Downsample screen from 144x160 to 72x80 for efficiency"""
        try:
            import cv2
            return cv2.resize(screen, (80, 72), interpolation=cv2.INTER_AREA)
        except ImportError:
            # Fallback: simple downsampling
            return screen[::2, ::2]

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation in Gymnasium format"""
        # Get raw screen and downsample
        screen = self.game.get_screen_array()
        screen_small = self._downsample_screen(screen)

        # Convert RGBA to RGB if needed (PyBoy sometimes returns 4 channels)
        if screen_small.shape[2] == 4:
            screen_small = screen_small[:, :, :3]  # Remove alpha channel
        elif screen_small.shape[2] != 3:
            # If it's grayscale or some other format, convert to RGB
            if len(screen_small.shape) == 2:
                screen_small = np.stack([screen_small] * 3, axis=-1)
            else:
                # Fallback: just take first 3 channels or pad to 3
                if screen_small.shape[2] < 3:
                    # Pad with zeros if less than 3 channels
                    pad_width = ((0, 0), (0, 0), (0, 3 - screen_small.shape[2]))
                    screen_small = np.pad(screen_small, pad_width, mode='constant')
                else:
                    # Take first 3 channels if more than 3
                    screen_small = screen_small[:, :, :3]

        # Get game state
        position = self.game.get_player_position()
        stats = self.game.get_player_stats()

        # Calculate derived values
        hp_ratio = int((stats['current_hp'] / max(stats['max_hp'], 1)) * 100)
        episode_progress = min(int((self.episode_steps / self.max_episode_steps) * 100), 100)

        observation = {
            'screen': screen_small.astype(np.uint8),
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
                episode_progress
            ], dtype=np.uint8)
        }

        return observation

    def _calculate_reward(self) -> float:
        """Calculate reward for current state"""
        reward = 0.0
        self.reward_components.clear()

        # Get current game state
        position = self.game.get_player_position()
        stats = self.game.get_player_stats()

        # Base time penalty (encourages efficiency)
        time_penalty = -0.01
        reward += time_penalty
        self.reward_components['time'] = time_penalty

        # Exploration rewards
        location_key = (position['x'], position['y'], position['map'])
        if location_key not in self.visited_locations:
            self.visited_locations.add(location_key)
            exploration_reward = 1.0
            reward += exploration_reward
            self.reward_components['exploration'] = exploration_reward

        # New map discovery
        if position['map'] not in self.visited_maps and position['map'] != 0:
            self.visited_maps.add(position['map'])
            map_reward = 20.0
            reward += map_reward
            self.reward_components['new_map'] = map_reward

        # Progress rewards (compare with previous state)
        if self.previous_stats:
            # Level progression
            level_diff = stats['level'] - self.previous_stats['level']
            if level_diff > 0:
                level_reward = level_diff * 50.0
                reward += level_reward
                self.reward_components['level'] = level_reward

            # Badge progression (major milestone)
            badge_diff = stats['badges'] - self.previous_stats['badges']
            if badge_diff > 0:
                badge_reward = badge_diff * 200.0
                reward += badge_reward
                self.reward_components['badge'] = badge_reward

            # Pokemon acquisition
            party_diff = stats['party_count'] - self.previous_stats['party_count']
            if party_diff > 0:
                pokemon_reward = party_diff * 100.0
                reward += pokemon_reward
                self.reward_components['pokemon'] = pokemon_reward

        # Health penalty (losing HP is bad)
        if stats['max_hp'] > 0:
            hp_ratio = stats['current_hp'] / stats['max_hp']
            if hp_ratio < 0.5:  # Low health penalty
                health_penalty = -10.0 * (0.5 - hp_ratio)
                reward += health_penalty
                self.reward_components['health'] = health_penalty

        # Death penalty (Pokemon fainting)
        if stats['current_hp'] == 0 and stats['max_hp'] > 0:
            death_penalty = -100.0
            reward += death_penalty
            self.reward_components['death'] = death_penalty

        # Update previous stats for next comparison
        self.previous_stats = stats.copy()

        return reward

    def _check_done(self) -> bool:
        """Check if episode should end"""
        stats = self.game.get_player_stats()

        # Episode length limit
        if self.episode_steps >= self.max_episode_steps:
            return True

        # Victory condition (all 8 badges)
        if stats['badges'] >= 8:
            return True

        # Failure condition (no Pokemon and no HP)
        if stats['party_count'] == 0 or (stats['current_hp'] == 0 and stats['max_hp'] > 0):
            return True

        return False

    def _get_info(self) -> Dict[str, Any]:
        """Get additional episode information"""
        position = self.game.get_player_position()
        stats = self.game.get_player_stats()

        return {
            'episode_steps': self.episode_steps,
            'episode_reward': self.episode_reward,
            'maps_visited': len(self.visited_maps),
            'locations_visited': len(self.visited_locations),
            'current_map': position['map'],
            'player_level': stats['level'],
            'badges_earned': stats['badges'],
            'pokemon_count': stats['party_count'],
            'reward_components': dict(self.reward_components)
        }

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one environment step"""
        # Execute action
        action_name = self.action_names[action]
        self.game.press_button(action_name)

        # Update episode tracking
        self.episode_steps += 1

        # Get new state
        observation = self._get_observation()
        reward = self._calculate_reward()
        terminated = self._check_done()
        truncated = False  # We handle episode length in terminated
        info = self._get_info()

        # Update episode reward
        self.episode_reward += reward

        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[
        Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment to starting state"""
        # Handle seed for reproducibility
        super().reset(seed=seed)

        # Reset episode tracking
        self.episode_steps = 0
        self.episode_reward = 0
        self.visited_maps.clear()
        self.visited_locations.clear()
        self.previous_stats = None
        self.reward_components.clear()

        # Reset game
        success = self.game.reset_game()
        if not success:
            print("Warning: Game reset may have failed")

        # Wait a moment for game to stabilize
        self.game.wait_frames(60)

        # Get initial observation
        observation = self._get_observation()

        # Initialize previous stats
        self.previous_stats = self.game.get_player_stats()

        # Return observation and info (Gymnasium format)
        info = self._get_info()
        return observation, info

    def render(self, mode: str = 'human'):
        """Render the environment"""
        if mode == 'rgb_array':
            return self.game.get_screen_array()
        elif mode == 'human':
            # Display current state info
            position = self.game.get_player_position()
            stats = self.game.get_player_stats()
            print(f"Step {self.episode_steps}: Pos({position['x']}, {position['y']}, {position['map']}) "
                  f"Level:{stats['level']} HP:{stats['current_hp']}/{stats['max_hp']} "
                  f"Badges:{stats['badges']} Reward:{self.episode_reward:.1f}")
        else:
            super().render()

    def close(self):
        """Clean up environment"""
        self.game.cleanup()


class TrainingCallback(BaseCallback):
    """
    Custom callback for Pokemon Red training

    Handles logging, checkpointing, and progress tracking
    """

    def __init__(self, save_freq: int = 10000, save_path: str = './models/', verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.save_path_models = os.path.join(save_path, 'models')
        self.save_path_logs = os.path.join(save_path, 'logs')

        # Create directories
        os.makedirs(self.save_path_models, exist_ok=True)
        os.makedirs(self.save_path_logs, exist_ok=True)

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.exploration_progress = []
        self.best_reward = -float('inf')

    def _on_step(self) -> bool:
        """Called after each environment step"""
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout"""
        # Log training progress
        if len(self.model.ep_info_buffer) > 0:
            recent_episodes = self.model.ep_info_buffer[-10:]  # Last 10 episodes

            rewards = [ep['r'] for ep in recent_episodes]
            lengths = [ep['l'] for ep in recent_episodes]

            mean_reward = np.mean(rewards)
            mean_length = np.mean(lengths)

            self.episode_rewards.extend(rewards)
            self.episode_lengths.extend(lengths)

            # Track best performance
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                # Save best model
                best_model_path = os.path.join(self.save_path_models, 'best_model')
                self.model.save(best_model_path)

                if self.verbose >= 1:
                    print(f"New best model saved! Reward: {mean_reward:.2f}")

            # Periodic logging
            if self.verbose >= 1 and self.num_timesteps % 1000 == 0:
                print(f"Step {self.num_timesteps}: Reward={mean_reward:.2f}, Length={mean_length:.0f}")

        # Periodic model saving
        if self.num_timesteps % self.save_freq == 0:
            model_path = os.path.join(self.save_path_models, f'model_{self.num_timesteps}')
            self.model.save(model_path)

            if self.verbose >= 1:
                print(f"Model checkpoint saved at step {self.num_timesteps}")


class EnhancedTrainingCallback(BaseCallback):
    """
    Enhanced callback with live visualizations and detailed progress tracking
    """

    def __init__(self, save_freq: int = 10000, save_path: str = './models/',
                 show_plots: bool = True, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.show_plots = show_plots
        self.save_path_models = os.path.join(save_path, 'models')
        self.save_path_logs = os.path.join(save_path, 'logs')

        # Create directories
        os.makedirs(self.save_path_models, exist_ok=True)
        os.makedirs(self.save_path_logs, exist_ok=True)

        # Training statistics with limited history for plotting
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.maps_discovered = deque(maxlen=1000)
        self.badges_earned = deque(maxlen=1000)
        self.exploration_scores = deque(maxlen=1000)

        # Timestep tracking for x-axis
        self.timesteps = deque(maxlen=1000)

        self.best_reward = -float('inf')
        self.best_exploration = 0

        # Live plotting setup
        if self.show_plots:
            self.setup_live_plots()

    def setup_live_plots(self):
        """Setup matplotlib for live plotting"""
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Pokemon Red RL Training Progress')

        # Configure subplots
        self.axes[0, 0].set_title('Episode Rewards')
        self.axes[0, 0].set_xlabel('Timestep')
        self.axes[0, 0].set_ylabel('Reward')

        self.axes[0, 1].set_title('Episode Length')
        self.axes[0, 1].set_xlabel('Timestep')
        self.axes[0, 1].set_ylabel('Steps')

        self.axes[1, 0].set_title('Maps Discovered')
        self.axes[1, 0].set_xlabel('Timestep')
        self.axes[1, 0].set_ylabel('Unique Maps')

        self.axes[1, 1].set_title('Badges Earned')
        self.axes[1, 1].set_xlabel('Timestep')
        self.axes[1, 1].set_ylabel('Badge Count')

        # Initialize empty line objects
        self.lines = {}
        for i, ax in enumerate(self.axes.flat):
            line, = ax.plot([], [], 'b-', alpha=0.7)
            self.lines[i] = line

        plt.tight_layout()
        plt.show(block=False)

    def update_plots(self):
        """Update the live plots with current data"""
        if not self.show_plots or len(self.timesteps) == 0:
            return

        try:
            timesteps = list(self.timesteps)

            # Update reward plot
            if len(self.episode_rewards) > 0:
                rewards = list(self.episode_rewards)
                self.lines[0].set_data(timesteps[-len(rewards):], rewards)
                self.axes[0, 0].relim()
                self.axes[0, 0].autoscale_view()

            # Update length plot
            if len(self.episode_lengths) > 0:
                lengths = list(self.episode_lengths)
                self.lines[1].set_data(timesteps[-len(lengths):], lengths)
                self.axes[0, 1].relim()
                self.axes[0, 1].autoscale_view()

            # Update maps plot
            if len(self.maps_discovered) > 0:
                maps = list(self.maps_discovered)
                self.lines[2].set_data(timesteps[-len(maps):], maps)
                self.axes[1, 0].relim()
                self.axes[1, 0].autoscale_view()

            # Update badges plot
            if len(self.badges_earned) > 0:
                badges = list(self.badges_earned)
                self.lines[3].set_data(timesteps[-len(badges):], badges)
                self.axes[1, 1].relim()
                self.axes[1, 1].autoscale_view()

            # Refresh the plot
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        except Exception as e:
            if self.verbose >= 1:
                print(f"Plot update error: {e}")

    def _on_step(self) -> bool:
        """Called after each environment step"""
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout"""
        # Process episode info if available
        if len(self.model.ep_info_buffer) > 0:
            recent_episodes = list(self.model.ep_info_buffer)

            for episode_info in recent_episodes:
                reward = episode_info['r']
                length = episode_info['l']

                # Try to get additional info if available
                maps = episode_info.get('maps_visited', 0)
                badges = episode_info.get('badges_earned', 0)

                # Store the data
                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                self.maps_discovered.append(maps)
                self.badges_earned.append(badges)
                self.timesteps.append(self.num_timesteps)

                # Track best performance
                if reward > self.best_reward:
                    self.best_reward = reward
                    # Save best model
                    best_model_path = os.path.join(self.save_path_models, 'best_model')
                    self.model.save(best_model_path)

                    if self.verbose >= 1:
                        print(f"\n🏆 New best model! Reward: {reward:.2f}, Maps: {maps}, Badges: {badges}")

            # Calculate rolling averages for display
            if len(self.episode_rewards) >= 10:
                recent_rewards = list(self.episode_rewards)[-10:]
                recent_lengths = list(self.episode_lengths)[-10:]
                recent_maps = list(self.maps_discovered)[-10:]
                recent_badges = list(self.badges_earned)[-10:]

                avg_reward = np.mean(recent_rewards)
                avg_length = np.mean(recent_lengths)
                avg_maps = np.mean(recent_maps)
                max_badges = max(recent_badges) if recent_badges else 0

                # Detailed logging every 1000 steps
                if self.verbose >= 1 and self.num_timesteps % 1000 == 0:
                    print(f"\n📊 Step {self.num_timesteps:,}")
                    print(f"   Reward: {avg_reward:.2f} (best: {self.best_reward:.2f})")
                    print(f"   Length: {avg_length:.0f} steps")
                    print(f"   Maps: {avg_maps:.1f} (recent)")
                    print(f"   Badges: {max_badges} (best recent)")

                    # Update plots
                    if self.show_plots:
                        self.update_plots()

        # Periodic model saving
        if self.num_timesteps % self.save_freq == 0:
            model_path = os.path.join(self.save_path_models, f'model_{self.num_timesteps}')
            self.model.save(model_path)

            if self.verbose >= 1:
                print(f"💾 Checkpoint saved at step {self.num_timesteps:,}")

    def cleanup(self):
        """Clean up plotting resources"""
        if self.show_plots:
            plt.close(self.fig)


class PokemonTrainer:
    """
    Complete Pokemon Red training system

    Handles model training, evaluation, and progress tracking
    """

    def __init__(self, rom_path: str, save_dir: str = './pokemon_training/'):
        """
        Initialize trainer

        Args:
            rom_path: Path to Pokemon Red ROM
            save_dir: Directory to save models and logs
        """
        self.rom_path = rom_path
        self.save_dir = save_dir

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Initialize environment
        self.env = None
        self.model = None

        # Training statistics
        self.training_stats = {
            'episodes_completed': 0,
            'total_timesteps': 0,
            'best_reward': -float('inf'),
            'training_start_time': None
        }

    def create_environment(self, headless: bool = True) -> PokemonRedGymEnv:
        """Create and validate Pokemon Red environment"""
        env = PokemonRedGymEnv(self.rom_path, headless=headless)

        # Validate environment
        print("Validating environment...")
        try:
            check_env(env)
            print("Environment validation passed!")
        except Exception as e:
            print(f"Environment validation failed: {e}")
            print("Continuing anyway - this might work with newer Stable-Baselines3 versions")

        return env

    def create_model(self, env: gym.Env, algorithm: str = 'PPO') -> Any:
        """Create RL model"""
        if algorithm == 'PPO':
            model = PPO(
                "MultiInputPolicy",
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=1,
                tensorboard_log=os.path.join(self.save_dir, 'tensorboard/'),
                device='auto'  # Automatically choose CPU/GPU
            )
        else:
            raise ValueError(f"Algorithm {algorithm} not supported")

        return model

    def train(self, total_timesteps: int = 100000, algorithm: str = 'PPO',
              show_game: bool = False, show_plots: bool = False):
        """
        Train Pokemon Red AI

        Args:
            total_timesteps: Number of training steps
            algorithm: RL algorithm to use
            show_game: Whether to show the game window during training
            show_plots: Whether to show live training plots
        """
        print(f"Starting Pokemon Red RL training...")
        print(f"Algorithm: {algorithm}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Save directory: {self.save_dir}")

        if show_game:
            print("🎮 Game window will be visible")
        if show_plots:
            print("📊 Live training plots enabled")

        # Create environment
        self.env = self.create_environment(headless=not show_game)
        self.env = Monitor(self.env, os.path.join(self.save_dir, 'monitor'))

        # Create model
        self.model = self.create_model(self.env, algorithm)

        # Use enhanced callback if plots are enabled
        if show_plots:
            callback = EnhancedTrainingCallback(
                save_freq=10000,
                save_path=self.save_dir,
                show_plots=True,
                verbose=1
            )
        else:
            callback = TrainingCallback(
                save_freq=10000,
                save_path=self.save_dir,
                verbose=1
            )

        # Start training
        self.training_stats['training_start_time'] = datetime.now()

        try:
            print("🚀 Training started! Press Ctrl+C to stop gracefully.")
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                tb_log_name=f"pokemon_red_{algorithm.lower()}",
                progress_bar=True
            )

            # Save final model
            final_model_path = os.path.join(self.save_dir, 'models', 'final_model')
            self.model.save(final_model_path)

            print(f"🎉 Training completed! Final model saved to {final_model_path}")

        except KeyboardInterrupt:
            print("⏹️  Training interrupted by user")
            # Save current model
            interrupt_model_path = os.path.join(self.save_dir, 'models', 'interrupted_model')
            self.model.save(interrupt_model_path)
            print(f"💾 Current model saved to {interrupt_model_path}")

        finally:
            if hasattr(callback, 'cleanup'):
                callback.cleanup()
            self.env.close()

    def test(self, model_path: str, episodes: int = 10, render: bool = True):
        """
        Test trained model

        Args:
            model_path: Path to trained model
            episodes: Number of test episodes
            render: Whether to display game
        """
        print(f"Testing model: {model_path}")

        # Load model
        self.model = PPO.load(model_path)

        # Create environment
        self.env = self.create_environment(headless=not render)

        # Run test episodes
        for episode in range(episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_steps = 0

            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)

                episode_reward += reward
                episode_steps += 1

                if render:
                    self.env.render()

                if terminated or truncated:
                    break

            print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
                  f"Steps={episode_steps}, Maps={info['maps_visited']}, "
                  f"Badges={info['badges_earned']}")

        self.env.close()

    def save_training_stats(self):
        """Save training statistics"""
        stats_path = os.path.join(self.save_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            # Convert datetime to string for JSON serialization
            stats_copy = self.training_stats.copy()
            if stats_copy['training_start_time']:
                stats_copy['training_start_time'] = stats_copy['training_start_time'].isoformat()
            json.dump(stats_copy, f, indent=2)


def main():
    """Main entry point for Pokemon Red RL training"""
    parser = argparse.ArgumentParser(description='Pokemon Red RL Training System')
    parser.add_argument('--rom', type=str, required=True, help='Path to Pokemon Red ROM file')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--test', action='store_true', help='Test a trained model')
    parser.add_argument('--model', type=str, help='Path to model for testing')
    parser.add_argument('--timesteps', type=int, default=100000, help='Training timesteps')
    parser.add_argument('--episodes', type=int, default=10, help='Test episodes')
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO'], help='RL algorithm')
    parser.add_argument('--save-dir', type=str, default='./pokemon_training/', help='Save directory')
    parser.add_argument('--headless', action='store_true', help='Run without display')

    # New monitoring options
    parser.add_argument('--show-game', action='store_true', help='Show game window during training')
    parser.add_argument('--show-plots', action='store_true', help='Show live training plots')
    parser.add_argument('--monitor-mode', action='store_true', help='Enable both game window and plots')

    args = parser.parse_args()

    # Handle monitor mode (enables both game and plots)
    if args.monitor_mode:
        args.show_game = True
        args.show_plots = True

    # Validate arguments
    if not os.path.exists(args.rom):
        print(f"Error: ROM file not found: {args.rom}")
        return

    if args.test and not args.model:
        print("Error: --model required when using --test")
        return

    if not args.train and not args.test:
        print("Error: Must specify either --train or --test")
        return

    # Initialize trainer
    trainer = PokemonTrainer(args.rom, args.save_dir)

    try:
        if args.train:
            print("=== POKEMON RED RL TRAINING ===")
            trainer.train(
                total_timesteps=args.timesteps,
                algorithm=args.algorithm,
                show_game=args.show_game,
                show_plots=args.show_plots
            )
            trainer.save_training_stats()

        if args.test:
            print("=== POKEMON RED RL TESTING ===")
            trainer.test(
                model_path=args.model,
                episodes=args.episodes,
                render=not args.headless
            )

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("Pokemon Red RL system completed!")


if __name__ == "__main__":
    main()