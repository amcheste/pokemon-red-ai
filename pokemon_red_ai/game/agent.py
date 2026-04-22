"""
Pokemon Red AI Agent for playing Pokemon Red using PyBoy emulator.

This is the main game interface that handles PyBoy initialization,
game automation, and provides a clean API for RL training.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Set
from pyboy import PyBoy

from .memory import (
    read_player_position,
    read_player_stats,
    read_game_state,
    get_comprehensive_state,
    is_in_game
)
from .controls import (
    ScreenType,
    detect_screen_type,
    press_button_basic,
    press_button_smart,
    wait_frames,
    wait_for_screen_change,
    spam_button,
    accept_default_name,
    get_screen_array,
    get_tilemap_background
)

logger = logging.getLogger(__name__)


class PokemonRedAgent:
    """
    AI Agent for playing Pokemon Red using PyBoy emulator.

    This is the core game interface that handles all PyBoy interactions
    and provides a clean API for RL training environments.
    """

    def __init__(self, rom_path: str, show_window: bool = True,
                 speed_multiplier: int = 1, enable_save_states: bool = False):
        """
        Initialize the Pokemon Red game.

        Args:
            rom_path: Path to Pokemon Red ROM file
            show_window: If True, shows PyBoy window; if False, runs headless
            speed_multiplier: Game speed multiplier (1 = normal, 0 = unlimited)
            enable_save_states: Whether to enable save states
        """
        self.rom_path = rom_path
        self.show_window = show_window
        self.speed_multiplier = speed_multiplier
        self.enable_save_states = enable_save_states

        # Initialize PyBoy with compatibility checks
        try:
            # Method 1: Try modern PyBoy 2.x syntax
            if show_window:
                self.pyboy = PyBoy(rom_path, window="SDL2")
            else:
                self.pyboy = PyBoy(rom_path, window="null")
            logger.info("PyBoy initialized with modern syntax")

        except (TypeError, ValueError) as e:
            logger.warning(f"Modern PyBoy syntax failed: {e}")
            try:
                # Method 2: Try legacy PyBoy 1.x syntax
                if show_window:
                    self.pyboy = PyBoy(rom_path, window_type="SDL2")
                else:
                    self.pyboy = PyBoy(rom_path, window_type="null")
                logger.info("PyBoy initialized with legacy syntax")

            except (TypeError, ValueError) as e2:
                logger.warning(f"Legacy PyBoy syntax failed: {e2}")
                # Method 3: Minimal initialization
                self.pyboy = PyBoy(rom_path)
                logger.info("PyBoy initialized with minimal syntax")

        # Set game speed if method exists
        if hasattr(self.pyboy, 'set_emulation_speed'):
            try:
                self.pyboy.set_emulation_speed(speed_multiplier)
                logger.debug(f"Emulation speed set to {speed_multiplier}")
            except Exception as e:
                logger.warning(f"Could not set emulation speed: {e}")
        else:
            logger.warning("set_emulation_speed method not available")

        # Direct property access for PyBoy 2.x
        self.memory = self.pyboy.memory
        self.screen = self.pyboy.screen

        # RL training extensions
        self.visited_locations: Set[tuple] = set()
        self.previous_stats: Dict[str, Any] = {}
        self.episode_steps = 0
        self.is_initialized = False

        logger.info(f"PokemonRedAgent initialized with ROM: {rom_path}")

    def press_button(self, button: str, hold_frames: int = 10, release_frames: int = 5) -> None:
        """
        Press and release a Game Boy button with proper timing.

        Args:
            button: Button name ('A', 'B', 'START', etc.)
            hold_frames: How long to hold button (10 frames = ~167ms at 60fps)
            release_frames: How long to wait after release (5 frames = ~83ms)
        """
        press_button_basic(self.pyboy, button, hold_frames, release_frames)

    def wait_frames(self, frames: int) -> None:
        """Wait for a specified number of frames (60 frames = 1 second)."""
        wait_frames(self.pyboy, frames)

    def get_screen_array(self) -> np.ndarray:
        """Get current screen as numpy array for ML/computer vision."""
        return get_screen_array(self.pyboy)

    def get_tilemap_background(self) -> np.ndarray:
        """Get background tilemap for efficient game state analysis."""
        return get_tilemap_background(self.pyboy)

    def get_current_screen_type(self) -> ScreenType:
        """Analyze current screen to determine game state."""
        return detect_screen_type(self.pyboy)

    def get_player_position(self) -> Dict[str, int]:
        """Get current player position and map information."""
        return read_player_position(self.memory)

    def get_player_stats(self) -> Dict[str, int]:
        """Get current player/Pokemon statistics."""
        return read_player_stats(self.memory)

    def get_game_state(self) -> Dict[str, int]:
        """Get game state indicators."""
        return read_game_state(self.memory)

    def get_comprehensive_state(self) -> Dict[str, Any]:
        """Get comprehensive game state information."""
        return get_comprehensive_state(self.memory)

    def is_in_game(self) -> bool:
        """Check if player is currently in the game world."""
        return is_in_game(self.memory)

    def skip_intro_sequence(self) -> bool:
        """
        Automatically navigate through Pokemon Red's intro sequence.

        Returns:
            True if successful, False if failed
        """
        logger.info("Starting intro sequence navigation...")

        # Wait for game to fully load (3 seconds)
        self.wait_frames(180)

        max_attempts = 10  # Maximum navigation attempts to prevent infinite loops

        for attempt in range(max_attempts):
            current_screen = self.get_current_screen_type()
            logger.debug(f"Attempt {attempt + 1}: Current screen = {current_screen.value}")

            if current_screen == ScreenType.TITLE_SCREEN:
                # Press START to advance from Pokemon logo screen
                new_screen = press_button_smart(
                    self.pyboy, 'START', current_screen, timeout_seconds=3
                )
                if new_screen == current_screen:
                    # Try A button if START doesn't work
                    new_screen = press_button_smart(
                        self.pyboy, 'A', current_screen, timeout_seconds=3
                    )

            elif current_screen == ScreenType.INTRO_ANIMATION:
                # Press A to skip Nidorino vs Gengar animation
                new_screen = press_button_smart(
                    self.pyboy, 'A', current_screen, timeout_seconds=3
                )
                if new_screen == current_screen:
                    # Try START if A doesn't work
                    new_screen = press_button_smart(
                        self.pyboy, 'START', current_screen, timeout_seconds=3
                    )

            elif current_screen == ScreenType.MAIN_MENU:
                # Press A to select NEW GAME
                new_screen = press_button_smart(
                    self.pyboy, 'A', current_screen, timeout_seconds=3
                )
                if new_screen == ScreenType.IN_GAME or new_screen != current_screen:
                    logger.info("Successfully started new game!")
                    return True

            elif current_screen == ScreenType.IN_GAME:
                logger.info("Already in game!")
                return True

            elif current_screen == ScreenType.UNKNOWN:
                # Try common buttons when screen type is unclear
                for button in ['START', 'A']:
                    press_button_smart(
                        self.pyboy, button, current_screen, timeout_seconds=1
                    )
                    self.wait_frames(60)  # 1 second between attempts

            # Brief pause between navigation attempts
            self.wait_frames(60)

        logger.warning("Failed to navigate intro sequence automatically")
        return False

    def skip_professor_oak_intro(self) -> None:
        """Skip through Professor Oak's introductory dialogue."""
        logger.info("Skipping Professor Oak introduction...")
        self.wait_frames(120)  # Wait 2 seconds for dialogue to start

        # Press A button 30 times to advance through all dialogue
        spam_button(self.pyboy, 'A', count=30, hold_frames=5, release_frames=20)

    def handle_naming_screen(self, name: str = "RED", is_player: bool = True) -> bool:
        """
        Handle Pokemon Red naming screen by accepting default names.

        Args:
            name: Name to use (currently just accepts default)
            is_player: True for player naming, False for rival naming

        Returns:
            True if successful
        """
        target = "player" if is_player else "rival"
        logger.info(f"Handling {target} naming screen...")

        return accept_default_name(self.pyboy, max_attempts=10)

    def complete_intro_dialogue(self) -> None:
        """Complete remaining intro dialogue after naming sequence."""
        logger.info("Completing intro dialogue...")

        # Continue pressing A to advance through dialogue (20 presses)
        spam_button(self.pyboy, 'A', count=20, hold_frames=5, release_frames=25)

        # Wait for world/lab to load (2 seconds)
        self.wait_frames(120)

    def take_initial_steps(self) -> None:
        """Take a few steps to verify movement system is working."""
        logger.info("Testing movement controls...")

        # Move down from starting position (2 steps)
        for _ in range(2):
            self.press_button('DOWN', hold_frames=15, release_frames=20)

        # Move left then right to test horizontal movement
        self.press_button('LEFT', hold_frames=15, release_frames=20)
        for _ in range(2):
            self.press_button('RIGHT', hold_frames=15, release_frames=20)

    def run_opening_sequence(self, player_name: str = "RED", rival_name: str = "BLUE") -> bool:
        """
        Complete the entire Pokemon Red opening sequence automatically.

        Args:
            player_name: Player name (currently accepts default)
            rival_name: Rival name (currently accepts default)

        Returns:
            True if successful, False if failed
        """
        try:
            logger.info("Starting Pokemon Red opening sequence...")

            # Step 1: Navigate through intro screens
            if not self.skip_intro_sequence():
                logger.warning("Failed to navigate intro screens automatically")

                # Wait for manual intervention (1 minute timeout)
                for i in range(60):
                    current_screen = self.get_current_screen_type()
                    if current_screen == ScreenType.IN_GAME:
                        break
                    if i % 10 == 0:
                        logger.info(f"Waiting for manual intervention... {i}/60 seconds")
                    self.wait_frames(60)
                else:
                    logger.error("Manual intervention timeout")
                    return False

            # Step 2: Skip Professor Oak's intro dialogue
            self.skip_professor_oak_intro()

            # Step 3: Handle naming screens
            logger.info("Handling player naming (accepting default)...")
            self.handle_naming_screen(player_name, is_player=True)

            logger.info("Handling rival naming (accepting default)...")
            self.handle_naming_screen(rival_name, is_player=False)

            # Step 4: Complete remaining intro dialogue
            self.complete_intro_dialogue()

            # Step 5: Verify game control is working
            self.take_initial_steps()

            # Display final status
            position = self.get_player_position()
            stats = self.get_player_stats()
            logger.info("Opening sequence completed successfully!")
            logger.info(f"Player position: {position}")
            logger.info(f"Player stats: {stats}")

            self.is_initialized = True
            return True

        except Exception as e:
            logger.error(f"Error during opening sequence: {e}")
            return False

    def reset_game(self) -> bool:
        """
        Reset the game to starting state for RL training.

        Returns:
            True if successful, False if failed
        """
        logger.info("Resetting game...")

        try:
            # Clean up current PyBoy instance
            self.pyboy.stop()
        except Exception as e:
            logger.warning(f"Warning during PyBoy cleanup: {e}")

        # Create new PyBoy instance
        try:
            window_type = "SDL2" if self.show_window else "null"
            try:
                # Try modern PyBoy syntax
                self.pyboy = PyBoy(
                    self.rom_path,
                    window=window_type
                )
            except TypeError:
                # Fallback for older versions
                self.pyboy = PyBoy(self.rom_path, window_type=window_type)

            # Set speed if available
            try:
                self.pyboy.set_emulation_speed(self.speed_multiplier)
            except AttributeError:
                pass

            self.memory = self.pyboy.memory
            self.screen = self.pyboy.screen
        except Exception as e:
            logger.error(f"Error recreating PyBoy instance: {e}")
            return False

        # Reset RL tracking
        self.visited_locations.clear()
        self.previous_stats.clear()
        self.episode_steps = 0
        self.is_initialized = False

        # Use the proven working opening sequence
        success = self.run_opening_sequence()
        if success:
            logger.info("Game reset completed successfully")
        else:
            logger.error("Game reset failed")

        return success

    def save_state(self, slot: int = 0) -> bool:
        """
        Save current game state.

        Args:
            slot: Save slot number (0-9)

        Returns:
            True if successful
        """
        if not self.enable_save_states:
            logger.warning("Save states not enabled")
            return False

        try:
            with open(f"state_{slot}.save", "wb") as f:
                self.pyboy.save_state(f)
            logger.info(f"Game state saved to slot {slot}")
            return True
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False

    def load_state(self, slot: int = 0) -> bool:
        """
        Load game state from save slot.

        Args:
            slot: Save slot number (0-9)

        Returns:
            True if successful
        """
        if not self.enable_save_states:
            logger.warning("Save states not enabled")
            return False

        try:
            with open(f"state_{slot}.save", "rb") as f:
                self.pyboy.load_state(f)
            logger.info(f"Game state loaded from slot {slot}")
            return True
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False

    def step(self, action: str) -> Dict[str, Any]:
        """
        Execute one game step with the given action.

        Args:
            action: Button to press ('A', 'B', 'UP', 'DOWN', etc.)

        Returns:
            Dictionary with game state after action
        """
        # Execute action
        self.press_button(action)

        # Update tracking
        self.episode_steps += 1

        # Get current state
        state = self.get_comprehensive_state()

        # Update visited locations for exploration tracking
        position = state['position']
        location_key = (position['x'], position['y'], position['map'])
        self.visited_locations.add(location_key)

        return state

    def get_exploration_progress(self) -> Dict[str, int]:
        """Get exploration progress metrics."""
        unique_maps = set()
        for x, y, map_id in self.visited_locations:
            if map_id != 0:  # Exclude non-game states
                unique_maps.add(map_id)

        return {
            'locations_visited': len(self.visited_locations),
            'unique_maps': len(unique_maps),
            'episode_steps': self.episode_steps
        }

    def cleanup(self) -> None:
        """Clean up PyBoy resources."""
        try:
            logger.info("Stopping PyBoy...")
            self.pyboy.stop()
            logger.info("PyBoy stopped successfully")
        except Exception as e:
            logger.error(f"Error during PyBoy cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def __del__(self):
        """Destructor."""
        try:
            self.cleanup()
        except:
            pass  # Ignore cleanup errors during destruction