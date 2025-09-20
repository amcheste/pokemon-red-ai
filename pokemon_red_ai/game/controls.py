"""
Pokemon Red input controls and screen detection utilities.

This module handles Game Boy button inputs and screen state detection
with compatibility across different PyBoy versions.
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple, List
from enum import Enum

logger = logging.getLogger(__name__)


class ScreenType(Enum):
    """Enumeration of Pokemon Red screen types."""
    TITLE_SCREEN = "title_screen"
    INTRO_ANIMATION = "intro_animation"
    MAIN_MENU = "main_menu"
    IN_GAME = "in_game"
    UNKNOWN = "unknown"


class GameBoyButton(Enum):
    """Game Boy button mappings."""
    A = "a"
    B = "b"
    SELECT = "select"
    START = "start"
    RIGHT = "right"
    LEFT = "left"
    UP = "up"
    DOWN = "down"


# Button mappings for different PyBoy versions
BUTTON_MAPPINGS = {
    'string': {  # PyBoy 2.x string identifiers
        'A': 'a',
        'B': 'b',
        'SELECT': 'select',
        'START': 'start',
        'RIGHT': 'right',
        'LEFT': 'left',
        'UP': 'up',
        'DOWN': 'down'
    },
    'numeric': {  # PyBoy 1.x numeric identifiers
        'A': 0,
        'B': 1,
        'SELECT': 2,
        'START': 3,
        'RIGHT': 4,
        'LEFT': 5,
        'UP': 6,
        'DOWN': 7
    }
}


def wait_frames(pyboy, frames: int) -> None:
    """
    Wait for a specified number of frames.

    Args:
        pyboy: PyBoy instance
        frames: Number of frames to wait (60 frames = 1 second)
    """
    for _ in range(frames):
        pyboy.tick()


def press_button_basic(pyboy, button: str, hold_frames: int = 10, release_frames: int = 5) -> None:
    """
    Press and release a Game Boy button with basic timing.

    Args:
        pyboy: PyBoy instance
        button: Button name ('A', 'B', 'START', etc.)
        hold_frames: How long to hold button (default: 10 frames = ~167ms at 60fps)
        release_frames: How long to wait after release (default: 5 frames = ~83ms)
    """
    try:
        # Try PyBoy 2.x button_press method first
        button_id = BUTTON_MAPPINGS['string'][button]
        pyboy.button_press(button_id)

        # Hold button for specified duration
        wait_frames(pyboy, hold_frames)

        # Release button
        pyboy.button_release(button_id)

        # Wait after release to prevent double-inputs
        wait_frames(pyboy, release_frames)

    except (AttributeError, KeyError):
        # Fallback to older send_input method if button_press unavailable
        try:
            button_id = BUTTON_MAPPINGS['numeric'][button]
            pyboy.send_input(button_id)
            wait_frames(pyboy, hold_frames)
            pyboy.send_input(button_id, False)
            wait_frames(pyboy, release_frames)
        except (AttributeError, KeyError) as e:
            logger.error(f"Failed to press button {button}: {e}")


def get_screen_array(pyboy) -> np.ndarray:
    """
    Get current screen as numpy array.

    Args:
        pyboy: PyBoy instance

    Returns:
        Screen image as numpy array
    """
    try:
        screen_image = pyboy.screen.image
        return np.array(screen_image)
    except Exception as e:
        logger.error(f"Failed to get screen array: {e}")
        return np.zeros((144, 160, 3), dtype=np.uint8)


def get_tilemap_background(pyboy) -> np.ndarray:
    """
    Get background tilemap for efficient game state analysis.

    Args:
        pyboy: PyBoy instance

    Returns:
        Background tilemap as numpy array
    """
    try:
        return np.array(pyboy.tilemap_background)
    except Exception as e:
        logger.error(f"Failed to get tilemap: {e}")
        return np.zeros((18, 20), dtype=np.uint8)


def detect_screen_type(pyboy) -> ScreenType:
    """
    Analyze current screen to determine game state.

    Args:
        pyboy: PyBoy instance

    Returns:
        Detected screen type
    """
    try:
        from .memory import read_memory_value, MEMORY_ADDRESSES

        # Read key memory indicators
        map_id = read_memory_value(pyboy.memory, MEMORY_ADDRESSES['map_id'])
        game_state = read_memory_value(pyboy.memory, MEMORY_ADDRESSES['game_state'])
        menu_state = read_memory_value(pyboy.memory, MEMORY_ADDRESSES['menu_state'])

        # If map_id != 0, we're in the actual game world
        if map_id != 0:
            return ScreenType.IN_GAME

        # Analyze screen content using tilemap
        try:
            tilemap = get_tilemap_background(pyboy)

            # Ensure tilemap has expected dimensions before analysis
            if tilemap.shape[0] < 13 or tilemap.shape[1] < 15:
                logger.debug(f"Tilemap too small for analysis: {tilemap.shape}")
                return ScreenType.UNKNOWN

            # Check for main menu pattern (NEW GAME/OPTION text area)
            if tilemap.shape[0] > 12 and tilemap.shape[1] > 14:
                menu_area = tilemap[8:13, 6:15]
                if np.any(menu_area > 0):
                    return ScreenType.MAIN_MENU

            # Check sprite density to detect intro animation
            sprite_density = np.count_nonzero(tilemap)
            if sprite_density > 50:
                return ScreenType.INTRO_ANIMATION

            # Check for title screen (Pokemon logo in top area)
            if tilemap.shape[0] > 8:
                top_area = tilemap[:8, :]
                if np.any(top_area > 0):
                    return ScreenType.TITLE_SCREEN

        except (IndexError, ValueError) as e:
            logger.debug(f"Tilemap analysis failed: {e}")
            # Fall through to memory-based detection

        # Fallback to memory-based detection
        if game_state == 0 and menu_state == 0:
            return ScreenType.TITLE_SCREEN
        elif menu_state in [1, 2, 3]:
            return ScreenType.MAIN_MENU

        return ScreenType.UNKNOWN

    except Exception as e:
        logger.error(f"Screen detection failed: {e}")
        return ScreenType.UNKNOWN


def wait_for_screen_change(pyboy, current_screen: ScreenType, timeout_seconds: int = 10) -> ScreenType:
    """
    Wait for the screen to change from current state.

    Args:
        pyboy: PyBoy instance
        current_screen: Current screen type to wait to change from
        timeout_seconds: Maximum time to wait

    Returns:
        New screen type, or current screen if timeout
    """
    timeout_frames = timeout_seconds * 60

    for frame in range(timeout_frames):
        new_screen = detect_screen_type(pyboy)
        if new_screen != current_screen and new_screen != ScreenType.UNKNOWN:
            logger.debug(f"Screen changed from {current_screen.value} to {new_screen.value}")
            return new_screen

        wait_frames(pyboy, 1)

        # Log progress every second
        if frame % 60 == 0 and frame > 0:
            seconds_elapsed = frame // 60
            logger.debug(f"Waiting for screen change... {seconds_elapsed}/{timeout_seconds}s")

    logger.warning(f"Screen change timeout after {timeout_seconds}s")
    return detect_screen_type(pyboy)


def press_button_smart(pyboy, button: str, current_screen: ScreenType,
                      timeout_seconds: int = 3) -> ScreenType:
    """
    Press button and wait for screen to change, trying multiple methods.

    Args:
        pyboy: PyBoy instance
        button: Button to press
        current_screen: Current screen type
        timeout_seconds: Timeout for screen change

    Returns:
        New screen type after button press
    """
    # Try different button press methods for PyBoy compatibility
    methods = [
        ("button_press/release", _method_button_press),
        ("send_input strings", _method_send_input_string),
        ("send_input numbers", _method_send_input_number)
    ]

    for method_name, method_func in methods:
        try:
            logger.debug(f"Trying {method_name} for button {button}")
            method_func(pyboy, button)

            # Wait for screen change
            new_screen = wait_for_screen_change(pyboy, current_screen, timeout_seconds)

            if new_screen != current_screen:
                logger.debug(f"Screen changed using {method_name}")
                return new_screen

        except Exception as e:
            logger.warning(f"Method {method_name} failed: {e}")
            continue

    logger.warning(f"All button press methods failed for {button}")
    return current_screen


def _method_button_press(pyboy, button: str) -> None:
    """Button press using PyBoy 2.x button_press/button_release."""
    button_id = BUTTON_MAPPINGS['string'][button]
    pyboy.button_press(button_id)
    wait_frames(pyboy, 15)  # Hold for 15 frames (~250ms)
    pyboy.button_release(button_id)


def _method_send_input_string(pyboy, button: str) -> None:
    """Button press using send_input with string identifiers."""
    button_id = BUTTON_MAPPINGS['string'][button]
    pyboy.send_input(button_id)
    wait_frames(pyboy, 15)
    pyboy.send_input(button_id, False)


def _method_send_input_number(pyboy, button: str) -> None:
    """Button press using send_input with numeric IDs (PyBoy 1.x style)."""
    button_id = BUTTON_MAPPINGS['numeric'][button]
    pyboy.send_input(button_id)
    wait_frames(pyboy, 15)
    pyboy.send_input(button_id, False)


def navigate_menu_cursor(pyboy, target_row: int, target_col: int,
                        max_rows: int = 10, max_cols: int = 10) -> None:
    """
    Navigate menu cursor to specific position.

    Args:
        pyboy: PyBoy instance
        target_row: Target row position
        target_col: Target column position
        max_rows: Maximum rows in menu
        max_cols: Maximum columns in menu
    """
    # Reset cursor to top-left corner
    for _ in range(max_rows):
        press_button_basic(pyboy, 'UP', hold_frames=8, release_frames=8)
    for _ in range(max_cols):
        press_button_basic(pyboy, 'LEFT', hold_frames=8, release_frames=8)

    # Move to target position
    for _ in range(target_row):
        press_button_basic(pyboy, 'DOWN', hold_frames=8, release_frames=8)
    for _ in range(target_col):
        press_button_basic(pyboy, 'RIGHT', hold_frames=8, release_frames=8)


def press_button_sequence(pyboy, buttons: List[str],
                         hold_frames: int = 10, between_frames: int = 20) -> None:
    """
    Press a sequence of buttons with timing.

    Args:
        pyboy: PyBoy instance
        buttons: List of button names to press
        hold_frames: Frames to hold each button
        between_frames: Frames to wait between button presses
    """
    for i, button in enumerate(buttons):
        press_button_basic(pyboy, button, hold_frames, release_frames=5)

        # Wait between buttons (except after last button)
        if i < len(buttons) - 1:
            wait_frames(pyboy, between_frames)


def spam_button(pyboy, button: str, count: int,
               hold_frames: int = 5, release_frames: int = 20) -> None:
    """
    Press a button multiple times quickly (useful for advancing dialogue).

    Args:
        pyboy: PyBoy instance
        button: Button to press
        count: Number of times to press
        hold_frames: Frames to hold button each press
        release_frames: Frames between presses
    """
    for _ in range(count):
        press_button_basic(pyboy, button, hold_frames, release_frames)


def accept_default_name(pyboy, max_attempts: int = 10) -> bool:
    """
    Accept default name on naming screen using multiple strategies.

    Args:
        pyboy: PyBoy instance
        max_attempts: Maximum attempts to accept name

    Returns:
        True if successful, False otherwise
    """
    # Wait for naming screen to load
    wait_frames(pyboy, 120)

    # Strategy 1: Press START to accept default name (most reliable)
    for attempt in range(max_attempts):
        press_button_basic(pyboy, 'START', hold_frames=20, release_frames=60)
        wait_frames(pyboy, 60)

    # Strategy 2: Navigate to END button manually
    navigate_menu_cursor(pyboy, 5, 9)  # Typical END button position

    # Try pressing A on END button
    for attempt in range(3):
        press_button_basic(pyboy, 'A', hold_frames=20, release_frames=60)
        wait_frames(pyboy, 60)

    # Strategy 3: Try alternative END button positions
    alternative_positions = [(5, 9), (4, 9), (5, 8), (4, 8)]

    for row, col in alternative_positions:
        navigate_menu_cursor(pyboy, row, col)
        press_button_basic(pyboy, 'A', hold_frames=20, release_frames=60)
        wait_frames(pyboy, 30)

    # Extra wait to ensure screen transition completes
    wait_frames(pyboy, 180)
    return True


def get_input_state(pyboy) -> Dict[str, bool]:
    """
    Get current state of all buttons (useful for debugging).

    Args:
        pyboy: PyBoy instance

    Returns:
        Dictionary mapping button names to pressed state
    """
    try:
        # This is implementation-specific and may not work on all PyBoy versions
        input_state = {}
        for button_name in BUTTON_MAPPINGS['string'].keys():
            # This would need to be implemented based on PyBoy's internal state
            input_state[button_name] = False  # Placeholder
        return input_state
    except Exception as e:
        logger.warning(f"Could not get input state: {e}")
        return {button: False for button in BUTTON_MAPPINGS['string'].keys()}