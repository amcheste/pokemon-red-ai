"""
Pokemon Red Game Interface Module

This module provides the core game interface components for interacting with
Pokemon Red through PyBoy emulation, including memory management, input controls,
and the main game class.
"""

# Import order matters to avoid circular dependencies
# First import utilities that don't depend on each other
from .memory import (
    MEMORY_ADDRESSES,
    MAP_IDS,
    BADGE_FLAGS,
    read_memory_value,
    read_player_position,
    read_player_stats,
    read_game_state,
    read_money,
    get_badge_count,
    has_badge,
    get_map_name,
    is_in_game,
    get_comprehensive_state
)

from .controls import (
    ScreenType,
    GameBoyButton,
    wait_frames,
    press_button_basic,
    get_screen_array,
    get_tilemap_background,
    detect_screen_type,
    wait_for_screen_change,
    press_button_smart,
    navigate_menu_cursor,
    press_button_sequence,
    spam_button,
    accept_default_name,
    get_input_state
)

# Then import the main game that depends on the utilities
from .agent import PokemonRedAgent

# Public API - what users should import
__all__ = [
    # Main game class
    "PokemonRedAgent",

    # Memory constants
    "MEMORY_ADDRESSES",
    "MAP_IDS",
    "BADGE_FLAGS",

    # Memory functions
    "read_memory_value",
    "read_player_position",
    "read_player_stats",
    "read_game_state",
    "read_money",
    "get_badge_count",
    "has_badge",
    "get_map_name",
    "is_in_game",
    "get_comprehensive_state",

    # Screen/control enums
    "ScreenType",
    "GameBoyButton",

    # Control functions
    "wait_frames",
    "press_button_basic",
    "get_screen_array",
    "get_tilemap_background",
    "detect_screen_type",
    "wait_for_screen_change",
    "press_button_smart",
    "navigate_menu_cursor",
    "press_button_sequence",
    "spam_button",
    "accept_default_name",
    "get_input_state",
]