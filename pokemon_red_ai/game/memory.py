"""
Pokemon Red memory address definitions and utilities.

This module contains well-documented memory addresses for Pokemon Red ROM hacking
and utilities for safely reading game state from memory.
"""

from typing import Dict, Union
import logging

logger = logging.getLogger(__name__)

# Pokemon Red memory addresses (well-documented from ROM hacking community)
MEMORY_ADDRESSES = {
    # Player position and map
    'player_x': 0xD362,  # Player X coordinate (0-255)
    'player_y': 0xD361,  # Player Y coordinate (0-255)
    'map_id': 0xD35E,  # Current map ID (0 = not in game, >0 = in game)

    # Player stats
    'player_level': 0xD18C,  # Current Pokemon level
    'current_hp_low': 0xD16C,  # Current HP low byte
    'current_hp_high': 0xD16D,  # Current HP high byte
    'max_hp_low': 0xD16E,  # Max HP low byte
    'max_hp_high': 0xD16F,  # Max HP high byte
    'badges': 0xD356,  # Badge bitfield (each bit = gym badge)

    # Game state indicators
    'game_state': 0xFF80,  # Overall game state flag
    'menu_state': 0xCC26,  # Current menu type (0 = no menu, >0 = in menu)

    # Pokemon party
    'party_count': 0xD163,  # Number of Pokemon in party (0-6)
    'party_species_1': 0xD164,  # First Pokemon species ID

    # Additional useful addresses
    'money_low': 0xD347,  # Money low byte
    'money_mid': 0xD348,  # Money middle byte
    'money_high': 0xD349,  # Money high byte
    'items_count': 0xD31D,  # Number of items in bag
    'pc_items_count': 0xD53A,  # Number of items in PC
}

# Map ID constants for easier reference
MAP_IDS = {
    'pallet_town': 1,
    'viridian_city': 2,
    'pewter_city': 3,
    'cerulean_city': 4,
    'lavender_town': 5,
    'vermilion_city': 6,
    'celadon_city': 7,
    'fuchsia_city': 8,
    'cinnabar_island': 9,
    'indigo_plateau': 10,
    'saffron_city': 11,
    'route_1': 12,
    'route_2': 13,
    'route_3': 14,
    'route_4': 15,
    'route_5': 16,
    'route_6': 17,
    'route_7': 18,
    'route_8': 19,
    'route_9': 20,
    'route_10': 21,
    'route_11': 22,
    'route_12': 23,
    'route_13': 24,
    'route_14': 25,
    'route_15': 26,
    'route_16': 27,
    'route_17': 28,
    'route_18': 29,
    'route_19': 30,
    'route_20': 31,
    'route_21': 32,
    'route_22': 33,
    'route_23': 34,
    'route_24': 35,
    'route_25': 36,
}

# Badge bit flags
BADGE_FLAGS = {
    'boulder': 0x01,  # Brock (Pewter City)
    'cascade': 0x02,  # Misty (Cerulean City)
    'thunder': 0x04,  # Lt. Surge (Vermilion City)
    'rainbow': 0x08,  # Erika (Celadon City)
    'soul': 0x10,  # Koga (Fuchsia City)
    'marsh': 0x20,  # Sabrina (Saffron City)
    'volcano': 0x40,  # Blaine (Cinnabar Island)
    'earth': 0x80,  # Giovanni (Viridian City)
}


def read_memory_value(memory, address: int, is_16bit: bool = False) -> int:
    """
    Safely read a value from Game Boy memory.

    Args:
        memory: PyBoy memory object
        address: Memory address to read from
        is_16bit: If True, read 16-bit value (little-endian)

    Returns:
        Memory value at address, or 0 if read fails
    """
    try:
        if is_16bit:
            # Game Boy uses little-endian: low byte first, then high byte
            low = memory[address]
            high = memory[address + 1]
            return low | (high << 8)
        else:
            return memory[address]
    except Exception as e:
        logger.warning(f"Failed to read memory at 0x{address:04X}: {e}")
        return 0


def read_player_position(memory) -> Dict[str, int]:
    """
    Read current player position and map information.

    Args:
        memory: PyBoy memory object

    Returns:
        Dictionary with 'x', 'y', and 'map' keys
    """
    return {
        'x': read_memory_value(memory, MEMORY_ADDRESSES['player_x']),
        'y': read_memory_value(memory, MEMORY_ADDRESSES['player_y']),
        'map': read_memory_value(memory, MEMORY_ADDRESSES['map_id'])
    }


def read_player_stats(memory) -> Dict[str, int]:
    """
    Read current player/Pokemon statistics.

    Args:
        memory: PyBoy memory object

    Returns:
        Dictionary with player stats
    """
    current_hp = read_memory_value(memory, MEMORY_ADDRESSES['current_hp_low'], is_16bit=True)
    max_hp = read_memory_value(memory, MEMORY_ADDRESSES['max_hp_low'], is_16bit=True)

    return {
        'level': read_memory_value(memory, MEMORY_ADDRESSES['player_level']),
        'current_hp': current_hp,
        'max_hp': max_hp,
        'hp_ratio': current_hp / max(max_hp, 1),  # Avoid division by zero
        'badges': read_memory_value(memory, MEMORY_ADDRESSES['badges']),
        'party_count': read_memory_value(memory, MEMORY_ADDRESSES['party_count'])
    }


def read_game_state(memory) -> Dict[str, int]:
    """
    Read game state indicators.

    Args:
        memory: PyBoy memory object

    Returns:
        Dictionary with game state information
    """
    return {
        'game_state': read_memory_value(memory, MEMORY_ADDRESSES['game_state']),
        'menu_state': read_memory_value(memory, MEMORY_ADDRESSES['menu_state']),
        'map_id': read_memory_value(memory, MEMORY_ADDRESSES['map_id']),
    }


def read_money(memory) -> int:
    """
    Read player's current money (24-bit value).

    Args:
        memory: PyBoy memory object

    Returns:
        Current money amount
    """
    low = read_memory_value(memory, MEMORY_ADDRESSES['money_low'])
    mid = read_memory_value(memory, MEMORY_ADDRESSES['money_mid'])
    high = read_memory_value(memory, MEMORY_ADDRESSES['money_high'])

    return low | (mid << 8) | (high << 16)


def get_badge_count(badges_value: int) -> int:
    """
    Count number of badges earned from badge bitfield.

    Args:
        badges_value: Raw badge bitfield value

    Returns:
        Number of badges earned (0-8)
    """
    return bin(badges_value).count('1')


def has_badge(badges_value: int, badge_name: str) -> bool:
    """
    Check if player has a specific badge.

    Args:
        badges_value: Raw badge bitfield value
        badge_name: Badge name (e.g., 'boulder', 'cascade')

    Returns:
        True if player has the badge
    """
    if badge_name not in BADGE_FLAGS:
        logger.warning(f"Unknown badge name: {badge_name}")
        return False

    return bool(badges_value & BADGE_FLAGS[badge_name])


def get_map_name(map_id: int) -> str:
    """
    Get human-readable map name from map ID.

    Args:
        map_id: Map ID from memory

    Returns:
        Map name string, or 'unknown' if not found
    """
    for name, id_val in MAP_IDS.items():
        if id_val == map_id:
            return name
    return f"unknown_map_{map_id}"


def is_in_game(memory) -> bool:
    """
    Check if player is currently in the game world (not in menus/intro).

    Args:
        memory: PyBoy memory object

    Returns:
        True if player is in the game world
    """
    map_id = read_memory_value(memory, MEMORY_ADDRESSES['map_id'])
    return map_id != 0


def get_comprehensive_state(memory) -> Dict[str, Union[int, float, str, Dict]]:
    """
    Get comprehensive game state information.

    Args:
        memory: PyBoy memory object

    Returns:
        Complete game state dictionary
    """
    position = read_player_position(memory)
    stats = read_player_stats(memory)
    game_state = read_game_state(memory)
    money = read_money(memory)

    return {
        'position': position,
        'stats': stats,
        'game_state': game_state,
        'money': money,
        'map_name': get_map_name(position['map']),
        'badge_count': get_badge_count(stats['badges']),
        'in_game': is_in_game(memory),
        'is_alive': stats['current_hp'] > 0,
    }