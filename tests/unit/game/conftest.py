"""
Agent-specific test fixtures and configuration.

This conftest.py file contains fixtures specifically for testing the
Pokemon Red agent components (agent.py, controls.py, memory.py).
"""

import pytest
import numpy as np
from unittest.mock import Mock, PropertyMock
from pathlib import Path

from pokemon_red_ai.game.memory import MEMORY_ADDRESSES, MAP_IDS
from pokemon_red_ai.game.controls import ScreenType


@pytest.fixture
def mock_pyboy_for_agent():
    """Create a comprehensive PyBoy mock specifically for agent testing."""
    mock = Mock()

    # Memory mock with specific behaviors for agent tests
    mock.memory = Mock()
    mock.memory.__getitem__ = Mock(return_value=0)

    # Screen mock
    mock.screen = Mock()
    mock.screen.image = np.zeros((144, 160, 3), dtype=np.uint8)

    # Button control mocks
    mock.button_press = Mock()
    mock.button_release = Mock()
    mock.send_input = Mock()

    # Tilemap mock - return realistic tilemap
    tilemap = np.zeros((18, 20), dtype=np.uint8)
    type(mock).tilemap_background = PropertyMock(return_value=tilemap)

    # Emulator control mocks
    mock.tick = Mock()
    mock.stop = Mock()
    mock.set_emulation_speed = Mock()

    # Save state mocks
    mock.save_state = Mock()
    mock.load_state = Mock()

    return mock


@pytest.fixture
def agent_memory_state():
    """Provide realistic memory state for game testing."""
    return {
        MEMORY_ADDRESSES['player_x']: 5,
        MEMORY_ADDRESSES['player_y']: 5,
        MEMORY_ADDRESSES['map_id']: MAP_IDS['pallet_town'],
        MEMORY_ADDRESSES['player_level']: 5,
        MEMORY_ADDRESSES['current_hp_low']: 25,
        MEMORY_ADDRESSES['current_hp_high']: 0,
        MEMORY_ADDRESSES['max_hp_low']: 30,
        MEMORY_ADDRESSES['max_hp_high']: 0,
        MEMORY_ADDRESSES['badges']: 0,
        MEMORY_ADDRESSES['party_count']: 1,
        MEMORY_ADDRESSES['game_state']: 1,
        MEMORY_ADDRESSES['menu_state']: 0,
        MEMORY_ADDRESSES['money_low']: 0,
        MEMORY_ADDRESSES['money_mid']: 3,  # 300 money
        MEMORY_ADDRESSES['money_high']: 0,
    }


@pytest.fixture
def mock_agent_memory(agent_memory_state):
    """Create a mock memory with realistic agent state."""
    mock = Mock()
    mock.__getitem__ = Mock(side_effect=lambda addr: agent_memory_state.get(addr, 0))
    return mock


@pytest.fixture
def title_screen_tilemap():
    """Create a tilemap that represents the title screen."""
    tilemap = np.zeros((18, 20), dtype=np.uint8)
    # Add some content in the top area (Pokemon logo)
    tilemap[:8, 5:15] = 50  # Simulate Pokemon logo
    return tilemap


@pytest.fixture
def intro_animation_tilemap():
    """Create a tilemap that represents the intro animation."""
    tilemap = np.random.randint(50, 200, (18, 20), dtype=np.uint8)
    return tilemap


@pytest.fixture
def main_menu_tilemap():
    """Create a tilemap that represents the main menu."""
    tilemap = np.zeros((18, 20), dtype=np.uint8)
    # Add content in the menu area (NEW GAME/OPTION)
    tilemap[8:13, 6:15] = 100  # Simulate menu text
    return tilemap


@pytest.fixture
def screen_detection_scenarios():
    """Provide different screen scenarios for testing detection logic."""
    return {
        'title_screen': {
            'memory_state': {
                MEMORY_ADDRESSES['map_id']: 0,
                MEMORY_ADDRESSES['game_state']: 0,
                MEMORY_ADDRESSES['menu_state']: 0
            },
            'tilemap': np.array([
                [0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                # ... more rows with Pokemon logo simulation
            ] + [[0] * 20 for _ in range(16)], dtype=np.uint8),
            'expected': ScreenType.TITLE_SCREEN
        },
        'intro_animation': {
            'memory_state': {
                MEMORY_ADDRESSES['map_id']: 0,
                MEMORY_ADDRESSES['game_state']: 0,
                MEMORY_ADDRESSES['menu_state']: 0
            },
            'tilemap': np.random.randint(50, 200, (18, 20), dtype=np.uint8),
            'expected': ScreenType.INTRO_ANIMATION
        },
        'main_menu': {
            'memory_state': {
                MEMORY_ADDRESSES['map_id']: 0,
                MEMORY_ADDRESSES['game_state']: 0,
                MEMORY_ADDRESSES['menu_state']: 1
            },
            'tilemap': np.array([
                [0] * 20 for _ in range(8)
            ] + [
                [0, 0, 0, 0, 0, 0, 100, 100, 100, 100, 100, 100, 100, 100, 0, 0, 0, 0, 0, 0]
                for _ in range(5)
            ] + [
                [0] * 20 for _ in range(5)
            ], dtype=np.uint8),
            'expected': ScreenType.MAIN_MENU
        },
        'in_game': {
            'memory_state': {
                MEMORY_ADDRESSES['map_id']: 1,
                MEMORY_ADDRESSES['game_state']: 1,
                MEMORY_ADDRESSES['menu_state']: 0
            },
            'tilemap': np.random.randint(10, 80, (18, 20), dtype=np.uint8),
            'expected': ScreenType.IN_GAME
        }
    }


@pytest.fixture
def mock_rom_file(tmp_path):
    """Create a temporary ROM file for agent testing."""
    rom_file = tmp_path / "test_pokemon.gb"
    # Create a 1MB file (Pokemon Red ROM size)
    with open(rom_file, 'wb') as f:
        f.write(b'\x00' * 1048576)
    return rom_file


@pytest.fixture
def mock_successful_agent():
    """Create a mock agent that behaves successfully for testing."""
    mock = Mock()
    mock.rom_path = "test.gb"
    mock.show_window = False
    mock.speed_multiplier = 1
    mock.save_state = False

    mock.press_button = Mock()
    mock.wait_frames = Mock()
    mock.get_screen_array = Mock(return_value=np.zeros((144, 160, 3), dtype=np.uint8))
    mock.get_player_position = Mock(return_value={'x': 5, 'y': 5, 'map': 1})
    mock.get_player_stats = Mock(return_value={
        'level': 5,
        'current_hp': 25,
        'max_hp': 30,
        'hp_ratio': 25/30,
        'badges': 0,
        'party_count': 1
    })
    mock.get_comprehensive_state = Mock(return_value={
        'position': {'x': 5, 'y': 5, 'map': 1},
        'stats': {
            'level': 5,
            'current_hp': 25,
            'max_hp': 30,
            'hp_ratio': 25/30,
            'badges': 0,
            'party_count': 1
        },
        'game_state': {'game_state': 1, 'menu_state': 0, 'map_id': 1},
        'money': 300,
        'map_name': 'pallet_town',
        'badge_count': 0,
        'in_game': True,
        'is_alive': True
    })

    mock.step = Mock(return_value=mock.get_comprehensive_state.return_value)
    mock.reset_game = Mock(return_value=True)
    mock.cleanup = Mock()

    return mock