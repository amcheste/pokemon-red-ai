"""
Tests for Pokemon Red memory reading utilities.

This module tests the memory management functions that read game state
from Pokemon Red's memory addresses.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from pokemon_red_ai.game.memory import (
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


class TestMemoryAddresses:
    """Test memory address constants and mappings."""

    def test_memory_addresses_exist(self):
        """Test that all required memory addresses are defined."""
        required_addresses = [
            'player_x', 'player_y', 'map_id', 'player_level',
            'current_hp_low', 'current_hp_high', 'max_hp_low', 'max_hp_high',
            'badges', 'game_state', 'menu_state', 'party_count'
        ]

        for addr in required_addresses:
            assert addr in MEMORY_ADDRESSES
            assert isinstance(MEMORY_ADDRESSES[addr], int)
            assert MEMORY_ADDRESSES[addr] >= 0

    def test_map_ids_exist(self):
        """Test that map ID constants are defined."""
        expected_maps = ['pallet_town', 'viridian_city', 'pewter_city', 'cerulean_city']

        for map_name in expected_maps:
            assert map_name in MAP_IDS
            assert isinstance(MAP_IDS[map_name], int)

    def test_badge_flags_exist(self):
        """Test that badge flag constants are defined."""
        expected_badges = ['boulder', 'cascade', 'thunder', 'rainbow', 'soul', 'marsh', 'volcano', 'earth']

        for badge in expected_badges:
            assert badge in BADGE_FLAGS
            assert isinstance(BADGE_FLAGS[badge], int)
            assert BADGE_FLAGS[badge] > 0


class TestReadMemoryValue:
    """Test basic memory reading functionality."""

    def test_read_memory_value_8bit(self):
        """Test reading 8-bit values from memory."""
        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(return_value=42)

        value = read_memory_value(mock_memory, 0xD000)

        assert value == 42
        mock_memory.__getitem__.assert_called_once_with(0xD000)

    def test_read_memory_value_16bit(self):
        """Test reading 16-bit values from memory (little-endian)."""
        mock_memory = Mock()
        # Low byte = 0x34 (52), High byte = 0x12 (18)
        # Result should be 0x1234 = 4660
        mock_memory.__getitem__ = Mock(side_effect=[0x34, 0x12])

        value = read_memory_value(mock_memory, 0xD000, is_16bit=True)

        assert value == 0x1234  # 4660
        assert mock_memory.__getitem__.call_count == 2

    def test_read_memory_value_error_handling(self):
        """Test error handling when memory read fails."""
        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(side_effect=IndexError("Memory error"))

        value = read_memory_value(mock_memory, 0xD000)

        assert value == 0  # Should return 0 on error

    def test_read_memory_value_16bit_error_handling(self):
        """Test error handling for 16-bit reads."""
        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(side_effect=KeyError("Memory error"))

        value = read_memory_value(mock_memory, 0xD000, is_16bit=True)

        assert value == 0


class TestReadPlayerPosition:
    """Test player position reading functions."""

    def test_read_player_position(self):
        """Test reading player position from memory."""
        mock_memory = Mock()

        def memory_side_effect(addr):
            memory_map = {
                MEMORY_ADDRESSES['player_x']: 15,
                MEMORY_ADDRESSES['player_y']: 25,
                MEMORY_ADDRESSES['map_id']: 3
            }
            return memory_map.get(addr, 0)

        mock_memory.__getitem__ = Mock(side_effect=memory_side_effect)

        position = read_player_position(mock_memory)

        assert position['x'] == 15
        assert position['y'] == 25
        assert position['map'] == 3

    def test_read_player_position_zero_values(self):
        """Test reading position when coordinates are zero."""
        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(return_value=0)

        position = read_player_position(mock_memory)

        assert position['x'] == 0
        assert position['y'] == 0
        assert position['map'] == 0


class TestReadPlayerData:
    """Test player/Pokemon statistics reading."""

    def test_read_player_stats(self):
        """Test reading player stats from memory."""
        mock_memory = Mock()

        # Create a side_effect function that properly handles the memory addresses
        def memory_side_effect(addr):
            memory_map = {
                MEMORY_ADDRESSES['player_level']: 15,
                MEMORY_ADDRESSES['current_hp_low']: 0x32,  # 50 low byte
                MEMORY_ADDRESSES['current_hp_high']: 0x00,  # 50 high byte
                MEMORY_ADDRESSES['max_hp_low']: 0x3C,      # 60 low byte
                MEMORY_ADDRESSES['max_hp_high']: 0x00,     # 60 high byte
                MEMORY_ADDRESSES['badges']: 3,
                MEMORY_ADDRESSES['party_count']: 2
            }
            return memory_map.get(addr, 0)

        mock_memory.__getitem__ = Mock(side_effect=memory_side_effect)

        stats = read_player_stats(mock_memory)

        assert stats['level'] == 15
        assert stats['current_hp'] == 50
        assert stats['max_hp'] == 60
        assert stats['badges'] == 3
        assert stats['party_count'] == 2
        assert abs(stats['hp_ratio'] - (50/60)) < 0.01  # Use approximate comparison

    def test_read_player_stats_zero_hp(self):
        """Test reading stats when HP is zero."""
        mock_memory = Mock()

        def memory_side_effect(addr):
            memory_map = {
                MEMORY_ADDRESSES['player_level']: 10,
                MEMORY_ADDRESSES['current_hp_low']: 0,     # 0 HP
                MEMORY_ADDRESSES['current_hp_high']: 0,
                MEMORY_ADDRESSES['max_hp_low']: 50,        # Max HP 50
                MEMORY_ADDRESSES['max_hp_high']: 0,
                MEMORY_ADDRESSES['badges']: 0,
                MEMORY_ADDRESSES['party_count']: 1
            }
            return memory_map.get(addr, 0)

        mock_memory.__getitem__ = Mock(side_effect=memory_side_effect)

        stats = read_player_stats(mock_memory)

        assert stats['level'] == 10
        assert stats['current_hp'] == 0
        assert stats['max_hp'] == 50
        assert stats['badges'] == 0
        assert stats['party_count'] == 1
        assert stats['hp_ratio'] == 0.0

    def test_read_player_stats_full_hp(self):
        """Test reading stats when at full HP."""
        mock_memory = Mock()

        def memory_side_effect(addr):
            memory_map = {
                MEMORY_ADDRESSES['player_level']: 20,
                MEMORY_ADDRESSES['current_hp_low']: 100,   # 100 HP
                MEMORY_ADDRESSES['current_hp_high']: 0,
                MEMORY_ADDRESSES['max_hp_low']: 100,       # Max HP 100
                MEMORY_ADDRESSES['max_hp_high']: 0,
                MEMORY_ADDRESSES['badges']: 5,
                MEMORY_ADDRESSES['party_count']: 6
            }
            return memory_map.get(addr, 0)

        mock_memory.__getitem__ = Mock(side_effect=memory_side_effect)

        stats = read_player_stats(mock_memory)

        assert stats['level'] == 20
        assert stats['current_hp'] == 100
        assert stats['max_hp'] == 100
        assert stats['badges'] == 5
        assert stats['party_count'] == 6
        # Use approximate comparison for floating point precision
        assert abs(stats['hp_ratio'] - 1.0) < 0.01

    def test_read_player_stats_max_hp_zero(self):
        """Test reading stats when max HP is zero (edge case)."""
        mock_memory = Mock()

        def memory_side_effect(addr):
            memory_map = {
                MEMORY_ADDRESSES['player_level']: 1,
                MEMORY_ADDRESSES['current_hp_low']: 0,
                MEMORY_ADDRESSES['current_hp_high']: 0,
                MEMORY_ADDRESSES['max_hp_low']: 0,  # Max HP is 0
                MEMORY_ADDRESSES['max_hp_high']: 0,
                MEMORY_ADDRESSES['badges']: 0,
                MEMORY_ADDRESSES['party_count']: 0
            }
            return memory_map.get(addr, 0)

        mock_memory.__getitem__ = Mock(side_effect=memory_side_effect)

        stats = read_player_stats(mock_memory)

        # Should handle division by zero gracefully
        assert stats['max_hp'] == 0
        assert stats['current_hp'] == 0
        # hp_ratio calculation should use max(max_hp, 1) to avoid division by zero
        assert stats['hp_ratio'] == 0.0


class TestGameState:
    """Test game state reading functions."""

    def test_read_game_state(self):
        """Test reading game state indicators."""
        mock_memory = Mock()

        def memory_side_effect(addr):
            memory_map = {
                MEMORY_ADDRESSES['game_state']: 1,
                MEMORY_ADDRESSES['menu_state']: 0,
                MEMORY_ADDRESSES['map_id']: 5
            }
            return memory_map.get(addr, 0)

        mock_memory.__getitem__ = Mock(side_effect=memory_side_effect)

        state = read_game_state(mock_memory)

        assert state['game_state'] == 1
        assert state['menu_state'] == 0
        assert state['map_id'] == 5

    def test_is_in_game_true(self):
        """Test is_in_game when player is in game world."""
        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(return_value=3)  # Non-zero map_id

        result = is_in_game(mock_memory)

        assert result is True

    def test_is_in_game_false(self):
        """Test is_in_game when player is not in game world."""
        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(return_value=0)  # Zero map_id

        result = is_in_game(mock_memory)

        assert result is False


class TestMoneyReading:
    """Test money reading functionality."""

    def test_read_money(self):
        """Test reading player money (24-bit value)."""
        mock_memory = Mock()

        def memory_side_effect(addr):
            memory_map = {
                MEMORY_ADDRESSES['money_low']: 0x50,    # Low byte (80)
                MEMORY_ADDRESSES['money_mid']: 0xC3,    # Mid byte (195)
                MEMORY_ADDRESSES['money_high']: 0x00    # High byte (0)
            }
            return memory_map.get(addr, 0)

        mock_memory.__getitem__ = Mock(side_effect=memory_side_effect)

        money = read_money(mock_memory)

        # Should be 0x00C350 = 50000
        expected = 0x50 | (0xC3 << 8) | (0x00 << 16)  # 80 + 49920 + 0 = 50000
        assert money == expected

    def test_read_money_max_value(self):
        """Test reading maximum money value."""
        mock_memory = Mock()

        def memory_side_effect(addr):
            memory_map = {
                MEMORY_ADDRESSES['money_low']: 0xFF,    # 255
                MEMORY_ADDRESSES['money_mid']: 0xFF,    # 255
                MEMORY_ADDRESSES['money_high']: 0xFF    # 255
            }
            return memory_map.get(addr, 0)

        mock_memory.__getitem__ = Mock(side_effect=memory_side_effect)

        money = read_money(mock_memory)

        # Should be 0xFFFFFF = 16777215
        assert money == 0xFFFFFF

    def test_read_money_zero(self):
        """Test reading zero money."""
        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(return_value=0)

        money = read_money(mock_memory)

        assert money == 0


class TestBadgeOperations:
    """Test badge-related utility functions."""

    def test_get_badge_count_zero(self):
        """Test badge count when no badges."""
        badge_count = get_badge_count(0)
        assert badge_count == 0

    def test_get_badge_count_some_badges(self):
        """Test badge count with some badges."""
        # Binary: 00001011 = 3 badges (bits 0, 1, 3 set)
        badge_value = 0b00001011  # 11 in decimal
        badge_count = get_badge_count(badge_value)
        assert badge_count == 3

    def test_get_badge_count_all_badges(self):
        """Test badge count with all 8 badges."""
        # Binary: 11111111 = 8 badges
        badge_value = 0b11111111  # 255 in decimal
        badge_count = get_badge_count(badge_value)
        assert badge_count == 8

    def test_has_badge_true(self):
        """Test has_badge when player has the badge."""
        # Binary: 00000001 = boulder badge (first bit set)
        badge_value = BADGE_FLAGS['boulder']
        result = has_badge(badge_value, 'boulder')
        assert result is True

    def test_has_badge_false(self):
        """Test has_badge when player doesn't have the badge."""
        badge_value = 0  # No badges
        result = has_badge(badge_value, 'boulder')
        assert result is False

    def test_has_badge_multiple(self):
        """Test has_badge with multiple badges."""
        # Boulder (0x01) + Cascade (0x02) = 0x03
        badge_value = BADGE_FLAGS['boulder'] | BADGE_FLAGS['cascade']

        assert has_badge(badge_value, 'boulder') is True
        assert has_badge(badge_value, 'cascade') is True
        assert has_badge(badge_value, 'thunder') is False

    def test_has_badge_invalid_name(self):
        """Test has_badge with invalid badge name."""
        result = has_badge(255, 'invalid_badge')
        assert result is False


class TestMapUtilities:
    """Test map-related utility functions."""

    def test_get_map_name_known_map(self):
        """Test getting name for known map."""
        map_name = get_map_name(MAP_IDS['pallet_town'])
        assert map_name == 'pallet_town'

    def test_get_map_name_unknown_map(self):
        """Test getting name for unknown map."""
        map_name = get_map_name(999)
        assert map_name == 'unknown_map_999'

    def test_get_map_name_zero(self):
        """Test getting name for map ID 0."""
        map_name = get_map_name(0)
        assert map_name == 'unknown_map_0'


class TestComprehensiveState:
    """Test comprehensive state reading function."""

    def test_get_comprehensive_state(self, mock_memory_from_state):
        """Test getting comprehensive game state."""
        mock_memory = mock_memory_from_state()

        state = get_comprehensive_state(mock_memory)

        # Verify structure
        assert 'position' in state
        assert 'stats' in state
        assert 'game_state' in state
        assert 'money' in state
        assert 'map_name' in state
        assert 'badge_count' in state
        assert 'in_game' in state
        assert 'is_alive' in state

        # Verify position data
        assert state['position']['x'] == 10
        assert state['position']['y'] == 10
        assert state['position']['map'] == 1

        # Verify stats
        assert state['stats']['level'] == 5
        assert state['stats']['current_hp'] == 20
        assert state['stats']['max_hp'] == 25

        # Verify derived values
        assert state['badge_count'] == 0  # No badges set
        assert state['in_game'] is True   # Map ID != 0
        assert state['is_alive'] is True  # HP > 0

    def test_get_comprehensive_state_dead_pokemon(self):
        """Test comprehensive state when Pokemon is unconscious."""
        def create_dead_pokemon_memory(addr):
            memory_map = {
                MEMORY_ADDRESSES['player_x']: 5,
                MEMORY_ADDRESSES['player_y']: 5,
                MEMORY_ADDRESSES['map_id']: 1,
                MEMORY_ADDRESSES['player_level']: 10,
                MEMORY_ADDRESSES['current_hp_low']: 0,     # Dead Pokemon
                MEMORY_ADDRESSES['current_hp_high']: 0,
                MEMORY_ADDRESSES['max_hp_low']: 50,
                MEMORY_ADDRESSES['max_hp_high']: 0,
                MEMORY_ADDRESSES['badges']: 1,  # One badge
                MEMORY_ADDRESSES['party_count']: 1,
                MEMORY_ADDRESSES['game_state']: 1,
                MEMORY_ADDRESSES['menu_state']: 0,
                MEMORY_ADDRESSES['money_low']: 0,
                MEMORY_ADDRESSES['money_mid']: 0,
                MEMORY_ADDRESSES['money_high']: 0,
            }
            return memory_map.get(addr, 0)

        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(side_effect=create_dead_pokemon_memory)

        state = get_comprehensive_state(mock_memory)

        assert state['stats']['current_hp'] == 0
        assert state['stats']['max_hp'] == 50
        assert state['badge_count'] == 1
        assert state['is_alive'] is False  # HP = 0


class TestMemoryPerformance:
    """Test memory reading performance."""

    def test_read_memory_performance(self, benchmark_runner):
        """Benchmark memory reading speed."""
        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(return_value=42)

        def read_op():
            read_memory_value(mock_memory, 0xD000)

        result = benchmark_runner.run('read_memory_8bit', read_op, iterations=1000)
        assert result['mean'] < 0.001  # Should be under 1ms

    def test_comprehensive_state_performance(self, benchmark_runner, mock_memory_from_state):
        """Benchmark comprehensive state retrieval."""
        mock_memory = mock_memory_from_state()

        def get_state():
            get_comprehensive_state(mock_memory)

        result = benchmark_runner.run('comprehensive_state', get_state, iterations=100)
        assert result['mean'] < 0.01  # Should be under 10ms


class TestErrorResilience:
    """Test error handling and resilience."""

    def test_memory_read_with_none_memory(self):
        """Test memory reading with None memory object."""
        with pytest.raises(Exception):
            read_memory_value(None, 0xD000)

    def test_memory_read_with_invalid_address(self):
        """Test memory reading with invalid address."""
        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(side_effect=KeyError("Invalid address"))

        # Should return 0 and not crash
        result = read_memory_value(mock_memory, 0xFFFFFF)
        assert result == 0

    def test_comprehensive_state_with_memory_errors(self):
        """Test comprehensive state when some memory reads fail."""
        def failing_memory_read(addr):
            # Only some addresses work
            working_addresses = {
                MEMORY_ADDRESSES['player_x']: 5,
                MEMORY_ADDRESSES['player_y']: 5,
                MEMORY_ADDRESSES['map_id']: 1
            }
            if addr in working_addresses:
                return working_addresses[addr]
            else:
                raise IndexError("Memory read failed")

        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(side_effect=failing_memory_read)

        # Should not crash, should return state with default values
        state = get_comprehensive_state(mock_memory)

        assert state['position']['x'] == 5  # This should work
        assert state['stats']['level'] == 0  # This should default to 0
        assert 'in_game' in state  # Should still have all expected keys