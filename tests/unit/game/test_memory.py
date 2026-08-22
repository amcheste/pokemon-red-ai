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
    read_16bit_big_endian,
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
            'player_x', 'player_y', 'map_id', 'player_name', 'player_level',
            'current_hp', 'max_hp',
            'badges', 'game_state', 'menu_state', 'party_count'
        ]

        for addr in required_addresses:
            assert addr in MEMORY_ADDRESSES
            assert isinstance(MEMORY_ADDRESSES[addr], int)
            assert MEMORY_ADDRESSES[addr] >= 0

    def test_hp_addresses_match_pret_pokered(self):
        """HP addresses must match the pret/pokered disassembly symbols.

        wPartyMon1HP is 0xD16C and wPartyMon1MaxHP is 0xD18D.  The bytes
        at 0xD16E/0xD16F are wPartyMon1BoxLevel/wPartyMon1Status — reading
        max HP from there was a real bug (see PR fixing party HP reads).
        """
        assert MEMORY_ADDRESSES['current_hp'] == 0xD16C
        assert MEMORY_ADDRESSES['max_hp'] == 0xD18D
        assert MEMORY_ADDRESSES['player_level'] == 0xD18C

    def test_map_and_name_addresses_match_pret_pokered(self):
        """wCurMap is 0xD35E and wPlayerName is 0xD158 per pret/pokered."""
        assert MEMORY_ADDRESSES['map_id'] == 0xD35E
        assert MEMORY_ADDRESSES['player_name'] == 0xD158

    def test_map_ids_exist(self):
        """Test that map ID constants are defined."""
        expected_maps = ['pallet_town', 'viridian_city', 'pewter_city', 'cerulean_city']

        for map_name in expected_maps:
            assert map_name in MAP_IDS
            assert isinstance(MAP_IDS[map_name], int)

    def test_map_ids_match_pret_pokered(self):
        """MAP_IDS must match constants/map_constants.asm in pret/pokered.

        Cities are $00-$0A (Pallet Town is genuinely 0, NOT 1 — the old
        table had every city off by one), $0B is unused, routes are
        $0C-$24, and indoor maps start at $25.
        """
        assert MAP_IDS['pallet_town'] == 0
        assert MAP_IDS['viridian_city'] == 1
        assert MAP_IDS['pewter_city'] == 2
        assert MAP_IDS['cerulean_city'] == 3
        assert MAP_IDS['lavender_town'] == 4
        assert MAP_IDS['vermilion_city'] == 5
        assert MAP_IDS['celadon_city'] == 6
        assert MAP_IDS['fuchsia_city'] == 7
        assert MAP_IDS['cinnabar_island'] == 8
        assert MAP_IDS['indigo_plateau'] == 9
        assert MAP_IDS['saffron_city'] == 10

        # Routes: ROUTE_n = $0C + (n - 1)
        for n in range(1, 26):
            assert MAP_IDS[f'route_{n}'] == 11 + n

        # Early-game indoor maps
        assert MAP_IDS['reds_house_1f'] == 37
        assert MAP_IDS['reds_house_2f'] == 38
        assert MAP_IDS['blues_house'] == 39
        assert MAP_IDS['oaks_lab'] == 40
        assert MAP_IDS['viridian_pokecenter'] == 41
        assert MAP_IDS['viridian_gym'] == 45
        assert MAP_IDS['viridian_forest'] == 51
        assert MAP_IDS['pewter_gym'] == 54
        assert MAP_IDS['mt_moon_1f'] == 59
        assert MAP_IDS['cerulean_gym'] == 65
        assert MAP_IDS['mt_moon_pokecenter'] == 68

    def test_map_ids_are_unique(self):
        """No two map names may share an ID (get_map_name relies on this)."""
        values = list(MAP_IDS.values())
        assert len(values) == len(set(values))

    def test_live_baseline_maps_resolve(self):
        """Maps observed live in the 2026-07-23 baseline grid must resolve.

        The post-intro save state yields visited map IDs
        {0, 12, 37, 38, 39, 40} in a real PyBoy session.
        """
        expected = {
            0: 'pallet_town',
            12: 'route_1',
            37: 'reds_house_1f',
            38: 'reds_house_2f',
            39: 'blues_house',
            40: 'oaks_lab',
        }
        for map_id, name in expected.items():
            assert get_map_name(map_id) == name

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

    def test_read_16bit_big_endian(self):
        """Gen 1 stat values are stored high byte first."""
        mock_memory = Mock()
        # High byte = 0x01, low byte = 0x2C -> 0x012C = 300
        mock_memory.__getitem__ = Mock(side_effect=[0x01, 0x2C])

        value = read_16bit_big_endian(mock_memory, 0xD16C)

        assert value == 300
        assert mock_memory.__getitem__.call_count == 2

    def test_read_16bit_big_endian_error_handling(self):
        """Test error handling for big-endian reads."""
        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(side_effect=IndexError("Memory error"))

        value = read_16bit_big_endian(mock_memory, 0xD16C)

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
                MEMORY_ADDRESSES['current_hp']: 0x00,      # 50, high byte first
                MEMORY_ADDRESSES['current_hp'] + 1: 0x32,
                MEMORY_ADDRESSES['max_hp']: 0x00,          # 60, high byte first
                MEMORY_ADDRESSES['max_hp'] + 1: 0x3C,
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
                MEMORY_ADDRESSES['current_hp']: 0,         # 0 HP
                MEMORY_ADDRESSES['current_hp'] + 1: 0,
                MEMORY_ADDRESSES['max_hp']: 0,             # Max HP 50
                MEMORY_ADDRESSES['max_hp'] + 1: 50,
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
                MEMORY_ADDRESSES['current_hp']: 0,         # 100 HP
                MEMORY_ADDRESSES['current_hp'] + 1: 100,
                MEMORY_ADDRESSES['max_hp']: 0,             # Max HP 100
                MEMORY_ADDRESSES['max_hp'] + 1: 100,
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
                MEMORY_ADDRESSES['current_hp']: 0,
                MEMORY_ADDRESSES['current_hp'] + 1: 0,
                MEMORY_ADDRESSES['max_hp']: 0,  # Max HP is 0
                MEMORY_ADDRESSES['max_hp'] + 1: 0,
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

    def test_read_player_stats_hp_is_big_endian(self):
        """Regression test: HP must decode big-endian (high byte first).

        A level-5 starter with 20 HP stores bytes 00 14 at wPartyMon1HP.
        The old little-endian decode returned 0x1400 = 5120 instead of 20.
        Uses a >255 value so the two byte orders cannot coincide.
        """
        def memory_side_effect(addr):
            memory_map = {
                MEMORY_ADDRESSES['current_hp']: 0x01,      # 0x012C = 300
                MEMORY_ADDRESSES['current_hp'] + 1: 0x2C,
                MEMORY_ADDRESSES['max_hp']: 0x01,          # 0x0158 = 344
                MEMORY_ADDRESSES['max_hp'] + 1: 0x58,
            }
            return memory_map.get(addr, 0)

        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(side_effect=memory_side_effect)

        stats = read_player_stats(mock_memory)

        assert stats['current_hp'] == 300  # not 0x2C01 = 11265
        assert stats['max_hp'] == 344      # not 0x5801 = 22529

    def test_read_player_stats_ignores_box_level_and_status(self):
        """Regression test: max HP comes from 0xD18D, not 0xD16E/0xD16F.

        0xD16E is wPartyMon1BoxLevel and 0xD16F is wPartyMon1Status.  The
        old code read max HP from those bytes, so a poisoned Pokemon's
        status byte corrupted the max-HP high byte.
        """
        def memory_side_effect(addr):
            memory_map = {
                0xD16C: 0x00, 0xD16D: 18,   # current HP = 18
                0xD16E: 5,                  # box level (garbage for HP)
                0xD16F: 0x08,               # status: poisoned
                0xD18D: 0x00, 0xD18E: 23,   # max HP = 23
            }
            return memory_map.get(addr, 0)

        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(side_effect=memory_side_effect)

        stats = read_player_stats(mock_memory)

        assert stats['current_hp'] == 18
        assert stats['max_hp'] == 23
        assert abs(stats['hp_ratio'] - (18 / 23)) < 0.01


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
        """is_in_game is True once the player name is set."""
        mock_memory = Mock()

        def memory_side_effect(addr):
            memory_map = {
                MEMORY_ADDRESSES['player_name']: 0x91,  # 'R' in Gen 1 text
                MEMORY_ADDRESSES['map_id']: 3,
            }
            return memory_map.get(addr, 0)

        mock_memory.__getitem__ = Mock(side_effect=memory_side_effect)

        assert is_in_game(mock_memory) is True

    def test_is_in_game_true_in_pallet_town(self):
        """Pallet Town's real map ID is 0 — must still count as in-game.

        This is exactly the case the old map_id != 0 check got wrong.
        """
        mock_memory = Mock()

        def memory_side_effect(addr):
            memory_map = {
                MEMORY_ADDRESSES['player_name']: 0x91,  # 'R' in Gen 1 text
                MEMORY_ADDRESSES['map_id']: 0,          # PALLET_TOWN
            }
            return memory_map.get(addr, 0)

        mock_memory.__getitem__ = Mock(side_effect=memory_side_effect)

        assert is_in_game(mock_memory) is True

    def test_is_in_game_false(self):
        """is_in_game is False during the intro (WRAM zero-filled)."""
        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(return_value=0)

        assert is_in_game(mock_memory) is False


class TestMoneyReading:
    """Test money reading functionality.

    wPlayerMoney (0xD347) is 3 bytes of big-endian binary-coded decimal —
    each nibble is one decimal digit, most-significant byte first.
    """

    def test_money_address_matches_pret_pokered(self):
        """wPlayerMoney is 0xD347 per the pret/pokered disassembly symbols."""
        assert MEMORY_ADDRESSES['money'] == 0xD347

    def test_read_money_starting_amount(self):
        """The starting ₽3000 is stored as bytes 00 30 00 (verified in PyBoy)."""
        mock_memory = Mock()

        def memory_side_effect(addr):
            memory_map = {
                MEMORY_ADDRESSES['money']: 0x00,        # Most-significant byte
                MEMORY_ADDRESSES['money'] + 1: 0x30,
                MEMORY_ADDRESSES['money'] + 2: 0x00,    # Least-significant byte
            }
            return memory_map.get(addr, 0)

        mock_memory.__getitem__ = Mock(side_effect=memory_side_effect)

        money = read_money(mock_memory)

        assert money == 3000

    def test_read_money_all_digits(self):
        """BCD bytes 12 34 56 decode to 123456."""
        mock_memory = Mock()

        def memory_side_effect(addr):
            memory_map = {
                MEMORY_ADDRESSES['money']: 0x12,
                MEMORY_ADDRESSES['money'] + 1: 0x34,
                MEMORY_ADDRESSES['money'] + 2: 0x56,
            }
            return memory_map.get(addr, 0)

        mock_memory.__getitem__ = Mock(side_effect=memory_side_effect)

        money = read_money(mock_memory)

        assert money == 123456

    def test_read_money_max_value(self):
        """Test reading maximum money value (999999 = BCD bytes 99 99 99)."""
        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(return_value=0x99)

        money = read_money(mock_memory)

        assert money == 999999

    def test_read_money_zero(self):
        """Test reading zero money."""
        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(return_value=0)

        money = read_money(mock_memory)

        assert money == 0

    def test_read_money_non_bcd_returns_zero(self):
        """Bytes with nibbles > 9 are not valid BCD — return 0, don't garble."""
        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(return_value=0xFF)

        money = read_money(mock_memory)

        assert money == 0

    def test_read_money_read_failure_returns_zero(self):
        """A failed memory read yields 0 from read_memory_value → money 0."""
        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(side_effect=IndexError("Memory error"))

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

    def test_get_map_name_zero_is_pallet_town(self):
        """Map ID 0 is Pallet Town in the pret/pokered numbering."""
        map_name = get_map_name(0)
        assert map_name == 'pallet_town'


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
        assert state['in_game'] is True   # Player name is set
        assert state['is_alive'] is True  # HP > 0
        assert state['map_name'] == 'viridian_city'  # Map 1 per pret/pokered

    def test_get_comprehensive_state_dead_pokemon(self):
        """Test comprehensive state when Pokemon is unconscious."""
        def create_dead_pokemon_memory(addr):
            memory_map = {
                MEMORY_ADDRESSES['player_x']: 5,
                MEMORY_ADDRESSES['player_y']: 5,
                MEMORY_ADDRESSES['map_id']: 1,
                MEMORY_ADDRESSES['player_level']: 10,
                MEMORY_ADDRESSES['current_hp']: 0,         # Dead Pokemon
                MEMORY_ADDRESSES['current_hp'] + 1: 0,
                MEMORY_ADDRESSES['max_hp']: 0,             # Max HP 50
                MEMORY_ADDRESSES['max_hp'] + 1: 50,
                MEMORY_ADDRESSES['badges']: 1,  # One badge
                MEMORY_ADDRESSES['party_count']: 1,
                MEMORY_ADDRESSES['game_state']: 1,
                MEMORY_ADDRESSES['menu_state']: 0,
                MEMORY_ADDRESSES['money']: 0,
                MEMORY_ADDRESSES['money'] + 1: 0,
                MEMORY_ADDRESSES['money'] + 2: 0,
            }
            return memory_map.get(addr, 0)

        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(side_effect=create_dead_pokemon_memory)

        state = get_comprehensive_state(mock_memory)

        assert state['stats']['current_hp'] == 0
        assert state['stats']['max_hp'] == 50
        assert state['badge_count'] == 1
        assert state['is_alive'] is False  # HP = 0


@pytest.mark.benchmark
class TestMemoryPerformance:
    """Test memory reading performance."""

    def test_read_memory_performance(self, benchmark_runner):
        """Benchmark memory reading speed."""
        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(return_value=42)

        def read_op():
            read_memory_value(mock_memory, 0xD000)

        result = benchmark_runner.run('read_memory_8bit', read_op, iterations=1000)
        assert result['median'] < 0.001  # Should be under 1ms

    def test_comprehensive_state_performance(self, benchmark_runner, mock_memory_from_state):
        """Benchmark comprehensive state retrieval."""
        mock_memory = mock_memory_from_state()

        def get_state():
            get_comprehensive_state(mock_memory)

        result = benchmark_runner.run('comprehensive_state', get_state, iterations=100)
        assert result['median'] < 0.01  # Should be under 10ms


class TestErrorResilience:
    """Test error handling and resilience."""

    def test_memory_read_with_none_memory(self):
        """Test memory reading with None memory object."""
        # The function handles None gracefully by returning 0, not raising an exception
        result = read_memory_value(None, 0xD000)
        assert result == 0

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

class TestReadPartyData:
    """Test per-Pokemon party struct reading (pret/pokered layout)."""

    @staticmethod
    def _memory_with_party(mons):
        """Build a mock memory holding `mons` party_structs.

        Each entry is (level, current_hp, max_hp).  HP values are
        written BIG-endian, as Gen 1 stores them.
        """
        from pokemon_red_ai.game.memory import (
            PARTY_MON_SIZE,
            PARTY_MON_HP_OFFSET,
            PARTY_MON_LEVEL_OFFSET,
            PARTY_MON_MAX_HP_OFFSET,
        )
        memory_map = {MEMORY_ADDRESSES['party_count']: len(mons)}
        base = MEMORY_ADDRESSES['party_data_start']
        for i, (level, current_hp, max_hp) in enumerate(mons):
            mon = base + i * PARTY_MON_SIZE
            memory_map[mon + PARTY_MON_HP_OFFSET] = (current_hp >> 8) & 0xFF
            memory_map[mon + PARTY_MON_HP_OFFSET + 1] = current_hp & 0xFF
            memory_map[mon + PARTY_MON_LEVEL_OFFSET] = level
            memory_map[mon + PARTY_MON_MAX_HP_OFFSET] = (max_hp >> 8) & 0xFF
            memory_map[mon + PARTY_MON_MAX_HP_OFFSET + 1] = max_hp & 0xFF

        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(
            side_effect=lambda addr: memory_map.get(addr, 0)
        )
        return mock_memory

    def test_empty_party(self):
        from pokemon_red_ai.game.memory import read_party_data
        assert read_party_data(self._memory_with_party([])) == []

    def test_single_pokemon_big_endian_hp(self):
        from pokemon_red_ai.game.memory import read_party_data
        party = read_party_data(self._memory_with_party([(5, 19, 20)]))
        assert party == [{'level': 5, 'current_hp': 19, 'max_hp': 20}]

    def test_hp_above_255_decodes_correctly(self):
        from pokemon_red_ai.game.memory import read_party_data
        # 300 = 0x012C: high byte 0x01 first (big-endian)
        party = read_party_data(self._memory_with_party([(80, 300, 312)]))
        assert party[0]['current_hp'] == 300
        assert party[0]['max_hp'] == 312

    def test_full_party_in_order(self):
        from pokemon_red_ai.game.memory import read_party_data
        mons = [(5 + i, 10 + i, 20 + i) for i in range(6)]
        party = read_party_data(self._memory_with_party(mons))
        assert len(party) == 6
        assert [m['level'] for m in party] == [5, 6, 7, 8, 9, 10]

    def test_party_count_clamped_to_six(self):
        from pokemon_red_ai.game.memory import read_party_data
        mock_memory = self._memory_with_party([(5, 10, 10)])
        # Corrupt count byte claims 99 party members
        original = mock_memory.__getitem__.side_effect
        count_addr = MEMORY_ADDRESSES['party_count']
        mock_memory.__getitem__.side_effect = (
            lambda addr: 99 if addr == count_addr else original(addr)
        )
        assert len(read_party_data(mock_memory)) == 6


class TestReadEventFlagCount:
    """Test wEventFlags popcount reading."""

    @staticmethod
    def _memory_with_flag_bytes(byte_map):
        """Mock memory with specific bytes in the wEventFlags array."""
        base = MEMORY_ADDRESSES['event_flags_start']
        memory_map = {base + offset: value
                      for offset, value in byte_map.items()}
        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(
            side_effect=lambda addr: memory_map.get(addr, 0)
        )
        return mock_memory

    def test_no_flags_set(self):
        from pokemon_red_ai.game.memory import read_event_flag_count
        assert read_event_flag_count(self._memory_with_flag_bytes({})) == 0

    def test_counts_bits_across_array(self):
        from pokemon_red_ai.game.memory import read_event_flag_count
        memory = self._memory_with_flag_bytes({
            0: 0b10110000,    # 3 bits
            100: 0xFF,        # 8 bits
            319: 0b00000001,  # last byte in range
        })
        assert read_event_flag_count(memory) == 12

    def test_bytes_outside_array_ignored(self):
        from pokemon_red_ai.game.memory import read_event_flag_count
        from pokemon_red_ai.game.memory import EVENT_FLAGS_SIZE
        memory = self._memory_with_flag_bytes({EVENT_FLAGS_SIZE: 0xFF})
        assert read_event_flag_count(memory) == 0

    def test_slice_read_supported(self):
        from pokemon_red_ai.game.memory import (
            read_event_flag_count, EVENT_FLAGS_SIZE,
        )

        class SliceMemory:
            """Fake memory supporting slice reads like PyBoy."""
            def __getitem__(self, key):
                if isinstance(key, slice):
                    block = [0] * EVENT_FLAGS_SIZE
                    block[0] = 0b00000111  # 3 bits
                    return block
                raise TypeError("only slices supported")

        assert read_event_flag_count(SliceMemory()) == 3


class TestComprehensiveStatePartyFields:
    """Test that get_comprehensive_state exposes the new fields."""

    def test_party_and_event_flag_count_present(self, mock_memory_from_state):
        mock_memory = mock_memory_from_state()
        state = get_comprehensive_state(mock_memory)
        assert 'party' in state
        assert isinstance(state['party'], list)
        assert 'event_flag_count' in state
        assert isinstance(state['event_flag_count'], int)
