"""
Tests for Pokemon Red event flag reader and tracker.

Covers:
- Flag ID -> address/bit conversion
- Single flag reads
- Boulder path flag reads
- Total event flag bit counting
- Battle state reads
- EventFlagTracker transition detection
"""

import pytest
from unittest.mock import Mock

from pokemon_red_ai.game.event_flags import (
    EVENT_FLAGS_START,
    EVENT_FLAGS_LENGTH,
    EVENT_FLAGS_END,
    BATTLE_STATE_ADDR,
    BOULDER_PATH_FLAGS,
    FLAG_REWARD_WEIGHTS,
    NUM_BOULDER_FLAGS,
    _flag_address_and_bit,
    read_event_flag,
    read_boulder_path_flags,
    count_boulder_path_flags,
    sum_all_event_flag_bits,
    read_battle_state,
    is_in_battle,
    EventFlagTracker,
)


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------

class TestEventFlagConstants:
    """Verify constant definitions are self-consistent."""

    def test_event_flags_start(self):
        assert EVENT_FLAGS_START == 0xD747

    def test_event_flags_length(self):
        assert EVENT_FLAGS_LENGTH == 320

    def test_event_flags_end(self):
        assert EVENT_FLAGS_END == EVENT_FLAGS_START + EVENT_FLAGS_LENGTH

    def test_boulder_path_has_18_flags(self):
        assert len(BOULDER_PATH_FLAGS) == 18

    def test_num_boulder_flags_matches(self):
        assert NUM_BOULDER_FLAGS == 18

    def test_reward_weights_cover_all_flags(self):
        """Every pre-registered flag must have a reward weight."""
        for name in BOULDER_PATH_FLAGS:
            assert name in FLAG_REWARD_WEIGHTS, (
                f"Missing reward weight for {name}"
            )

    def test_reward_weights_are_positive(self):
        for name, weight in FLAG_REWARD_WEIGHTS.items():
            assert weight > 0, f"Weight for {name} must be positive"

    def test_flag_ids_within_array_bounds(self):
        """All flag IDs must resolve to addresses within the event array."""
        for name, flag_id in BOULDER_PATH_FLAGS.items():
            addr, bit = _flag_address_and_bit(flag_id)
            assert EVENT_FLAGS_START <= addr < EVENT_FLAGS_END, (
                f"{name} (ID=0x{flag_id:03X}) maps to 0x{addr:04X} "
                f"which is outside [{EVENT_FLAGS_START:#06x}, {EVENT_FLAGS_END:#06x})"
            )
            assert 0 <= bit <= 7

    def test_battle_state_address(self):
        assert BATTLE_STATE_ADDR == 0xD057


# ---------------------------------------------------------------------------
# Address/bit conversion
# ---------------------------------------------------------------------------

class TestFlagAddressAndBit:
    """Test the flag ID -> (address, bit) helper."""

    def test_flag_zero(self):
        addr, bit = _flag_address_and_bit(0)
        assert addr == EVENT_FLAGS_START
        assert bit == 0

    def test_flag_seven(self):
        """Bit 7 of the first byte."""
        addr, bit = _flag_address_and_bit(7)
        assert addr == EVENT_FLAGS_START
        assert bit == 7

    def test_flag_eight(self):
        """First bit of the second byte."""
        addr, bit = _flag_address_and_bit(8)
        assert addr == EVENT_FLAGS_START + 1
        assert bit == 0

    def test_flag_0x045(self):
        """The first pre-registered flag."""
        addr, bit = _flag_address_and_bit(0x045)
        expected_byte = 0x045 // 8  # = 8
        expected_bit = 0x045 % 8    # = 5
        assert addr == EVENT_FLAGS_START + expected_byte
        assert bit == expected_bit


# ---------------------------------------------------------------------------
# Single flag reads
# ---------------------------------------------------------------------------

class TestReadEventFlag:
    """Test reading individual event flags from memory."""

    def _mock_memory(self, addr_to_val):
        """Create a mock memory that returns values for specific addresses."""
        mock = Mock()
        mock.__getitem__ = Mock(
            side_effect=lambda a: addr_to_val.get(a, 0)
        )
        return mock

    def test_flag_not_set(self):
        mem = self._mock_memory({})
        assert read_event_flag(mem, 0x045) is False

    def test_flag_is_set(self):
        # Flag 0x045 → byte offset 8 → address 0xD74F, bit 5
        addr = EVENT_FLAGS_START + (0x045 // 8)
        bit = 0x045 % 8
        mem = self._mock_memory({addr: (1 << bit)})
        assert read_event_flag(mem, 0x045) is True

    def test_other_bits_dont_interfere(self):
        """Setting all bits except the target should return False."""
        addr = EVENT_FLAGS_START + (0x045 // 8)
        bit = 0x045 % 8
        all_except = 0xFF ^ (1 << bit)
        mem = self._mock_memory({addr: all_except})
        assert read_event_flag(mem, 0x045) is False

    def test_all_bits_set_returns_true(self):
        addr = EVENT_FLAGS_START + (0x045 // 8)
        mem = self._mock_memory({addr: 0xFF})
        assert read_event_flag(mem, 0x045) is True

    def test_memory_error_returns_false(self):
        """If memory read fails, flag should be False (not crash)."""
        mock = Mock()
        mock.__getitem__ = Mock(side_effect=Exception("memory error"))
        # read_memory_value catches exceptions and returns 0
        assert read_event_flag(mock, 0x045) is False


# ---------------------------------------------------------------------------
# Boulder path flag reads
# ---------------------------------------------------------------------------

class TestReadBoulderPathFlags:
    """Test reading all 18 pre-registered flags at once."""

    def test_all_false_on_fresh_game(self):
        mock = Mock()
        mock.__getitem__ = Mock(return_value=0)
        flags = read_boulder_path_flags(mock)

        assert len(flags) == 18
        assert all(v is False for v in flags.values())

    def test_specific_flag_true(self):
        """Set exactly one flag and verify only that one reads True."""
        flag_name = 'EVENT_GOT_STARTER'
        flag_id = BOULDER_PATH_FLAGS[flag_name]
        addr = EVENT_FLAGS_START + (flag_id // 8)
        bit = flag_id % 8

        def mem_read(a):
            if a == addr:
                return 1 << bit
            return 0

        mock = Mock()
        mock.__getitem__ = Mock(side_effect=mem_read)
        flags = read_boulder_path_flags(mock)

        assert flags[flag_name] is True
        # Other flags that share the same byte might also be True if
        # their bit is coincidentally set; just check the target
        false_count = sum(1 for v in flags.values() if not v)
        assert false_count >= 16  # At most 2 could share the byte

    def test_count_with_no_flags(self):
        mock = Mock()
        mock.__getitem__ = Mock(return_value=0)
        assert count_boulder_path_flags(mock) == 0

    def test_count_with_all_flags(self):
        """Set every byte in the event array to 0xFF."""
        mock = Mock()
        mock.__getitem__ = Mock(return_value=0xFF)
        assert count_boulder_path_flags(mock) == 18


# ---------------------------------------------------------------------------
# Total event flag bit counting
# ---------------------------------------------------------------------------

class TestSumAllEventFlagBits:
    """Test the coarse total-bits-set metric."""

    def test_zero_on_fresh_game(self):
        mock = Mock()
        mock.__getitem__ = Mock(return_value=0)
        assert sum_all_event_flag_bits(mock) == 0

    def test_all_bits_set(self):
        mock = Mock()
        mock.__getitem__ = Mock(return_value=0xFF)
        # 320 bytes * 8 bits = 2560
        assert sum_all_event_flag_bits(mock) == 320 * 8

    def test_one_byte_set(self):
        """Only the first byte has bits set."""
        def mem_read(addr):
            if addr == EVENT_FLAGS_START:
                return 0b10101010  # 4 bits set
            return 0

        mock = Mock()
        mock.__getitem__ = Mock(side_effect=mem_read)
        assert sum_all_event_flag_bits(mock) == 4


# ---------------------------------------------------------------------------
# Battle state
# ---------------------------------------------------------------------------

class TestBattleState:

    def test_overworld(self):
        mock = Mock()
        mock.__getitem__ = Mock(return_value=0)
        assert read_battle_state(mock) == 0
        assert is_in_battle(mock) is False

    def test_wild_battle(self):
        def mem_read(addr):
            if addr == BATTLE_STATE_ADDR:
                return 1
            return 0
        mock = Mock()
        mock.__getitem__ = Mock(side_effect=mem_read)
        assert read_battle_state(mock) == 1
        assert is_in_battle(mock) is True

    def test_trainer_battle(self):
        def mem_read(addr):
            if addr == BATTLE_STATE_ADDR:
                return 2
            return 0
        mock = Mock()
        mock.__getitem__ = Mock(side_effect=mem_read)
        assert read_battle_state(mock) == 2
        assert is_in_battle(mock) is True


# ---------------------------------------------------------------------------
# EventFlagTracker
# ---------------------------------------------------------------------------

class TestEventFlagTracker:
    """Test the transition-detecting tracker used by the reward calculator."""

    def _make_memory(self, flags_set=None):
        """Return a mock memory where specified flag names are set."""
        flags_set = flags_set or []
        set_addresses = {}

        for name in flags_set:
            flag_id = BOULDER_PATH_FLAGS[name]
            addr = EVENT_FLAGS_START + (flag_id // 8)
            bit = flag_id % 8
            set_addresses[addr] = set_addresses.get(addr, 0) | (1 << bit)

        mock = Mock()
        mock.__getitem__ = Mock(
            side_effect=lambda a: set_addresses.get(a, 0)
        )
        return mock

    def test_reset_clears_state(self):
        tracker = EventFlagTracker()
        tracker.reset()
        assert tracker.flags_triggered == 0
        assert tracker.progress_fraction == 0.0

    def test_reset_with_memory_takes_snapshot(self):
        """Flags already set at reset time should NOT fire on first update."""
        mem = self._make_memory(['EVENT_GOT_STARTER'])
        tracker = EventFlagTracker()
        tracker.reset(memory=mem)

        # Same state — no transitions
        newly = tracker.update(mem)
        assert newly == []
        assert tracker.flags_triggered == 0

    def test_detects_new_flag(self):
        tracker = EventFlagTracker()
        mem_0 = self._make_memory([])
        tracker.reset(memory=mem_0)

        mem_1 = self._make_memory(['EVENT_GOT_STARTER'])
        newly = tracker.update(mem_1)

        assert newly == ['EVENT_GOT_STARTER']
        assert tracker.flags_triggered == 1

    def test_does_not_double_count(self):
        tracker = EventFlagTracker()
        tracker.reset(memory=self._make_memory([]))

        mem_1 = self._make_memory(['EVENT_GOT_STARTER'])
        tracker.update(mem_1)
        # Second call with same flag still set
        newly = tracker.update(mem_1)

        assert newly == []
        assert tracker.flags_triggered == 1

    def test_multiple_flags_at_once(self):
        tracker = EventFlagTracker()
        tracker.reset(memory=self._make_memory([]))

        flags = [
            'EVENT_FOLLOWED_OAK_INTO_LAB',
            'EVENT_GOT_STARTER',
            'EVENT_BATTLED_RIVAL_IN_OAKS_LAB',
        ]
        mem = self._make_memory(flags)
        newly = tracker.update(mem)

        assert set(newly) == set(flags)
        assert tracker.flags_triggered == 3

    def test_incremental_progress(self):
        tracker = EventFlagTracker()
        tracker.reset(memory=self._make_memory([]))

        # Step 1: one flag
        newly_1 = tracker.update(
            self._make_memory(['EVENT_FOLLOWED_OAK_INTO_LAB'])
        )
        assert len(newly_1) == 1

        # Step 2: add a second
        newly_2 = tracker.update(
            self._make_memory([
                'EVENT_FOLLOWED_OAK_INTO_LAB',
                'EVENT_GOT_STARTER',
            ])
        )
        assert newly_2 == ['EVENT_GOT_STARTER']
        assert tracker.flags_triggered == 2

    def test_progress_fraction(self):
        tracker = EventFlagTracker()
        tracker.reset(memory=self._make_memory([]))

        tracker.update(self._make_memory(['EVENT_GOT_STARTER']))
        assert tracker.progress_fraction == pytest.approx(1.0 / 18)

    def test_triggered_flags_property(self):
        tracker = EventFlagTracker()
        tracker.reset(memory=self._make_memory([]))
        tracker.update(self._make_memory(['EVENT_BEAT_BROCK']))

        triggered = tracker.triggered_flags
        assert triggered['EVENT_BEAT_BROCK'] is True
        assert triggered['EVENT_GOT_STARTER'] is False

    def test_flag_that_unsets_does_not_re_trigger(self):
        """If a flag flips 1->0->1 it should not fire a second reward."""
        tracker = EventFlagTracker()
        tracker.reset(memory=self._make_memory([]))

        # Flag turns on
        tracker.update(self._make_memory(['EVENT_GOT_STARTER']))
        # Flag turns off (unlikely in-game, but test the guard)
        tracker.update(self._make_memory([]))
        # Flag turns on again
        newly = tracker.update(self._make_memory(['EVENT_GOT_STARTER']))

        assert newly == []
        assert tracker.flags_triggered == 1
