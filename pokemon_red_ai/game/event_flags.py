"""
Event flag definitions and reader utilities for Pokemon Red.

Event flags are a 320-byte bit-array starting at ``wEventFlags`` (0xD747)
that the game engine sets as the player completes story beats, defeats
trainers, collects items, etc.  Each flag occupies one bit::

    address = EVENT_FLAGS_START + (flag_id // 8)
    bit     = flag_id % 8

The 15 flags in ``BOULDER_PATH_FLAGS`` are the pre-registered Boulder-Badge
path events from ``paper/analysis_plan.md`` section 9, corrected to match
the canonical names and IDs in the ``pret/pokered`` disassembly
(``constants/event_constants.asm``).

.. note::
   Pre-pilot correction (2026-05-09): four events in the original §9 list
   (``EVENT_BEAT_RIVAL_IN_OAKS_LAB``, ``EVENT_DELIVERED_OAKS_PARCEL``,
   ``EVENT_BEAT_PEWTER_GYM_TRAINER_1``, ``EVENT_GOT_POKEMON_FROM_FAN_CLUB_CHAIRMAN``)
   do not exist in ``pret/pokered`` and were dropped.  The canary was
   re-pointed to ``EVENT_GOT_BIKE_VOUCHER`` (post-Vermilion, requires
   passing Brock).  See ``paper/compute_ledger.md`` for the full record.

.. note::
   If you discover an ID is off by one, fix it here **and** log the
   correction in ``paper/compute_ledger.md`` (any change after main
   experiments begin is a protocol deviation).

Usage::

    from pokemon_red_ai.game.event_flags import (
        read_event_flag, read_all_event_flags, BOULDER_PATH_FLAGS
    )

    # Check a single flag
    got_starter = read_event_flag(memory, BOULDER_PATH_FLAGS['EVENT_GOT_STARTER'])

    # Count total event progress (simpler metric)
    total = sum_all_event_flag_bits(memory)
"""

import logging
from typing import Dict, List, Tuple

from .memory import read_memory_value

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Base address of the event flag bit-array (``wEventFlags`` in pokered).
EVENT_FLAGS_START: int = 0xD747

#: Length of the event flag array in bytes (320 bytes = 2560 possible flags).
EVENT_FLAGS_LENGTH: int = 320

#: End address (exclusive) of the event flag array.
EVENT_FLAGS_END: int = EVENT_FLAGS_START + EVENT_FLAGS_LENGTH  # 0xD887

#: Battle state address (``wIsInBattle``).  0 = overworld, 1 = wild battle,
#: 2 = trainer battle.  Useful for battle-aware reward shaping.
BATTLE_STATE_ADDR: int = 0xD057


# ---------------------------------------------------------------------------
# Pre-registered Boulder Badge path flags  (analysis_plan.md section 9)
# ---------------------------------------------------------------------------
# Each value is a *flag ID* — a non-negative integer.  The reader below
# converts it to (byte_offset, bit_position) automatically.
#
# All IDs verified against the canonical ``pret/pokered`` disassembly:
#   https://github.com/pret/pokered/blob/master/constants/event_constants.asm
#
# IDs were resolved by emulating the rgbds const_def/const/const_skip/
# const_next macros against the file; full table is reproducible by
# running ``scripts/verify_event_flag_ids.py``.

BOULDER_PATH_FLAGS: Dict[str, int] = {
    # --- Pallet Town / Oak's Lab ---
    'EVENT_FOLLOWED_OAK_INTO_LAB':            0x000,   # Oak leads you to lab (game start)
    'EVENT_GOT_TOWN_MAP':                     0x018,   # Daisy gives Town Map
    'EVENT_GOT_STARTER':                      0x022,   # Chose Bulbasaur/Charmander/Squirtle
    'EVENT_BATTLED_RIVAL_IN_OAKS_LAB':        0x023,   # First rival battle (always won)
    'EVENT_GOT_POKEBALLS_FROM_OAK':           0x024,   # Oak gives Poke Balls
    'EVENT_GOT_POKEDEX':                      0x025,   # Received Pokedex (after parcel delivery)

    # --- Oak's Parcel quest (Route 1 → Viridian → back to Pallet) ---
    'EVENT_GOT_OAKS_PARCEL':                  0x039,   # Picked up parcel in Viridian Mart

    # --- Pewter City Gym ---
    'EVENT_BEAT_PEWTER_GYM_TRAINER_0':        0x072,   # Jr. Trainer in Pewter Gym
    'EVENT_GOT_TM34':                         0x076,   # Brock gives TM34 (Bide)
    'EVENT_BEAT_BROCK':                       0x077,   # Defeated Brock (primary milestone)

    # --- Viridian Forest ---
    'EVENT_BEAT_VIRIDIAN_FOREST_TRAINER_0':   0x562,   # Bug Catcher 1
    'EVENT_BEAT_VIRIDIAN_FOREST_TRAINER_1':   0x563,   # Bug Catcher 2
    'EVENT_BEAT_VIRIDIAN_FOREST_TRAINER_2':   0x564,   # Bug Catcher 3

    # --- Optional stretch milestones ---
    'EVENT_BEAT_ROUTE22_RIVAL_1ST_BATTLE':    0x525,   # Route 22 rival (optional)

    # --- Sanity canary (should NOT fire before Brock) ---
    'EVENT_GOT_BIKE_VOUCHER':                 0x151,   # Vermilion Fan Club Chairman gift
}

#: Total number of pre-registered flags.
NUM_BOULDER_FLAGS: int = len(BOULDER_PATH_FLAGS)
assert NUM_BOULDER_FLAGS == 15, (
    f"Expected 15 pre-registered flags, got {NUM_BOULDER_FLAGS}. "
    f"If you change the set, update paper/analysis_plan.md section 9 "
    f"and log the deviation in paper/compute_ledger.md."
)


# ---------------------------------------------------------------------------
# Reward weights for each flag (used by EventProgressRewardCalculator)
# ---------------------------------------------------------------------------
# Larger reward for harder-to-reach milestones.  The total reward budget
# across all flags is ~575 (the same magnitude as a badge reward in the
# existing reward calculators) so the reward scale stays in the same regime.

FLAG_REWARD_WEIGHTS: Dict[str, float] = {
    # --- Pallet Town / Lab  (easy, early) ---
    'EVENT_FOLLOWED_OAK_INTO_LAB':            5.0,
    'EVENT_GOT_TOWN_MAP':                     5.0,
    'EVENT_GOT_STARTER':                      15.0,
    'EVENT_BATTLED_RIVAL_IN_OAKS_LAB':        15.0,
    'EVENT_GOT_POKEBALLS_FROM_OAK':           10.0,
    'EVENT_GOT_POKEDEX':                      30.0,   # implies parcel delivery completed

    # --- Parcel quest (requires Route 1 traversal) ---
    'EVENT_GOT_OAKS_PARCEL':                  15.0,

    # --- Viridian Forest (requires combat) ---
    'EVENT_BEAT_VIRIDIAN_FOREST_TRAINER_0':   35.0,
    'EVENT_BEAT_VIRIDIAN_FOREST_TRAINER_1':   35.0,
    'EVENT_BEAT_VIRIDIAN_FOREST_TRAINER_2':   35.0,

    # --- Pewter Gym  (requires reaching Pewter + winning battles) ---
    'EVENT_BEAT_PEWTER_GYM_TRAINER_0':        50.0,
    'EVENT_GOT_TM34':                         50.0,
    'EVENT_BEAT_BROCK':                       200.0,

    # --- Optional stretch ---
    'EVENT_BEAT_ROUTE22_RIVAL_1ST_BATTLE':    75.0,

    # --- Canary (should NOT fire before Brock — reward is low) ---
    'EVENT_GOT_BIKE_VOUCHER':                 1.0,
}


# ---------------------------------------------------------------------------
# Reader functions
# ---------------------------------------------------------------------------

def _flag_address_and_bit(flag_id: int) -> Tuple[int, int]:
    """Convert a flag ID to an absolute memory address and bit position.

    Args:
        flag_id: Non-negative integer flag ID.

    Returns:
        ``(address, bit)`` where ``address`` is in ``[0xD747, 0xD887)`` and
        ``bit`` is in ``[0, 7]``.
    """
    byte_offset = flag_id >> 3       # flag_id // 8
    bit_position = flag_id & 0x07    # flag_id % 8
    address = EVENT_FLAGS_START + byte_offset
    return address, bit_position


def read_event_flag(memory, flag_id: int) -> bool:
    """Read a single event flag from memory.

    Args:
        memory: PyBoy memory object (supports ``memory[address]``).
        flag_id: Flag ID from ``BOULDER_PATH_FLAGS`` or any valid flag ID.

    Returns:
        ``True`` if the flag is set, ``False`` otherwise.
    """
    address, bit = _flag_address_and_bit(flag_id)
    byte_value = read_memory_value(memory, address)
    return bool(byte_value & (1 << bit))


def read_boulder_path_flags(memory) -> Dict[str, bool]:
    """Read all 15 pre-registered Boulder Badge path flags.

    Args:
        memory: PyBoy memory object.

    Returns:
        Dict mapping flag names to their current boolean state.
    """
    return {
        name: read_event_flag(memory, flag_id)
        for name, flag_id in BOULDER_PATH_FLAGS.items()
    }


def count_boulder_path_flags(memory) -> int:
    """Count how many of the 15 pre-registered flags are set.

    This is the *event-flag coverage* secondary metric from
    ``analysis_plan.md`` section 4.

    Args:
        memory: PyBoy memory object.

    Returns:
        Integer count in ``[0, 15]``.
    """
    return sum(
        read_event_flag(memory, flag_id)
        for flag_id in BOULDER_PATH_FLAGS.values()
    )


def sum_all_event_flag_bits(memory) -> int:
    """Count total set bits across the entire 320-byte event flag array.

    This is a coarser but fully general progress metric.  It does **not**
    distinguish which flags are set — just how many.

    Args:
        memory: PyBoy memory object.

    Returns:
        Total number of set bits (max theoretical: 2560).
    """
    total = 0
    for offset in range(EVENT_FLAGS_LENGTH):
        byte_val = read_memory_value(memory, EVENT_FLAGS_START + offset)
        total += bin(byte_val).count('1')
    return total


def read_battle_state(memory) -> int:
    """Read the current battle state.

    Args:
        memory: PyBoy memory object.

    Returns:
        0 = overworld, 1 = wild battle, 2 = trainer battle.
    """
    return read_memory_value(memory, BATTLE_STATE_ADDR)


def is_in_battle(memory) -> bool:
    """Check whether the player is currently in any battle.

    Args:
        memory: PyBoy memory object.

    Returns:
        ``True`` if in a wild or trainer battle.
    """
    return read_battle_state(memory) != 0


# ---------------------------------------------------------------------------
# Flag change detection (used by EventProgressRewardCalculator)
# ---------------------------------------------------------------------------

class EventFlagTracker:
    """Track event flag transitions for reward calculation.

    Maintains a snapshot of the pre-registered flags and detects 0 -> 1
    transitions.  Each flag fires at most once per episode (calling
    ``reset()`` clears the snapshot).

    Usage::

        tracker = EventFlagTracker()

        # In env.reset():
        tracker.reset(memory)

        # In env.step():
        newly_set = tracker.update(memory)
        for flag_name in newly_set:
            reward += FLAG_REWARD_WEIGHTS[flag_name]
    """

    def __init__(self) -> None:
        self._previous: Dict[str, bool] = {}
        self._ever_triggered: Dict[str, bool] = {}

    def reset(self, memory=None) -> None:
        """Reset tracker state for a new episode.

        Args:
            memory: If provided, take an initial snapshot so the first
                ``update()`` only fires for genuinely new flags.
        """
        if memory is not None:
            self._previous = read_boulder_path_flags(memory)
        else:
            self._previous = {name: False for name in BOULDER_PATH_FLAGS}
        self._ever_triggered = {name: False for name in BOULDER_PATH_FLAGS}

    def update(self, memory) -> List[str]:
        """Read current flags and return names of newly-set flags.

        A flag is considered *newly set* if it was ``False`` in the previous
        snapshot and is ``True`` now, **and** has not already been triggered
        in this episode (prevents double-counting on consecutive steps).

        Args:
            memory: PyBoy memory object.

        Returns:
            List of flag names that transitioned 0 -> 1 since the last call.
        """
        current = read_boulder_path_flags(memory)
        newly_set: List[str] = []

        for name, is_set in current.items():
            if is_set and not self._previous.get(name, False):
                if not self._ever_triggered.get(name, False):
                    newly_set.append(name)
                    self._ever_triggered[name] = True

        self._previous = current
        return newly_set

    @property
    def flags_triggered(self) -> int:
        """Number of unique flags triggered so far this episode."""
        return sum(self._ever_triggered.values())

    @property
    def triggered_flags(self) -> Dict[str, bool]:
        """Copy of the triggered-flags dict for logging."""
        return dict(self._ever_triggered)

    @property
    def progress_fraction(self) -> float:
        """Fraction of pre-registered flags triggered (0.0 to 1.0)."""
        return self.flags_triggered / NUM_BOULDER_FLAGS
