"""
Tests for EventProgressRewardCalculator.

This is the paper reward function — the only reward calculator used in
all three treatments (pixel, symbolic, hybrid).  These tests verify
that event-flag transitions, exploration, progress, milestones, and
badge bit-counting all behave correctly.
"""

import pytest
from copy import deepcopy
from unittest.mock import patch

from pokemon_red_ai.environment.rewards import (
    EventProgressRewardCalculator,
    EventRewardConfig,
    create_reward_calculator,
)
from pokemon_red_ai.game.event_flags import (
    BOULDER_PATH_FLAGS,
    FLAG_REWARD_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(
    x=10, y=10, map_id=1,
    level=5, hp=20, max_hp=25, badges=0, party_count=1,
    event_flags=None,
):
    """Build a minimal game state dict for testing."""
    return {
        'position': {'x': x, 'y': y, 'map': map_id},
        'stats': {
            'level': level,
            'current_hp': hp,
            'max_hp': max_hp,
            'badges': badges,
            'party_count': party_count,
        },
        'event_flags': event_flags,
    }


def _flags(**overrides):
    """Return a full event_flags dict with all flags False, then apply overrides."""
    base = {name: False for name in BOULDER_PATH_FLAGS}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestEventRewardFactory:
    """Test that the 'events' strategy works in the factory."""

    def test_create_via_factory(self):
        calc = create_reward_calculator('events')
        assert isinstance(calc, EventProgressRewardCalculator)


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------

class TestEventRewardLifecycle:

    def test_reset_clears_state(self):
        calc = EventProgressRewardCalculator()
        calc.calculate_reward(_state(event_flags=_flags(EVENT_GOT_STARTER=True)))

        calc.reset()

        assert len(calc.visited_locations) == 0
        assert len(calc.visited_maps) == 0
        assert calc.previous_state is None

    def test_first_step_gives_exploration(self):
        calc = EventProgressRewardCalculator()
        reward = calc.calculate_reward(_state(event_flags=_flags()))

        assert 'exploration' in calc.reward_components
        assert calc.reward_components['exploration'] > 0

    def test_first_step_gives_new_map(self):
        calc = EventProgressRewardCalculator()
        calc.calculate_reward(_state(event_flags=_flags()))

        assert 'new_map' in calc.reward_components
        assert calc.reward_components['new_map'] > 0


# ---------------------------------------------------------------------------
# Event flag rewards
# ---------------------------------------------------------------------------

class TestEventFlagRewards:

    @pytest.fixture
    def calc(self):
        c = EventProgressRewardCalculator()
        c.reset()
        return c

    def test_single_flag_fires_reward(self, calc):
        # Step 1: no flags
        calc.calculate_reward(_state(event_flags=_flags()))

        # Step 2: flag turns on
        reward = calc.calculate_reward(
            _state(x=11, event_flags=_flags(EVENT_GOT_STARTER=True))
        )

        assert 'event_flags' in calc.reward_components
        expected_weight = FLAG_REWARD_WEIGHTS['EVENT_GOT_STARTER']
        assert calc.reward_components['event_flags'] == pytest.approx(expected_weight)

    def test_flag_does_not_double_fire(self, calc):
        calc.calculate_reward(_state(event_flags=_flags()))

        calc.calculate_reward(
            _state(x=11, event_flags=_flags(EVENT_GOT_STARTER=True))
        )

        # Same flag still set on next step
        calc.calculate_reward(
            _state(x=12, event_flags=_flags(EVENT_GOT_STARTER=True))
        )
        assert 'event_flags' not in calc.reward_components

    def test_multiple_flags_fire_independently(self, calc):
        calc.calculate_reward(_state(event_flags=_flags()))

        calc.calculate_reward(
            _state(
                x=11,
                event_flags=_flags(
                    EVENT_FOLLOWED_OAK_INTO_LAB=True,
                    EVENT_GOT_STARTER=True,
                ),
            )
        )

        expected = (
            FLAG_REWARD_WEIGHTS['EVENT_FOLLOWED_OAK_INTO_LAB']
            + FLAG_REWARD_WEIGHTS['EVENT_GOT_STARTER']
        )
        assert calc.reward_components['event_flags'] == pytest.approx(expected)

    def test_brock_flag_gives_large_reward(self, calc):
        calc.calculate_reward(_state(event_flags=_flags()))

        calc.calculate_reward(
            _state(x=11, event_flags=_flags(EVENT_BEAT_BROCK=True))
        )

        assert calc.reward_components['event_flags'] == pytest.approx(
            FLAG_REWARD_WEIGHTS['EVENT_BEAT_BROCK']
        )
        assert FLAG_REWARD_WEIGHTS['EVENT_BEAT_BROCK'] >= 100

    def test_event_flag_scale_multiplier(self):
        config = EventRewardConfig(event_flag_scale=2.0)
        calc = EventProgressRewardCalculator(config=config)
        calc.reset()

        calc.calculate_reward(_state(event_flags=_flags()))
        calc.calculate_reward(
            _state(x=11, event_flags=_flags(EVENT_GOT_STARTER=True))
        )

        expected = FLAG_REWARD_WEIGHTS['EVENT_GOT_STARTER'] * 2.0
        assert calc.reward_components['event_flags'] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Missing event_flags key (graceful degradation)
# ---------------------------------------------------------------------------

class TestMissingEventFlags:

    def test_no_crash_without_event_flags(self):
        calc = EventProgressRewardCalculator()
        calc.reset()

        # No 'event_flags' key at all
        reward = calc.calculate_reward(_state(event_flags=None))
        assert isinstance(reward, float)

    def test_warning_emitted_once(self, caplog):
        import logging
        calc = EventProgressRewardCalculator()
        calc.reset()

        with caplog.at_level(logging.WARNING, logger='pokemon_red_ai.environment.rewards'):
            calc.calculate_reward(_state(event_flags=None))
            calc.calculate_reward(_state(x=11, event_flags=None))

        warnings = [r for r in caplog.records if 'event_flags' in r.message]
        assert len(warnings) == 1  # Only warns once


# ---------------------------------------------------------------------------
# Milestone bonuses
# ---------------------------------------------------------------------------

class TestMilestones:

    def test_5_flag_milestone(self):
        config = EventRewardConfig()
        calc = EventProgressRewardCalculator(config=config)
        calc.reset()

        # Trigger 5 flags at once
        five_flags = {
            'EVENT_FOLLOWED_OAK_INTO_LAB': True,
            'EVENT_GOT_STARTER': True,
            'EVENT_BATTLED_RIVAL_IN_OAKS_LAB': True,
            'EVENT_BEAT_RIVAL_IN_OAKS_LAB': True,
            'EVENT_GOT_OAKS_PARCEL': True,
        }
        calc.calculate_reward(_state(event_flags=_flags()))
        calc.calculate_reward(_state(x=11, event_flags=_flags(**five_flags)))

        assert 'milestone_5' in calc.reward_components
        assert calc.reward_components['milestone_5'] == config.milestone_thresholds[5]

    def test_milestone_fires_only_once(self):
        config = EventRewardConfig()
        calc = EventProgressRewardCalculator(config=config)
        calc.reset()

        five_flags = {
            'EVENT_FOLLOWED_OAK_INTO_LAB': True,
            'EVENT_GOT_STARTER': True,
            'EVENT_BATTLED_RIVAL_IN_OAKS_LAB': True,
            'EVENT_BEAT_RIVAL_IN_OAKS_LAB': True,
            'EVENT_GOT_OAKS_PARCEL': True,
        }
        calc.calculate_reward(_state(event_flags=_flags()))
        calc.calculate_reward(_state(x=11, event_flags=_flags(**five_flags)))

        # Add another flag — still at 6, milestone 5 already claimed
        six_flags = {**five_flags, 'EVENT_DELIVERED_OAKS_PARCEL': True}
        calc.calculate_reward(_state(x=12, event_flags=_flags(**six_flags)))

        assert 'milestone_5' not in calc.reward_components


# ---------------------------------------------------------------------------
# Exploration rewards
# ---------------------------------------------------------------------------

class TestEventExploration:

    @pytest.fixture
    def calc(self):
        c = EventProgressRewardCalculator()
        c.reset()
        return c

    def test_new_tile_gives_reward(self, calc):
        calc.calculate_reward(_state(x=10, y=10, event_flags=_flags()))
        calc.calculate_reward(_state(x=11, y=10, event_flags=_flags()))

        assert 'exploration' in calc.reward_components

    def test_revisit_gives_no_exploration(self, calc):
        calc.calculate_reward(_state(x=10, y=10, event_flags=_flags()))
        calc.calculate_reward(_state(x=10, y=10, event_flags=_flags()))

        assert 'exploration' not in calc.reward_components

    def test_new_map_gives_bonus(self, calc):
        calc.calculate_reward(_state(map_id=1, event_flags=_flags()))
        calc.calculate_reward(_state(x=11, map_id=2, event_flags=_flags()))

        assert 'new_map' in calc.reward_components


# ---------------------------------------------------------------------------
# Progress rewards (level-ups, badges)
# ---------------------------------------------------------------------------

class TestEventProgress:

    @pytest.fixture
    def calc(self):
        c = EventProgressRewardCalculator()
        c.reset()
        return c

    def test_level_up_reward(self, calc):
        calc.calculate_reward(_state(level=5, event_flags=_flags()))
        calc.calculate_reward(_state(x=11, level=6, event_flags=_flags()))

        assert 'level' in calc.reward_components
        assert calc.reward_components['level'] > 0

    def test_badge_reward_with_bit_counting(self, calc):
        """Badge diff should use popcount, not raw integer diff."""
        # badges=0 -> badges=1 (Boulder Badge bit 0x01)
        calc.calculate_reward(_state(badges=0, event_flags=_flags()))
        calc.calculate_reward(
            _state(x=11, badges=0x01, event_flags=_flags())
        )

        assert 'badge' in calc.reward_components
        assert calc.reward_components['badge'] == pytest.approx(
            EventRewardConfig().badge_reward_multiplier
        )

    def test_badge_diff_uses_popcount_not_int_diff(self, calc):
        """Cascade badge (0x02) should give the same reward as Boulder (0x01).

        The old code used ``stats['badges'] - prev_badges`` which would
        give 0x02 - 0x01 = 1 for the second badge but only because
        of coincidence.  Going from 0x01 to 0x04 (skip cascade, get thunder)
        would give 0x04 - 0x01 = 3 instead of 1.  Our code uses popcount.
        """
        calc.calculate_reward(_state(badges=0x01, event_flags=_flags()))
        calc.calculate_reward(
            _state(x=11, badges=0x05, event_flags=_flags())  # 0x01|0x04
        )

        # Should be 1 badge gained (popcount 0x05 - popcount 0x01 = 2 - 1 = 1)
        assert calc.reward_components['badge'] == pytest.approx(
            EventRewardConfig().badge_reward_multiplier
        )


# ---------------------------------------------------------------------------
# Time and death penalties
# ---------------------------------------------------------------------------

class TestEventPenalties:

    def test_time_penalty_every_step(self):
        calc = EventProgressRewardCalculator()
        calc.reset()
        calc.calculate_reward(_state(event_flags=_flags()))

        assert 'time' in calc.reward_components
        assert calc.reward_components['time'] < 0

    def test_death_penalty(self):
        calc = EventProgressRewardCalculator()
        calc.reset()
        calc.calculate_reward(_state(hp=0, max_hp=25, event_flags=_flags()))

        assert 'death' in calc.reward_components
        assert calc.reward_components['death'] < 0

    def test_no_death_penalty_if_max_hp_zero(self):
        calc = EventProgressRewardCalculator()
        calc.reset()
        calc.calculate_reward(_state(hp=0, max_hp=0, event_flags=_flags()))

        assert 'death' not in calc.reward_components


# ---------------------------------------------------------------------------
# get_event_progress() for logging
# ---------------------------------------------------------------------------

class TestEventProgressReport:

    def test_progress_report_structure(self):
        calc = EventProgressRewardCalculator()
        calc.reset()

        calc.calculate_reward(_state(event_flags=_flags()))
        calc.calculate_reward(
            _state(x=11, event_flags=_flags(EVENT_GOT_STARTER=True))
        )

        report = calc.get_event_progress()
        assert report['flags_triggered'] == 1
        assert report['flags_total'] == 18
        assert report['progress_fraction'] == pytest.approx(1 / 18)
        assert 'EVENT_GOT_STARTER' in report['triggered_names']

    def test_progress_report_empty_after_reset(self):
        calc = EventProgressRewardCalculator()
        calc.reset()

        report = calc.get_event_progress()
        assert report['flags_triggered'] == 0
        assert report['triggered_names'] == []


# ---------------------------------------------------------------------------
# EventRewardConfig
# ---------------------------------------------------------------------------

class TestEventRewardConfig:

    def test_defaults(self):
        config = EventRewardConfig()
        assert config.exploration_reward == 3.0
        assert config.time_penalty == -0.0005
        assert config.event_flag_scale == 1.0
        assert 5 in config.milestone_thresholds

    def test_custom_config(self):
        config = EventRewardConfig(
            exploration_reward=10.0,
            event_flag_scale=0.5,
        )
        assert config.exploration_reward == 10.0
        assert config.event_flag_scale == 0.5
