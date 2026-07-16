"""
Tests for PleinesRewardCalculator.

Verifies a faithful reproduction of the reward function from
Pleines et al. 2025, "Pokemon Red via Reinforcement Learning"
(arXiv:2502.19920, section II-F): event reward (+2 per newly set
wEventFlags bit), navigation reward (+0.005 per new coordinate per
episode), healing reward (Eq. 1), and level reward (Eq. 2).
"""

import pytest

from pokemon_red_ai.environment.rewards import (
    PleinesRewardCalculator,
    PleinesRewardConfig,
    create_reward_calculator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mon(level=5, current_hp=20, max_hp=20):
    return {'level': level, 'current_hp': current_hp, 'max_hp': max_hp}


def _state(x=10, y=10, map_id=1, party=None, event_flag_count=0):
    """Build a minimal game state dict for testing."""
    return {
        'position': {'x': x, 'y': y, 'map': map_id},
        'stats': {
            'level': 5,
            'current_hp': 20,
            'max_hp': 20,
            'badges': 0,
            'party_count': len(party) if party is not None else 0,
        },
        'party': party if party is not None else [],
        'event_flag_count': event_flag_count,
    }


@pytest.fixture
def calc():
    return PleinesRewardCalculator()


def _settle(calc, state):
    """Feed a state once to establish the delta baseline."""
    calc.calculate_reward(state)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestPleinesFactory:

    def test_factory_creates_pleines_calculator(self):
        calculator = create_reward_calculator('pleines')
        assert isinstance(calculator, PleinesRewardCalculator)

    def test_default_config_matches_paper_scalars(self):
        cfg = PleinesRewardConfig()
        assert cfg.event_reward == 2.0
        assert cfg.navigation_reward == 0.005
        assert cfg.heal_reward_coefficient == 2.5
        assert cfg.level_potential_scale == 0.5
        assert cfg.level_sum_threshold == 22.0
        assert cfg.level_downscale_divisor == 4.0


# ---------------------------------------------------------------------------
# Event reward
# ---------------------------------------------------------------------------

class TestEventReward:

    def test_no_event_reward_on_first_step(self, calc):
        reward = calc.calculate_reward(_state(event_flag_count=7))
        assert 'event' not in calc.get_reward_breakdown()

    def test_plus_two_per_newly_set_flag(self, calc):
        _settle(calc, _state(event_flag_count=7))
        calc.calculate_reward(_state(event_flag_count=8))
        assert calc.get_reward_breakdown()['event'] == pytest.approx(2.0)

    def test_multiple_flags_in_one_step(self, calc):
        _settle(calc, _state(event_flag_count=7))
        calc.calculate_reward(_state(event_flag_count=10))
        assert calc.get_reward_breakdown()['event'] == pytest.approx(6.0)

    def test_no_negative_reward_when_count_drops(self, calc):
        # Some flags are transient in-game; a drop must not punish.
        _settle(calc, _state(event_flag_count=10))
        calc.calculate_reward(_state(event_flag_count=9))
        assert 'event' not in calc.get_reward_breakdown()

    def test_baseline_flags_in_save_state_not_rewarded(self, calc):
        # A save state starts with flags already set; only NEW flags pay.
        _settle(calc, _state(event_flag_count=50))
        calc.calculate_reward(_state(x=11, event_flag_count=50))
        assert 'event' not in calc.get_reward_breakdown()


# ---------------------------------------------------------------------------
# Navigation reward
# ---------------------------------------------------------------------------

class TestNavigationReward:

    def test_new_coordinate_rewarded(self, calc):
        calc.calculate_reward(_state(x=10, y=10))
        assert calc.get_reward_breakdown()['navigation'] == pytest.approx(0.005)

    def test_revisited_coordinate_not_rewarded(self, calc):
        calc.calculate_reward(_state(x=10, y=10))
        calc.calculate_reward(_state(x=10, y=10))
        assert 'navigation' not in calc.get_reward_breakdown()

    def test_same_xy_different_map_is_new(self, calc):
        calc.calculate_reward(_state(x=10, y=10, map_id=1))
        calc.calculate_reward(_state(x=10, y=10, map_id=2))
        assert calc.get_reward_breakdown()['navigation'] == pytest.approx(0.005)

    def test_visited_set_resets_per_episode(self, calc):
        calc.calculate_reward(_state(x=10, y=10))
        calc.reset()
        calc.calculate_reward(_state(x=10, y=10))
        assert calc.get_reward_breakdown()['navigation'] == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# Healing reward (Eq. 1)
# ---------------------------------------------------------------------------

class TestHealingReward:

    def test_fractional_heal_formula(self, calc):
        # 20 -> 40 out of 40 max: fraction 0.5, reward 2.5 * 0.5 = 1.25
        _settle(calc, _state(party=[_mon(current_hp=20, max_hp=40)]))
        calc.calculate_reward(_state(party=[_mon(current_hp=40, max_hp=40)]))
        assert calc.get_reward_breakdown()['healing'] == pytest.approx(1.25)

    def test_full_party_heal_sums_fractions(self, calc):
        before = [_mon(current_hp=10, max_hp=20), _mon(current_hp=5, max_hp=30)]
        after = [_mon(current_hp=20, max_hp=20), _mon(current_hp=30, max_hp=30)]
        _settle(calc, _state(party=before))
        calc.calculate_reward(_state(party=after))
        # (10/20) + (25/30) = 0.5 + 0.8333 -> * 2.5
        expected = 2.5 * (0.5 + 25 / 30)
        assert calc.get_reward_breakdown()['healing'] == pytest.approx(expected)

    def test_damage_not_rewarded(self, calc):
        _settle(calc, _state(party=[_mon(current_hp=20, max_hp=20)]))
        calc.calculate_reward(_state(party=[_mon(current_hp=5, max_hp=20)]))
        assert 'healing' not in calc.get_reward_breakdown()

    def test_catching_pokemon_is_not_a_heal(self, calc):
        # Party grows by a full-HP member; no heal reward may fire.
        _settle(calc, _state(party=[_mon(current_hp=20, max_hp=20)]))
        calc.calculate_reward(_state(
            party=[_mon(current_hp=20, max_hp=20), _mon(current_hp=15, max_hp=15)]
        ))
        assert 'healing' not in calc.get_reward_breakdown()

    def test_level_up_hp_gain_counts_as_heal(self, calc):
        # Paper: heal reward "is triggered when a Pokemon levels up".
        _settle(calc, _state(party=[_mon(level=5, current_hp=18, max_hp=20)]))
        calc.calculate_reward(
            _state(party=[_mon(level=6, current_hp=23, max_hp=25)])
        )
        assert calc.get_reward_breakdown()['healing'] == pytest.approx(2.5 * 5 / 25)

    def test_zero_max_hp_member_skipped(self, calc):
        _settle(calc, _state(party=[_mon(current_hp=0, max_hp=0)]))
        calc.calculate_reward(_state(party=[_mon(current_hp=0, max_hp=0)]))
        assert 'healing' not in calc.get_reward_breakdown()

    def test_starter_acquisition_stub_is_not_a_heal(self, calc):
        # Live-repro regression: during capture the game bumps
        # party_count before filling the party_struct, so one step
        # reads a {level: 0, hp: 0/0} stub.  When the real data lands,
        # the 0 -> full-HP transition must NOT pay a heal reward
        # (the level reward for the new capture is legitimate).
        _settle(calc, _state(party=[]))
        calc.calculate_reward(
            _state(party=[_mon(level=0, current_hp=0, max_hp=0)])
        )
        calc.calculate_reward(
            _state(party=[_mon(level=5, current_hp=20, max_hp=20)])
        )
        breakdown = calc.get_reward_breakdown()
        assert 'healing' not in breakdown
        assert breakdown['level'] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Level reward (Eq. 2)
# ---------------------------------------------------------------------------

class TestLevelReward:

    def test_below_threshold_half_point_per_level(self, calc):
        _settle(calc, _state(party=[_mon(level=5)]))
        calc.calculate_reward(_state(party=[_mon(level=6)]))
        assert calc.get_reward_breakdown()['level'] == pytest.approx(0.5)

    def test_above_threshold_downscaled_by_four(self, calc):
        # Sum 30 -> 31, both past the knee at 22: marginal 0.5/4 = 0.125
        _settle(calc, _state(party=[_mon(level=30)]))
        calc.calculate_reward(_state(party=[_mon(level=31)]))
        assert calc.get_reward_breakdown()['level'] == pytest.approx(0.125)

    def test_potential_is_continuous_at_knee(self, calc):
        # min(x, (x-22)/4 + 22) equals x exactly at x = 22.
        assert calc._level_potential(22) == pytest.approx(0.5 * 22)
        assert calc._level_potential(21.999) == pytest.approx(
            0.5 * 21.999, rel=1e-6
        )

    def test_crossing_threshold_splits_marginal_gain(self, calc):
        # Sum 21 -> 23: one full level below the knee (0.5) plus one
        # downscaled level above it (0.125).
        _settle(calc, _state(party=[_mon(level=21)]))
        calc.calculate_reward(_state(party=[_mon(level=23)]))
        assert calc.get_reward_breakdown()['level'] == pytest.approx(0.625)

    def test_level_sum_drop_not_punished(self, calc):
        # Depositing a Pokemon lowers the sum; no negative reward.
        _settle(calc, _state(party=[_mon(level=10), _mon(level=8)]))
        calc.calculate_reward(_state(party=[_mon(level=10)]))
        assert 'level' not in calc.get_reward_breakdown()

    def test_first_party_appearance_rewards_starter(self, calc):
        # Empty party -> starter at level 5: potential goes 0 -> 2.5.
        _settle(calc, _state(party=[]))
        calc.calculate_reward(_state(party=[_mon(level=5)]))
        assert calc.get_reward_breakdown()['level'] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Composition and robustness
# ---------------------------------------------------------------------------

class TestComposition:

    def test_components_sum_to_total(self, calc):
        _settle(calc, _state(
            x=10, party=[_mon(level=5, current_hp=10, max_hp=20)],
            event_flag_count=3,
        ))
        total = calc.calculate_reward(_state(
            x=11,
            party=[_mon(level=6, current_hp=25, max_hp=25)],
            event_flag_count=4,
        ))
        breakdown = calc.get_reward_breakdown()
        assert total == pytest.approx(sum(breakdown.values()))
        assert set(breakdown) == {'event', 'navigation', 'healing', 'level'}

    def test_missing_keys_degrade_gracefully(self, calc):
        state = _state()
        del state['party']
        del state['event_flag_count']
        # Must not raise; navigation still works.
        reward = calc.calculate_reward(state)
        assert reward == pytest.approx(0.005)

    def test_reset_clears_delta_baselines(self, calc):
        _settle(calc, _state(party=[_mon(level=10)], event_flag_count=5))
        calc.reset()
        # After reset the first step re-establishes baselines silently.
        calc.calculate_reward(_state(x=99, party=[_mon(level=10)],
                                     event_flag_count=5))
        breakdown = calc.get_reward_breakdown()
        assert 'event' not in breakdown
        assert 'level' not in breakdown

    def test_no_time_penalty_or_death_penalty(self, calc):
        # The Pleines reward has exactly four components — no extras.
        _settle(calc, _state())
        calc.calculate_reward(_state(party=[_mon(current_hp=0, max_hp=20)]))
        breakdown = calc.get_reward_breakdown()
        assert 'time' not in breakdown
        assert 'death' not in breakdown
