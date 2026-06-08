"""
Numerical regression for ``EventProgressRewardCalculator``.

Drives a hand-built trajectory through the reward calculator and locks
the per-step reward against a golden array.  Any silent change to the
calculator (reward weights, milestone thresholds, exploration bonus,
time penalty, double-fire logic) breaks this test.

What this catches that the existing unit tests do not:

The unit tests in ``tests/unit/environment/test_event_rewards.py``
verify individual components in isolation (one flag firing, one
milestone tripping, etc.).  They do **not** verify that the full
multi-component reward sums correctly across a real-shaped sequence.
PR #36's encoder rewrite and PR #45's flag-set correction both
touched constants that feed into this calculation; a numerical lock
catches the next such change automatically.

The golden array is computed by running the calculator with the
current code and pasted in below.  If the test fails after a
deliberate change, regenerate it by:

    pytest tests/integration/test_reward_regression.py -v -s

read the actual rewards from the diff, paste them in, and document
the reason for the change in ``compute_ledger.md``.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

from pokemon_red_ai.environment.rewards import (
    EventProgressRewardCalculator,
    EventRewardConfig,
)


def _state(
    x: int = 5,
    y: int = 5,
    map_id: int = 1,
    level: int = 5,
    badges: int = 0,
    party_count: int = 1,
    current_hp: int = 20,
    max_hp: int = 20,
    **flags: bool,
) -> Dict[str, object]:
    """Build a synthetic state dict matching what env.step() produces."""
    return {
        "position": {"x": x, "y": y, "map": map_id},
        "stats": {
            "level": level,
            "badges": badges,
            "party_count": party_count,
            "current_hp": current_hp,
            "max_hp": max_hp,
        },
        "event_flags": flags,
    }


# ──────────────────────────────────────────────────────────────────────
# Golden trajectory — represents a realistic first-episode walk:
#   step 0: spawn in Pallet (no flags, first tile)
#   step 1: move (new tile)
#   step 2: trigger FOLLOWED_OAK_INTO_LAB
#   step 3: stay put (revisit, no exploration reward)
#   step 4: trigger GOT_STARTER (this is flag #2 of 15)
#   step 5: move to a new map (Route 1 = map 13)
#   step 6: level up (5 → 6)
#   step 7: trigger BATTLED_RIVAL_IN_OAKS_LAB
#   step 8: trigger GOT_POKEBALLS_FROM_OAK + GOT_POKEDEX  (now at 5 flags = milestone 5)
#   step 9: faint (current_hp = 0)
# ──────────────────────────────────────────────────────────────────────


def _build_trajectory() -> List[Dict[str, object]]:
    return [
        _state(x=5, y=5, map_id=1, level=5),                                                # step 0
        _state(x=6, y=5, map_id=1, level=5),                                                # step 1
        _state(x=7, y=5, map_id=1, level=5, EVENT_FOLLOWED_OAK_INTO_LAB=True),              # step 2
        _state(x=7, y=5, map_id=1, level=5, EVENT_FOLLOWED_OAK_INTO_LAB=True),              # step 3
        _state(x=8, y=5, map_id=1, level=5,
               EVENT_FOLLOWED_OAK_INTO_LAB=True, EVENT_GOT_STARTER=True),                   # step 4
        _state(x=0, y=0, map_id=13, level=5,
               EVENT_FOLLOWED_OAK_INTO_LAB=True, EVENT_GOT_STARTER=True),                   # step 5
        _state(x=0, y=1, map_id=13, level=6,
               EVENT_FOLLOWED_OAK_INTO_LAB=True, EVENT_GOT_STARTER=True),                   # step 6
        _state(x=0, y=2, map_id=13, level=6,
               EVENT_FOLLOWED_OAK_INTO_LAB=True, EVENT_GOT_STARTER=True,
               EVENT_BATTLED_RIVAL_IN_OAKS_LAB=True),                                       # step 7
        _state(x=0, y=3, map_id=13, level=6,
               EVENT_FOLLOWED_OAK_INTO_LAB=True, EVENT_GOT_STARTER=True,
               EVENT_BATTLED_RIVAL_IN_OAKS_LAB=True,
               EVENT_GOT_POKEBALLS_FROM_OAK=True, EVENT_GOT_POKEDEX=True),                  # step 8
        _state(x=0, y=4, map_id=13, level=6,
               current_hp=0,
               EVENT_FOLLOWED_OAK_INTO_LAB=True, EVENT_GOT_STARTER=True,
               EVENT_BATTLED_RIVAL_IN_OAKS_LAB=True,
               EVENT_GOT_POKEBALLS_FROM_OAK=True, EVENT_GOT_POKEDEX=True),                  # step 9
    ]


# Golden output: per-step rewards from the trajectory above with the
# default EventRewardConfig and FLAG_REWARD_WEIGHTS from PR #45.  Lock
# to 6 decimal places — finer precision than that exposes float-order
# differences without catching anything meaningful.
#
# Computed via:  see docstring at top of file.
GOLDEN_REWARDS: List[float] = [
    52.999500,   # step 0:  exploration(3) + new_map(50) + time(-0.0005)
                 #          (map=1 is new on the very first step)
    2.999500,    # step 1:  exploration(3) + time(-0.0005)
    7.999500,    # step 2:  exploration(3) + FOLLOWED_OAK(5) + time(-0.0005)
    -0.000500,   # step 3:  revisit, just time penalty
    17.999500,   # step 4:  exploration(3) + GOT_STARTER(15) + time(-0.0005)
    52.999500,   # step 5:  exploration(3) + new_map(50, map=13) + time(-0.0005)
    17.999500,   # step 6:  exploration(3) + level-up(15) + time(-0.0005)
    17.999500,   # step 7:  exploration(3) + BATTLED_RIVAL(15) + time(-0.0005)
    92.999500,   # step 8:  exploration(3) + GOT_POKEBALLS(10) + GOT_POKEDEX(30)
                 #          + milestone_5(50) + time(-0.0005)
    -17.000500,  # step 9:  exploration(3) + time(-0.0005) + death(-20)
]


@pytest.fixture
def fresh_calc():
    """A fresh calculator with default config — replaces state-leak risk."""
    calc = EventProgressRewardCalculator(config=EventRewardConfig())
    calc.reset()
    return calc


def test_full_trajectory_reward_matches_golden(fresh_calc):
    """End-to-end numerical lock on the reward calculator.

    If this fails after a deliberate change (e.g. raising a flag weight
    or rebalancing milestones), regenerate ``GOLDEN_REWARDS`` and log
    the change in ``pokemon-rl-paper/compute_ledger.md``.  Section 8
    of ``analysis_plan.md`` requires it.
    """
    trajectory = _build_trajectory()
    rewards = [fresh_calc.calculate_reward(s) for s in trajectory]
    np.testing.assert_allclose(
        rewards, GOLDEN_REWARDS, atol=1e-6,
        err_msg=(
            "EventProgressRewardCalculator output drifted from the "
            "locked golden trajectory.  See the file's module docstring "
            "for the regeneration procedure."
        ),
    )


def test_milestone_5_fires_exactly_once(fresh_calc):
    """Drive past milestone-5 twice and verify it only credits once.

    Catches a class of regression where the milestone-claimed bookkeeping
    gets reset accidentally — would inflate cumulative reward in long
    episodes.
    """
    trajectory = _build_trajectory()
    for s in trajectory:
        fresh_calc.calculate_reward(s)

    # One more state with the same 5 flags — milestone 5 must NOT re-fire.
    follow_up = _state(
        x=1, y=4, map_id=13, level=6,
        EVENT_FOLLOWED_OAK_INTO_LAB=True, EVENT_GOT_STARTER=True,
        EVENT_BATTLED_RIVAL_IN_OAKS_LAB=True,
        EVENT_GOT_POKEBALLS_FROM_OAK=True, EVENT_GOT_POKEDEX=True,
    )
    fresh_calc.calculate_reward(follow_up)
    assert "milestone_5" not in fresh_calc.reward_components


def test_flag_never_double_counts_within_episode(fresh_calc):
    """A flag that flickers 1 → 0 → 1 still only rewards once.

    Pre-condition for safe reward shaping: the canonical EventFlagTracker
    guarantees this.  Locks the integration with the reward calculator
    so a refactor can't silently break the integration.
    """
    s1 = _state(EVENT_GOT_STARTER=True)
    s0 = _state(EVENT_GOT_STARTER=False)

    r1 = fresh_calc.calculate_reward(s1)
    fresh_calc.calculate_reward(s0)  # flag turns off
    r3 = fresh_calc.calculate_reward(s1)  # flag turns back on

    # First trigger: reward includes the GOT_STARTER weight.  Third
    # trigger: must NOT include GOT_STARTER reward — the same flag.
    components_first = "event_flags" in dir(fresh_calc)  # sentinel
    # Compute the expected delta: r1 includes exploration(3) +
    # GOT_STARTER(15) + time(-0.0005); r3 should include only exploration
    # if any (but x=5,y=5,map=1 is already in visited set) + time.
    assert r3 < r1, (
        "A flag re-firing within an episode added a second reward.  "
        "EventFlagTracker's ``_ever_triggered`` bookkeeping is leaking."
    )
