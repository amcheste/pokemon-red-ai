"""
Performance regression smoke test.

``compute_plan.md`` budgets the 33-hour 9-pilot grid on a measured
~91 fps single-env throughput (~250-350 fps with 4 parallel envs).
The dominant cost is PyBoy emulation, not the policy network.  This
test does NOT exercise PyBoy — it exercises the surrounding code
(SB3 predict + env.step wrappers + VecEnv plumbing) to catch
regressions in the *non-PyBoy* portion of the loop.

What this catches:

- A future change that adds per-step torch tensor allocation in the
  policy forward pass (memory churn → 10× slowdown).
- A bug where ``model.predict`` accidentally re-builds the LSTM state
  every call rather than caching it.
- A VecEnv refactor that adds per-step subprocess IPC even on
  DummyVecEnv.

Loose threshold (≥50 fps over 500 iterations on CPU) chosen to avoid
flakiness on slow CI runners while still catching catastrophic
regressions.  On an M3 Max this runs in ~50 ms.  If you see this test
take >10 s on CI, investigate.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from stable_baselines3.common.vec_env import DummyVecEnv


# Minimum acceptable throughput for the non-PyBoy portion of the loop.
# Catastrophically loose on purpose — anything below this is a bug,
# not just a slow CI runner.  PyBoy itself does ~91 fps; this test
# excludes PyBoy so the floor is much higher.
_MIN_FPS = 50.0
_N_STEPS = 500


@pytest.mark.slow
def test_predict_step_loop_throughput(mock_env_factory):
    """Time ``model.predict()`` + ``env.step()`` for 500 iterations on
    a DummyVecEnv-wrapped mock env.  Must exceed ``_MIN_FPS``.
    """
    from sb3_contrib import RecurrentPPO

    env = DummyVecEnv([mock_env_factory])
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=0,
        n_steps=16,
        batch_size=16,
        learning_rate=1e-4,
        device="cpu",
    )

    obs = env.reset()
    state = None
    starts = np.ones((env.num_envs,), dtype=bool)

    # Warmup — torch / numpy lazy-init, dispatch caching, etc.
    for _ in range(20):
        action, state = model.predict(
            obs, state=state, episode_start=starts, deterministic=False,
        )
        starts = np.zeros_like(starts)
        obs, _, _, _ = env.step(action)

    # Timed loop.
    state, starts = None, np.ones((env.num_envs,), dtype=bool)
    obs = env.reset()
    t0 = time.perf_counter()
    for _ in range(_N_STEPS):
        action, state = model.predict(
            obs, state=state, episode_start=starts, deterministic=False,
        )
        starts = np.zeros_like(starts)
        obs, _, _, _ = env.step(action)
    elapsed = time.perf_counter() - t0

    fps = _N_STEPS / elapsed
    # The assertion message includes the actual fps so a CI failure
    # makes the regression visible without re-running locally.
    assert fps >= _MIN_FPS, (
        f"RecurrentPPO predict+step throughput regressed: "
        f"{fps:.1f} fps over {_N_STEPS} iterations "
        f"({elapsed * 1000 / _N_STEPS:.2f} ms/step), "
        f"floor is {_MIN_FPS} fps.  "
        "This excludes PyBoy entirely — the regression is in SB3, "
        "the VecEnv wrapper, or torch.  See compute_plan.md for the "
        "expected baseline."
    )


@pytest.mark.slow
def test_reward_calculator_throughput():
    """The reward calculator runs once per env-step on every parallel
    env, so its per-call cost matters at scale.  Must process at
    least 10k state dicts per second (~100 µs each) on CPU.
    """
    from pokemon_red_ai.environment.rewards import (
        EventProgressRewardCalculator,
        EventRewardConfig,
    )

    calc = EventProgressRewardCalculator(config=EventRewardConfig())
    calc.reset()

    # A representative state dict — exercises the slowest path
    # (event-flag iteration + exploration set lookups).
    state = {
        "position": {"x": 5, "y": 5, "map": 1},
        "stats": {
            "level": 5, "badges": 0, "party_count": 1,
            "current_hp": 20, "max_hp": 20,
        },
        "event_flags": {
            "EVENT_FOLLOWED_OAK_INTO_LAB": True,
            "EVENT_GOT_STARTER": False,
        },
    }

    n = 10_000
    t0 = time.perf_counter()
    for i in range(n):
        # Vary x so exploration set grows linearly — closer to real
        # env behaviour.
        state["position"]["x"] = i % 200
        calc.calculate_reward(state)
    elapsed = time.perf_counter() - t0

    per_call_us = (elapsed / n) * 1e6
    assert per_call_us < 100.0, (
        f"EventProgressRewardCalculator.calculate_reward is slow: "
        f"{per_call_us:.1f} µs/call (target <100 µs).  "
        "At 4 parallel envs × 100M steps × 100 µs = 40k seconds "
        "of pure reward-calc cost — meaningful at pilot scale."
    )
