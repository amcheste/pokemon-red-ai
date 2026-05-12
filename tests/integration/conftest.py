"""
Shared fixtures for integration tests.

The integration suite tests behaviour that crosses module boundaries:
- the seeding chain through SB3
- the reward calculator on a synthetic trajectory
- the eval-script's metric schema
- runtime performance of the model-env loop

None of these require a real ROM or PyBoy — they use the lightweight
mock env below.  Tests that need a real ROM are gated by
``@pytest.mark.rom`` and live elsewhere.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
import numpy as np
import pytest


class DeterministicMockEnv(gym.Env):
    """A minimal gymnasium env that is fully deterministic given its seed.

    Used by the determinism and performance tests so any nondeterminism
    observed in the action sequence has to come from the policy / seeding
    chain, not from the environment itself.  Observations are uniform
    random in [0, 1] driven by a per-env :class:`numpy.random.Generator`
    that is reseeded on every ``reset(seed=...)`` call.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        obs_dim: int = 8,
        n_actions: int = 4,
        episode_len: int = 64,
    ) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(n_actions)
        self.episode_len = episode_len
        self._rng: Optional[np.random.Generator] = None
        self._step = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        # Per-env Generator with explicit seed so two envs with the same
        # seed produce bit-identical observations.  Fallback to 0 if
        # the caller didn't pass a seed.
        self._rng = np.random.default_rng(seed if seed is not None else 0)
        self._step = 0
        return self._observe(), {}

    def step(
        self, action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step += 1
        # Reward is a deterministic function of (action, step), so it
        # encodes whatever action sequence the policy produces.
        reward = float(action) * 0.01 + self._step * 0.0001
        terminated = self._step >= self.episode_len
        truncated = False
        return self._observe(), reward, terminated, truncated, {"step": self._step}

    def _observe(self) -> np.ndarray:
        assert self._rng is not None
        return self._rng.random(self.observation_space.shape, dtype=np.float32)


@pytest.fixture
def mock_env():
    """Single-instance DeterministicMockEnv (raw, no VecEnv wrapper)."""
    return DeterministicMockEnv()


@pytest.fixture
def mock_env_factory():
    """A callable that returns fresh DeterministicMockEnv instances.

    Use this when constructing a ``DummyVecEnv`` / ``SubprocVecEnv``.
    """
    return lambda: DeterministicMockEnv()
