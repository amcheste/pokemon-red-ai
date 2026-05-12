"""
Full-stack seeding determinism — the gate on pilot reproducibility.

The paper claims that the three reported seeds (42, 123, 456) are
genuine replicates of the same trajectory distribution.  That claim
requires that two invocations with the same seed, same code, and same
hardware produce bit-identical action sequences.

These tests verify that the seeding chain in
``scripts/seed_utils.seed_everything`` plus ``VecEnv.seed(seed)`` plus
``RecurrentPPO(seed=...)`` actually achieves that, end-to-end, without
needing a real ROM.  If a future change to the SB3 stack or to the
seed_utils function silently breaks determinism, these tests fail
loudly — *before* anyone burns 33 GPU-hours on a non-reproducible
pilot grid.

The mock env (see ``conftest.py``) generates observations from a
seeded numpy Generator, so the only sources of stochasticity that
remain are the policy network init and the action sampler — exactly
the things ``seed_everything`` is supposed to nail down.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

# Project root on sys.path so scripts.seed_utils imports cleanly even
# when the test runs from the integration/ subdir.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.seed_utils import seed_everything
from stable_baselines3.common.vec_env import DummyVecEnv


def _run_episode(seed: int, mock_env_factory, n_actions: int = 16) -> List[int]:
    """Build a fresh RecurrentPPO + DummyVecEnv with the given seed and
    return the first ``n_actions`` actions the policy emits.

    Built per-call so each invocation goes through the full seeding
    chain exactly the way ``scripts/train.py`` does.
    """
    from sb3_contrib import RecurrentPPO

    seed_everything(seed)
    env = DummyVecEnv([mock_env_factory])
    env.seed(seed)

    # Tiny hyperparameters so the test runs in well under a second.
    # n_steps must be ≥ batch_size; we don't actually train here so
    # specific values don't matter beyond letting the constructor pass.
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        seed=seed,
        verbose=0,
        n_steps=16,
        batch_size=16,
        learning_rate=1e-4,
        device="cpu",  # CPU is deterministic; CUDA/MPS may not be.
    )

    obs = env.reset()
    state = None
    starts = np.ones((env.num_envs,), dtype=bool)
    actions: List[int] = []
    for _ in range(n_actions):
        action, state = model.predict(
            obs, state=state, episode_start=starts, deterministic=False,
        )
        starts = np.zeros_like(starts)
        obs, _, dones, _ = env.step(action)
        actions.append(int(action[0]))
        if dones[0]:
            starts = dones.astype(bool)

    return actions


@pytest.mark.slow
def test_same_seed_produces_same_actions(mock_env_factory):
    """Two RecurrentPPO instances with the same seed must emit
    bit-identical action sequences on a deterministic mock env.

    This is the load-bearing claim that makes the pilot grid scientific.
    """
    actions_a = _run_episode(seed=42, mock_env_factory=mock_env_factory)
    actions_b = _run_episode(seed=42, mock_env_factory=mock_env_factory)

    assert actions_a == actions_b, (
        f"Seeding chain is not deterministic.\n"
        f"  Run A: {actions_a}\n"
        f"  Run B: {actions_b}\n"
        "Investigate: seed_everything coverage (Python random, NumPy, "
        "torch, PYTHONHASHSEED, SB3), VecEnv.seed() per-rank propagation, "
        "and any cudnn / MPS nondeterminism.  This MUST hold before "
        "pilots launch — the paper's three-seed design assumes it."
    )


@pytest.mark.slow
def test_different_seeds_produce_different_actions(mock_env_factory):
    """Sanity check: seeding actually does something.

    If seed_everything were a no-op, both runs would still match
    because the underlying RNGs are time-default.  This test catches
    the case where the seeding chain is missed entirely (e.g. a
    refactor that drops the seed_everything call) by asserting that
    different seeds produce visibly different outputs.
    """
    actions_42 = _run_episode(seed=42, mock_env_factory=mock_env_factory)
    actions_123 = _run_episode(seed=123, mock_env_factory=mock_env_factory)

    # Two random 16-length sequences over 4 actions match with
    # probability 4^-16 ≈ 2.3e-10; if they ever do, we have bigger
    # problems than this test failing.
    assert actions_42 != actions_123, (
        "Seeding chain looks broken: seeds 42 and 123 produced identical "
        "action sequences.  Likely cause: seed_everything is being "
        "called but its result isn't propagating to the SB3 policy "
        "(check `seed=` is actually being passed to the model constructor)."
    )


@pytest.mark.slow
def test_env_seed_propagates_per_rank(mock_env_factory):
    """``VecEnv.seed(seed)`` sends ``seed + rank`` to each child.

    For a 4-env DummyVecEnv, the initial observations across the 4
    sub-envs must differ from each other (rank-decorrelation) and the
    full set must be reproducible across runs with the same base seed.
    """
    def first_obs_per_rank(seed: int) -> List[bytes]:
        env = DummyVecEnv([mock_env_factory for _ in range(4)])
        env.seed(seed)
        obs = env.reset()
        # Hash the per-env observation so we compare reproducibly.
        return [obs[i].tobytes() for i in range(4)]

    run_a = first_obs_per_rank(seed=42)
    run_b = first_obs_per_rank(seed=42)
    run_c = first_obs_per_rank(seed=999)

    # Same seed → identical observations across runs.
    assert run_a == run_b, (
        "VecEnv.seed(seed) is not deterministic across runs with the "
        "same base seed."
    )
    # Within a run, different ranks must produce different observations
    # (otherwise SubprocVecEnv parallel rollouts would just be the same
    # trajectory copied N times).
    assert len(set(run_a)) == 4, (
        "VecEnv.seed(seed) is not propagating `seed + rank` to each "
        "child env — all 4 ranks produced the same observation.  This "
        "would silently correlate parallel-env rollouts."
    )
    # And different base seeds must produce different observations
    # somewhere (rank-0 is enough).
    assert run_a[0] != run_c[0], (
        "Base seed has no effect on rank-0 observations."
    )
