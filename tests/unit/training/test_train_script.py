"""
Unit tests for the scripts/train.py training script.

Tests argument parsing, seeding, and config assembly without actually
running a training loop (no ROM or PyBoy needed).
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# The script adds project root to sys.path itself, but for test
# collection we import directly.
from scripts.train import (
    _make_env_factory,
    _make_vec_env,
    build_parser,
)
# set_global_seeds was removed; seed_everything is the canonical replacement
# and lives in scripts.seed_utils.  It seeds Python random, NumPy, torch
# (CPU/CUDA/MPS), PYTHONHASHSEED, and SB3 in one call.
from scripts.seed_utils import seed_everything


class TestArgumentParser:
    """Test CLI argument parsing."""

    def test_required_rom_argument(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])  # missing --rom

    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["--rom", "test.gb"])

        assert args.rom == "test.gb"
        assert args.algorithm == "RecurrentPPO"
        assert args.reward_strategy == "events"
        assert args.observation_type == "multi_modal"
        assert args.total_timesteps == 1_000_000
        assert args.max_episode_steps == 15_000
        assert args.save_freq == 50_000
        assert args.wandb_project == "pokemon-red-ai"
        assert args.no_wandb is False
        assert args.seed is None
        assert args.show_game is False

    def test_override_algorithm(self):
        parser = build_parser()
        args = parser.parse_args(["--rom", "test.gb", "--algorithm", "PPO"])
        assert args.algorithm == "PPO"

    def test_override_seed(self):
        parser = build_parser()
        args = parser.parse_args(["--rom", "test.gb", "--seed", "42"])
        assert args.seed == 42

    def test_no_wandb_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--rom", "test.gb", "--no-wandb"])
        assert args.no_wandb is True

    def test_save_state_option(self):
        parser = build_parser()
        args = parser.parse_args([
            "--rom", "test.gb",
            "--save-state", "states/intro.state",
        ])
        assert args.save_state == "states/intro.state"

    def test_learning_rate_override(self):
        parser = build_parser()
        args = parser.parse_args([
            "--rom", "test.gb",
            "--learning-rate", "0.0001",
        ])
        assert args.learning_rate == pytest.approx(0.0001)

    def test_lstm_hidden_size(self):
        parser = build_parser()
        args = parser.parse_args([
            "--rom", "test.gb",
            "--lstm-hidden-size", "512",
        ])
        assert args.lstm_hidden_size == 512

    def test_n_envs_default_is_one(self):
        parser = build_parser()
        args = parser.parse_args(["--rom", "test.gb"])
        assert args.n_envs == 1

    def test_n_envs_override(self):
        parser = build_parser()
        args = parser.parse_args([
            "--rom", "test.gb",
            "--n-envs", "8",
        ])
        assert args.n_envs == 8

    @pytest.mark.parametrize("strategy", [
        "standard", "exploration", "progress", "sparse", "events",
    ])
    def test_reward_strategy_choices(self, strategy):
        parser = build_parser()
        args = parser.parse_args([
            "--rom", "test.gb",
            "--reward-strategy", strategy,
        ])
        assert args.reward_strategy == strategy

    def test_invalid_algorithm_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--rom", "test.gb", "--algorithm", "DQN"])

    def test_wandb_entity(self):
        parser = build_parser()
        args = parser.parse_args([
            "--rom", "test.gb",
            "--wandb-entity", "my-team",
        ])
        assert args.wandb_entity == "my-team"


class TestGlobalSeeds:
    """Test reproducibility seeding via scripts.seed_utils.seed_everything."""

    def test_numpy_deterministic(self):
        seed_everything(123)
        a = np.random.rand(5)
        seed_everything(123)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        seed_everything(1)
        a = np.random.rand(5)
        seed_everything(2)
        b = np.random.rand(5)
        assert not np.array_equal(a, b)

    def test_torch_deterministic(self):
        import torch
        seed_everything(42)
        a = torch.rand(5)
        seed_everything(42)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_python_random_seeded(self):
        """seed_everything must seed Python's stdlib random too — train.py's
        previous set_global_seeds did, and SB3 relies on it for some ops."""
        import random
        seed_everything(7)
        a = [random.random() for _ in range(5)]
        seed_everything(7)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_python_hash_seed_set(self):
        """PYTHONHASHSEED must be set so dict-iteration order is reproducible
        in subprocess children of SubprocVecEnv."""
        import os
        seed_everything(99)
        assert os.environ["PYTHONHASHSEED"] == "99"

    def test_per_rank_offset(self):
        """seed_offset shifts the effective seed so parallel envs get
        independent but reproducible streams."""
        seed_everything(10, seed_offset=0)
        a = np.random.rand(3)
        seed_everything(10, seed_offset=1)
        b = np.random.rand(3)
        assert not np.array_equal(a, b)
        # Rank-1 should equal seed=11 with no offset.
        seed_everything(11, seed_offset=0)
        c = np.random.rand(3)
        np.testing.assert_array_equal(b, c)


# ──────────────────────────────────────────────────────────────────────
# Vectorised environment construction
# ──────────────────────────────────────────────────────────────────────


def _fake_args(tmp_path, **overrides):
    """Build a minimal args namespace for env-factory tests."""
    import argparse
    args = argparse.Namespace(
        rom="fake.gb",
        save_state=None,
        max_episode_steps=15_000,
        reward_strategy="events",
        observation_type="pixel",
        save_dir=str(tmp_path),
        show_game=False,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class TestMakeEnvFactory:
    """The factory must be picklable and produce a Monitor-wrapped env."""

    def test_factory_is_callable(self, tmp_path):
        args = _fake_args(tmp_path)
        factory = _make_env_factory(0, args, reward_config=None)
        assert callable(factory)

    def test_factory_uses_per_rank_monitor_path(self, tmp_path, monkeypatch):
        """Two envs at different ranks must write to different monitor files
        so SubprocVecEnv copies don't clobber each other's logs."""
        args = _fake_args(tmp_path)

        captured_monitor_paths = []

        def fake_env(*a, **kw):
            return MagicMock()

        def fake_monitor(env, filename, **kwargs):
            captured_monitor_paths.append(filename)
            return env

        monkeypatch.setattr("scripts.train.PokemonRedGymEnv", fake_env)
        monkeypatch.setattr("scripts.train.Monitor", fake_monitor)

        _make_env_factory(0, args, None)()
        _make_env_factory(1, args, None)()

        assert len(captured_monitor_paths) == 2
        assert captured_monitor_paths[0] != captured_monitor_paths[1]
        assert captured_monitor_paths[0].endswith("monitor.0")
        assert captured_monitor_paths[1].endswith("monitor.1")

    def test_factory_passes_save_state_path(self, tmp_path, monkeypatch):
        args = _fake_args(tmp_path, save_state="states/foo.state")
        captured_kwargs = {}

        def fake_env(**kw):
            captured_kwargs.update(kw)
            return MagicMock()

        monkeypatch.setattr("scripts.train.PokemonRedGymEnv", fake_env)
        monkeypatch.setattr(
            "scripts.train.Monitor", lambda env, *a, **kw: env
        )
        _make_env_factory(0, args, None)()
        assert captured_kwargs["save_state_path"] == "states/foo.state"


class TestMakeVecEnv:
    """The dispatcher must pick DummyVecEnv vs SubprocVecEnv correctly."""

    def test_n_envs_one_uses_dummy(self, tmp_path, monkeypatch):
        """Single env path must stay on DummyVecEnv to avoid subprocess
        overhead — preserves the legacy behaviour bit-for-bit."""
        args = _fake_args(tmp_path)

        class FakeDummyVecEnv:
            def __init__(self, env_fns, **kwargs):
                self.env_fns = env_fns
                self.num_envs = len(env_fns)

        monkeypatch.setattr("scripts.train.DummyVecEnv", FakeDummyVecEnv)
        # Stub PokemonRedGymEnv + Monitor so the factory is safe to call
        # (though _make_vec_env doesn't actually invoke them at n_envs=1
        # construction time — DummyVecEnv calls them lazily).
        monkeypatch.setattr(
            "scripts.train.PokemonRedGymEnv", lambda **kw: MagicMock()
        )
        monkeypatch.setattr(
            "scripts.train.Monitor", lambda env, *a, **kw: env
        )

        vec = _make_vec_env(args, n_envs=1, reward_config=None)
        assert isinstance(vec, FakeDummyVecEnv)
        assert vec.num_envs == 1
        # Dispatcher correctly chose Dummy over Subproc.
        assert not isinstance(vec, SubprocVecEnv)

    def test_n_envs_above_one_uses_subproc(self, tmp_path, monkeypatch):
        """Multi-env path must switch to SubprocVecEnv (the whole point)."""
        args = _fake_args(tmp_path)
        # SubprocVecEnv inspection without spawning real subprocesses:
        # we patch SubprocVecEnv itself to a lightweight mock so the
        # type / arity check happens at the dispatcher level.
        captured_factories = []

        class FakeSubprocVecEnv:
            def __init__(self, env_fns, **kwargs):
                captured_factories.extend(env_fns)
                self.env_fns = env_fns
                self.num_envs = len(env_fns)

        monkeypatch.setattr("scripts.train.SubprocVecEnv", FakeSubprocVecEnv)

        vec = _make_vec_env(args, n_envs=4, reward_config=None)
        assert isinstance(vec, FakeSubprocVecEnv)
        assert vec.num_envs == 4
        assert len(captured_factories) == 4
        # Each factory must be callable and distinct (per-rank closures)
        assert all(callable(f) for f in captured_factories)
        assert len(set(id(f) for f in captured_factories)) == 4

    def test_n_envs_zero_clamps_via_train_function(self):
        """The dispatcher itself doesn't clamp; that's done in train()
        via `max(1, int(args.n_envs))`.  This documents the contract:
        callers must clamp before calling _make_vec_env."""
        # No assertion — this is a documentation test.  If clamping
        # logic moves into the dispatcher in the future, add a real
        # behavioural test here.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
