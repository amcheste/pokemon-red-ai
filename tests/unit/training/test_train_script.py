"""
Unit tests for the scripts/train.py training script.

Tests argument parsing, seeding, and config assembly without actually
running a training loop (no ROM or PyBoy needed).
"""

import pytest
import numpy as np
from unittest.mock import patch

# The script adds project root to sys.path itself, but for test
# collection we import directly.
from scripts.train import build_parser, set_global_seeds


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
    """Test reproducibility seeding."""

    def test_numpy_deterministic(self):
        set_global_seeds(123)
        a = np.random.rand(5)
        set_global_seeds(123)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        set_global_seeds(1)
        a = np.random.rand(5)
        set_global_seeds(2)
        b = np.random.rand(5)
        assert not np.array_equal(a, b)

    def test_torch_deterministic(self):
        import torch
        set_global_seeds(42)
        a = torch.rand(5)
        set_global_seeds(42)
        b = torch.rand(5)
        assert torch.equal(a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
