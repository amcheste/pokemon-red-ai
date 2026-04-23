"""
Unit tests for scripts/create_save_states.py.

Tests CLI parsing, validation logic, and state creation flow
using mocked PyBoy so no ROM is needed.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open

from scripts.create_save_states import (
    build_parser,
    validate_save_state,
    validate_all,
    create_post_intro_state,
    SAVE_STATE_SPECS,
    REDS_HOUSE_2F,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def save_dir(tmp_path):
    """Provide a temporary save directory."""
    d = tmp_path / "save_states"
    d.mkdir()
    return str(d)


@pytest.fixture
def fake_state_file(save_dir):
    """Create a fake .state file with some content."""
    path = os.path.join(save_dir, "s0_post_intro.state")
    with open(path, "wb") as f:
        f.write(b"\x00" * 1024)  # Fake state data
    return path


@pytest.fixture
def mock_agent():
    """Create a mock PokemonRedAgent."""
    agent = MagicMock()
    agent.memory = MagicMock()
    agent.pyboy = MagicMock()
    agent.load_save_state = Mock(return_value=True)
    agent.save_save_state = Mock(return_value=True)
    agent.run_opening_sequence = Mock(return_value=True)
    agent.wait_frames = Mock()
    return agent


# ──────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ──────────────────────────────────────────────────────────────────────


class TestArgumentParser:
    def test_requires_rom(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["--rom", "game.gb"])
        assert args.rom == "game.gb"
        assert args.save_dir == "save_states"
        assert args.interactive is False
        assert args.validate_only is False
        assert args.skip_validation is False
        assert args.save_name is None

    def test_interactive_mode(self):
        parser = build_parser()
        args = parser.parse_args(["--rom", "game.gb", "--interactive"])
        assert args.interactive is True

    def test_validate_only(self):
        parser = build_parser()
        args = parser.parse_args(["--rom", "game.gb", "--validate-only"])
        assert args.validate_only is True

    def test_interactive_and_validate_mutually_exclusive(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--rom", "game.gb",
                "--interactive",
                "--validate-only",
            ])

    def test_custom_save_dir(self):
        parser = build_parser()
        args = parser.parse_args(["--rom", "game.gb", "--save-dir", "/tmp/states"])
        assert args.save_dir == "/tmp/states"

    def test_save_name(self):
        parser = build_parser()
        args = parser.parse_args([
            "--rom", "game.gb",
            "--interactive",
            "--save-name", "s2_pre_brock.state",
        ])
        assert args.save_name == "s2_pre_brock.state"

    def test_skip_validation(self):
        parser = build_parser()
        args = parser.parse_args(["--rom", "game.gb", "--skip-validation"])
        assert args.skip_validation is True


# ──────────────────────────────────────────────────────────────────────
# Save state spec definitions
# ──────────────────────────────────────────────────────────────────────


class TestSaveStateSpecs:
    def test_post_intro_spec_exists(self):
        assert "s0_post_intro" in SAVE_STATE_SPECS

    def test_all_specs_have_required_keys(self):
        required = {"filename", "description", "expected_map_id", "expected_badges"}
        for key, spec in SAVE_STATE_SPECS.items():
            assert required.issubset(spec.keys()), f"Spec {key} missing keys"

    def test_filenames_end_with_state(self):
        for key, spec in SAVE_STATE_SPECS.items():
            assert spec["filename"].endswith(".state"), (
                f"Spec {key} filename should end with .state"
            )

    def test_post_intro_expects_player_bedroom(self):
        spec = SAVE_STATE_SPECS["s0_post_intro"]
        assert spec["expected_map_id"] == REDS_HOUSE_2F  # Player's bedroom
        assert spec["expected_badges"] == 0

    def test_post_brock_expects_boulder_badge(self):
        spec = SAVE_STATE_SPECS["s3_post_brock"]
        assert spec["expected_badges"] == 0x01  # Boulder badge


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────


class TestValidation:
    def test_missing_file_returns_false(self, save_dir):
        spec = SAVE_STATE_SPECS["s0_post_intro"]
        result = validate_save_state(
            "game.gb",
            os.path.join(save_dir, "nonexistent.state"),
            spec,
        )
        assert result is False

    def test_empty_file_returns_false(self, save_dir):
        path = os.path.join(save_dir, "empty.state")
        with open(path, "wb") as f:
            pass  # 0 bytes

        spec = SAVE_STATE_SPECS["s0_post_intro"]
        result = validate_save_state("game.gb", path, spec)
        assert result is False

    @patch("scripts.create_save_states.PokemonRedAgent")
    def test_valid_post_intro_passes(self, MockAgent, fake_state_file):
        agent = MagicMock()
        MockAgent.return_value = agent

        agent.load_save_state.return_value = True
        # Map ID 1 = Pallet Town, position (5, 3)
        agent.memory.__getitem__ = lambda self, addr: {
            0xD35E: REDS_HOUSE_2F,  # map_id = Player's bedroom
            0xD362: 9,    # player_x
            0xD361: 4,    # player_y
            0xD18C: 0,    # level (no Pokemon yet)
            0xD16C: 0,    # current_hp_low
            0xD16D: 0,    # current_hp_high
            0xD16E: 0,    # max_hp_low
            0xD16F: 0,    # max_hp_high
            0xD356: 0,    # badges = 0
            0xD163: 0,    # party_count = 0
        }.get(addr, 0)

        spec = SAVE_STATE_SPECS["s0_post_intro"]
        result = validate_save_state("game.gb", fake_state_file, spec)
        assert result is True

    @patch("scripts.create_save_states.PokemonRedAgent")
    def test_wrong_map_fails(self, MockAgent, fake_state_file):
        agent = MagicMock()
        MockAgent.return_value = agent

        agent.load_save_state.return_value = True
        # Map ID 3 = Pewter City (wrong for post-intro)
        agent.memory.__getitem__ = lambda self, addr: {
            0xD35E: 3,    # map_id = Pewter (wrong!)
            0xD362: 5,
            0xD361: 3,
            0xD18C: 5,
            0xD16C: 20,
            0xD16D: 0,
            0xD16E: 20,
            0xD16F: 0,
            0xD356: 0,
            0xD163: 0,
        }.get(addr, 0)

        spec = SAVE_STATE_SPECS["s0_post_intro"]
        result = validate_save_state("game.gb", fake_state_file, spec)
        assert result is False

    @patch("scripts.create_save_states.PokemonRedAgent")
    def test_load_failure_returns_false(self, MockAgent, fake_state_file):
        agent = MagicMock()
        MockAgent.return_value = agent
        agent.load_save_state.return_value = False

        spec = SAVE_STATE_SPECS["s0_post_intro"]
        result = validate_save_state("game.gb", fake_state_file, spec)
        assert result is False

    @patch("scripts.create_save_states.PokemonRedAgent")
    def test_agent_exception_returns_false(self, MockAgent, fake_state_file):
        MockAgent.side_effect = RuntimeError("PyBoy init failed")

        spec = SAVE_STATE_SPECS["s0_post_intro"]
        result = validate_save_state("game.gb", fake_state_file, spec)
        assert result is False

    @patch("scripts.create_save_states.PokemonRedAgent")
    def test_pyboy_stop_called_on_cleanup(self, MockAgent, fake_state_file):
        agent = MagicMock()
        MockAgent.return_value = agent
        agent.load_save_state.return_value = True
        agent.memory.__getitem__ = lambda self, addr: {
            0xD35E: REDS_HOUSE_2F, 0xD362: 9, 0xD361: 4,
            0xD18C: 0, 0xD16C: 0, 0xD16D: 0,
            0xD16E: 0, 0xD16F: 0, 0xD356: 0, 0xD163: 0,
        }.get(addr, 0)

        spec = SAVE_STATE_SPECS["s0_post_intro"]
        validate_save_state("game.gb", fake_state_file, spec)
        agent.pyboy.stop.assert_called_once()


class TestValidateAll:
    @patch("scripts.create_save_states.validate_save_state")
    def test_skips_missing_files(self, mock_validate, save_dir):
        results = validate_all("game.gb", save_dir)
        # No files exist, so all should be None
        for val in results.values():
            assert val is None
        mock_validate.assert_not_called()

    @patch("scripts.create_save_states.validate_save_state", return_value=True)
    def test_validates_existing_files(self, mock_validate, fake_state_file, save_dir):
        results = validate_all("game.gb", save_dir)
        assert results["s0_post_intro"] is True
        mock_validate.assert_called_once()


# ──────────────────────────────────────────────────────────────────────
# State creation
# ──────────────────────────────────────────────────────────────────────


class TestCreatePostIntro:
    @patch("scripts.create_save_states.PokemonRedAgent")
    @patch("scripts.create_save_states.read_player_position")
    @patch("scripts.create_save_states.read_player_stats")
    def test_creates_state_on_success(
        self, mock_stats, mock_pos, MockAgent, save_dir
    ):
        agent = MagicMock()
        MockAgent.return_value = agent

        agent.run_opening_sequence.return_value = True
        agent.save_save_state.return_value = True
        mock_pos.return_value = {"map": REDS_HOUSE_2F, "x": 9, "y": 4}
        mock_stats.return_value = {
            "party_count": 0, "badges": 0, "level": 0,
            "current_hp": 0, "max_hp": 0, "hp_ratio": 0,
        }

        result = create_post_intro_state("game.gb", save_dir)
        assert result is True
        agent.run_opening_sequence.assert_called_once()
        agent.save_save_state.assert_called_once()

    @patch("scripts.create_save_states.PokemonRedAgent")
    def test_fails_when_opening_fails(self, MockAgent, save_dir):
        agent = MagicMock()
        MockAgent.return_value = agent
        agent.run_opening_sequence.return_value = False

        result = create_post_intro_state("game.gb", save_dir)
        assert result is False
        agent.save_save_state.assert_not_called()

    @patch("scripts.create_save_states.PokemonRedAgent")
    @patch("scripts.create_save_states.read_player_position")
    @patch("scripts.create_save_states.read_player_stats")
    def test_fails_when_map_id_zero(
        self, mock_stats, mock_pos, MockAgent, save_dir
    ):
        agent = MagicMock()
        MockAgent.return_value = agent
        agent.run_opening_sequence.return_value = True
        mock_pos.return_value = {"map": 0, "x": 0, "y": 0}
        mock_stats.return_value = {
            "party_count": 0, "badges": 0, "level": 0,
            "current_hp": 0, "max_hp": 0, "hp_ratio": 0,
        }

        result = create_post_intro_state("game.gb", save_dir)
        assert result is False

    @patch("scripts.create_save_states.PokemonRedAgent")
    @patch("scripts.create_save_states.read_player_position")
    @patch("scripts.create_save_states.read_player_stats")
    def test_fails_when_save_fails(
        self, mock_stats, mock_pos, MockAgent, save_dir
    ):
        agent = MagicMock()
        MockAgent.return_value = agent
        agent.run_opening_sequence.return_value = True
        agent.save_save_state.return_value = False
        mock_pos.return_value = {"map": REDS_HOUSE_2F, "x": 9, "y": 4}
        mock_stats.return_value = {
            "party_count": 0, "badges": 0, "level": 0,
            "current_hp": 0, "max_hp": 0, "hp_ratio": 0,
        }

        result = create_post_intro_state("game.gb", save_dir)
        assert result is False

    @patch("scripts.create_save_states.PokemonRedAgent")
    def test_pyboy_cleaned_up_on_failure(self, MockAgent, save_dir):
        agent = MagicMock()
        MockAgent.return_value = agent
        agent.run_opening_sequence.side_effect = RuntimeError("boom")

        create_post_intro_state("game.gb", save_dir)
        agent.pyboy.stop.assert_called_once()

    @patch("scripts.create_save_states.PokemonRedAgent")
    @patch("scripts.create_save_states.read_player_position")
    @patch("scripts.create_save_states.read_player_stats")
    def test_saves_to_correct_path(
        self, mock_stats, mock_pos, MockAgent, save_dir
    ):
        agent = MagicMock()
        MockAgent.return_value = agent
        agent.run_opening_sequence.return_value = True
        agent.save_save_state.return_value = True
        mock_pos.return_value = {"map": REDS_HOUSE_2F, "x": 9, "y": 4}
        mock_stats.return_value = {
            "party_count": 0, "badges": 0, "level": 0,
            "current_hp": 0, "max_hp": 0, "hp_ratio": 0,
        }

        create_post_intro_state("game.gb", save_dir)

        expected_path = os.path.join(save_dir, "s0_post_intro.state")
        agent.save_save_state.assert_called_once_with(expected_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
