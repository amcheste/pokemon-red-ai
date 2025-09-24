"""
Unit tests for Pokemon Red AI agent module.

Tests the PokemonRedAgent class and game automation.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock

from pokemon_red_ai.game.agent import PokemonRedAgent
from pokemon_red_ai.game.memory import MEMORY_ADDRESSES
from pokemon_red_ai.game.controls import ScreenType


@pytest.fixture(autouse=True)
def mock_pyboy_class(monkeypatch, mock_pyboy):
    """Automatically mock PyBoy class for all tests."""

    def mock_pyboy_init(*args, **kwargs):
        return mock_pyboy

    monkeypatch.setattr('pokemon_red_ai.game.agent.PyBoy', mock_pyboy_init)
    return mock_pyboy

class TestAgentInitialization:
    """Test agent initialization."""

    def test_agent_init_basic(self, mock_rom_file):
        """Test basic agent initialization."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file), show_window=False)

            assert agent.rom_path == str(mock_rom_file)
            assert agent.show_window is False
            assert agent.episode_steps == 0
            assert len(agent.visited_locations) == 0
            MockPyBoy.assert_called_once()

    def test_agent_init_with_window(self, mock_rom_file):
        """Test initialization with window display."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file), show_window=True)

            assert agent.show_window is True

    def test_agent_init_with_speed_multiplier(self, mock_rom_file):
        """Test initialization with speed multiplier."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            mock_pyboy.set_emulation_speed = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file), speed_multiplier=2)

            assert agent.speed_multiplier == 2

    def test_agent_init_fallback_syntax(self, mock_rom_file):
        """Test initialization with fallback PyBoy syntax."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            # Simulate modern syntax failing
            MockPyBoy.side_effect = [TypeError("Modern syntax failed"), Mock()]

            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            MockPyBoy.return_value = mock_pyboy

            # Should still succeed with fallback
            agent = PokemonRedAgent(str(mock_rom_file))

            assert agent is not None


class TestAgentButtonControls:
    """Test agent button control methods."""

    def test_press_button(self, mock_rom_file):
        """Test pressing buttons."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            mock_pyboy.button_press = Mock()
            mock_pyboy.button_release = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))
            agent.press_button('A')

            mock_pyboy.button_press.assert_called()
            mock_pyboy.button_release.assert_called()

    def test_press_button_custom_timing(self, mock_rom_file):
        """Test pressing button with custom timing."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            mock_pyboy.button_press = Mock()
            mock_pyboy.button_release = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))
            agent.press_button('B', hold_frames=20, release_frames=10)

            # Should tick for hold + release frames
            assert mock_pyboy.tick.call_count == 30

    def test_wait_frames(self, mock_rom_file):
        """Test waiting for frames."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))
            agent.wait_frames(60)

            assert mock_pyboy.tick.call_count == 60


class TestAgentScreenMethods:
    """Test agent screen-related methods."""

    def test_get_screen_array(self, mock_rom_file):
        """Test getting screen array."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_screen = Mock()
            mock_screen.image = np.zeros((144, 160, 3), dtype=np.uint8)
            mock_pyboy.screen = mock_screen
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))
            screen = agent.get_screen_array()

            assert isinstance(screen, np.ndarray)
            assert screen.shape == (144, 160, 3)

    def test_get_tilemap_background(self, mock_rom_file):
        """Test getting tilemap."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            mock_tilemap = np.zeros((18, 20), dtype=np.uint8)
            type(mock_pyboy).tilemap_background = PropertyMock(return_value=mock_tilemap)
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))
            tilemap = agent.get_tilemap_background()

            assert isinstance(tilemap, np.ndarray)
            assert tilemap.shape == (18, 20)

    def test_get_current_screen_type(self, mock_rom_file):
        """Test getting current screen type."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_memory = Mock()
            mock_memory.__getitem__ = Mock(return_value=1)  # In game
            mock_pyboy.memory = mock_memory
            mock_pyboy.screen = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))
            screen_type = agent.get_current_screen_type()

            assert screen_type == ScreenType.IN_GAME


class TestAgentGameState:
    """Test agent game state methods."""

    def test_get_player_position(self, mock_rom_file):
        """Test getting player position."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_memory = Mock()
            mock_memory.__getitem__ = Mock(side_effect=[10, 20, 1])
            mock_pyboy.memory = mock_memory
            mock_pyboy.screen = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))
            position = agent.get_player_position()

            assert position == {'x': 10, 'y': 20, 'map': 1}

    def test_get_player_stats(self, mock_rom_file, sample_memory_state):
        """Test getting player stats."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_memory = Mock()

            def memory_side_effect(addr):
                if addr == MEMORY_ADDRESSES['player_level']:
                    return 15
                elif addr == MEMORY_ADDRESSES['current_hp_low']:
                    return 50
                elif addr == MEMORY_ADDRESSES['current_hp_high']:
                    return 0
                elif addr == MEMORY_ADDRESSES['max_hp_low']:
                    return 60
                elif addr == MEMORY_ADDRESSES['max_hp_high']:
                    return 0
                elif addr == MEMORY_ADDRESSES['badges']:
                    return 3
                elif addr == MEMORY_ADDRESSES['party_count']:
                    return 2
                return 0

            mock_memory.__getitem__ = Mock(side_effect=memory_side_effect)
            mock_pyboy.memory = mock_memory
            mock_pyboy.screen = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))
            stats = agent.get_player_stats()

            assert stats['level'] == 15
            assert stats['current_hp'] == 50
            assert stats['max_hp'] == 60
            assert stats['badges'] == 3

    def test_get_comprehensive_state(self, mock_rom_file):
        """Test getting comprehensive game state."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_memory = Mock()
            mock_memory.__getitem__ = Mock(return_value=0)
            mock_pyboy.memory = mock_memory
            mock_pyboy.screen = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))
            state = agent.get_comprehensive_state()

            assert 'position' in state
            assert 'stats' in state
            assert 'map_name' in state

    def test_is_in_game(self, mock_rom_file):
        """Test checking if in game."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_memory = Mock()
            mock_memory.__getitem__ = Mock(return_value=1)
            mock_pyboy.memory = mock_memory
            mock_pyboy.screen = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))

            assert agent.is_in_game() is True


class TestAgentStep:
    """Test agent step functionality."""

    def test_step_updates_tracking(self, mock_rom_file):
        """Test that step updates episode tracking."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.memory.__getitem__ = Mock(return_value=0)
            mock_pyboy.screen = Mock()
            mock_pyboy.button_press = Mock()
            mock_pyboy.button_release = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))
            initial_steps = agent.episode_steps

            state = agent.step('RIGHT')

            assert agent.episode_steps == initial_steps + 1
            assert isinstance(state, dict)

    def test_step_updates_visited_locations(self, mock_rom_file):
        """Test that step updates visited locations."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_memory = Mock()
            mock_memory.__getitem__ = Mock(side_effect=[5, 10, 1, 0, 0, 0, 0, 0, 0])
            mock_pyboy.memory = mock_memory
            mock_pyboy.screen = Mock()
            mock_pyboy.button_press = Mock()
            mock_pyboy.button_release = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))
            agent.step('DOWN')

            assert len(agent.visited_locations) > 0

    def test_get_exploration_progress(self, mock_rom_file):
        """Test getting exploration progress."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))
            agent.visited_locations = {(1, 2, 1), (3, 4, 1), (5, 6, 2)}
            agent.episode_steps = 100

            progress = agent.get_exploration_progress()

            assert progress['locations_visited'] == 3
            assert progress['unique_maps'] == 2
            assert progress['episode_steps'] == 100


class TestAgentGameAutomation:
    """Test game automation methods."""

    def test_skip_intro_sequence(self, mock_rom_file):
        """Test skipping intro sequence."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            mock_pyboy.button_press = Mock()
            mock_pyboy.button_release = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))

            with patch.object(agent, 'get_current_screen_type', return_value=ScreenType.IN_GAME):
                result = agent.skip_intro_sequence()

            assert result is True

    def test_run_opening_sequence(self, mock_rom_file):
        """Test running complete opening sequence."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))

            # Mock all the sub-methods
            with patch.object(agent, 'skip_intro_sequence', return_value=True), \
                    patch.object(agent, 'skip_professor_oak_intro'), \
                    patch.object(agent, 'handle_naming_screen', return_value=True), \
                    patch.object(agent, 'complete_intro_dialogue'), \
                    patch.object(agent, 'take_initial_steps'), \
                    patch.object(agent, 'get_current_screen_type', return_value=ScreenType.IN_GAME), \
                    patch.object(agent, 'get_player_position', return_value={'x': 5, 'y': 10, 'map': 1}), \
                    patch.object(agent, 'get_player_stats', return_value={'level': 5}):
                result = agent.run_opening_sequence()

            assert result is True
            assert agent.is_initialized is True


class TestAgentReset:
    """Test agent reset functionality."""

    def test_reset_game(self, mock_rom_file):
        """Test resetting the game."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            mock_pyboy.stop = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))
            agent.visited_locations.add((1, 2, 3))
            agent.episode_steps = 100

            with patch.object(agent, 'run_opening_sequence', return_value=True):
                result = agent.reset_game()

            assert result is True
            assert len(agent.visited_locations) == 0
            assert agent.episode_steps == 0

    def test_reset_game_handles_errors(self, mock_rom_file):
        """Test reset handles errors gracefully."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            mock_pyboy.stop = Mock(side_effect=Exception("Stop error"))
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))

            # Should handle error and try to continue
            with patch.object(agent, 'run_opening_sequence', return_value=True):
                result = agent.reset_game()

            # Should still attempt reset despite error
            assert isinstance(result, bool)


class TestAgentSaveStates:
    """Test save state functionality."""

    def test_save_state(self, mock_rom_file, tmp_path):
        """Test saving game state."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            mock_pyboy.save_state = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file), enable_save_states=True)

            with patch('builtins.open', create=True):
                result = agent.save_state(slot=0)

            # Should succeed if save_state enabled
            assert result is True

    def test_save_state_disabled(self, mock_rom_file):
        """Test save state when disabled."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file), enable_save_states=False)
            result = agent.save_state(slot=0)

            assert result is False

    def test_load_state(self, mock_rom_file):
        """Test loading game state."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            mock_pyboy.load_state = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file), enable_save_states=True)

            with patch('builtins.open', create=True):
                with patch('os.path.exists', return_value=True):
                    result = agent.load_state(slot=0)

            assert result is True


class TestAgentCleanup:
    """Test agent cleanup and resource management."""

    def test_cleanup(self, mock_rom_file):
        """Test cleanup method."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            mock_pyboy.stop = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))
            agent.cleanup()

            mock_pyboy.stop.assert_called_once()

    def test_cleanup_handles_errors(self, mock_rom_file):
        """Test cleanup handles errors gracefully."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            mock_pyboy.stop = Mock(side_effect=Exception("Stop error"))
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))

            # Should not raise exception
            agent.cleanup()

    def test_context_manager_enter(self, mock_rom_file):
        """Test context manager __enter__."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            MockPyBoy.return_value = mock_pyboy

            with PokemonRedAgent(str(mock_rom_file)) as agent:
                assert agent is not None
                assert isinstance(agent, PokemonRedAgent)

    def test_context_manager_exit(self, mock_rom_file):
        """Test context manager __exit__."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            mock_pyboy.stop = Mock()
            MockPyBoy.return_value = mock_pyboy

            with PokemonRedAgent(str(mock_rom_file)) as agent:
                pass

            # Should have called cleanup
            mock_pyboy.stop.assert_called()

    def test_destructor(self, mock_rom_file):
        """Test __del__ destructor."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            mock_pyboy.stop = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))
            del agent

            # Cleanup should have been called
            mock_pyboy.stop.assert_called()


class TestAgentIntegration:
    """Integration tests for agent."""

    @pytest.mark.integration
    def test_agent_full_workflow(self, mock_rom_file):
        """Test complete agent workflow."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_memory = Mock()
            mock_memory.__getitem__ = Mock(return_value=0)
            mock_pyboy.memory = mock_memory
            mock_pyboy.screen = Mock()
            mock_pyboy.button_press = Mock()
            mock_pyboy.button_release = Mock()
            MockPyBoy.return_value = mock_pyboy

            # Create agent
            agent = PokemonRedAgent(str(mock_rom_file))

            # Initialize game
            with patch.object(agent, 'run_opening_sequence', return_value=True):
                agent.run_opening_sequence()

            # Take some steps
            for _ in range(5):
                agent.step('RIGHT')

            # Check state
            assert agent.episode_steps == 5

            # Reset
            with patch.object(agent, 'run_opening_sequence', return_value=True):
                agent.reset_game()

            assert agent.episode_steps == 0

            # Cleanup
            agent.cleanup()

    @pytest.mark.integration
    def test_agent_with_exploration(self, mock_rom_file):
        """Test agent exploration tracking."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_memory = Mock()

            # Simulate movement through different positions
            positions = [(0, 0, 1), (1, 0, 1), (2, 0, 1), (2, 1, 1), (2, 1, 2)]
            position_idx = [0]

            def memory_side_effect(addr):
                x, y, map_id = positions[min(position_idx[0], len(positions) - 1)]
                if addr == MEMORY_ADDRESSES['player_x']:
                    return x
                elif addr == MEMORY_ADDRESSES['player_y']:
                    return y
                elif addr == MEMORY_ADDRESSES['map_id']:
                    return map_id
                return 0

            mock_memory.__getitem__ = Mock(side_effect=memory_side_effect)
            mock_pyboy.memory = mock_memory
            mock_pyboy.screen = Mock()
            mock_pyboy.button_press = Mock()
            mock_pyboy.button_release = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))

            # Take steps and move position
            for i in range(len(positions)):
                position_idx[0] = i
                agent.step('RIGHT')

            progress = agent.get_exploration_progress()

            assert progress['locations_visited'] == len(positions)
            assert progress['unique_maps'] == 2


class TestAgentErrorHandling:
    """Test agent error handling."""

    def test_agent_handles_memory_errors(self, mock_rom_file):
        """Test agent handles memory read errors."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_memory = Mock()
            mock_memory.__getitem__ = Mock(side_effect=Exception("Memory error"))
            mock_pyboy.memory = mock_memory
            mock_pyboy.screen = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))

            # Should handle error gracefully
            position = agent.get_player_position()
            assert isinstance(position, dict)

    def test_agent_handles_screen_errors(self, mock_rom_file):
        """Test agent handles screen read errors."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_screen = Mock()
            mock_screen.image = Mock(side_effect=Exception("Screen error"))
            mock_pyboy.screen = mock_screen
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))

            # Should handle error gracefully
            screen = agent.get_screen_array()
            assert isinstance(screen, np.ndarray)

    def test_agent_handles_button_errors(self, mock_rom_file):
        """Test agent handles button press errors."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            # Make both button methods fail to force fallback
            mock_pyboy.button_press = Mock(side_effect=AttributeError("Button press not available"))
            mock_pyboy.button_release = Mock()
            mock_pyboy.send_input = Mock()  # This should work as fallback
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))

            # Should handle error gracefully by falling back to send_input
            agent.press_button('A')

            # Verify fallback was used
            assert mock_pyboy.send_input.call_count == 2  # Press and release


# Performance tests
@pytest.mark.slow
class TestAgentPerformance:
    """Performance tests for agent."""

    def test_agent_step_performance(self, benchmark_runner, mock_rom_file):
        """Benchmark agent step performance."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.memory.__getitem__ = Mock(return_value=0)
            mock_pyboy.screen = Mock()
            mock_pyboy.button_press = Mock()
            mock_pyboy.button_release = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))

            def step_op():
                agent.step('RIGHT')

            result = benchmark_runner.run('agent_step', step_op, iterations=100)
            assert result['mean'] < 0.05  # Should be under 50ms

    def test_agent_state_reading_performance(self, benchmark_runner, mock_rom_file):
        """Benchmark state reading performance."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_memory = Mock()
            mock_memory.__getitem__ = Mock(return_value=0)
            mock_pyboy.memory = mock_memory
            mock_pyboy.screen = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))

            def state_op():
                agent.get_comprehensive_state()

            result = benchmark_runner.run('get_state', state_op, iterations=100)
            assert result['mean'] < 0.01  # Should be under 10ms


# Parameterized tests
class TestAgentParameterized:
    """Parameterized tests for agent."""

    @pytest.mark.parametrize("action", ['A', 'B', 'START', 'SELECT', 'UP', 'DOWN', 'LEFT', 'RIGHT'])
    def test_step_with_all_actions(self, mock_rom_file, action):
        """Test stepping with all possible actions."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.memory.__getitem__ = Mock(return_value=0)
            mock_pyboy.screen = Mock()
            mock_pyboy.button_press = Mock()
            mock_pyboy.button_release = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))
            state = agent.step(action)

            assert isinstance(state, dict)
            mock_pyboy.button_press.assert_called()

    @pytest.mark.parametrize("speed", [0, 1, 2, 5, 10])
    def test_agent_with_different_speeds(self, mock_rom_file, speed):
        """Test agent with different speed multipliers."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            mock_pyboy.set_emulation_speed = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file), speed_multiplier=speed)

            assert agent.speed_multiplier == speed


# Edge case tests
class TestAgentEdgeCases:
    """Test edge cases for agent."""

    def test_agent_with_invalid_rom_path(self):
        """Test agent with invalid ROM path."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            MockPyBoy.side_effect = FileNotFoundError("ROM not found")

            with pytest.raises(FileNotFoundError):
                PokemonRedAgent("nonexistent.gb")

    def test_agent_step_without_initialization(self, mock_rom_file):
        """Test stepping before initialization."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.memory.__getitem__ = Mock(return_value=0)
            mock_pyboy.screen = Mock()
            mock_pyboy.button_press = Mock()
            mock_pyboy.button_release = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))

            # Should work even without initialization
            state = agent.step('A')
            assert isinstance(state, dict)

    def test_agent_with_max_visited_locations(self, mock_rom_file):
        """Test agent with many visited locations."""
        with patch('pokemon_red_ai.game.agent.PyBoy') as MockPyBoy:
            mock_pyboy = Mock()
            mock_pyboy.memory = Mock()
            mock_pyboy.screen = Mock()
            MockPyBoy.return_value = mock_pyboy

            agent = PokemonRedAgent(str(mock_rom_file))

            # Add many locations
            for i in range(10000):
                agent.visited_locations.add((i % 256, i // 256, 1))

            progress = agent.get_exploration_progress()
            assert progress['locations_visited'] == 10000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])