"""
Unit tests for Pokemon Red AI controls module.

Tests button controls, screen detection, and input handling.
"""

import pytest
import numpy as np
from unittest.mock import Mock, PropertyMock, patch

from pokemon_red_ai.game.controls import (
    ScreenType,
    GameBoyButton,
    wait_frames,
    press_button_basic,
    get_screen_array,
    get_tilemap_background,
    detect_screen_type,
    wait_for_screen_change,
    press_button_smart,
    navigate_menu_cursor,
    press_button_sequence,
    spam_button,
    accept_default_name,
    get_input_state,
    BUTTON_MAPPINGS
)
from pokemon_red_ai.game.memory import MEMORY_ADDRESSES


class TestWaitFrames:
    """Test frame waiting functionality."""

    def test_wait_frames_basic(self):
        """Test waiting for specified frames."""
        mock_pyboy = Mock()

        wait_frames(mock_pyboy, 10)

        assert mock_pyboy.tick.call_count == 10

    def test_wait_zero_frames(self):
        """Test waiting for zero frames."""
        mock_pyboy = Mock()

        wait_frames(mock_pyboy, 0)

        assert mock_pyboy.tick.call_count == 0

    def test_wait_many_frames(self):
        """Test waiting for many frames."""
        mock_pyboy = Mock()

        wait_frames(mock_pyboy, 1000)

        assert mock_pyboy.tick.call_count == 1000


class TestButtonPress:
    """Test button press functionality."""

    def test_press_button_basic_pyboy2(self):
        """Test button press with PyBoy 2.x API."""
        mock_pyboy = Mock()
        mock_pyboy.button_press = Mock()
        mock_pyboy.button_release = Mock()

        press_button_basic(mock_pyboy, 'A', hold_frames=5, release_frames=3)

        mock_pyboy.button_press.assert_called_once_with('a')
        mock_pyboy.button_release.assert_called_once_with('a')
        assert mock_pyboy.tick.call_count == 8  # 5 hold + 3 release

    def test_press_button_fallback_to_send_input(self):
        """Test fallback to send_input when button_press unavailable."""
        mock_pyboy = Mock()
        mock_pyboy.button_press = Mock(side_effect=AttributeError)
        mock_pyboy.send_input = Mock()

        press_button_basic(mock_pyboy, 'A', hold_frames=5, release_frames=3)

        assert mock_pyboy.send_input.call_count == 2  # Press and release

    @pytest.mark.parametrize("button,expected_id", [
        ('A', 'a'),
        ('B', 'b'),
        ('START', 'start'),
        ('SELECT', 'select'),
        ('UP', 'up'),
        ('DOWN', 'down'),
        ('LEFT', 'left'),
        ('RIGHT', 'right'),
    ])
    def test_press_all_buttons(self, button, expected_id):
        """Test pressing all available buttons."""
        mock_pyboy = Mock()
        mock_pyboy.button_press = Mock()
        mock_pyboy.button_release = Mock()

        press_button_basic(mock_pyboy, button)

        mock_pyboy.button_press.assert_called_with(expected_id)
        mock_pyboy.button_release.assert_called_with(expected_id)

    def test_press_button_custom_timing(self):
        """Test button press with custom timing."""
        mock_pyboy = Mock()
        mock_pyboy.button_press = Mock()
        mock_pyboy.button_release = Mock()

        press_button_basic(mock_pyboy, 'B', hold_frames=20, release_frames=10)

        assert mock_pyboy.tick.call_count == 30


class TestScreenFunctions:
    """Test screen-related functions."""

    def test_get_screen_array(self):
        """Test getting screen as numpy array."""
        mock_pyboy = Mock()
        mock_screen = Mock()
        mock_screen.image = np.zeros((144, 160, 3), dtype=np.uint8)
        mock_pyboy.screen = mock_screen

        screen = get_screen_array(mock_pyboy)

        assert isinstance(screen, np.ndarray)
        assert screen.shape == (144, 160, 3)
        assert screen.dtype == np.uint8

    def test_get_screen_array_error_handling(self):
        """Test screen array error handling."""
        mock_pyboy = Mock()
        mock_pyboy.screen = Mock()
        mock_pyboy.screen.image = Mock(side_effect=Exception("Screen error"))

        screen = get_screen_array(mock_pyboy)

        assert isinstance(screen, np.ndarray)
        assert screen.shape == (144, 160, 3)

    def test_get_tilemap_background(self):
        """Test getting background tilemap."""
        mock_pyboy = Mock()
        mock_tilemap = np.random.randint(0, 256, (18, 20), dtype=np.uint8)
        type(mock_pyboy).tilemap_background = PropertyMock(return_value=mock_tilemap)

        tilemap = get_tilemap_background(mock_pyboy)

        assert isinstance(tilemap, np.ndarray)
        assert tilemap.shape == (18, 20)

    def test_get_tilemap_error_handling(self):
        """Test tilemap error handling."""
        mock_pyboy = Mock()
        type(mock_pyboy).tilemap_background = PropertyMock(side_effect=Exception("Tilemap error"))

        tilemap = get_tilemap_background(mock_pyboy)

        assert isinstance(tilemap, np.ndarray)
        assert tilemap.shape == (18, 20)


class TestScreenDetection:
    """Test screen type detection."""

    def test_detect_screen_type_in_game(self):
        """Test detection when in game world."""
        mock_pyboy = Mock()
        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(return_value=1)  # map_id = 1
        mock_pyboy.memory = mock_memory

        screen_type = detect_screen_type(mock_pyboy)

        assert screen_type == ScreenType.IN_GAME

    def test_detect_screen_type_title_screen(self):
        """Test detection of title screen."""
        mock_pyboy = Mock()
        mock_memory = Mock()

        def memory_side_effect(addr):
            if addr == MEMORY_ADDRESSES['map_id']:
                return 0
            elif addr == MEMORY_ADDRESSES['game_state']:
                return 0
            elif addr == MEMORY_ADDRESSES['menu_state']:
                return 0
            return 0

        mock_memory.__getitem__ = Mock(side_effect=memory_side_effect)
        mock_pyboy.memory = mock_memory

        # Mock tilemap with Pokemon logo in top area
        mock_tilemap = np.zeros((18, 20), dtype=np.uint8)
        mock_tilemap[:8, :] = 100
        type(mock_pyboy).tilemap_background = PropertyMock(return_value=mock_tilemap)

        screen_type = detect_screen_type(mock_pyboy)

        assert screen_type == ScreenType.TITLE_SCREEN

    def test_detect_screen_type_main_menu(self):
        """Test detection of main menu."""
        mock_pyboy = Mock()
        mock_memory = Mock()

        def memory_side_effect(addr):
            if addr == MEMORY_ADDRESSES['map_id']:
                return 0
            elif addr == MEMORY_ADDRESSES['menu_state']:
                return 1  # Menu state indicates main menu
            return 0

        mock_memory.__getitem__ = Mock(side_effect=memory_side_effect)
        mock_pyboy.memory = mock_memory

        # Mock tilemap with menu content
        mock_tilemap = np.zeros((18, 20), dtype=np.uint8)
        mock_tilemap[8:13, 6:15] = 50  # Menu area
        type(mock_pyboy).tilemap_background = PropertyMock(return_value=mock_tilemap)

        screen_type = detect_screen_type(mock_pyboy)

        assert screen_type == ScreenType.MAIN_MENU

    def test_detect_screen_type_intro_animation(self):
        """Test detection of intro animation."""
        mock_pyboy = Mock()
        mock_memory = Mock()

        def memory_side_effect(addr):
            if addr == MEMORY_ADDRESSES['map_id']:
                return 0
            return 0

        mock_memory.__getitem__ = Mock(side_effect=memory_side_effect)
        mock_pyboy.memory = mock_memory

        # Mock tilemap with high sprite density
        mock_tilemap = np.random.randint(50, 200, (18, 20), dtype=np.uint8)
        type(mock_pyboy).tilemap_background = PropertyMock(return_value=mock_tilemap)

        screen_type = detect_screen_type(mock_pyboy)

        assert screen_type == ScreenType.INTRO_ANIMATION

    def test_detect_screen_type_error_handling(self):
        """Test screen detection handles errors gracefully."""
        mock_pyboy = Mock()
        mock_pyboy.memory = Mock()
        mock_pyboy.memory.__getitem__ = Mock(side_effect=Exception("Memory error"))

        screen_type = detect_screen_type(mock_pyboy)

        assert screen_type == ScreenType.UNKNOWN


class TestScreenChangeWaiting:
    """Test waiting for screen changes."""

    def test_wait_for_screen_change_success(self):
        """Test successfully waiting for screen change."""
        mock_pyboy = Mock()

        # Simulate screen changing after 3 ticks
        screen_sequence = [
            ScreenType.TITLE_SCREEN,
            ScreenType.TITLE_SCREEN,
            ScreenType.TITLE_SCREEN,
            ScreenType.IN_GAME
        ]

        with patch('pokemon_red_ai.game.controls.detect_screen_type', side_effect=screen_sequence):
            new_screen = wait_for_screen_change(mock_pyboy, ScreenType.TITLE_SCREEN, timeout_seconds=1)

        assert new_screen == ScreenType.IN_GAME

    def test_wait_for_screen_change_timeout(self):
        """Test timeout when screen doesn't change."""
        mock_pyboy = Mock()

        # Screen never changes
        with patch('pokemon_red_ai.game.controls.detect_screen_type', return_value=ScreenType.TITLE_SCREEN):
            new_screen = wait_for_screen_change(mock_pyboy, ScreenType.TITLE_SCREEN, timeout_seconds=1)

        assert new_screen == ScreenType.TITLE_SCREEN

    def test_wait_for_screen_change_immediate(self):
        """Test when screen has already changed."""
        mock_pyboy = Mock()

        with patch('pokemon_red_ai.game.controls.detect_screen_type', return_value=ScreenType.IN_GAME):
            new_screen = wait_for_screen_change(mock_pyboy, ScreenType.TITLE_SCREEN, timeout_seconds=1)

        assert new_screen == ScreenType.IN_GAME


class TestSmartButtonPress:
    """Test smart button press with screen detection."""

    def test_press_button_smart_success(self):
        """Test smart button press succeeds."""
        mock_pyboy = Mock()
        mock_pyboy.button_press = Mock()
        mock_pyboy.button_release = Mock()

        with patch('pokemon_red_ai.game.controls.wait_for_screen_change', return_value=ScreenType.IN_GAME):
            new_screen = press_button_smart(mock_pyboy, 'A', ScreenType.TITLE_SCREEN)

        assert new_screen == ScreenType.IN_GAME
        mock_pyboy.button_press.assert_called()

    def test_press_button_smart_all_methods(self):
        """Test smart button tries all methods."""
        mock_pyboy = Mock()
        # Simulate all methods failing
        mock_pyboy.button_press = Mock(side_effect=Exception("Failed"))
        mock_pyboy.send_input = Mock(side_effect=Exception("Failed"))

        with patch('pokemon_red_ai.game.controls.wait_for_screen_change', return_value=ScreenType.TITLE_SCREEN):
            new_screen = press_button_smart(mock_pyboy, 'A', ScreenType.TITLE_SCREEN)

        # Should try multiple methods
        assert mock_pyboy.button_press.called or mock_pyboy.send_input.called


class TestMenuNavigation:
    """Test menu navigation functions."""

    def test_navigate_menu_cursor_basic(self):
        """Test basic menu cursor navigation."""
        mock_pyboy = Mock()
        mock_pyboy.button_press = Mock()
        mock_pyboy.button_release = Mock()

        navigate_menu_cursor(mock_pyboy, 2, 3)

        # Should press UP and LEFT to reset, then DOWN and RIGHT to target
        assert mock_pyboy.button_press.called

    def test_navigate_menu_cursor_top_left(self):
        """Test navigating to top-left corner."""
        mock_pyboy = Mock()
        mock_pyboy.button_press = Mock()
        mock_pyboy.button_release = Mock()

        navigate_menu_cursor(mock_pyboy, 0, 0, max_rows=5, max_cols=5)

        # Should only reset to top-left
        assert mock_pyboy.button_press.called

    def test_navigate_menu_cursor_bottom_right(self):
        """Test navigating to bottom-right corner."""
        mock_pyboy = Mock()
        mock_pyboy.button_press = Mock()
        mock_pyboy.button_release = Mock()

        navigate_menu_cursor(mock_pyboy, 9, 9, max_rows=10, max_cols=10)

        # Should press many DOWN and RIGHT
        assert mock_pyboy.button_press.called


class TestButtonSequences:
    """Test button sequence functions."""

    def test_press_button_sequence(self):
        """Test pressing a sequence of buttons."""
        mock_pyboy = Mock()
        mock_pyboy.button_press = Mock()
        mock_pyboy.button_release = Mock()

        buttons = ['A', 'B', 'START']
        press_button_sequence(mock_pyboy, buttons)

        # Should press each button
        assert mock_pyboy.button_press.call_count == 3
        assert mock_pyboy.button_release.call_count == 3

    def test_press_empty_sequence(self):
        """Test pressing empty sequence."""
        mock_pyboy = Mock()
        mock_pyboy.button_press = Mock()

        press_button_sequence(mock_pyboy, [])

        assert mock_pyboy.button_press.call_count == 0

    def test_spam_button(self):
        """Test spamming a button."""
        mock_pyboy = Mock()
        mock_pyboy.button_press = Mock()
        mock_pyboy.button_release = Mock()

        spam_button(mock_pyboy, 'A', count=5)

        assert mock_pyboy.button_press.call_count == 5
        assert mock_pyboy.button_release.call_count == 5

    def test_spam_button_custom_timing(self):
        """Test spamming button with custom timing."""
        mock_pyboy = Mock()
        mock_pyboy.button_press = Mock()
        mock_pyboy.button_release = Mock()

        spam_button(mock_pyboy, 'B', count=3, hold_frames=10, release_frames=5)

        # 3 presses * (10 hold + 5 release) = 45 ticks
        assert mock_pyboy.tick.call_count == 45


class TestNamingScreen:
    """Test naming screen functions."""

    def test_accept_default_name(self):
        """Test accepting default name."""
        mock_pyboy = Mock()
        mock_pyboy.button_press = Mock()
        mock_pyboy.button_release = Mock()

        result = accept_default_name(mock_pyboy, max_attempts=3)

        assert result is True
        assert mock_pyboy.tick.call_count > 0

    def test_accept_default_name_with_error_handling(self):
        """Test accepting name handles errors."""
        mock_pyboy = Mock()
        mock_pyboy.button_press = Mock(side_effect=Exception("Error"))
        mock_pyboy.send_input = Mock()

        result = accept_default_name(mock_pyboy, max_attempts=2)

        # Should still complete
        assert result is True


class TestInputState:
    """Test input state functions."""

    def test_get_input_state(self):
        """Test getting input state."""
        mock_pyboy = Mock()

        state = get_input_state(mock_pyboy)

        assert isinstance(state, dict)
        assert 'A' in state
        assert 'B' in state
        assert isinstance(state['A'], bool)

    def test_get_input_state_error_handling(self):
        """Test input state error handling."""
        mock_pyboy = Mock()
        type(mock_pyboy).input_state = PropertyMock(side_effect=Exception("Error"))

        state = get_input_state(mock_pyboy)

        # Should return default state
        assert isinstance(state, dict)


class TestButtonMappings:
    """Test button mapping constants."""

    def test_string_button_mappings_complete(self):
        """Test string button mappings are complete."""
        required_buttons = ['A', 'B', 'SELECT', 'START', 'RIGHT', 'LEFT', 'UP', 'DOWN']

        for button in required_buttons:
            assert button in BUTTON_MAPPINGS['string']

    def test_numeric_button_mappings_complete(self):
        """Test numeric button mappings are complete."""
        required_buttons = ['A', 'B', 'SELECT', 'START', 'RIGHT', 'LEFT', 'UP', 'DOWN']

        for button in required_buttons:
            assert button in BUTTON_MAPPINGS['numeric']
            assert isinstance(BUTTON_MAPPINGS['numeric'][button], int)


class TestEnumTypes:
    """Test enum types."""

    def test_screen_type_enum(self):
        """Test ScreenType enum values."""
        assert ScreenType.TITLE_SCREEN.value == "title_screen"
        assert ScreenType.INTRO_ANIMATION.value == "intro_animation"
        assert ScreenType.MAIN_MENU.value == "main_menu"
        assert ScreenType.IN_GAME.value == "in_game"
        assert ScreenType.UNKNOWN.value == "unknown"

    def test_gameboy_button_enum(self):
        """Test GameBoyButton enum values."""
        assert GameBoyButton.A.value == "a"
        assert GameBoyButton.B.value == "b"
        assert GameBoyButton.START.value == "start"
        assert GameBoyButton.SELECT.value == "select"


# Integration tests
@pytest.mark.integration
class TestControlsIntegration:
    """Integration tests for controls."""

    def test_button_press_and_screen_detection(self):
        """Test button press affects screen detection."""
        mock_pyboy = Mock()
        mock_pyboy.button_press = Mock()
        mock_pyboy.button_release = Mock()
        mock_memory = Mock()

        # Initially at title screen
        def memory_side_effect_before(addr):
            if addr == MEMORY_ADDRESSES['map_id']:
                return 0
            return 0

        mock_memory.__getitem__ = Mock(side_effect=memory_side_effect_before)
        mock_pyboy.memory = mock_memory

        initial_screen = detect_screen_type(mock_pyboy)

        # Press button
        press_button_basic(mock_pyboy, 'A')

        # Simulate screen changed to in-game
        def memory_side_effect_after(addr):
            if addr == MEMORY_ADDRESSES['map_id']:
                return 1
            return 0

        mock_memory.__getitem__ = Mock(side_effect=memory_side_effect_after)

        new_screen = detect_screen_type(mock_pyboy)

        assert initial_screen != new_screen


# Performance tests
@pytest.mark.slow
class TestControlsPerformance:
    """Performance tests for controls."""

    def test_button_press_performance(self, benchmark_runner):
        """Benchmark button press speed."""
        mock_pyboy = Mock()
        mock_pyboy.button_press = Mock()
        mock_pyboy.button_release = Mock()

        def press_op():
            press_button_basic(mock_pyboy, 'A')

        result = benchmark_runner.run('button_press', press_op, iterations=1000)
        assert result['mean'] < 0.01

    def test_screen_detection_performance(self, benchmark_runner):
        """Benchmark screen detection speed."""
        mock_pyboy = Mock()
        mock_memory = Mock()
        mock_memory.__getitem__ = Mock(return_value=1)
        mock_pyboy.memory = mock_memory

        def detect_op():
            detect_screen_type(mock_pyboy)

        result = benchmark_runner.run('screen_detection', detect_op, iterations=1000)
        assert result['mean'] < 0.005


if __name__ == '__main__':
    pytest.main([__file__, '-v'])