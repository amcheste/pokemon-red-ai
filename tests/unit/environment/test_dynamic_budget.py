"""
Tests for the Pleines et al. (2025) dynamic episode budget.

With ``dynamic_episode_budget=True`` an episode starts with a budget
of ``dynamic_budget_initial`` steps and gains
``dynamic_budget_per_event`` steps for every event flag set since
reset, replacing the fixed ``max_episode_steps`` truncation
(paper section II-E).
"""

import pytest

from pokemon_red_ai.environment import PokemonRedGymEnv


def _make_env(mock_agent_class, mock_rom_file, **kwargs):
    env = PokemonRedGymEnv(
        rom_path=mock_rom_file,
        headless=True,
        dynamic_episode_budget=True,
        dynamic_budget_initial=100,
        dynamic_budget_per_event=25,
        **kwargs,
    )
    return env


def _set_event_count(env, count):
    env.game.get_comprehensive_state.return_value['event_flag_count'] = count


class TestDynamicEpisodeBudget:

    def test_disabled_by_default(self, mock_agent_class, mock_rom_file):
        env = PokemonRedGymEnv(rom_path=mock_rom_file, headless=True,
                               max_episode_steps=500)
        assert env.dynamic_episode_budget is False
        env.episode_steps = 500
        terminated, truncated = env._check_done()
        assert truncated is True

    def test_truncates_at_initial_budget_without_events(
            self, mock_agent_class, mock_rom_file):
        env = _make_env(mock_agent_class, mock_rom_file)
        _set_event_count(env, 4)
        env.reset()
        env.episode_steps = 99
        assert env._check_done() == (False, False)
        env.episode_steps = 100
        terminated, truncated = env._check_done()
        assert truncated is True

    def test_completed_events_extend_budget(
            self, mock_agent_class, mock_rom_file):
        env = _make_env(mock_agent_class, mock_rom_file)
        _set_event_count(env, 4)
        env.reset()
        # Two events completed since reset: budget 100 + 2*25 = 150
        _set_event_count(env, 6)
        env.episode_steps = 149
        assert env._check_done() == (False, False)
        env.episode_steps = 150
        terminated, truncated = env._check_done()
        assert truncated is True

    def test_save_state_baseline_flags_do_not_extend(
            self, mock_agent_class, mock_rom_file):
        # Flags already set at reset (from a save state) must not
        # extend the budget — only NEW events count.
        env = _make_env(mock_agent_class, mock_rom_file)
        _set_event_count(env, 50)
        env.reset()
        env.episode_steps = 100
        terminated, truncated = env._check_done()
        assert truncated is True

    def test_fixed_cap_ignored_when_dynamic(
            self, mock_agent_class, mock_rom_file):
        # A small max_episode_steps must not truncate a dynamic episode.
        env = _make_env(mock_agent_class, mock_rom_file,
                        max_episode_steps=10)
        _set_event_count(env, 0)
        env.reset()
        env.episode_steps = 50
        assert env._check_done() == (False, False)

    def test_missing_event_count_falls_back_gracefully(
            self, mock_agent_class, mock_rom_file):
        # State without 'event_flag_count' behaves like zero events.
        env = _make_env(mock_agent_class, mock_rom_file)
        env.reset()
        env.episode_steps = 100
        terminated, truncated = env._check_done()
        assert truncated is True

    def test_paper_defaults(self, mock_agent_class, mock_rom_file):
        env = PokemonRedGymEnv(rom_path=mock_rom_file, headless=True,
                               dynamic_episode_budget=True)
        assert env.dynamic_budget_initial == 10_240
        assert env.dynamic_budget_per_event == 2_048
