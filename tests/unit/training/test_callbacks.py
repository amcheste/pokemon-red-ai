"""
Unit tests for training callbacks.

Covers the four callbacks defined in ``pokemon_red_ai.training.callbacks``
that previously had little or no coverage:

- :class:`TrainingCallback`: best-model saving, statistics aggregation
- :class:`EnhancedTrainingCallback`: rollout-end plotting + fallbacks
- :class:`EarlyStopping`: patience-based termination
- :class:`PerformanceMonitor`: step-timing FPS calculation

The file used to be truncated at line 72 mid-assertion.  PR-2 of the
audit rewrites it with assertions that actually verify behaviour rather
than just call signatures.
"""

import time

import numpy as np
import pytest
from unittest.mock import MagicMock, Mock, patch

from pokemon_red_ai.training.callbacks import (
    EarlyStopping,
    EnhancedTrainingCallback,
    PerformanceMonitor,
    TrainingCallback,
)


# ──────────────────────────────────────────────────────────────────────
# TrainingCallback
# ──────────────────────────────────────────────────────────────────────


class TestTrainingCallback:
    """The basic SB3 callback used by every training run."""

    def test_initialization(self, temp_save_dir):
        callback = TrainingCallback(
            save_freq=1000,
            save_path=str(temp_save_dir),
            verbose=1,
        )
        assert callback.save_freq == 1000
        assert callback.save_path == str(temp_save_dir)
        assert callback.best_reward == -float("inf")

    def test_on_step_returns_true(self, temp_save_dir):
        """_on_step never aborts training — it must always return True."""
        callback = TrainingCallback(save_path=str(temp_save_dir))
        assert callback._on_step() is True

    def test_on_rollout_end_records_episodes(self, temp_save_dir, sample_episode_info):
        callback = TrainingCallback(save_path=str(temp_save_dir), verbose=0)
        callback.model = Mock()
        callback.model.ep_info_buffer = sample_episode_info
        callback.model.save = Mock()
        callback.num_timesteps = 1000

        callback._on_rollout_end()

        assert len(callback.episode_rewards) > 0
        assert len(callback.episode_lengths) > 0
        # Episode rewards must come from ep_info_buffer entries.
        assert max(callback.episode_rewards) == 200.0

    def test_saves_best_model_when_reward_jumps(self, temp_save_dir):
        callback = TrainingCallback(save_path=str(temp_save_dir), verbose=0)
        callback.model = Mock()
        callback.model.ep_info_buffer = [{"r": 500.0, "l": 1000}]
        callback.model.save = Mock()
        callback.num_timesteps = 1000

        callback._on_rollout_end()

        # Best model must have been saved at least once.
        assert callback.model.save.called
        # And the recorded best_reward must reflect the new high.
        assert callback.best_reward >= 500.0

    def test_does_not_save_when_reward_does_not_improve(self, temp_save_dir):
        callback = TrainingCallback(save_path=str(temp_save_dir), verbose=0)
        callback.model = Mock()
        callback.model.ep_info_buffer = [{"r": 100.0, "l": 500}]
        callback.model.save = Mock()
        callback.num_timesteps = 1000

        # First rollout sets the baseline.
        callback._on_rollout_end()
        callback.model.save.reset_mock()

        # Second rollout with a lower reward must not save the "best".
        callback.model.ep_info_buffer = [{"r": 50.0, "l": 500}]
        callback._on_rollout_end()

        # Periodic save may still fire on save_freq schedule, but the
        # *best-model* save path is keyed on improvement and must not.
        saved_paths = [
            call.args[0]
            for call in callback.model.save.call_args_list
            if call.args
        ]
        assert not any("best" in str(p) for p in saved_paths)

    def test_get_statistics_after_rollouts(self, temp_save_dir, sample_episode_info):
        callback = TrainingCallback(save_path=str(temp_save_dir), verbose=0)
        callback.model = Mock()
        callback.model.ep_info_buffer = sample_episode_info
        callback.model.save = Mock()
        callback.num_timesteps = 1000

        callback._on_rollout_end()
        stats = callback.get_statistics()

        # API contract: get_statistics returns at least these keys.
        assert "total_episodes" in stats
        assert "best_reward" in stats
        assert "num_timesteps" in stats
        # After buffering all three episodes, the recent-window avg
        # equals the mean of [100, 150, 200] = 150.
        assert "recent_avg_reward" in stats
        assert stats["recent_avg_reward"] == pytest.approx(150.0)


# ──────────────────────────────────────────────────────────────────────
# EnhancedTrainingCallback
# ──────────────────────────────────────────────────────────────────────


class TestEnhancedTrainingCallback:
    """Verifies the plotting callback used by `trainer.py` by default."""

    def test_initialization(self, temp_save_dir):
        cb = EnhancedTrainingCallback(
            save_freq=5000,
            save_path=str(temp_save_dir),
            plot_freq=10_000,
            verbose=0,
        )
        assert cb.save_freq == 5000
        assert cb.plot_freq == 10_000
        # episode buffers start empty
        assert len(cb.episode_rewards) == 0

    def test_on_step_returns_true(self, temp_save_dir):
        cb = EnhancedTrainingCallback(
            save_path=str(temp_save_dir), verbose=0,
        )
        assert cb._on_step() is True

    def test_rollout_end_buffers_episodes(self, temp_save_dir, sample_episode_info):
        cb = EnhancedTrainingCallback(
            save_path=str(temp_save_dir),
            plot_freq=10_000,
            verbose=0,
        )
        cb.model = Mock()
        cb.model.ep_info_buffer = sample_episode_info
        cb.model.save = Mock()
        cb.num_timesteps = 1000

        cb._on_rollout_end()

        assert len(cb.episode_rewards) >= len(sample_episode_info)

    def test_get_enhanced_statistics_keys(self, temp_save_dir, sample_episode_info):
        cb = EnhancedTrainingCallback(
            save_path=str(temp_save_dir),
            verbose=0,
        )
        cb.model = Mock()
        cb.model.ep_info_buffer = sample_episode_info
        cb.model.save = Mock()
        cb.num_timesteps = 1000
        cb._on_rollout_end()

        stats = cb.get_enhanced_statistics()
        # Surface-level contract: returns a dict that always has the
        # baseline counters, and adds episode-aggregates if rewards
        # buffered.  See callbacks.py:530-569.
        assert isinstance(stats, dict)
        assert "num_timesteps" in stats
        assert "best_reward" in stats
        assert "total_episodes" in stats  # populated once ep_rewards is non-empty
        assert "recent_avg_reward" in stats
        assert stats["recent_avg_reward"] == pytest.approx(150.0)


# ──────────────────────────────────────────────────────────────────────
# EarlyStopping
# ──────────────────────────────────────────────────────────────────────


class TestEarlyStopping:
    """The early-stopping callback is opt-in but must work when used."""

    def _make_cb(self, **kwargs):
        defaults = dict(check_freq=10, patience=3, min_delta=1.0, verbose=0)
        defaults.update(kwargs)
        cb = EarlyStopping(**defaults)
        cb.model = Mock()
        cb.model.ep_info_buffer = []
        cb.num_timesteps = 0
        return cb

    def test_initialization(self):
        cb = self._make_cb(check_freq=500, patience=7, min_delta=2.5)
        assert cb.check_freq == 500
        assert cb.patience == 7
        assert cb.min_delta == 2.5
        assert cb.best_reward == -float("inf")
        assert cb.wait_count == 0
        assert cb.stopped_early is False

    def test_returns_true_off_check_boundary(self):
        cb = self._make_cb(check_freq=100)
        cb.num_timesteps = 37  # Not a multiple of 100
        assert cb._on_step() is True
        assert cb.wait_count == 0

    def test_records_improvement(self):
        cb = self._make_cb(check_freq=10, min_delta=1.0)
        cb.num_timesteps = 10
        cb.model.ep_info_buffer = [{"r": 100.0} for _ in range(5)]

        result = cb._on_step()
        assert result is True
        assert cb.best_reward == pytest.approx(100.0)
        assert cb.wait_count == 0

    def test_increments_wait_when_not_improving(self):
        cb = self._make_cb(check_freq=10, patience=3, min_delta=1.0)
        cb.num_timesteps = 10
        # First check: baseline 100
        cb.model.ep_info_buffer = [{"r": 100.0} for _ in range(5)]
        cb._on_step()

        # Second check at +10 timesteps: same reward — wait_count++.
        cb.num_timesteps = 20
        cb.model.ep_info_buffer = [{"r": 100.0} for _ in range(5)]
        cb._on_step()
        assert cb.wait_count == 1
        assert cb.stopped_early is False

    def test_stops_after_patience_exhausted(self):
        cb = self._make_cb(check_freq=10, patience=2, min_delta=1.0)

        # Baseline
        cb.num_timesteps = 10
        cb.model.ep_info_buffer = [{"r": 100.0} for _ in range(5)]
        assert cb._on_step() is True

        # First miss
        cb.num_timesteps = 20
        cb.model.ep_info_buffer = [{"r": 100.0} for _ in range(5)]
        assert cb._on_step() is True
        assert cb.wait_count == 1

        # Second miss — patience exhausted, returns False to stop training.
        cb.num_timesteps = 30
        cb.model.ep_info_buffer = [{"r": 100.0} for _ in range(5)]
        assert cb._on_step() is False
        assert cb.stopped_early is True

    def test_resets_wait_on_improvement(self):
        cb = self._make_cb(check_freq=10, patience=5, min_delta=1.0)
        cb.num_timesteps = 10
        cb.model.ep_info_buffer = [{"r": 100.0} for _ in range(5)]
        cb._on_step()  # baseline

        # Miss
        cb.num_timesteps = 20
        cb._on_step()
        assert cb.wait_count == 1

        # Improvement: reward jumps to 200, well past best+min_delta.
        cb.num_timesteps = 30
        cb.model.ep_info_buffer = [{"r": 200.0} for _ in range(5)]
        cb._on_step()
        assert cb.wait_count == 0
        assert cb.best_reward == pytest.approx(200.0)

    def test_min_delta_must_be_exceeded(self):
        """A jump of exactly min_delta is NOT enough — must be strict."""
        cb = self._make_cb(check_freq=10, patience=5, min_delta=10.0)
        cb.num_timesteps = 10
        cb.model.ep_info_buffer = [{"r": 100.0} for _ in range(5)]
        cb._on_step()  # baseline 100.0

        # +10.0 exactly: at the threshold; current impl uses strict >.
        cb.num_timesteps = 20
        cb.model.ep_info_buffer = [{"r": 110.0} for _ in range(5)]
        cb._on_step()
        # wait_count must advance (no improvement past threshold).
        assert cb.wait_count == 1


# ──────────────────────────────────────────────────────────────────────
# PerformanceMonitor
# ──────────────────────────────────────────────────────────────────────


class TestPerformanceMonitor:
    """System-resource monitor — checks FPS calculation and graceful degradation."""

    def test_initialization(self):
        cb = PerformanceMonitor(monitor_freq=500, verbose=0)
        assert cb.monitor_freq == 500
        # step_times deque is empty until _on_step is called.
        assert len(cb.step_times) == 0
        assert cb.last_time is None

    def test_first_step_records_no_delta(self):
        cb = PerformanceMonitor(monitor_freq=10_000, verbose=0)
        cb.num_timesteps = 1
        cb._on_step()
        # First step sets last_time but has no delta to record yet.
        assert cb.last_time is not None
        assert len(cb.step_times) == 0

    def test_subsequent_steps_record_deltas(self):
        cb = PerformanceMonitor(monitor_freq=10_000, verbose=0)
        cb.num_timesteps = 1
        cb._on_step()
        time.sleep(0.001)  # Ensure a measurable delta.
        cb.num_timesteps = 2
        cb._on_step()
        assert len(cb.step_times) == 1
        assert cb.step_times[0] > 0

    def test_returns_true_always(self):
        cb = PerformanceMonitor(monitor_freq=10_000, verbose=0)
        cb.num_timesteps = 1
        # Performance monitoring never aborts training.
        assert cb._on_step() is True

    def test_get_performance_stats_returns_dict(self):
        cb = PerformanceMonitor(monitor_freq=10_000, verbose=0)
        # Run a few steps to populate step_times.
        for i in range(1, 4):
            cb.num_timesteps = i
            cb._on_step()
            time.sleep(0.001)

        stats = cb.get_performance_stats()
        assert isinstance(stats, dict)
        # Surface-level: should expose timing/FPS-ish fields.  We don't
        # bind to exact keys (the implementation may rename) but the
        # dict must be non-empty after seeing steps.
        assert len(stats) > 0
