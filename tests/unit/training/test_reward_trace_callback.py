"""
Unit tests for RewardComponentTraceCallback (AMC-233).

The callback persists per-step reward-component firings to a gzip
NDJSON trace so the reward-hacking analysis can work per seed, per
step.  These tests drive the SB3 callback interface directly with fake
``locals`` dicts and verify the on-disk format round-trips through
``iter_reward_trace``.
"""

import gzip
import json
from unittest.mock import Mock

import pytest

from pokemon_red_ai.training.callbacks import (
    REWARD_TRACE_FORMAT,
    RewardComponentTraceCallback,
    iter_reward_trace,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _make_callback(tmp_path, n_envs=1, **kwargs):
    cb = RewardComponentTraceCallback(
        trace_path=str(tmp_path / "reward_trace.ndjson.gz"),
        **kwargs,
    )
    env = Mock()
    env.num_envs = n_envs
    model = Mock()
    model.get_env = Mock(return_value=env)
    cb.model = model
    cb.num_timesteps = 0
    cb.n_calls = 0
    cb.locals = {}
    cb.globals = {}
    return cb


def _step(cb, infos, dones=None, n_envs=1):
    """Drive one _on_step with the given per-env infos/dones."""
    if dones is None:
        dones = [False] * len(infos)
    cb.num_timesteps += n_envs
    cb.locals = {"dones": dones, "infos": infos}
    return cb._on_step()


def _read_trace(cb):
    cb._on_training_end()
    header, records = iter_reward_trace(cb.trace_path)
    return header, list(records)


# ──────────────────────────────────────────────────────────────────────
# Format and round-trip
# ──────────────────────────────────────────────────────────────────────


class TestTraceFormat:
    def test_header_written_on_training_start(self, tmp_path):
        cb = _make_callback(tmp_path)
        cb._on_training_start()

        header, records = _read_trace(cb)

        assert header["format"] == REWARD_TRACE_FORMAT
        assert header["sparse"] is True
        assert header["n_envs"] == 1
        assert records == []

    def test_component_step_recorded(self, tmp_path):
        cb = _make_callback(tmp_path)
        cb._on_training_start()

        _step(cb, [{"reward_components": {"navigation": 0.005}}])

        header, records = _read_trace(cb)
        assert len(records) == 1
        rec = records[0]
        assert rec["t"] == 1
        assert rec["s"] == 0
        assert rec["env"] == 0
        assert rec["ep"] == 0
        assert rec["c"] == {"navigation": 0.005}
        assert "done" not in rec

    def test_sparse_empty_steps_not_recorded(self, tmp_path):
        """Steps where nothing fired produce no record (sparse encoding)."""
        cb = _make_callback(tmp_path)
        cb._on_training_start()

        _step(cb, [{"reward_components": {}}])
        _step(cb, [{}])  # no reward_components key at all
        _step(cb, [{"reward_components": {"event": 2.0}}])

        _, records = _read_trace(cb)
        assert len(records) == 1
        # Per-env step index reflects the two skipped steps
        assert records[0]["s"] == 2

    def test_episode_end_always_recorded(self, tmp_path):
        """done=True emits a record even when no component fired."""
        cb = _make_callback(tmp_path)
        cb._on_training_start()

        _step(cb, [{"reward_components": {}}], dones=[True])

        _, records = _read_trace(cb)
        assert len(records) == 1
        assert records[0]["done"] == 1
        assert "c" not in records[0]

    def test_episode_index_increments_after_done(self, tmp_path):
        cb = _make_callback(tmp_path)
        cb._on_training_start()

        _step(cb, [{"reward_components": {"event": 2.0}}], dones=[True])
        _step(cb, [{"reward_components": {"event": 4.0}}])

        _, records = _read_trace(cb)
        assert [r["ep"] for r in records] == [0, 1]

    def test_multi_env_indexed_separately(self, tmp_path):
        cb = _make_callback(tmp_path, n_envs=2)
        cb._on_training_start()

        _step(
            cb,
            [
                {"reward_components": {"navigation": 0.005}},
                {"reward_components": {"event": 2.0}},
            ],
            dones=[False, True],
            n_envs=2,
        )
        _step(
            cb,
            [
                {"reward_components": {}},
                {"reward_components": {"healing": 1.25}},
            ],
            n_envs=2,
        )

        header, records = _read_trace(cb)
        assert header["n_envs"] == 2
        by_env = {}
        for r in records:
            by_env.setdefault(r["env"], []).append(r)
        assert len(by_env[0]) == 1
        assert len(by_env[1]) == 2
        # Env 1 finished an episode on step 0; its next record is ep 1
        assert by_env[1][0]["done"] == 1
        assert by_env[1][1]["ep"] == 1

    def test_non_numeric_component_values_skipped(self, tmp_path):
        cb = _make_callback(tmp_path)
        cb._on_training_start()

        _step(
            cb,
            [{"reward_components": {"event": 2.0, "weird": "not-a-number"}}],
        )

        _, records = _read_trace(cb)
        assert records[0]["c"] == {"event": 2.0}

    def test_full_trace_survives_10k_steps(self, tmp_path):
        """Every firing step lands in the file across flush boundaries."""
        cb = _make_callback(tmp_path, flush_every=100)
        cb._on_training_start()

        fired = 0
        for i in range(10_000):
            if i % 7 == 0:
                infos = [{"reward_components": {"navigation": 0.005}}]
                fired += 1
            else:
                infos = [{}]
            _step(cb, infos)

        _, records = _read_trace(cb)
        assert len(records) == fired
        # Trace preserves per-env step ordering
        steps = [r["s"] for r in records]
        assert steps == sorted(steps)
        assert steps[0] == 0 and steps[-1] == 9996


# ──────────────────────────────────────────────────────────────────────
# Flushing and failure behavior
# ──────────────────────────────────────────────────────────────────────


class TestFlushAndFailure:
    def test_rollout_end_flushes(self, tmp_path):
        cb = _make_callback(tmp_path, flush_every=1_000_000)
        cb._on_training_start()

        _step(cb, [{"reward_components": {"event": 2.0}}])
        cb._on_rollout_end()

        # Readable without _on_training_end because rollout end flushed
        header, records = iter_reward_trace(cb.trace_path)
        assert len(list(records)) == 1

    def test_missing_locals_is_noop(self, tmp_path):
        cb = _make_callback(tmp_path)
        cb._on_training_start()
        cb.locals = {}

        assert cb._on_step() is True

        _, records = _read_trace(cb)
        assert records == []

    def test_unwritable_path_disables_without_raising(self, tmp_path):
        cb = RewardComponentTraceCallback(
            trace_path="/dev/null/not-a-dir/reward_trace.ndjson.gz",
        )
        env = Mock()
        env.num_envs = 1
        model = Mock()
        model.get_env = Mock(return_value=env)
        cb.model = model
        cb.num_timesteps = 0
        cb.locals = {}

        cb._on_training_start()  # must not raise
        assert cb._disabled is True

        # Subsequent steps are safe no-ops
        assert _step(cb, [{"reward_components": {"event": 2.0}}]) is True
        cb._on_training_end()

    def test_double_training_end_safe(self, tmp_path):
        cb = _make_callback(tmp_path)
        cb._on_training_start()
        _step(cb, [{"reward_components": {"event": 2.0}}])
        cb._on_training_end()
        cb._on_training_end()  # second close must not raise

        header, records = iter_reward_trace(cb.trace_path)
        assert len(list(records)) == 1


# ──────────────────────────────────────────────────────────────────────
# Reader helper
# ──────────────────────────────────────────────────────────────────────


class TestIterRewardTrace:
    def test_truncated_tail_stops_gracefully(self, tmp_path):
        """A run killed mid-write leaves a partial gzip member — the
        reader must return everything up to the last complete flush."""
        cb = _make_callback(tmp_path, flush_every=1)
        cb._on_training_start()
        _step(cb, [{"reward_components": {"event": 2.0}}])

        # Simulate a crash mid-append: dangling partial gzip member
        with open(cb.trace_path, "ab") as fh:
            fh.write(b"\x1f\x8b\x08\x00partial-garbage")

        header, records = iter_reward_trace(cb.trace_path)

        assert header["format"] == REWARD_TRACE_FORMAT
        recs = list(records)  # must not raise
        assert len(recs) == 1
        assert recs[0]["c"] == {"event": 2.0}

    def test_streams_without_loading_all(self, tmp_path):
        path = tmp_path / "trace.ndjson.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(json.dumps({"format": REWARD_TRACE_FORMAT}) + "\n")
            for i in range(5):
                fh.write(json.dumps({"t": i, "s": i, "env": 0, "ep": 0}) + "\n")

        header, records = iter_reward_trace(str(path))

        assert header["format"] == REWARD_TRACE_FORMAT
        first = next(records)
        assert first["t"] == 0
        assert len(list(records)) == 4
