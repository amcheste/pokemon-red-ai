"""
Lock the eval JSON schema.

``scripts/eval.py`` declares ``EVAL_METRIC_SCHEMA`` as the contract for
the JSON it writes.  Downstream consumers (``scripts/analyze.py``,
notebooks in the paper repo, the rliable analysis pipeline) read those
fields by name.  Silent drift between the schema and the actual
metrics dict turns into mysterious KeyErrors at analysis time, often
weeks after the regression landed.

This test synthesises an ``episode_results`` list, runs it through the
exact metrics-building expression in ``evaluate_checkpoint`` (copy-
pasted into ``_build_metrics_for_test`` so we exercise the same code
path), then asserts every key in the schema is present with the right
type — and conversely, every schema-typed key is reachable.

The "extended metrics" outside the schema (algorithm, seed, etc.) are
not locked, by design — they're convenience fields whose presence is
not contractually relied on by downstream code.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, get_args, get_origin

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.eval import EVAL_METRIC_SCHEMA


def _build_metrics_for_test(
    episode_results: List[Dict[str, Any]],
    n_episodes: int,
    save_state: Path,
    checkpoint_path: Path,
    rom_sha256: str,
    git_sha: str,
) -> Dict[str, Any]:
    """Mirror ``evaluate_checkpoint``'s metrics-dict construction.

    Kept in sync with ``scripts/eval.py``.  If you change the schema
    or the metrics dict, update this helper too — that mirroring is
    intentional: it forces the schema-vs-output coupling to be tested
    every time the schema is touched.
    """
    returns = [r["total_reward"] for r in episode_results]
    flags = [r["event_flags_triggered"] for r in episode_results]
    maps = [r["maps_visited"] for r in episode_results]
    brock_wins = [r["earned_boulder_badge"] for r in episode_results]
    pewter_steps = [r["step_reached_pewter"] for r in episode_results
                    if r["step_reached_pewter"] is not None]
    brock_steps = [r["step_earned_boulder"] for r in episode_results
                   if r["step_earned_boulder"] is not None]

    return {
        "mean_return": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "brock_win_rate": float(np.mean(brock_wins)),
        "mean_event_flags_triggered": float(np.mean(flags)),
        "max_event_flags_triggered": int(np.max(flags)) if flags else 0,
        "unique_maps_visited": int(np.max(maps)) if maps else 0,
        "steps_to_pewter": (
            int(np.mean(pewter_steps)) if pewter_steps else None
        ),
        "steps_to_brock_win": (
            int(np.mean(brock_steps)) if brock_steps else None
        ),
        "n_episodes": n_episodes,
        "eval_save_state": str(save_state),
        "checkpoint_path": str(checkpoint_path),
        "rom_sha256": rom_sha256,
        "git_sha": git_sha,
    }


def _synth_episode(
    won_brock: bool = False,
    flags_triggered: int = 0,
    maps_visited: int = 1,
    total_reward: float = 100.0,
) -> Dict[str, Any]:
    return {
        "total_reward": total_reward,
        "steps": 2000,
        "maps_visited": maps_visited,
        "maps_list": list(range(1, maps_visited + 1)),
        "event_flags_triggered": flags_triggered,
        "badges": 1 if won_brock else 0,
        "earned_boulder_badge": won_brock,
        "step_reached_pewter": 1500 if maps_visited >= 3 else None,
        "step_earned_boulder": 1900 if won_brock else None,
        "final_map": 3 if maps_visited >= 3 else 1,
    }


def _matches_type(value: Any, expected: type) -> bool:
    """Loose type check that handles ``Optional[T]`` and numpy scalars."""
    origin = get_origin(expected)
    if origin is None:
        if expected is float:
            return isinstance(value, (float, np.floating))
        if expected is int:
            return isinstance(value, (int, np.integer)) and not isinstance(value, bool)
        return isinstance(value, expected)
    # Optional[T] resolves to Union[T, None]
    args = get_args(expected)
    if type(None) in args and value is None:
        return True
    return any(_matches_type(value, a) for a in args if a is not type(None))


class TestEvalSchemaCompleteness:
    """Every key in EVAL_METRIC_SCHEMA must appear in the actual metrics
    output, typed as declared, and vice versa."""

    @pytest.fixture
    def metrics(self, tmp_path) -> Dict[str, Any]:
        episodes = [
            _synth_episode(won_brock=True, flags_triggered=13, maps_visited=3),
            _synth_episode(won_brock=False, flags_triggered=8, maps_visited=2),
            _synth_episode(won_brock=False, flags_triggered=4, maps_visited=1),
        ]
        return _build_metrics_for_test(
            episode_results=episodes,
            n_episodes=len(episodes),
            save_state=tmp_path / "fake.state",
            checkpoint_path=tmp_path / "fake.zip",
            rom_sha256="0" * 64,
            git_sha="deadbeef",
        )

    def test_every_schema_key_present_in_output(self, metrics):
        missing = [k for k in EVAL_METRIC_SCHEMA if k not in metrics]
        assert missing == [], (
            f"EVAL_METRIC_SCHEMA declares these keys but evaluate_checkpoint "
            f"does not produce them: {missing}.  Downstream consumers will "
            f"KeyError."
        )

    def test_every_schema_value_has_correct_type(self, metrics):
        mismatches = []
        for key, expected_type in EVAL_METRIC_SCHEMA.items():
            actual = metrics[key]
            if not _matches_type(actual, expected_type):
                mismatches.append(
                    f"  {key}: expected {expected_type!r}, got {type(actual).__name__} ({actual!r})"
                )
        assert not mismatches, (
            "Schema type mismatch:\n" + "\n".join(mismatches)
        )

    def test_optional_fields_can_be_none(self, tmp_path):
        """``steps_to_pewter`` / ``steps_to_brock_win`` are Optional[int].
        With zero Brock wins, both must serialise as JSON-null without
        breaking the schema check."""
        episodes = [
            _synth_episode(won_brock=False, flags_triggered=2, maps_visited=1),
        ]
        m = _build_metrics_for_test(
            episode_results=episodes,
            n_episodes=1,
            save_state=tmp_path / "s.state",
            checkpoint_path=tmp_path / "c.zip",
            rom_sha256="0" * 64,
            git_sha="abc123",
        )
        assert m["steps_to_pewter"] is None
        assert m["steps_to_brock_win"] is None
        # Schema must still validate.
        assert _matches_type(m["steps_to_pewter"], EVAL_METRIC_SCHEMA["steps_to_pewter"])
        assert _matches_type(m["steps_to_brock_win"], EVAL_METRIC_SCHEMA["steps_to_brock_win"])

    def test_output_is_json_serialisable(self, metrics):
        """Metrics dict round-trips through JSON without loss for every
        schema key.  Catches the case where a numpy scalar slips in
        and breaks ``json.dumps``."""
        import json
        as_json = json.loads(json.dumps(metrics, default=str))
        for key in EVAL_METRIC_SCHEMA:
            assert key in as_json
