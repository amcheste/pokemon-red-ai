"""
Unit tests for ``pokemon_red_ai.analysis.comparison`` (AMC-79).

Covers:
* Treatment detection from run names
* Grouping runs by treatment with deterministic ordering
* Learning curve construction (smoothing, alignment, mean/std bands)
* Aggregate summary table (IQM + bootstrap CIs without rliable)
* Milestone race-chart aggregation
* Final-window performance bars
* Publication-style figure export to PDF / SVG / PNG
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from pokemon_red_ai.analysis.comparison import (
    KNOWN_TREATMENTS,
    LearningCurve,
    UNKNOWN_TREATMENT,
    _iqm,
    _iqm_bootstrap_ci,
    detect_treatment,
    export_figure,
    final_performance,
    group_runs_by_treatment,
    learning_curves_with_bands,
    milestone_first_episode,
    plot_final_performance_bars,
    plot_learning_curves,
    setup_publication_style,
    treatment_color,
    treatment_label,
    treatment_summary_table,
)


# ──────────────────────────────────────────────────────────────────────
# Lightweight stand-in for ``RunData`` from ``scripts/monitor.py``.
# The comparison module duck-types on ``name``, ``monitor_df``,
# ``dashboard_state`` so we don't need the real class here.
# ──────────────────────────────────────────────────────────────────────


@dataclass
class FakeRun:
    name: str
    monitor_df: Optional[pd.DataFrame] = None
    dashboard_state: Optional[Dict[str, Any]] = None

    @property
    def episode_count(self) -> int:
        if self.dashboard_state:
            return int(self.dashboard_state.get("episode_count", 0))
        if self.monitor_df is not None:
            return len(self.monitor_df)
        return 0

    @property
    def num_timesteps(self) -> int:
        if self.dashboard_state:
            return int(self.dashboard_state.get("num_timesteps", 0))
        return 0

    @property
    def best_reward(self) -> Optional[float]:
        if self.monitor_df is not None and "r" in self.monitor_df.columns:
            return float(self.monitor_df["r"].max())
        return None


def _monitor_df(rewards: List[float]) -> pd.DataFrame:
    return pd.DataFrame({"r": rewards, "l": [100] * len(rewards), "t": list(range(len(rewards)))})


def _make_run(
    name: str,
    rewards: Optional[List[float]] = None,
    *,
    flags: Optional[Dict[str, int]] = None,
    badges: Optional[List[int]] = None,
    map_visit_counts: Optional[Dict[str, int]] = None,
) -> FakeRun:
    n_rewards = len(rewards) if rewards else 0
    state: Dict[str, Any] = {
        "episode_count": n_rewards,
        "num_timesteps": n_rewards * 100,
        "flag_first_triggered": flags or {},
        "map_visit_counts": map_visit_counts or {},
    }
    if rewards is not None and badges is not None:
        state["episodes"] = [
            {"reward": r, "badges": b, "episode": i + 1}
            for i, (r, b) in enumerate(zip(rewards, badges))
        ]
    elif rewards is not None:
        state["episodes"] = [
            {"reward": r, "badges": 0, "episode": i + 1}
            for i, r in enumerate(rewards)
        ]
    return FakeRun(
        name=name,
        monitor_df=_monitor_df(rewards) if rewards else None,
        dashboard_state=state,
    )


# ──────────────────────────────────────────────────────────────────────
# detect_treatment
# ──────────────────────────────────────────────────────────────────────


class TestDetectTreatment:
    @pytest.mark.parametrize("name,expected", [
        ("rppo-pixel-seed42", "pixel"),
        ("pixel_seed42", "pixel"),
        ("PIXEL-SEED-42", "pixel"),
        ("RecurrentPPO_symbolic_42", "symbolic"),
        ("hybrid-events-s7", "hybrid"),
        ("multi_modal-rppo-s1", "multi_modal"),
        ("screen_only_run", "screen_only"),
        ("minimal-baseline", "minimal"),
    ])
    def test_known_treatments(self, name, expected):
        assert detect_treatment(name) == expected

    @pytest.mark.parametrize("name", [
        "baseline-2026-01-01",
        "experiment_xyz",
        "",
        "random-name-with-no-treatment",
    ])
    def test_unknown_falls_back(self, name):
        assert detect_treatment(name) == UNKNOWN_TREATMENT

    def test_first_match_wins_when_multiple(self):
        # If a name has both, we return the first token-position match;
        # this is a safety net for misnamed runs.
        # `pixel` appears first → pixel
        assert detect_treatment("pixel-vs-symbolic") == "pixel"


# ──────────────────────────────────────────────────────────────────────
# group_runs_by_treatment
# ──────────────────────────────────────────────────────────────────────


class TestGroupRuns:
    def test_groups_by_detected_treatment(self):
        runs = [
            _make_run("pixel-s1"),
            _make_run("symbolic-s1"),
            _make_run("pixel-s2"),
            _make_run("hybrid-s1"),
        ]
        grouped = group_runs_by_treatment(runs)
        assert set(grouped.keys()) == {"pixel", "symbolic", "hybrid"}
        assert len(grouped["pixel"]) == 2
        assert len(grouped["symbolic"]) == 1
        assert len(grouped["hybrid"]) == 1

    def test_deterministic_order_known_first(self):
        runs = [
            _make_run("hybrid-s1"),
            _make_run("symbolic-s1"),
            _make_run("pixel-s1"),
            _make_run("zzz-unknown"),
        ]
        grouped = group_runs_by_treatment(runs)
        keys = list(grouped.keys())
        # KNOWN_TREATMENTS order: pixel, symbolic, hybrid → unknown trails
        assert keys.index("pixel") < keys.index("symbolic") < keys.index("hybrid")
        assert keys[-1] == UNKNOWN_TREATMENT

    def test_unknown_runs_grouped_under_unknown_key(self):
        grouped = group_runs_by_treatment([_make_run("foobar")])
        assert UNKNOWN_TREATMENT in grouped
        assert len(grouped[UNKNOWN_TREATMENT]) == 1

    def test_empty_input(self):
        assert group_runs_by_treatment([]) == {}


# ──────────────────────────────────────────────────────────────────────
# Treatment color / label helpers
# ──────────────────────────────────────────────────────────────────────


class TestTreatmentDisplay:
    def test_known_treatments_have_colors(self):
        for t in ("pixel", "symbolic", "hybrid"):
            assert treatment_color(t).startswith("#")

    def test_unknown_falls_back(self):
        # A grey hex
        assert treatment_color("nonexistent").startswith("#")

    def test_label_capitalises(self):
        assert treatment_label("pixel") == "Pixel"
        assert treatment_label("multi_modal") == "Multi-Modal"

    def test_label_unknown_titlecases(self):
        assert treatment_label("foo") == "Foo"


# ──────────────────────────────────────────────────────────────────────
# learning_curves_with_bands
# ──────────────────────────────────────────────────────────────────────


class TestLearningCurves:
    def test_single_treatment_single_seed(self):
        runs = [_make_run("pixel-s1", rewards=list(range(50)))]
        curves = learning_curves_with_bands(runs, smooth_window=1)
        assert len(curves) == 1
        c = curves[0]
        assert c.treatment == "pixel"
        assert c.n_seeds == 1
        assert len(c.episode) == 50
        # std collapses to 0 with one seed
        assert np.allclose(c.std, 0.0)

    def test_multiple_seeds_compute_band(self):
        runs = [
            _make_run("pixel-s1", rewards=[1.0, 2.0, 3.0, 4.0]),
            _make_run("pixel-s2", rewards=[3.0, 4.0, 5.0, 6.0]),
            _make_run("pixel-s3", rewards=[5.0, 6.0, 7.0, 8.0]),
        ]
        curves = learning_curves_with_bands(runs, smooth_window=1)
        c = curves[0]
        assert c.n_seeds == 3
        # mean of (1, 3, 5) at episode 1 = 3.0
        assert c.mean[0] == pytest.approx(3.0)
        # std (sample, ddof=1) at episode 1 of (1,3,5) = sqrt(((-2)^2+0^2+2^2)/2) = 2.0
        assert c.std[0] == pytest.approx(2.0)

    def test_seeds_aligned_to_shortest(self):
        runs = [
            _make_run("pixel-s1", rewards=list(range(10))),
            _make_run("pixel-s2", rewards=list(range(5))),  # shortest
            _make_run("pixel-s3", rewards=list(range(20))),
        ]
        curves = learning_curves_with_bands(runs, smooth_window=1)
        assert len(curves[0].episode) == 5

    def test_smoothing_reduces_noise(self):
        runs = [
            _make_run(
                "pixel-s1",
                rewards=[0.0 if i % 2 == 0 else 100.0 for i in range(100)],
            ),
        ]
        # No smoothing — peaks at 100
        c_raw = learning_curves_with_bands(runs, smooth_window=1)[0]
        # Smoothed — should attenuate peaks
        c_smoothed = learning_curves_with_bands(runs, smooth_window=20)[0]
        assert float(c_smoothed.mean.max()) < float(c_raw.mean.max())

    def test_max_episodes_truncates(self):
        runs = [_make_run("pixel-s1", rewards=list(range(100)))]
        curves = learning_curves_with_bands(runs, smooth_window=1, max_episodes=10)
        assert len(curves[0].episode) == 10

    def test_treatment_with_no_data_omitted(self):
        runs = [
            _make_run("pixel-s1", rewards=[1.0, 2.0]),
            _make_run("symbolic-s1", rewards=None),  # no monitor_df / no episodes
        ]
        curves = learning_curves_with_bands(runs)
        treatments = {c.treatment for c in curves}
        assert "pixel" in treatments
        assert "symbolic" not in treatments

    def test_to_dataframe(self):
        c = LearningCurve(
            treatment="pixel",
            episode=np.array([1, 2, 3]),
            mean=np.array([1.0, 2.0, 3.0]),
            std=np.array([0.1, 0.2, 0.3]),
            n_seeds=3,
        )
        df = c.to_dataframe()
        assert list(df.columns) == ["treatment", "episode", "mean", "std", "lower", "upper"]
        assert df["upper"].iloc[0] == pytest.approx(1.1)
        assert df["lower"].iloc[1] == pytest.approx(1.8)


# ──────────────────────────────────────────────────────────────────────
# IQM helpers
# ──────────────────────────────────────────────────────────────────────


class TestIQMHelpers:
    def test_iqm_drops_tails(self):
        # IQM of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] uses values in [25th, 75th]
        # percentiles (= [2.25, 6.75]) → numpy includes 3, 4, 5, 6 in mask
        # 25th percentile = 2.25, 75th = 6.75; values 3..6 satisfy >= 2.25 AND <= 6.75
        scores = np.array(list(range(10)), dtype=float)
        # mean(3,4,5,6) = 4.5
        assert _iqm(scores) == pytest.approx(4.5)

    def test_iqm_small_sample_uses_mean(self):
        assert _iqm(np.array([1.0, 5.0])) == pytest.approx(3.0)
        assert _iqm(np.array([7.0])) == pytest.approx(7.0)

    def test_iqm_constant_array(self):
        assert _iqm(np.array([5.0] * 10)) == pytest.approx(5.0)

    def test_iqm_empty_returns_nan(self):
        assert np.isnan(_iqm(np.array([])))

    def test_bootstrap_ci_brackets_point_estimate(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rng = np.random.default_rng(42)
        lo, hi = _iqm_bootstrap_ci(scores, n_boot=2000, confidence=0.95, rng=rng)
        # IQM of a 5-element array uses inner ~3 values; CI should be sane
        assert lo <= _iqm(scores) <= hi
        assert lo >= scores.min()
        assert hi <= scores.max() + 1e-9


# ──────────────────────────────────────────────────────────────────────
# treatment_summary_table
# ──────────────────────────────────────────────────────────────────────


class TestTreatmentSummary:
    def test_summary_contains_iqm_and_ci(self):
        runs = [
            _make_run("pixel-s1", rewards=[10.0, 20.0, 30.0]),
            _make_run("pixel-s2", rewards=[15.0, 25.0, 35.0]),
            _make_run("pixel-s3", rewards=[20.0, 30.0, 40.0]),
        ]
        df = treatment_summary_table(runs, n_boot=200, rng_seed=42)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["treatment"] == "pixel"
        assert row["n_seeds"] == 3
        # best per-seed = 30, 35, 40
        assert row["mean"] == pytest.approx(35.0)
        assert row["median"] == pytest.approx(35.0)
        assert row["iqm_lo"] <= row["iqm"] <= row["iqm_hi"]

    def test_multiple_treatments(self):
        runs = [
            _make_run("pixel-s1", rewards=[10.0, 100.0]),
            _make_run("symbolic-s1", rewards=[5.0, 50.0]),
            _make_run("hybrid-s1", rewards=[20.0, 200.0]),
        ]
        df = treatment_summary_table(runs, n_boot=100, rng_seed=42)
        assert set(df["treatment"]) == {"pixel", "symbolic", "hybrid"}

    def test_runs_without_rewards_skipped(self):
        runs = [
            _make_run("pixel-s1", rewards=[10.0, 20.0]),
            _make_run("pixel-s2", rewards=None),
        ]
        df = treatment_summary_table(runs, n_boot=100, rng_seed=42)
        assert df.iloc[0]["n_seeds"] == 1

    def test_single_seed_yields_zero_std(self):
        runs = [_make_run("pixel-s1", rewards=[5.0, 10.0])]
        df = treatment_summary_table(runs, n_boot=100, rng_seed=42)
        assert df.iloc[0]["std"] == 0.0

    def test_custom_metric_fn(self):
        runs = [
            _make_run("pixel-s1", rewards=[1.0, 2.0]),
            _make_run("pixel-s2", rewards=[3.0, 4.0]),
        ]
        # Use mean instead of best
        df = treatment_summary_table(
            runs,
            metric_fn=lambda r: float(r.monitor_df["r"].mean()),
            n_boot=100,
            rng_seed=42,
        )
        # mean rewards: 1.5, 3.5 → overall mean = 2.5
        assert df.iloc[0]["mean"] == pytest.approx(2.5)


# ──────────────────────────────────────────────────────────────────────
# milestone_first_episode
# ──────────────────────────────────────────────────────────────────────


class TestMilestoneRace:
    def test_per_treatment_aggregation(self):
        runs = [
            _make_run("pixel-s1", flags={"GOT_STARTER": 5, "BEAT_BROCK": 50}),
            _make_run("pixel-s2", flags={"GOT_STARTER": 10, "BEAT_BROCK": 60}),
            _make_run("symbolic-s1", flags={"GOT_STARTER": 3, "BEAT_BROCK": 100}),
        ]
        df = milestone_first_episode(runs, aggregator="median")
        # Median first-episode for GOT_STARTER, pixel: median(5, 10) = 7.5
        pixel_starter = df[
            (df["treatment"] == "pixel") & (df["flag"] == "GOT_STARTER")
        ].iloc[0]
        assert pixel_starter["first_episode"] == pytest.approx(7.5)
        # Min aggregator
        df_min = milestone_first_episode(runs, aggregator="min")
        pixel_starter_min = df_min[
            (df_min["treatment"] == "pixel") & (df_min["flag"] == "GOT_STARTER")
        ].iloc[0]
        assert pixel_starter_min["first_episode"] == 5

    def test_flag_not_triggered_in_treatment_is_nan(self):
        runs = [
            _make_run("pixel-s1", flags={"GOT_STARTER": 5}),
            _make_run("symbolic-s1", flags={"GOT_STARTER": 3, "BEAT_BROCK": 50}),
        ]
        df = milestone_first_episode(runs)
        pixel_brock = df[
            (df["treatment"] == "pixel") & (df["flag"] == "BEAT_BROCK")
        ].iloc[0]
        assert np.isnan(pixel_brock["first_episode"])
        assert pixel_brock["n_seeds_triggered"] == 0

    def test_flags_whitelist(self):
        runs = [_make_run("pixel-s1", flags={"X": 1, "Y": 2, "Z": 3})]
        df = milestone_first_episode(runs, flags=["X", "Z"])
        assert set(df["flag"]) == {"X", "Z"}

    def test_invalid_aggregator_raises(self):
        with pytest.raises(ValueError):
            milestone_first_episode([_make_run("pixel-s1")], aggregator="garbage")

    def test_empty_runs_yields_empty_df(self):
        df = milestone_first_episode([])
        assert df.empty


# ──────────────────────────────────────────────────────────────────────
# final_performance
# ──────────────────────────────────────────────────────────────────────


class TestFinalPerformance:
    def test_uses_last_window(self):
        runs = [
            _make_run("pixel-s1", rewards=[0.0] * 100 + [10.0] * 50),
            _make_run("pixel-s2", rewards=[0.0] * 100 + [20.0] * 50),
        ]
        df = final_performance(runs, window=50)
        row = df.iloc[0]
        assert row["treatment"] == "pixel"
        # Mean across seeds of (10.0, 20.0) = 15.0
        assert row["mean"] == pytest.approx(15.0)

    def test_window_larger_than_episodes_uses_all(self):
        runs = [_make_run("pixel-s1", rewards=[1.0, 2.0, 3.0])]
        df = final_performance(runs, window=100)
        # mean of all 3 = 2.0
        assert df.iloc[0]["mean"] == pytest.approx(2.0)

    def test_skips_empty_runs(self):
        runs = [
            _make_run("pixel-s1", rewards=[1.0, 2.0]),
            _make_run("pixel-s2"),  # no rewards
        ]
        df = final_performance(runs)
        assert df.iloc[0]["n_seeds"] == 1


# ──────────────────────────────────────────────────────────────────────
# Plotting + export
# ──────────────────────────────────────────────────────────────────────


class TestPlotting:
    def test_setup_publication_style_runs(self):
        # Just ensure no exceptions
        setup_publication_style()

    def test_plot_learning_curves_returns_figure(self):
        runs = [
            _make_run("pixel-s1", rewards=[1.0, 2.0, 3.0]),
            _make_run("pixel-s2", rewards=[2.0, 3.0, 4.0]),
        ]
        curves = learning_curves_with_bands(runs, smooth_window=1)
        fig = plot_learning_curves(curves)
        assert fig is not None
        assert len(fig.axes) == 1
        # Title set
        assert fig.axes[0].get_title()

    def test_plot_final_performance_bars(self):
        df = pd.DataFrame(
            {
                "treatment": ["pixel", "symbolic"],
                "n_seeds": [3, 3],
                "mean": [10.0, 5.0],
                "std": [1.0, 0.5],
                "window": [50, 50],
            }
        )
        fig = plot_final_performance_bars(df)
        assert fig is not None
        # Two bars
        bars = fig.axes[0].patches
        assert len(bars) == 2

    def test_plot_handles_empty_data(self):
        fig = plot_final_performance_bars(pd.DataFrame())
        assert fig is not None  # axis is turned off but figure exists

    def test_export_pdf(self, tmp_path):
        runs = [_make_run("pixel-s1", rewards=[1.0, 2.0, 3.0])]
        curves = learning_curves_with_bands(runs, smooth_window=1)
        fig = plot_learning_curves(curves)
        out = export_figure(fig, str(tmp_path / "out.pdf"), dpi=300)
        assert out.exists()
        assert out.stat().st_size > 0
        assert out.suffix == ".pdf"

    def test_export_svg(self, tmp_path):
        runs = [_make_run("pixel-s1", rewards=[1.0, 2.0])]
        curves = learning_curves_with_bands(runs, smooth_window=1)
        fig = plot_learning_curves(curves)
        out = export_figure(fig, str(tmp_path / "out.svg"))
        assert out.exists()
        assert out.suffix == ".svg"

    def test_export_png(self, tmp_path):
        runs = [_make_run("pixel-s1", rewards=[1.0, 2.0])]
        curves = learning_curves_with_bands(runs, smooth_window=1)
        fig = plot_learning_curves(curves)
        out = export_figure(fig, str(tmp_path / "out.png"))
        assert out.exists()
        assert out.suffix == ".png"

    def test_export_unsupported_extension_raises(self, tmp_path):
        runs = [_make_run("pixel-s1", rewards=[1.0])]
        curves = learning_curves_with_bands(runs, smooth_window=1)
        fig = plot_learning_curves(curves)
        with pytest.raises(ValueError):
            export_figure(fig, str(tmp_path / "out.tiff"))

    def test_export_creates_parent_dir(self, tmp_path):
        runs = [_make_run("pixel-s1", rewards=[1.0])]
        curves = learning_curves_with_bands(runs, smooth_window=1)
        fig = plot_learning_curves(curves)
        out_path = tmp_path / "nested" / "deep" / "out.pdf"
        out = export_figure(fig, str(out_path))
        assert out.exists()


# ──────────────────────────────────────────────────────────────────────
# CLI parser for compare.py
# ──────────────────────────────────────────────────────────────────────


class TestCompareScriptParser:
    def test_default_args(self):
        from scripts.compare import build_parser
        args = build_parser().parse_args([])
        assert args.runs_dir == "./training_output"
        assert args.smooth_window == 20
        assert args.final_window == 50
        assert args.n_boot == 2000

    def test_overrides(self):
        from scripts.compare import build_parser
        args = build_parser().parse_args(
            ["--runs-dir", "/tmp/foo", "--smooth-window", "5", "--n-boot", "500"]
        )
        assert args.runs_dir == "/tmp/foo"
        assert args.smooth_window == 5
        assert args.n_boot == 500
