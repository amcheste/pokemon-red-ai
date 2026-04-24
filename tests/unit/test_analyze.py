"""
Tests for scripts/analyze.py — rliable analysis pipeline.

Covers data loading, score matrix construction, aggregate metric
computation, probability of improvement, performance profiles,
and figure generation.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
import pytest

# Use non-interactive backend for CI
matplotlib.use("Agg")

import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.analyze import (
    build_score_matrix,
    compute_aggregate_metrics,
    compute_performance_profiles,
    compute_probability_of_improvement,
    load_csv,
    load_eval_jsons,
    load_sample_efficiency_csv,
    plot_aggregate_metrics,
    plot_performance_profiles,
    plot_probability_of_improvement,
    plot_sample_efficiency,
    print_summary_report,
    METRIC_KEYS,
    TREATMENT_COLORS,
    TREATMENT_DISPLAY,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _make_eval_json(
    brock_win_rate: float = 0.5,
    mean_return: float = 800.0,
    seed: int = 0,
    treatment: str = "hybrid",
) -> dict:
    """Create a synthetic eval JSON matching eval.py output schema."""
    return {
        "brock_win_rate": brock_win_rate,
        "mean_return": mean_return,
        "return_std": 50.0,
        "mean_event_flags_triggered": 8.0,
        "max_event_flags_triggered": 12,
        "unique_maps_visited": 7,
        "steps_to_pewter": 5000,
        "steps_to_brock_win": 12000,
        "n_episodes": 20,
        "eval_save_state": "save_states/s0_post_intro.state",
        "checkpoint_path": f"models/{treatment}_seed{seed}.zip",
        "git_sha": "abc123",
        "algorithm": "RecurrentPPO",
        "observation_type": treatment,
        "seed": seed,
    }


@pytest.fixture
def eval_results_dir(tmp_path):
    """Create a temp directory with 3 treatments × 5 seeds of eval JSONs."""
    rng = np.random.RandomState(42)

    treatments = {
        "pixel": {"base_wr": 0.15, "base_return": 500},
        "symbolic": {"base_wr": 0.45, "base_return": 850},
        "hybrid": {"base_wr": 0.55, "base_return": 920},
    }

    for treatment, params in treatments.items():
        tdir = tmp_path / treatment
        tdir.mkdir()
        for seed in range(5):
            data = _make_eval_json(
                brock_win_rate=max(0, params["base_wr"] + rng.normal(0, 0.1)),
                mean_return=params["base_return"] + rng.normal(0, 50),
                seed=seed,
                treatment=treatment,
            )
            (tdir / f"seed_{seed}.json").write_text(json.dumps(data, indent=2))

    return tmp_path


@pytest.fixture
def csv_results(tmp_path):
    """Create a temp CSV with results."""
    csv_path = tmp_path / "results.csv"
    rng = np.random.RandomState(42)

    lines = ["treatment,seed,brock_win_rate,mean_return,mean_event_flags_triggered,unique_maps_visited"]
    for treatment, base_wr in [("pixel", 0.15), ("symbolic", 0.45), ("hybrid", 0.55)]:
        for seed in range(5):
            wr = max(0, base_wr + rng.normal(0, 0.1))
            ret = 500 + rng.normal(0, 50)
            lines.append(f"{treatment},{seed},{wr:.4f},{ret:.1f},8.0,7")

    csv_path.write_text("\n".join(lines))
    return csv_path


@pytest.fixture
def efficiency_csv(tmp_path):
    """Create a temp CSV with sample efficiency data."""
    csv_path = tmp_path / "efficiency.csv"
    rng = np.random.RandomState(42)

    lines = ["treatment,seed,timestep,brock_win_rate"]
    steps = [1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000]

    for treatment, base_wr in [("pixel", 0.1), ("symbolic", 0.4), ("hybrid", 0.5)]:
        for seed in range(3):
            for i, step in enumerate(steps):
                # Scores increase with timestep
                wr = min(1.0, max(0, base_wr * (i + 1) / len(steps) + rng.normal(0, 0.05)))
                lines.append(f"{treatment},{seed},{step},{wr:.4f}")

    csv_path.write_text("\n".join(lines))
    return csv_path


# ──────────────────────────────────────────────────────────────────────
# Data loading tests
# ──────────────────────────────────────────────────────────────────────


class TestLoadEvalJsons:
    """Tests for loading evaluation JSON files."""

    def test_loads_all_treatments(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        assert set(results.keys()) == {"pixel", "symbolic", "hybrid"}

    def test_correct_seed_count(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        for treatment, evals in results.items():
            assert len(evals) == 5, f"{treatment} should have 5 seeds"

    def test_metrics_present(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        for treatment, evals in results.items():
            for e in evals:
                assert "brock_win_rate" in e
                assert "mean_return" in e
                assert "mean_event_flags_triggered" in e

    def test_source_file_tracked(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        for evals in results.values():
            for e in evals:
                assert "_source_file" in e

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_eval_jsons(tmp_path / "nonexistent")

    def test_empty_dir_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No eval JSONs found"):
            load_eval_jsons(empty)

    def test_ignores_non_json_files(self, tmp_path):
        tdir = tmp_path / "pixel"
        tdir.mkdir()
        # Write one valid JSON and one non-JSON file
        (tdir / "seed_0.json").write_text(json.dumps(_make_eval_json()))
        (tdir / "notes.txt").write_text("not a json file")

        results = load_eval_jsons(tmp_path)
        assert len(results["pixel"]) == 1


class TestLoadCsv:
    """Tests for CSV data loading."""

    def test_loads_all_treatments(self, csv_results):
        results = load_csv(csv_results)
        assert set(results.keys()) == {"pixel", "symbolic", "hybrid"}

    def test_correct_seed_count(self, csv_results):
        results = load_csv(csv_results)
        for treatment, evals in results.items():
            assert len(evals) == 5

    def test_numeric_conversion(self, csv_results):
        results = load_csv(csv_results)
        for evals in results.values():
            for e in evals:
                assert isinstance(e["brock_win_rate"], float)


class TestLoadSampleEfficiencyCsv:
    """Tests for sample efficiency CSV loading."""

    def test_has_frames(self, efficiency_csv):
        data = load_sample_efficiency_csv(efficiency_csv)
        assert "frames" in data
        assert len(data["frames"]) == 5  # 5 timestep values

    def test_has_treatments(self, efficiency_csv):
        data = load_sample_efficiency_csv(efficiency_csv)
        treatments = {k for k in data if k != "frames"}
        assert treatments == {"pixel", "symbolic", "hybrid"}

    def test_correct_shape(self, efficiency_csv):
        data = load_sample_efficiency_csv(efficiency_csv)
        for treatment in ["pixel", "symbolic", "hybrid"]:
            assert data[treatment].shape == (3, 5)  # 3 seeds × 5 timesteps

    def test_frames_sorted(self, efficiency_csv):
        data = load_sample_efficiency_csv(efficiency_csv)
        frames = data["frames"]
        assert np.all(frames[:-1] <= frames[1:])


# ──────────────────────────────────────────────────────────────────────
# Score matrix tests
# ──────────────────────────────────────────────────────────────────────


class TestBuildScoreMatrix:
    """Tests for score matrix construction."""

    def test_shape(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")

        for treatment, arr in scores.items():
            assert arr.shape == (5, 1), f"{treatment} should be (5, 1)"

    def test_values_match_input(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")

        for treatment, evals in results.items():
            expected = [e["brock_win_rate"] for e in evals]
            np.testing.assert_array_almost_equal(
                scores[treatment].flatten(), expected,
            )

    def test_normalization(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(
            results, metric="mean_return", normalize=True,
        )

        for arr in scores.values():
            assert arr.min() >= 0.0
            assert arr.max() <= 1.0

    def test_normalization_with_custom_max(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(
            results, metric="mean_return",
            normalize=True, max_score=1500.0,
        )

        for arr in scores.values():
            # All values should be < 1.0 since max_score=1500 > all returns
            assert arr.max() < 1.0

    def test_missing_metric_defaults_to_zero(self, tmp_path):
        tdir = tmp_path / "pixel"
        tdir.mkdir()
        data = {"brock_win_rate": 0.5}  # Missing mean_return
        (tdir / "seed_0.json").write_text(json.dumps(data))

        results = load_eval_jsons(tmp_path)
        scores = build_score_matrix(results, metric="mean_return")
        assert scores["pixel"][0, 0] == 0.0


# ──────────────────────────────────────────────────────────────────────
# rliable computation tests
# ──────────────────────────────────────────────────────────────────────


class TestComputeAggregateMetrics:
    """Tests for IQM, mean, median, optimality gap computation."""

    def test_returns_four_metrics(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")

        point_est, interval_est = compute_aggregate_metrics(scores, reps=100)

        assert set(point_est.keys()) == {"IQM", "Mean", "Median", "Optimality Gap"}
        assert set(interval_est.keys()) == {"IQM", "Mean", "Median", "Optimality Gap"}

    def test_all_treatments_present(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")

        point_est, interval_est = compute_aggregate_metrics(scores, reps=100)

        for metric_name in ["IQM", "Mean"]:
            assert set(point_est[metric_name].keys()) == {"pixel", "symbolic", "hybrid"}

    def test_ci_contains_point_estimate(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")

        point_est, interval_est = compute_aggregate_metrics(scores, reps=100)

        for metric_name in ["IQM", "Mean", "Median"]:
            for treatment in scores:
                val = point_est[metric_name][treatment]
                ci = interval_est[metric_name][treatment]
                # CI should contain the point estimate (within bootstrap noise)
                assert ci[0] <= val + 0.05, (
                    f"{metric_name}/{treatment}: CI lo {ci[0]} > val {val}"
                )
                assert ci[1] >= val - 0.05, (
                    f"{metric_name}/{treatment}: CI hi {ci[1]} < val {val}"
                )

    def test_iqm_within_range(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")

        point_est, _ = compute_aggregate_metrics(scores, reps=100)

        for treatment in scores:
            iqm = point_est["IQM"][treatment]
            raw_mean = float(scores[treatment].mean())
            # IQM should be in a reasonable range around the mean
            assert 0.0 <= iqm <= 1.0, f"IQM out of range: {iqm}"

    def test_pixel_lower_than_symbolic(self, eval_results_dir):
        """Validates the synthetic data has expected ordering."""
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")

        point_est, _ = compute_aggregate_metrics(scores, reps=100)

        assert point_est["IQM"]["pixel"] < point_est["IQM"]["symbolic"]


class TestComputeProbabilityOfImprovement:
    """Tests for pairwise probability of improvement."""

    def test_returns_all_pairs(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")

        poi_point, poi_ci = compute_probability_of_improvement(scores, reps=100)

        # 3 treatments → 6 ordered pairs (A,B) and (B,A)
        assert len(poi_point) == 6
        assert len(poi_ci) == 6

    def test_poi_between_0_and_1(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")

        poi_point, _ = compute_probability_of_improvement(scores, reps=100)

        for key, val in poi_point.items():
            assert 0.0 <= val <= 1.0, f"POI {key} out of range: {val}"

    def test_complementary_pairs(self, eval_results_dir):
        """P(A > B) + P(B > A) should sum to 1.0."""
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")

        poi_point, _ = compute_probability_of_improvement(scores, reps=100)

        treatments = sorted(scores.keys())
        for i, t_a in enumerate(treatments):
            for j, t_b in enumerate(treatments):
                if i >= j:
                    continue
                ab = poi_point[(t_a, t_b)]
                ba = poi_point[(t_b, t_a)]
                assert abs(ab + ba - 1.0) < 1e-6, (
                    f"P({t_a}>{t_b}) + P({t_b}>{t_a}) = {ab + ba}, expected 1.0"
                )

    def test_ci_bounds_ordered(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")

        _, poi_ci = compute_probability_of_improvement(scores, reps=100)

        for key, ci in poi_ci.items():
            assert ci[0] <= ci[1], f"CI for {key}: lo={ci[0]} > hi={ci[1]}"

    def test_symbolic_beats_pixel(self, eval_results_dir):
        """Synthetic data is designed so symbolic >> pixel."""
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")

        poi_point, _ = compute_probability_of_improvement(scores, reps=100)

        assert poi_point[("symbolic", "pixel")] > 0.7


class TestComputePerformanceProfiles:
    """Tests for performance profile computation."""

    def test_returns_correct_shape(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")

        taus, profiles, profile_cis = compute_performance_profiles(
            scores, reps=100,
        )

        assert len(taus) == 51  # default linspace(0, 1, 51)
        for treatment in scores:
            assert profiles[treatment].shape == (51,)
            assert profile_cis[treatment].shape == (2, 51)

    def test_profiles_monotone_decreasing(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")

        taus, profiles, _ = compute_performance_profiles(scores, reps=100)

        for treatment, profile in profiles.items():
            # Performance profiles should be monotonically non-increasing
            for i in range(len(profile) - 1):
                assert profile[i] >= profile[i + 1] - 0.01, (
                    f"{treatment}: profile[{i}]={profile[i]:.3f} < "
                    f"profile[{i+1}]={profile[i+1]:.3f}"
                )

    def test_starts_at_1_ends_at_0(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")

        taus, profiles, _ = compute_performance_profiles(scores, reps=100)

        for treatment, profile in profiles.items():
            # At tau=0, all runs should score >= 0
            assert profile[0] >= 0.9, (
                f"{treatment}: profile at tau=0 should be ~1.0, got {profile[0]}"
            )


# ──────────────────────────────────────────────────────────────────────
# Plotting tests (smoke tests — verify no crashes, output exists)
# ──────────────────────────────────────────────────────────────────────


class TestPlotAggregateMetrics:
    """Smoke tests for aggregate metric bar chart."""

    def test_generates_figure(self, eval_results_dir, tmp_path):
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")
        point_est, interval_est = compute_aggregate_metrics(scores, reps=100)

        output = tmp_path / "test_iqm.png"
        fig = plot_aggregate_metrics(
            point_est, interval_est,
            metric_name="IQM",
            output_path=output,
        )

        assert output.exists()
        assert output.stat().st_size > 0
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_returns_figure_object(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")
        point_est, interval_est = compute_aggregate_metrics(scores, reps=100)

        fig = plot_aggregate_metrics(point_est, interval_est, metric_name="IQM")

        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotPerformanceProfiles:
    """Smoke tests for performance profile plot."""

    def test_generates_figure(self, eval_results_dir, tmp_path):
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")
        taus, profiles, cis = compute_performance_profiles(scores, reps=100)

        output = tmp_path / "test_profiles.png"
        fig = plot_performance_profiles(
            taus, profiles, cis, output_path=output,
        )

        assert output.exists()
        assert output.stat().st_size > 0
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotProbabilityOfImprovement:
    """Smoke tests for POI heatmap."""

    def test_generates_figure(self, eval_results_dir, tmp_path):
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")
        poi_point, poi_ci = compute_probability_of_improvement(scores, reps=100)

        output = tmp_path / "test_poi.png"
        fig = plot_probability_of_improvement(
            poi_point, poi_ci, output_path=output,
        )

        assert output.exists()
        assert output.stat().st_size > 0
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotSampleEfficiency:
    """Smoke tests for sample efficiency curves."""

    def test_generates_figure(self, efficiency_csv, tmp_path):
        data = load_sample_efficiency_csv(efficiency_csv)
        frames = data.pop("frames")

        output = tmp_path / "test_efficiency.png"
        fig = plot_sample_efficiency(
            frames, data,
            output_path=output,
            reps=50,  # fast for tests
        )

        assert output.exists()
        assert output.stat().st_size > 0
        import matplotlib.pyplot as plt
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Summary report tests
# ──────────────────────────────────────────────────────────────────────


class TestPrintSummaryReport:
    """Tests for the text report generation."""

    def test_contains_treatment_names(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")
        point_est, interval_est = compute_aggregate_metrics(scores, reps=100)

        report = print_summary_report(
            results, point_est, interval_est,
        )

        assert "Pixel" in report
        assert "Symbolic" in report
        assert "Hybrid" in report

    def test_contains_iqm(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")
        point_est, interval_est = compute_aggregate_metrics(scores, reps=100)

        report = print_summary_report(
            results, point_est, interval_est,
        )

        assert "IQM" in report

    def test_contains_poi_when_provided(self, eval_results_dir):
        results = load_eval_jsons(eval_results_dir)
        scores = build_score_matrix(results, metric="brock_win_rate")
        point_est, interval_est = compute_aggregate_metrics(scores, reps=100)
        poi_point, poi_ci = compute_probability_of_improvement(scores, reps=100)

        report = print_summary_report(
            results, point_est, interval_est,
            poi_point, poi_ci,
        )

        assert "PROBABILITY OF IMPROVEMENT" in report


# ──────────────────────────────────────────────────────────────────────
# Constants / config tests
# ──────────────────────────────────────────────────────────────────────


class TestConstants:
    """Tests for module-level constants."""

    def test_all_treatments_have_colors(self):
        for treatment in ["pixel", "symbolic", "hybrid"]:
            assert treatment in TREATMENT_COLORS

    def test_all_treatments_have_display_names(self):
        for treatment in ["pixel", "symbolic", "hybrid"]:
            assert treatment in TREATMENT_DISPLAY

    def test_metric_keys_valid(self):
        expected = {
            "brock_win_rate",
            "mean_return",
            "mean_event_flags_triggered",
            "unique_maps_visited",
        }
        assert set(METRIC_KEYS.keys()) == expected


# ──────────────────────────────────────────────────────────────────────
# CLI parser tests
# ──────────────────────────────────────────────────────────────────────


class TestCLI:
    """Tests for the argument parser."""

    def test_results_dir_arg(self):
        from scripts.analyze import build_parser
        parser = build_parser()
        args = parser.parse_args(["--results-dir", "eval_results/"])
        assert args.results_dir == Path("eval_results/")

    def test_csv_arg(self):
        from scripts.analyze import build_parser
        parser = build_parser()
        args = parser.parse_args(["--csv", "results.csv"])
        assert args.csv == Path("results.csv")

    def test_mutual_exclusion(self):
        from scripts.analyze import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--results-dir", "dir/", "--csv", "file.csv"])

    def test_default_reps(self):
        from scripts.analyze import build_parser
        parser = build_parser()
        args = parser.parse_args(["--results-dir", "eval_results/"])
        assert args.reps == 2000

    def test_custom_reps(self):
        from scripts.analyze import build_parser
        parser = build_parser()
        args = parser.parse_args(["--results-dir", "eval_results/", "--reps", "500"])
        assert args.reps == 500

    def test_plot_choices(self):
        from scripts.analyze import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "--results-dir", "eval_results/",
            "--plots", "iqm", "poi",
        ])
        assert args.plots == ["iqm", "poi"]
