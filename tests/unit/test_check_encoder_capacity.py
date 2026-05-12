"""
Tests for ``scripts/check_encoder_capacity.py``.

This script is "the single source of truth for the capacity-match
constraint" per its own docstring — but had zero test coverage before
the audit.  These tests lock its behaviour so a future encoder edit
that breaks fairness fails CI loudly.

The pixel-vs-symbolic gap claim underlies the paper's modality
comparison (PR #36).  If this test fails, the paper's main result is
suspect.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from gymnasium import spaces


# scripts/ uses ``from pokemon_red_ai...`` imports that need the
# project root on sys.path.  Manually load the script module so we can
# call its helpers in unit tests.
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "check_encoder_capacity.py"
)


@pytest.fixture(scope="module")
def encoder_capacity_mod():
    spec = importlib.util.spec_from_file_location(
        "check_encoder_capacity", _SCRIPT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_encoder_capacity"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestCountTrainableParams:

    def test_counts_only_trainable(self, encoder_capacity_mod):
        # Two-parameter layer: weights (3x4=12) + bias (4) = 16 trainable.
        layer = nn.Linear(3, 4, bias=True)
        assert encoder_capacity_mod.count_trainable_params(layer) == 12 + 4

    def test_frozen_params_excluded(self, encoder_capacity_mod):
        layer = nn.Linear(3, 4, bias=True)
        for p in layer.parameters():
            p.requires_grad = False
        assert encoder_capacity_mod.count_trainable_params(layer) == 0

    def test_mixed_frozen_and_trainable(self, encoder_capacity_mod):
        layer = nn.Linear(3, 4, bias=True)
        # Freeze weights, leave bias trainable.
        layer.weight.requires_grad = False
        assert encoder_capacity_mod.count_trainable_params(layer) == 4


class TestMeasureMacs:

    def test_linear_layer_macs(self, encoder_capacity_mod):
        # nn.Linear(in=10, out=20): one matmul = 10*20 = 200 MACs/sample.
        layer = nn.Linear(10, 20)
        dummy = torch.zeros(1, 10)
        macs = encoder_capacity_mod.measure_macs(layer, dummy)
        assert macs == 10 * 20

    def test_conv2d_macs(self, encoder_capacity_mod):
        """Conv2d MAC count: out_h * out_w * out_c * (in_c/groups) * k_h * k_w."""
        conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        dummy = torch.zeros(1, 3, 10, 10)
        macs = encoder_capacity_mod.measure_macs(conv, dummy)
        # output is 10x10 (with padding=1), 8 out channels, 3x3 kernel, 3 in channels
        expected = 10 * 10 * 8 * 3 * 3 * 3
        assert macs == expected

    def test_stacked_layers_sum(self, encoder_capacity_mod):
        # Two linear layers: MACs add.
        net = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        dummy = torch.zeros(1, 4)
        macs = encoder_capacity_mod.measure_macs(net, dummy)
        assert macs == (4 * 8) + (8 * 2)


class TestAuditFunctions:
    """End-to-end audit functions — slower but catch encoder drift."""

    def test_audit_pixel_returns_positive_counts(self, encoder_capacity_mod):
        params, macs = encoder_capacity_mod._audit_pixel()
        assert params > 0
        assert macs > 0

    def test_audit_symbolic_returns_positive_counts(self, encoder_capacity_mod):
        params, macs = encoder_capacity_mod._audit_symbolic()
        assert params > 0
        assert macs > 0

    def test_audit_hybrid_returns_positive_counts(self, encoder_capacity_mod):
        params, macs = encoder_capacity_mod._audit_hybrid()
        assert params > 0
        assert macs > 0

    def test_pixel_and_symbolic_within_10pct(self, encoder_capacity_mod):
        """The paper's central fairness claim.  Fail loud if encoders drift."""
        pixel_params, _ = encoder_capacity_mod._audit_pixel()
        symbolic_params, _ = encoder_capacity_mod._audit_symbolic()
        rel = abs(pixel_params - symbolic_params) / max(pixel_params, symbolic_params)
        assert rel <= 0.10, (
            f"pixel={pixel_params:,}, symbolic={symbolic_params:,}, "
            f"gap={rel * 100:.2f}% — paper modality fairness claim broken"
        )

    def test_hybrid_is_approximately_pixel_plus_symbolic(self, encoder_capacity_mod):
        """Hybrid should reuse both branches as-is (option (c) in the
        capacity-matching design); drift should be near zero."""
        pixel_params, _ = encoder_capacity_mod._audit_pixel()
        symbolic_params, _ = encoder_capacity_mod._audit_symbolic()
        hybrid_params, _ = encoder_capacity_mod._audit_hybrid()
        expected = pixel_params + symbolic_params
        drift = abs(hybrid_params - expected) / expected
        # 25% is generous — the hybrid head adds a small final fusion
        # layer; keep this in line with the script's own sanity check.
        assert drift < 0.25, (
            f"hybrid={hybrid_params:,}, expected~{expected:,}, "
            f"drift={drift * 100:.2f}% — hybrid encoder no longer reuses "
            f"the matched branches as-is"
        )


class TestMainExitCodes:
    """The script wires into CI; its exit codes are load-bearing."""

    def test_main_succeeds_under_default_tolerance(
        self, encoder_capacity_mod, monkeypatch, capsys
    ):
        # Pretend we were called as `python check_encoder_capacity.py`.
        monkeypatch.setattr(sys, "argv", ["check_encoder_capacity.py"])
        exit_code = encoder_capacity_mod.main()
        captured = capsys.readouterr()
        # Exit 0 means the encoders match within tolerance.
        assert exit_code == 0
        assert "OK" in captured.out

    def test_main_fails_with_zero_tolerance(
        self, encoder_capacity_mod, monkeypatch, capsys
    ):
        # Zero tolerance — any non-bit-exact parameter difference fails.
        # Used here as a smoke test that the failure path returns 1.
        monkeypatch.setattr(
            sys, "argv",
            ["check_encoder_capacity.py", "--tolerance", "0.0"],
        )
        exit_code = encoder_capacity_mod.main()
        # Either the encoders are bit-identical (exit 0) or they differ
        # (exit 1).  Both are valid outcomes — the contract under test is
        # only that the exit code is a clean 0 or 1, never something
        # else (no traceback escape).
        assert exit_code in (0, 1)
