#!/usr/bin/env python3
"""
Capacity audit for the three paper observation treatments.

Reports trainable parameter counts and per-forward FLOPs for the pixel,
symbolic, and hybrid feature extractors, and asserts that pixel and
symbolic match to within 10% on parameter count. The hybrid encoder is
expected to be roughly the sum of the two single-modality encoders by
construction (option (c) in the capacity-matching design).

This script is the single source of truth for the capacity-match
constraint. Run it whenever you touch SymbolicFeaturesExtractor,
HybridFeaturesExtractor, or any policy_kwargs feeding into the paper
treatments. It exits with status 1 if the constraint fails so it can
be wired into CI.

Usage:
    python scripts/check_encoder_capacity.py
    python scripts/check_encoder_capacity.py --tolerance 0.10  # default

Methodology rationale: see the encoder section docstring in
pokemon_red_ai/training/models.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

# Make ``pokemon_red_ai`` importable when running this script directly
# (matches the pattern used by scripts/train.py and scripts/eval.py).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import NatureCNN

from pokemon_red_ai.environment.observations import (
    SYMBOLIC_DIM,
    create_hybrid_observation_space,
    create_pixel_observation_space,
    create_symbolic_observation_space,
)
from pokemon_red_ai.training.models import (
    PAPER_SYMBOLIC_DIM,
    HybridFeaturesExtractor,
    SymbolicFeaturesExtractor,
)


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def measure_macs(module: nn.Module, dummy_input) -> int:
    """
    Run one forward pass and accumulate MACs from Conv2d and Linear
    submodules via forward hooks. One MAC = one multiply-accumulate;
    FLOPs ≈ 2 × MACs by the common convention used in CNN benchmarks.
    """
    macs = [0]
    handles = []

    def conv_hook(mod, _inp, out):
        out_h, out_w = out.shape[-2], out.shape[-1]
        k_h, k_w = mod.kernel_size
        per_sample = (
            out_h * out_w * mod.out_channels
            * (mod.in_channels // mod.groups) * k_h * k_w
        )
        macs[0] += per_sample

    def linear_hook(mod, _inp, out):
        macs[0] += mod.in_features * mod.out_features

    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    module.eval()
    with torch.no_grad():
        module(dummy_input)

    for h in handles:
        h.remove()
    return macs[0]


def _to_chw_image_space(space: spaces.Box) -> spaces.Box:
    """
    Convert an HWC image Box space to CHW, mirroring SB3's
    VecTransposeImage wrapper applied at training time. Required because
    NatureCNN constructs its conv stack from the observation_space's shape
    and only accepts channels-first layouts.
    """
    h, w, c = space.shape
    return spaces.Box(low=0, high=255, shape=(c, h, w), dtype=space.dtype)


def _dummy_image_chw(chw_space: spaces.Box) -> torch.Tensor:
    # Normalize to [0, 1] like SB3's preprocess_obs does for image spaces.
    arr = np.random.randint(0, 256, size=(1, *chw_space.shape), dtype=np.uint8)
    return torch.from_numpy(arr).float() / 255.0


def _dummy_vector(space: spaces.Box) -> torch.Tensor:
    arr = np.random.uniform(
        low=float(space.low.min()),
        high=float(space.high.max()),
        size=(1, *space.shape),
    ).astype(np.float32)
    return torch.from_numpy(arr)


def _audit_pixel() -> Tuple[int, int]:
    chw_space = _to_chw_image_space(create_pixel_observation_space())
    extractor = NatureCNN(chw_space, features_dim=256)
    macs = measure_macs(extractor, _dummy_image_chw(chw_space))
    return count_trainable_params(extractor), macs


def _audit_symbolic() -> Tuple[int, int]:
    space = create_symbolic_observation_space()
    extractor = SymbolicFeaturesExtractor(space, features_dim=256)
    macs = measure_macs(extractor, _dummy_vector(space))
    return count_trainable_params(extractor), macs


def _audit_hybrid() -> Tuple[int, int]:
    hybrid_space = create_hybrid_observation_space()
    chw_screen = _to_chw_image_space(hybrid_space.spaces["screen"])
    transposed = spaces.Dict({
        "screen": chw_screen,
        "game_state": hybrid_space.spaces["game_state"],
    })
    extractor = HybridFeaturesExtractor(transposed, features_dim=512)
    dummy = {
        "screen": _dummy_image_chw(chw_screen),
        "game_state": _dummy_vector(hybrid_space.spaces["game_state"]),
    }
    macs = measure_macs(extractor, dummy)
    return count_trainable_params(extractor), macs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tolerance", type=float, default=0.10,
        help="Max allowed relative gap between pixel and symbolic params "
             "(default: 0.10 == 10%%).",
    )
    args = parser.parse_args()

    # Sanity check: the symbolic input dim assumed by the encoder must
    # match the observation builder. Otherwise the param target drifts
    # silently when someone adds a feature.
    if PAPER_SYMBOLIC_DIM != SYMBOLIC_DIM:
        print(
            f"FAIL: PAPER_SYMBOLIC_DIM ({PAPER_SYMBOLIC_DIM}) != "
            f"observations.SYMBOLIC_DIM ({SYMBOLIC_DIM}). The symbolic "
            "encoder's parameter budget was chosen for a specific input "
            "dim; update SYMBOLIC_HIDDEN_DIM and re-run this check.",
            file=sys.stderr,
        )
        return 1

    pixel_params, pixel_macs = _audit_pixel()
    symbolic_params, symbolic_macs = _audit_symbolic()
    hybrid_params, hybrid_macs = _audit_hybrid()

    rows = [
        ("pixel    (NatureCNN, features_dim=256)",       pixel_params,    pixel_macs),
        ("symbolic (3-layer MLP 29->640->640->256)",     symbolic_params, symbolic_macs),
        ("hybrid   (NatureCNN(256) + symbolic(256))",    hybrid_params,   hybrid_macs),
    ]
    print()
    print(f"{'Encoder':<48} {'Params':>12} {'MACs':>14} {'~FLOPs':>14}")
    print("-" * 92)
    for name, params, macs in rows:
        print(f"{name:<48} {params:>12,} {macs:>14,} {2 * macs:>14,}")
    print()

    # Param-count match check (the strict constraint).
    diff = abs(pixel_params - symbolic_params)
    base = max(pixel_params, symbolic_params)
    rel = diff / base
    print(f"pixel vs symbolic params: {rel * 100:.2f}% gap "
          f"(tolerance {args.tolerance * 100:.0f}%)")

    # Hybrid sanity: should be roughly pixel + symbolic.
    expected_hybrid = pixel_params + symbolic_params
    hybrid_drift = abs(hybrid_params - expected_hybrid) / expected_hybrid
    print(f"hybrid vs pixel+symbolic: {hybrid_drift * 100:.2f}% drift "
          "(should be ~0; both halves are reused as-is)")

    if rel > args.tolerance:
        print(
            f"\nFAIL: pixel and symbolic encoders differ by "
            f"{rel * 100:.2f}% on parameter count, exceeding the "
            f"{args.tolerance * 100:.0f}% tolerance. Adjust "
            "SYMBOLIC_HIDDEN_DIM in pokemon_red_ai/training/models.py.",
            file=sys.stderr,
        )
        return 1

    print("\nOK: encoders are capacity-matched within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
