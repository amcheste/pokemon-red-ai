#!/usr/bin/env python3
"""
Fixed evaluation harness for the Pokemon Red AI paper experiments.

Runs a locked, deterministic evaluation protocol on a trained checkpoint:

- N evaluation episodes (default 20, matches analysis_plan.md S3)
- Fixed starting save state (default s0_post_intro.state)
- Deterministic policy (argmax action)
- Exploration bonuses disabled at eval time
- Same seed for all eval episodes of the same checkpoint (default 42)

Outputs a metrics dict matching ``EVAL_METRIC_SCHEMA`` which is consumed by
the rliable analysis notebooks in ``paper/notebooks/``.

Locked fields:
    - ``n_episodes = 20``
    - ``seed = 42``
    - ``deterministic = True``

Changing any locked field after main experiments begin is a deviation from the
pre-registered plan and must be logged in ``paper/compute_ledger.md``.

Usage::

    # Evaluate a RecurrentPPO checkpoint
    python scripts/eval.py \\
        --checkpoint training_output/models/best_model.zip \\
        --rom PokemonRed.gb

    # Evaluate with custom save state and output file
    python scripts/eval.py \\
        --checkpoint training_output/models/best_model.zip \\
        --rom PokemonRed.gb \\
        --save-state save_states/s0_post_intro.state \\
        --output paper/notebooks/eval_results/hybrid_seed42.json

    # Quick smoke test (override locked episode count)
    python scripts/eval.py \\
        --checkpoint model.zip --rom PokemonRed.gb \\
        --n-episodes 3 --allow-override
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pokemon_red_ai.environment import PokemonRedGymEnv
from pokemon_red_ai.game.memory import MAP_IDS, BADGE_FLAGS
from pokemon_red_ai.game.rom import (
    compute_rom_sha256,
    RomHashMismatchError,
)

# Eval reproducibility: locked deterministic=True per analysis_plan.md S3.
# seed_everything covers Python random, NumPy, torch (CPU/CUDA/MPS),
# PYTHONHASHSEED, and SB3 — strictly more than the previous np.random.seed.
try:
    from scripts.seed_utils import seed_everything
except ImportError:
    from seed_utils import seed_everything

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Schema & locked defaults
# ──────────────────────────────────────────────────────────────────────

# Schema for the JSON blob this script emits. Notebooks in paper/notebooks/
# assume these exact keys -- do not rename without updating downstream consumers.
EVAL_METRIC_SCHEMA: Dict[str, type] = {
    "brock_win_rate": float,              # fraction of episodes that earned Boulder Badge
    "mean_event_flags_triggered": float,  # mean across episodes
    "max_event_flags_triggered": int,     # best single episode
    "unique_maps_visited": int,           # max across episodes
    "mean_return": float,                 # raw PPO episodic return
    "return_std": float,
    "steps_to_pewter": Optional[int],     # mean, None if no episode reached Pewter
    "steps_to_brock_win": Optional[int],  # mean, None if no Brock wins
    "n_episodes": int,
    "eval_save_state": str,
    "checkpoint_path": str,
    "rom_sha256": str,
    "git_sha": str,
}

# Locked defaults per analysis_plan.md S3
LOCKED_N_EPISODES = 20
LOCKED_SEED = 42

# Game constants for milestone detection
PEWTER_CITY_MAP_ID = MAP_IDS["pewter_city"]  # 3
BOULDER_BADGE_BIT = BADGE_FLAGS["boulder"]    # 0x01


# ──────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────


def load_model(checkpoint_path: Path, algorithm: str):
    """
    Load a trained SB3 model from a checkpoint.

    Args:
        checkpoint_path: Path to the ``.zip`` checkpoint.
        algorithm: One of 'PPO' or 'RecurrentPPO'.

    Returns:
        The loaded model instance.
    """
    if algorithm == "RecurrentPPO":
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(str(checkpoint_path))
    else:
        from stable_baselines3 import PPO
        model = PPO.load(str(checkpoint_path))

    logger.info(
        f"Loaded {algorithm} checkpoint from {checkpoint_path} "
        f"({sum(p.numel() for p in model.policy.parameters()):,} params)"
    )
    return model


def detect_algorithm(checkpoint_path: Path) -> str:
    """
    Try to detect whether a checkpoint is PPO or RecurrentPPO.

    Attempts RecurrentPPO first (it fails fast on PPO checkpoints).
    Falls back to PPO.
    """
    try:
        from sb3_contrib import RecurrentPPO
        RecurrentPPO.load(str(checkpoint_path))
        return "RecurrentPPO"
    except Exception:
        pass

    try:
        from stable_baselines3 import PPO
        PPO.load(str(checkpoint_path))
        return "PPO"
    except Exception:
        pass

    raise ValueError(
        f"Could not load checkpoint as PPO or RecurrentPPO: {checkpoint_path}"
    )


# ──────────────────────────────────────────────────────────────────────
# Single episode rollout
# ──────────────────────────────────────────────────────────────────────


def run_episode(
    env: PokemonRedGymEnv,
    model,
    is_recurrent: bool,
    deterministic: bool = True,
    max_steps: int = 15_000,
) -> Dict[str, Any]:
    """
    Run a single evaluation episode and collect metrics.

    Returns a dict with per-episode results including game milestones.
    """
    obs, info = env.reset()

    # RecurrentPPO LSTM state tracking
    lstm_states = None
    episode_start = True

    total_reward = 0.0
    steps = 0
    step_reached_pewter = None
    step_earned_boulder = None
    maps_seen = set()
    max_event_flags = 0

    while steps < max_steps:
        # Get action from policy
        if is_recurrent:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=deterministic,
            )
            episode_start = False
        else:
            action, _ = model.predict(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # Track map visits
        current_map = info.get("current_map", 0)
        if current_map != 0:
            maps_seen.add(current_map)

        # Detect Pewter City arrival
        if step_reached_pewter is None and current_map == PEWTER_CITY_MAP_ID:
            step_reached_pewter = steps
            logger.debug(f"  Reached Pewter City at step {steps}")

        # Detect Boulder Badge
        badges = info.get("badges_earned", 0)
        if step_earned_boulder is None and (badges & BOULDER_BADGE_BIT):
            step_earned_boulder = steps
            logger.debug(f"  Earned Boulder Badge at step {steps}")

        # Track event flags
        event_progress = info.get("event_progress", {})
        flags_triggered = event_progress.get("flags_triggered", 0)
        max_event_flags = max(max_event_flags, flags_triggered)

        if terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "steps": steps,
        "maps_visited": len(maps_seen),
        "maps_list": sorted(maps_seen),
        "event_flags_triggered": max_event_flags,
        "badges": info.get("badges_earned", 0),
        "earned_boulder_badge": step_earned_boulder is not None,
        "step_reached_pewter": step_reached_pewter,
        "step_earned_boulder": step_earned_boulder,
        "final_map": info.get("current_map", 0),
        "player_level": info.get("player_level", 0),
        "terminated": terminated,
        "truncated": truncated,
    }


# ──────────────────────────────────────────────────────────────────────
# Main evaluation function
# ──────────────────────────────────────────────────────────────────────


def evaluate_checkpoint(
    checkpoint_path: Path,
    rom_path: Path,
    save_state: Path,
    algorithm: str = "auto",
    n_episodes: int = LOCKED_N_EPISODES,
    deterministic: bool = True,
    seed: int = LOCKED_SEED,
    max_episode_steps: int = 15_000,
    observation_type: str = "multi_modal",
    allow_override: bool = False,
    expected_rom_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the fixed evaluation protocol on a trained checkpoint.

    Args:
        checkpoint_path: Path to a stable-baselines3 ``.zip`` checkpoint.
        rom_path: Path to the Pokemon Red ROM file.
        save_state: Path to a PyBoy ``.state`` file. The eval env is reset
            to this state before every episode.
        algorithm: 'PPO', 'RecurrentPPO', or 'auto' (detect from checkpoint).
        n_episodes: Number of evaluation episodes. Locked at 20 for the paper.
        deterministic: Whether to use argmax actions.
        seed: Evaluation seed. Locked at 42 for the paper.
        max_episode_steps: Maximum steps per episode before truncation.
        observation_type: Observation representation to use.
        allow_override: If True, suppresses ValueError on non-locked params.

    Returns:
        A dict matching ``EVAL_METRIC_SCHEMA``.
    """
    # Enforce locked values unless explicitly overridden
    if not allow_override:
        if n_episodes != LOCKED_N_EPISODES:
            raise ValueError(
                f"n_episodes={n_episodes} deviates from the pre-registered "
                f"n_episodes={LOCKED_N_EPISODES}. Pass --allow-override or "
                f"log the deviation in paper/compute_ledger.md."
            )
        if seed != LOCKED_SEED:
            raise ValueError(
                f"seed={seed} deviates from the pre-registered seed={LOCKED_SEED}. "
                f"Pass --allow-override or log the deviation in "
                f"paper/compute_ledger.md."
            )

    # Validate inputs
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if not save_state.exists():
        raise FileNotFoundError(f"Save state not found: {save_state}")

    # Always compute the ROM hash so it ends up in the output JSON.
    # If --rom-sha256 was supplied, abort on mismatch.
    rom_sha256 = compute_rom_sha256(rom_path)
    logger.info("ROM sha256: %s  (%s)", rom_sha256, rom_path)
    if expected_rom_sha256 is not None and (
        rom_sha256.lower() != expected_rom_sha256.lower()
    ):
        raise RomHashMismatchError(
            f"ROM at {rom_path} has SHA-256 {rom_sha256} but eval was "
            f"asked for {expected_rom_sha256}.  Aborting to keep results "
            f"comparable across runs."
        )

    # Auto-detect algorithm if needed
    if algorithm == "auto":
        logger.info("Auto-detecting algorithm from checkpoint...")
        algorithm = detect_algorithm(checkpoint_path)
        logger.info(f"Detected: {algorithm}")

    is_recurrent = algorithm == "RecurrentPPO"

    # Load model
    model = load_model(checkpoint_path, algorithm)

    # Create environment
    logger.info("Creating evaluation environment...")
    env = PokemonRedGymEnv(
        rom_path=str(rom_path),
        headless=True,
        max_episode_steps=max_episode_steps,
        reward_strategy="events",
        observation_type=observation_type,
        save_state_path=str(save_state),
    )

    # Set seed for reproducibility.  deterministic_torch=True enforces
    # bit-reproducible torch ops — slower than training, but eval is
    # short (20 episodes) and reproducibility is the whole point.
    seed_everything(seed, deterministic_torch=True)
    env.seed(seed)

    # Run episodes
    logger.info("=" * 60)
    logger.info(f"EVALUATION: {n_episodes} episodes, seed={seed}")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Algorithm:  {algorithm}")
    logger.info(f"  Save state: {save_state}")
    logger.info(f"  Max steps:  {max_episode_steps}")
    logger.info("=" * 60)

    episode_results: List[Dict[str, Any]] = []
    start_time = time.time()

    try:
        for i in range(n_episodes):
            logger.info(f"Episode {i + 1}/{n_episodes}...")
            result = run_episode(
                env, model, is_recurrent,
                deterministic=deterministic,
                max_steps=max_episode_steps,
            )
            episode_results.append(result)

            logger.info(
                f"  reward={result['total_reward']:.1f}  "
                f"steps={result['steps']}  "
                f"maps={result['maps_visited']}  "
                f"flags={result['event_flags_triggered']}  "
                f"badges={result['badges']}  "
                f"level={result['player_level']}"
            )
    finally:
        env.close()

    elapsed = time.time() - start_time
    logger.info(f"Evaluation complete in {elapsed:.1f}s")

    # ── Aggregate metrics ───────────────────────────────────────────
    returns = [r["total_reward"] for r in episode_results]
    flags = [r["event_flags_triggered"] for r in episode_results]
    maps = [r["maps_visited"] for r in episode_results]
    brock_wins = [r["earned_boulder_badge"] for r in episode_results]
    pewter_steps = [r["step_reached_pewter"] for r in episode_results
                    if r["step_reached_pewter"] is not None]
    brock_steps = [r["step_earned_boulder"] for r in episode_results
                   if r["step_earned_boulder"] is not None]

    metrics: Dict[str, Any] = {
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
        "git_sha": _git_sha(),
        # Extended metrics (not in schema, but useful)
        "algorithm": algorithm,
        "observation_type": observation_type,
        "seed": seed,
        "max_episode_steps": max_episode_steps,
        "wall_clock_seconds": round(elapsed, 1),
        "episode_details": episode_results,
    }

    # Log summary
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Mean return:       {metrics['mean_return']:.2f} +/- {metrics['return_std']:.2f}")
    logger.info(f"  Brock win rate:    {metrics['brock_win_rate']:.1%}")
    logger.info(f"  Event flags (mean): {metrics['mean_event_flags_triggered']:.1f}")
    logger.info(f"  Event flags (max):  {metrics['max_event_flags_triggered']}")
    logger.info(f"  Maps visited (max): {metrics['unique_maps_visited']}")
    if metrics["steps_to_pewter"] is not None:
        logger.info(f"  Steps to Pewter:   {metrics['steps_to_pewter']}")
    if metrics["steps_to_brock_win"] is not None:
        logger.info(f"  Steps to Brock:    {metrics['steps_to_brock_win']}")

    return metrics


# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────


def _git_sha() -> str:
    """Return the current git SHA, or 'unknown' if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fixed evaluation harness for Pokemon Red AI paper runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to a stable-baselines3 .zip checkpoint.",
    )
    p.add_argument(
        "--rom", type=Path, required=True,
        help="Path to Pokemon Red ROM (.gb) file.",
    )
    p.add_argument(
        "--rom-sha256", type=str, default=None,
        help=(
            "Expected SHA-256 of the ROM.  If set, eval aborts on "
            "mismatch.  Always recorded in the output JSON so the paper "
            "can match eval results to a specific ROM dump."
        ),
    )
    p.add_argument(
        "--save-state", type=Path,
        default=Path("save_states/s0_post_intro.state"),
        help="PyBoy .state file to reset the eval env to before each episode.",
    )
    p.add_argument(
        "--algorithm", type=str, default="auto",
        choices=["auto", "PPO", "RecurrentPPO"],
        help="RL algorithm. 'auto' detects from checkpoint.",
    )
    p.add_argument(
        "--observation-type", type=str, default="multi_modal",
        choices=["multi_modal", "screen_only", "minimal", "pixel", "symbolic", "hybrid"],
        help="Observation representation.",
    )
    p.add_argument(
        "--n-episodes", type=int, default=LOCKED_N_EPISODES,
        help=f"Number of eval episodes (locked at {LOCKED_N_EPISODES} for the paper).",
    )
    p.add_argument(
        "--seed", type=int, default=LOCKED_SEED,
        help=f"Eval seed (locked at {LOCKED_SEED} for the paper).",
    )
    p.add_argument(
        "--max-episode-steps", type=int, default=15_000,
        help="Maximum steps per episode before truncation.",
    )
    p.add_argument(
        "--allow-override", action="store_true",
        help="Allow deviating from locked n_episodes/seed values.",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Optional JSON file to write metrics to. If omitted, prints to stdout.",
    )
    p.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v INFO, -vv DEBUG).",
    )

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    level = {0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(name)s %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        metrics = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            rom_path=args.rom,
            save_state=args.save_state,
            algorithm=args.algorithm,
            n_episodes=args.n_episodes,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            observation_type=args.observation_type,
            allow_override=args.allow_override,
            expected_rom_sha256=args.rom_sha256,
        )
    except (ValueError, FileNotFoundError, RomHashMismatchError) as e:
        logger.error(str(e))
        return 1

    # Write output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2, default=str))
        logger.info(f"Wrote metrics to {args.output}")
    else:
        print(json.dumps(metrics, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
