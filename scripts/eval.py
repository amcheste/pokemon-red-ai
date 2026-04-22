"""
Fixed evaluation harness for the Pokémon Red AI paper experiments.

Runs a locked, deterministic evaluation protocol on a trained checkpoint:

- N evaluation episodes (default 20, matches analysis_plan.md §3)
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

Usage:
    python -m scripts.eval \\
        --checkpoint training_output/models/best_model.zip \\
        --save-state save_states/s0_post_intro.state \\
        --output paper/notebooks/eval_results/symbolic_seed0.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Schema for the JSON blob this script emits. Notebooks in paper/notebooks/
# assume these exact keys — do not rename without updating downstream consumers.
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
    "git_sha": str,
}


# Locked defaults per analysis_plan.md §3
LOCKED_N_EPISODES = 20
LOCKED_SEED = 42


def evaluate_checkpoint(
    checkpoint_path: Path,
    save_state: Path,
    n_episodes: int = LOCKED_N_EPISODES,
    deterministic: bool = True,
    seed: int = LOCKED_SEED,
) -> Dict[str, Any]:
    """
    Run the fixed evaluation protocol on a trained checkpoint.

    This function is intentionally a skeleton in PR #0. The real implementation
    follows in PR #2, once these dependencies from PR #1 land:
      - Save state loading in PokemonRedGymEnv.reset(save_state=...)
      - Event flag reading in pokemon_red_ai/game/memory.py
      - RecurrentPPO checkpoint support in training/models.py

    Args:
        checkpoint_path: Path to a stable-baselines3 ``.zip`` checkpoint.
        save_state: Path to a PyBoy ``.state`` file. The eval env is reset to
            this state before every episode.
        n_episodes: Number of evaluation episodes. Locked at 20 for the paper.
        deterministic: Whether to use argmax actions. Locked True for the paper.
        seed: Evaluation seed. Locked at 42 for the paper.

    Returns:
        A dict matching ``EVAL_METRIC_SCHEMA``.

    Raises:
        NotImplementedError: Until PR #2 lands.
        ValueError: If ``n_episodes`` or ``seed`` deviate from locked values
            without an explicit override flag.
    """
    if n_episodes != LOCKED_N_EPISODES:
        raise ValueError(
            f"n_episodes={n_episodes} deviates from the pre-registered "
            f"n_episodes={LOCKED_N_EPISODES}. Log the deviation in "
            f"paper/compute_ledger.md before re-running with a different value."
        )
    if seed != LOCKED_SEED:
        raise ValueError(
            f"seed={seed} deviates from the pre-registered seed={LOCKED_SEED}. "
            f"Log the deviation in paper/compute_ledger.md before re-running "
            f"with a different seed."
        )

    # TODO(PR #2): Implement once PR #1 lands save state loading and event flag reads.
    #   1. Construct a single PokemonRedGymEnv with reward=events, eval_mode=True
    #   2. Load the RecurrentPPO checkpoint
    #   3. Run n_episodes rollouts, reset from save_state each time
    #   4. Collect per-episode: event flags triggered, maps visited, return,
    #      steps-to-Pewter, steps-to-Brock-win
    #   5. Aggregate into EVAL_METRIC_SCHEMA dict
    raise NotImplementedError(
        "eval.py is a skeleton in PR #0. Real implementation follows in PR #2 "
        "once save state loading and event flag reading land."
    )


def _git_sha() -> str:
    """Return the current git SHA, or 'unknown' if not in a repo."""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Fixed evaluation harness for Pokémon Red AI paper runs."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a stable-baselines3 .zip checkpoint.",
    )
    parser.add_argument(
        "--save-state",
        type=Path,
        default=Path("save_states/s0_post_intro.state"),
        help="PyBoy .state file to reset the eval env to before each episode.",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=LOCKED_N_EPISODES,
        help=f"Number of eval episodes (locked at {LOCKED_N_EPISODES} for the paper).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=LOCKED_SEED,
        help=f"Eval seed (locked at {LOCKED_SEED} for the paper).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to write metrics to. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        save_state=args.save_state,
        n_episodes=args.n_episodes,
        seed=args.seed,
    )
    metrics["git_sha"] = _git_sha()
    metrics["checkpoint_path"] = str(args.checkpoint)
    metrics["eval_save_state"] = str(args.save_state)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2, default=str))
        logger.info("Wrote metrics to %s", args.output)
    else:
        print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
