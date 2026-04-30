#!/usr/bin/env python3
"""
Pokemon Red RL training script with Weights & Biases tracking.

This is the primary entry point for research training runs.  It wires
together the gym environment, RecurrentPPO model, event-flag reward
calculator, and W&B logging into a single reproducible pipeline.

Usage (minimal)::

    python scripts/train.py --rom path/to/PokemonRed.gb

Usage (full research run)::

    python scripts/train.py \
        --rom path/to/PokemonRed.gb \
        --save-state states/post_intro.state \
        --algorithm RecurrentPPO \
        --reward-strategy events \
        --total-timesteps 1_000_000 \
        --wandb-project pokemon-red-ai \
        --wandb-run-name "rppo-events-seed42" \
        --seed 42

Run ``python scripts/train.py --help`` for all options.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path so ``pokemon_red_ai`` is importable when
# running the script directly (outside of ``pip install -e .``).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
)

from pokemon_red_ai.environment import PokemonRedGymEnv, RewardConfig
from pokemon_red_ai.training.models import (
    create_model,
    get_model_config,
)
from pokemon_red_ai.training.callbacks import (
    TrainingCallback,
    MonitoringCallback,
    MONITORED_INFO_KEYS,
)
from pokemon_red_ai.training.alerts import (
    TrainingAlertCallback,
    DesktopChannel,
    SlackChannel,
    load_alert_config,
    channels_from_config,
)
from pokemon_red_ai.utils import create_directories

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a Pokemon Red RL agent with W&B tracking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ─────────────────────────────────────────────────────
    p.add_argument(
        "--rom", required=True, type=str,
        help="Path to Pokemon Red ROM (.gb) file.",
    )

    # ── Environment ──────────────────────────────────────────────────
    p.add_argument(
        "--save-state", type=str, default=None,
        help="Path to a PyBoy .state file for fast episode resets.",
    )
    p.add_argument(
        "--reward-strategy", type=str, default="events",
        choices=["standard", "exploration", "progress", "sparse", "events"],
        help="Reward calculator strategy.",
    )
    p.add_argument(
        "--observation-type", type=str, default="multi_modal",
        choices=["multi_modal", "screen_only", "minimal", "pixel", "symbolic", "hybrid"],
        help="Observation representation.",
    )
    p.add_argument(
        "--max-episode-steps", type=int, default=15_000,
        help="Maximum steps per episode before truncation.",
    )

    # ── Algorithm ────────────────────────────────────────────────────
    p.add_argument(
        "--algorithm", type=str, default="RecurrentPPO",
        choices=["PPO", "RecurrentPPO"],
        help="RL algorithm.",
    )
    p.add_argument(
        "--total-timesteps", type=int, default=1_000_000,
        help="Total environment steps to train.",
    )
    p.add_argument(
        "--learning-rate", type=float, default=None,
        help="Override default learning rate.",
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Override default batch size.",
    )
    p.add_argument(
        "--n-steps", type=int, default=None,
        help="Rollout length (steps per update).",
    )
    p.add_argument(
        "--ent-coef", type=float, default=None,
        help="Entropy coefficient (higher = more exploration).",
    )
    p.add_argument(
        "--lstm-hidden-size", type=int, default=256,
        help="LSTM hidden size (RecurrentPPO only).",
    )
    p.add_argument(
        "--n-envs", type=int, default=1,
        help="Parallel environments via SubprocVecEnv.  Each runs its own "
             "PyBoy instance on a separate CPU core.  Recommended: 4 on M3 Max, "
             "8+ on cloud high-CPU instances.  Set to 1 for the legacy "
             "single-env path (DummyVecEnv).",
    )

    # ── Reproducibility ──────────────────────────────────────────────
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility.",
    )

    # ── Saving ───────────────────────────────────────────────────────
    p.add_argument(
        "--save-dir", type=str, default="./training_output/",
        help="Directory for checkpoints, logs, and artifacts.",
    )
    p.add_argument(
        "--save-freq", type=int, default=50_000,
        help="Checkpoint save frequency (in timesteps).",
    )

    # ── W&B ──────────────────────────────────────────────────────────
    p.add_argument(
        "--wandb-project", type=str, default="pokemon-red-ai",
        help="W&B project name.",
    )
    p.add_argument(
        "--wandb-run-name", type=str, default=None,
        help="W&B run name (auto-generated if omitted).",
    )
    p.add_argument(
        "--wandb-entity", type=str, default=None,
        help="W&B entity (team or username).",
    )
    p.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging entirely.",
    )
    p.add_argument(
        "--screen-capture-freq", type=int, default=10,
        help="Episodes between game-screen captures logged to W&B. 0 disables.",
    )

    # ── Alerts (AMC-78) ──────────────────────────────────────────────
    p.add_argument(
        "--enable-alerts", action="store_true",
        help="Enable training milestone alerts (badges, plateaus, "
             "checkpoints).  See --alerts-config for full configuration.",
    )
    p.add_argument(
        "--alerts-config", type=str, default=None,
        help="Path to a YAML/JSON alerts config file.  Overrides any "
             "channel-specific CLI flags.",
    )
    p.add_argument(
        "--alerts-desktop", action="store_true",
        help="Enable desktop notifications (macOS osascript).",
    )
    p.add_argument(
        "--alerts-slack-webhook", type=str, default=None,
        help="Slack incoming webhook URL.  Falls back to "
             "$SLACK_WEBHOOK_URL.",
    )
    p.add_argument(
        "--alerts-plateau-episodes", type=int, default=50,
        help="Episodes without best-reward improvement before plateau "
             "alert fires.",
    )
    p.add_argument(
        "--alerts-checkpoint-freq", type=int, default=100_000,
        help="Timesteps between checkpoint alerts (0 to disable).",
    )

    # ── Misc ─────────────────────────────────────────────────────────
    p.add_argument(
        "--show-game", action="store_true",
        help="Show the PyBoy emulator window during training.",
    )
    p.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase verbosity (-v INFO, -vv DEBUG).",
    )

    return p


# ──────────────────────────────────────────────────────────────────────
# Seeding
# ──────────────────────────────────────────────────────────────────────

def set_global_seeds(seed: int) -> None:
    """Set seeds for numpy, torch, and Python stdlib."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info(f"Global seeds set to {seed}")


# ──────────────────────────────────────────────────────────────────────
# Vectorised environment construction
# ──────────────────────────────────────────────────────────────────────


def _make_env_factory(
    rank: int,
    args: argparse.Namespace,
    reward_config: Optional[RewardConfig],
):
    """Return a callable that constructs one PokemonRedGymEnv + Monitor.

    The returned callable is what SubprocVecEnv / DummyVecEnv invokes
    inside each subprocess (or the main process for DummyVecEnv).  It
    is deliberately picklable: it captures only the arguments needed
    to build the env, no shared state.

    ``rank`` is the per-env index (0..n_envs-1).  Each env writes its
    Monitor CSV to a rank-suffixed path so the parallel envs don't
    overwrite each other's logs.
    """
    def _init():
        env = PokemonRedGymEnv(
            rom_path=args.rom,
            headless=not args.show_game,
            max_episode_steps=args.max_episode_steps,
            reward_strategy=args.reward_strategy,
            reward_config=reward_config,
            observation_type=args.observation_type,
            save_state_path=args.save_state,
        )
        # Per-env Monitor file.  ``info_keywords`` makes Monitor copy
        # our custom per-step metrics into ``ep_info_buffer``, which is
        # what the W&B callback reads for aggregate episode stats.
        monitor_path = os.path.join(args.save_dir, f"monitor.{rank}")
        env = Monitor(
            env,
            monitor_path,
            info_keywords=MONITORED_INFO_KEYS,
        )
        return env
    return _init


def _make_vec_env(
    args: argparse.Namespace,
    n_envs: int,
    reward_config: Optional[RewardConfig],
) -> VecEnv:
    """Build a SubprocVecEnv (n_envs > 1) or DummyVecEnv (n_envs == 1).

    DummyVecEnv preserves the legacy single-env code path bit-for-bit:
    one PokemonRedGymEnv runs in the main process, no subprocess
    overhead.  SubprocVecEnv spawns one process per env, each running
    its own PyBoy instance — this is the path that yields throughput
    proportional to CPU core count.
    """
    env_fns = [
        _make_env_factory(rank, args, reward_config)
        for rank in range(n_envs)
    ]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)


# ──────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """Run a full training pipeline."""

    # ── Seeding ──────────────────────────────────────────────────────
    if args.seed is not None:
        set_global_seeds(args.seed)

    # ── Directories ──────────────────────────────────────────────────
    create_directories(args.save_dir)

    # ── Environment ──────────────────────────────────────────────────
    n_envs = max(1, int(args.n_envs))
    logger.info(f"Creating {n_envs} parallel environment(s)...")

    # Build custom reward config for exploration-focused events strategy
    reward_config = None
    if args.reward_strategy == "exploration":
        reward_config = RewardConfig(
            time_penalty=-0.001,
            exploration_reward=5.0,
            new_map_reward=100.0,
            level_reward_multiplier=25.0,
            badge_reward_multiplier=150.0,
            pokemon_reward_multiplier=75.0,
            low_health_threshold=0.3,
            health_penalty_multiplier=5.0,
            death_penalty=-50.0,
            money_reward_multiplier=0.005,
            battle_victory_reward=15.0,
            item_acquisition_reward=8.0,
        )

    env = _make_vec_env(args, n_envs, reward_config)

    logger.info(
        f"Environment ready  "
        f"obs={args.observation_type}  "
        f"reward={args.reward_strategy}  "
        f"max_steps={args.max_episode_steps}  "
        f"n_envs={n_envs}"
    )

    # ── Model ────────────────────────────────────────────────────────
    logger.info(f"Creating {args.algorithm} model...")

    model_overrides = {}
    if args.learning_rate is not None:
        model_overrides["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        model_overrides["batch_size"] = args.batch_size
    if args.n_steps is not None:
        model_overrides["n_steps"] = args.n_steps
    if args.ent_coef is not None:
        model_overrides["ent_coef"] = args.ent_coef
    if args.seed is not None:
        model_overrides["seed"] = args.seed

    # RecurrentPPO-specific args
    if args.algorithm == "RecurrentPPO":
        model_overrides["lstm_hidden_size"] = args.lstm_hidden_size

    model = create_model(
        algorithm=args.algorithm,
        env=env,
        tensorboard_log=os.path.join(args.save_dir, "tensorboard"),
        observation_type=args.observation_type,
        **model_overrides,
    )

    # Collect the effective config for W&B logging
    effective_config = get_model_config(args.algorithm)
    effective_config.update(model_overrides)
    effective_config.update(
        {
            "algorithm": args.algorithm,
            "reward_strategy": args.reward_strategy,
            "observation_type": args.observation_type,
            "max_episode_steps": args.max_episode_steps,
            "total_timesteps": args.total_timesteps,
            "save_state": args.save_state or "none",
            "seed": args.seed,
            "lstm_hidden_size": args.lstm_hidden_size,
        }
    )

    param_count = sum(p.numel() for p in model.policy.parameters())
    logger.info(f"Model created — {param_count:,} parameters")

    # ── W&B ──────────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb

            run_name = args.wandb_run_name or (
                f"{args.algorithm.lower()}-{args.reward_strategy}"
                f"{f'-s{args.seed}' if args.seed is not None else ''}"
            )

            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=effective_config,
                tags=[
                    args.algorithm,
                    args.reward_strategy,
                    args.observation_type,
                ],
                save_code=True,
            )
            logger.info(f"W&B run started: {wandb_run.url}")

        except ImportError:
            logger.warning(
                "wandb not installed — training will proceed without "
                "W&B logging.  Install with: pip install wandb"
            )
        except Exception as exc:
            logger.warning(f"W&B init failed ({exc}) — continuing without W&B")

    # ── Callbacks ────────────────────────────────────────────────────
    callbacks = [
        TrainingCallback(
            save_freq=args.save_freq,
            save_path=args.save_dir,
            verbose=1,
        ),
    ]

    if wandb_run is not None:
        callbacks.append(
            MonitoringCallback(
                save_freq=args.save_freq,
                save_path=args.save_dir,
                screen_capture_freq=args.screen_capture_freq,
                verbose=1,
            )
        )

    # ── Alerts (AMC-78) ─────────────────────────────────────────────
    alert_callback = _build_alert_callback(args)
    if alert_callback is not None:
        callbacks.append(alert_callback)

    callback_list = CallbackList(callbacks)

    # ── Train ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info(f"  Algorithm:    {args.algorithm}")
    logger.info(f"  Timesteps:    {args.total_timesteps:,}")
    logger.info(f"  Reward:       {args.reward_strategy}")
    logger.info(f"  Observation:  {args.observation_type}")
    logger.info(f"  Save dir:     {args.save_dir}")
    logger.info(f"  W&B:          {'enabled' if wandb_run else 'disabled'}")
    if args.seed is not None:
        logger.info(f"  Seed:         {args.seed}")
    logger.info("=" * 60)

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback_list,
            tb_log_name=f"pokemon_{args.algorithm.lower()}",
            progress_bar=True,
        )

        # Save final model
        final_path = os.path.join(args.save_dir, "models", "final_model")
        model.save(final_path)
        logger.info(f"Final model saved: {final_path}")

        # Upload final model artifact to W&B
        if wandb_run is not None:
            try:
                import wandb

                art = wandb.Artifact(
                    name="final-model",
                    type="model",
                    description="Final trained model",
                )
                art.add_file(final_path + ".zip")
                wandb_run.log_artifact(art)
            except Exception as exc:
                logger.warning(f"Final artifact upload failed: {exc}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        interrupted_path = os.path.join(
            args.save_dir, "models", "interrupted_model"
        )
        model.save(interrupted_path)
        logger.info(f"Interrupted model saved: {interrupted_path}")

    except Exception as exc:
        logger.exception("Training crashed with unhandled exception")
        if alert_callback is not None:
            try:
                alert_callback.notify_crash(exc)
            except Exception:
                logger.warning("Failed to dispatch crash alert")
        raise

    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()
            logger.info("W&B run finished")


# ──────────────────────────────────────────────────────────────────────
# Alert callback wiring
# ──────────────────────────────────────────────────────────────────────


def _build_alert_callback(
    args: argparse.Namespace,
) -> Optional[TrainingAlertCallback]:
    """Construct a :class:`TrainingAlertCallback` from CLI args.

    Returns ``None`` when alerts are not enabled.  Honors:

    * ``--alerts-config`` — full config file (YAML or JSON), wins over
      individual channel flags.
    * ``--alerts-desktop`` — adds :class:`DesktopChannel`.
    * ``--alerts-slack-webhook`` — adds :class:`SlackChannel`.

    The callback always includes :class:`LogChannel` automatically.
    """
    if not (args.enable_alerts or args.alerts_config or args.alerts_desktop
            or args.alerts_slack_webhook):
        return None

    channels = []

    if args.alerts_config:
        try:
            cfg = load_alert_config(args.alerts_config)
            channels.extend(channels_from_config(cfg))
            triggers = cfg.get("triggers", {}) or {}
            plateau_episodes = int(
                triggers.get("plateau_episodes", args.alerts_plateau_episodes)
            )
            checkpoint_freq = int(
                triggers.get("checkpoint_alert_freq", args.alerts_checkpoint_freq)
            )
        except Exception as exc:
            logger.warning(
                f"Failed to load --alerts-config {args.alerts_config}: {exc}. "
                f"Falling back to CLI flags."
            )
            plateau_episodes = args.alerts_plateau_episodes
            checkpoint_freq = args.alerts_checkpoint_freq
    else:
        plateau_episodes = args.alerts_plateau_episodes
        checkpoint_freq = args.alerts_checkpoint_freq

    if args.alerts_desktop:
        channels.append(DesktopChannel())
    if args.alerts_slack_webhook:
        slack = SlackChannel(webhook_url=args.alerts_slack_webhook)
        if slack.is_configured():
            channels.append(slack)
        else:
            logger.warning(
                "--alerts-slack-webhook set but webhook URL is empty; "
                "skipping Slack channel."
            )

    callback = TrainingAlertCallback(
        channels=channels,
        plateau_episodes=plateau_episodes,
        checkpoint_alert_freq=checkpoint_freq,
        verbose=1,
    )
    logger.info(
        f"Alerts enabled — channels: "
        f"{[type(c).__name__ for c in callback.channels]}"
    )
    return callback


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Logging
    level = {0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(name)s %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    train(args)


if __name__ == "__main__":
    main()
