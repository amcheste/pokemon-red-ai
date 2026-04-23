#!/usr/bin/env python3
"""
Create curriculum save states for Pokemon Red RL training.

Runs the ROM through the opening sequence and saves PyBoy ``.state``
files at key game checkpoints.  These save states let the training
script skip the intro on every episode reset, dramatically reducing
wall-clock time.

Automatically created save states
----------------------------------
``s0_post_intro.state``
    Player is in their bedroom in Pallet Town, immediately after
    Prof. Oak's intro and naming screens.  This is the standard
    starting point for RL training.

Manual save states (via ``--interactive``)
------------------------------------------
The script can also drop into an interactive loop where you play
the game and press ``Ctrl+C`` to save at arbitrary checkpoints.
This is useful for creating curriculum states deeper in the game
(e.g. pre-Brock, post-Brock) that are too complex to automate.

Usage::

    # Automatic (creates post-intro save state)
    python scripts/create_save_states.py --rom PokemonRed.gb

    # Interactive (play the game and save manually)
    python scripts/create_save_states.py --rom PokemonRed.gb --interactive

    # Validate existing save states
    python scripts/create_save_states.py --rom PokemonRed.gb --validate-only
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pokemon_red_ai.game.agent import PokemonRedAgent
from pokemon_red_ai.game.memory import (
    MAP_IDS,
    BADGE_FLAGS,
    read_player_position,
    read_player_stats,
    read_memory_value,
    MEMORY_ADDRESSES,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Save state definitions
# ──────────────────────────────────────────────────────────────────────

# Each entry defines the expected game state for validation.
# Map 40 = Red's House 2F (player's bedroom) — this is the actual
# starting map, not the Pallet Town overworld (map 1).
REDS_HOUSE_2F = 40

SAVE_STATE_SPECS = {
    "s0_post_intro": {
        "filename": "s0_post_intro.state",
        "description": "Post-intro: Pallet Town, player's bedroom",
        "expected_map_id": REDS_HOUSE_2F,
        "expected_badges": 0,
        "min_party_count": 0,  # No Pokemon yet
    },
    "s1_post_starter": {
        "filename": "s1_post_starter.state",
        "description": "Post-starter: Pallet Town, has first Pokemon",
        "expected_map_id": None,  # Could be in Oak's lab or Pallet
        "expected_badges": 0,
        "min_party_count": 1,
    },
    "s2_pre_brock": {
        "filename": "s2_pre_brock.state",
        "description": "Pre-Brock: Pewter City, before gym battle",
        "expected_map_id": MAP_IDS["pewter_city"],
        "expected_badges": 0,
        "min_party_count": 1,
    },
    "s3_post_brock": {
        "filename": "s3_post_brock.state",
        "description": "Post-Brock: has Boulder Badge",
        "expected_map_id": None,  # Could be in gym or Pewter
        "expected_badges": BADGE_FLAGS["boulder"],
        "min_party_count": 1,
    },
}


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────


def validate_save_state(
    rom_path: str,
    state_path: str,
    spec: dict,
) -> bool:
    """
    Load a save state and verify the game is in the expected condition.

    Returns True if all checks pass.
    """
    if not os.path.isfile(state_path):
        logger.warning(f"  State file not found: {state_path}")
        return False

    file_size = os.path.getsize(state_path)
    if file_size == 0:
        logger.error(f"  State file is empty: {state_path}")
        return False

    agent = None
    try:
        agent = PokemonRedAgent(rom_path, show_window=False, speed_multiplier=0)
        success = agent.load_save_state(state_path)
        if not success:
            logger.error(f"  PyBoy failed to load state: {state_path}")
            return False

        # Let the game settle after state load
        agent.wait_frames(30)

        # Read game state
        pos = read_player_position(agent.memory)
        stats = read_player_stats(agent.memory)

        logger.info(f"  Map ID: {pos['map']}  Position: ({pos['x']}, {pos['y']})")
        logger.info(
            f"  Party: {stats['party_count']}  "
            f"Badges: {stats['badges']:08b}  "
            f"Level: {stats['level']}"
        )

        passed = True

        # Check map
        expected_map = spec.get("expected_map_id")
        if expected_map is not None and pos["map"] != expected_map:
            logger.error(
                f"  FAIL map_id: expected {expected_map}, got {pos['map']}"
            )
            passed = False

        # Check badges
        expected_badges = spec.get("expected_badges", 0)
        if expected_badges > 0:
            if not (stats["badges"] & expected_badges):
                logger.error(
                    f"  FAIL badges: expected {expected_badges:#04x} set, "
                    f"got {stats['badges']:#04x}"
                )
                passed = False
        elif expected_badges == 0 and stats["badges"] != 0:
            # If we expect 0 badges but have some, that's a warning not a failure
            logger.warning(
                f"  WARN badges: expected 0, got {stats['badges']:#04x}"
            )

        # Check party
        min_party = spec.get("min_party_count", 0)
        if stats["party_count"] < min_party:
            logger.error(
                f"  FAIL party_count: expected >= {min_party}, "
                f"got {stats['party_count']}"
            )
            passed = False

        if passed:
            logger.info("  PASS")
        return passed

    except Exception as e:
        logger.error(f"  Validation error: {e}")
        return False
    finally:
        if agent is not None:
            try:
                agent.pyboy.stop()
            except Exception:
                pass


def validate_all(rom_path: str, save_dir: str) -> dict:
    """Validate all save states that exist in the directory."""
    results = {}
    for key, spec in SAVE_STATE_SPECS.items():
        state_path = os.path.join(save_dir, spec["filename"])
        if os.path.isfile(state_path):
            logger.info(f"Validating {spec['description']}...")
            results[key] = validate_save_state(rom_path, state_path, spec)
        else:
            logger.info(f"Skipping {spec['description']} (file not found)")
            results[key] = None

    return results


# ──────────────────────────────────────────────────────────────────────
# Automatic creation
# ──────────────────────────────────────────────────────────────────────


def create_post_intro_state(rom_path: str, save_dir: str) -> bool:
    """
    Run the opening sequence and save state right after the intro.

    The agent automates:
    1. Title screen / copyright notices
    2. Prof. Oak's introduction
    3. Player and rival naming (accepts defaults: RED / BLUE)
    4. Intro dialogue completion

    Saves ``s0_post_intro.state`` when the player is standing in
    Pallet Town with game control active.
    """
    spec = SAVE_STATE_SPECS["s0_post_intro"]
    state_path = os.path.join(save_dir, spec["filename"])

    logger.info("=" * 60)
    logger.info("Creating post-intro save state")
    logger.info("=" * 60)

    agent = None
    try:
        agent = PokemonRedAgent(
            rom_path, show_window=False, speed_multiplier=0
        )

        logger.info("Running opening sequence (this takes ~30-60 seconds)...")
        start = time.time()
        success = agent.run_opening_sequence()
        elapsed = time.time() - start

        if not success:
            logger.error(
                f"Opening sequence failed after {elapsed:.1f}s. "
                "The ROM may not be compatible or PyBoy may need updating."
            )
            return False

        logger.info(f"Opening sequence completed in {elapsed:.1f}s")

        # Verify we're in the game
        pos = read_player_position(agent.memory)
        stats = read_player_stats(agent.memory)
        logger.info(
            f"Position: map={pos['map']} ({pos['x']}, {pos['y']})  "
            f"Party: {stats['party_count']}"
        )

        if pos["map"] == 0:
            logger.error(
                "map_id is 0 after opening sequence -- player is not in-game. "
                "The intro may not have completed properly."
            )
            return False

        # Save the state
        logger.info(f"Saving state to {state_path}...")
        ok = agent.save_save_state(state_path)
        if not ok:
            logger.error("save_save_state() returned False")
            return False

        try:
            size = os.path.getsize(state_path)
            logger.info(f"State saved ({size:,} bytes)")
        except OSError:
            logger.info("State saved (size unknown)")

        return True

    except Exception as e:
        logger.error(f"Failed to create post-intro state: {e}")
        return False
    finally:
        if agent is not None:
            try:
                agent.pyboy.stop()
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────
# Interactive mode
# ──────────────────────────────────────────────────────────────────────


def run_interactive(rom_path: str, save_dir: str, save_name: str) -> bool:
    """
    Open the game with a visible window and let the user play.

    The user plays normally and presses ``Ctrl+C`` when they want
    to save a state.  The state is saved under ``save_name`` in
    ``save_dir``.

    If a post-intro state exists, it is loaded first to skip the
    opening sequence.
    """
    state_path = os.path.join(save_dir, save_name)
    post_intro = os.path.join(save_dir, "s0_post_intro.state")

    logger.info("=" * 60)
    logger.info("Interactive mode")
    logger.info(f"  Save target: {state_path}")
    logger.info("  Play the game, then press Ctrl+C to save.")
    logger.info("=" * 60)

    agent = None
    try:
        agent = PokemonRedAgent(
            rom_path, show_window=True, speed_multiplier=1
        )

        # Load post-intro state if available to skip intro
        if os.path.isfile(post_intro):
            logger.info(f"Loading post-intro state to skip intro...")
            agent.load_save_state(post_intro)
            agent.wait_frames(30)
        else:
            logger.info("No post-intro state found, running opening sequence...")
            agent.run_opening_sequence()

        pos = read_player_position(agent.memory)
        stats = read_player_stats(agent.memory)
        logger.info(
            f"Ready! Map={pos['map']} ({pos['x']},{pos['y']})  "
            f"Party={stats['party_count']}  Badges={stats['badges']:08b}"
        )
        logger.info("Press Ctrl+C to save state and exit.")

        # Let the user play — PyBoy handles input via SDL2 window
        while True:
            agent.pyboy.tick()


    except KeyboardInterrupt:
        logger.info("\nCtrl+C received — saving state...")

        if agent is None:
            logger.error("Cannot save: agent not initialized")
            return False

        pos = read_player_position(agent.memory)
        stats = read_player_stats(agent.memory)
        logger.info(
            f"Current state: Map={pos['map']} ({pos['x']},{pos['y']})  "
            f"Party={stats['party_count']}  "
            f"Level={stats['level']}  "
            f"Badges={stats['badges']:08b}"
        )

        ok = agent.save_save_state(state_path)

        if ok:
            try:
                size_str = f" ({os.path.getsize(state_path):,} bytes)"
            except OSError:
                size_str = ""
            logger.info(f"State saved to {state_path}{size_str}")
        else:
            logger.error("Failed to save state!")
        return ok

    except Exception as e:
        logger.error(f"Interactive mode error: {e}")
        return False
    finally:
        if agent is not None:
            try:
                agent.pyboy.stop()
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create and validate curriculum save states for RL training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--rom", required=True, type=str,
        help="Path to Pokemon Red ROM (.gb) file.",
    )
    p.add_argument(
        "--save-dir", type=str, default="save_states",
        help="Directory for save state files.",
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--interactive", action="store_true",
        help="Open game window for manual play; Ctrl+C to save.",
    )
    mode.add_argument(
        "--validate-only", action="store_true",
        help="Only validate existing save states, don't create new ones.",
    )

    p.add_argument(
        "--save-name", type=str, default=None,
        help="Filename for interactive save (default: prompted).",
    )
    p.add_argument(
        "--skip-validation", action="store_true",
        help="Skip post-creation validation step.",
    )
    p.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v INFO, -vv DEBUG).",
    )

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Logging
    level = {0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(name)s %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate ROM
    if not os.path.isfile(args.rom):
        logger.error(f"ROM file not found: {args.rom}")
        return 1

    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Validate-only mode ──────────────────────────────────────────
    if args.validate_only:
        results = validate_all(args.rom, args.save_dir)
        found = {k: v for k, v in results.items() if v is not None}
        if not found:
            logger.warning("No save state files found to validate.")
            return 1
        failed = [k for k, v in found.items() if not v]
        if failed:
            logger.error(f"Validation failed for: {', '.join(failed)}")
            return 1
        logger.info("All existing save states validated successfully.")
        return 0

    # ── Interactive mode ────────────────────────────────────────────
    if args.interactive:
        save_name = args.save_name
        if not save_name:
            # List available spec names
            print("\nAvailable save state slots:")
            for key, spec in SAVE_STATE_SPECS.items():
                path = os.path.join(args.save_dir, spec["filename"])
                exists = " [exists]" if os.path.isfile(path) else ""
                print(f"  {spec['filename']:30s}  {spec['description']}{exists}")
            print()
            save_name = input("Enter filename (e.g. s2_pre_brock.state): ").strip()
            if not save_name:
                logger.error("No filename provided.")
                return 1

        ok = run_interactive(args.rom, args.save_dir, save_name)
        return 0 if ok else 1

    # ── Automatic mode (default) ────────────────────────────────────
    logger.info("Creating curriculum save states...")

    ok = create_post_intro_state(args.rom, args.save_dir)
    if not ok:
        logger.error("Failed to create post-intro save state.")
        return 1

    # Validate what we just created
    if not args.skip_validation:
        logger.info("\nValidating created save state...")
        spec = SAVE_STATE_SPECS["s0_post_intro"]
        state_path = os.path.join(args.save_dir, spec["filename"])
        valid = validate_save_state(args.rom, state_path, spec)
        if not valid:
            logger.error("Post-creation validation FAILED!")
            return 1
        logger.info("Validation passed.")

    logger.info("\nDone! Save states are ready for training.")
    logger.info(f"  Use with: python scripts/train.py --rom {args.rom} "
                f"--save-state {args.save_dir}/s0_post_intro.state")

    return 0


if __name__ == "__main__":
    sys.exit(main())
