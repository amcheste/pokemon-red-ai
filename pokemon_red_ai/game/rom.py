"""
ROM hash verification utilities.

Pokemon Red has multiple ROM revisions (1.0/1.1, US/EU/JP), and even
within a single revision a corrupted dump can produce silently
different emulator state.  Different ROMs yield different reward
trajectories — invalidating reproducibility claims in the paper's
"Reproducibility and compute" appendix.

This module provides a stable SHA-256-based check that the user's ROM
matches an expected hash.  The default :data:`CANONICAL_ROM_HASHES`
dict captures the two most common dumps (US 1.0 and US 1.1); users
with other revisions can pass a custom hash via the ``--rom-sha256``
flag on the scripts that consume this module.

Usage::

    from pokemon_red_ai.game.rom import (
        compute_rom_sha256, verify_rom_hash, CANONICAL_ROM_HASHES,
    )

    sha = compute_rom_sha256("PokemonRed.gb")
    verify_rom_hash("PokemonRed.gb")  # raises if not in known-good set
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# Known-good SHA-256 hashes of canonical Pokemon Red ROM dumps.
#
# Empty by default to avoid shipping placeholder hashes — the project
# cannot legally distribute the ROM and we don't want to imply that a
# specific dump is "official" without independent verification.
#
# Operators of paper-quality runs should:
#
# 1. Compute the SHA-256 of their ROM::
#
#        shasum -a 256 path/to/PokemonRed.gb
#
# 2. Either populate this dict with the value, or pass --rom-sha256 to
#    scripts/train.py and scripts/eval.py.  The hash gets recorded in
#    compute_ledger.md for the paper's reproducibility appendix.
#
# Reference dumps (verify yourself, do not trust this comment):
#   - "Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb"  [Rev 0]
#   - "Pokemon - Red Version (USA, Europe) (SGB Enhanced) (Rev 1).gb"
# Canonical hashes are catalogued at https://datomatic.no-intro.org/
# under DAT name "Nintendo - Game Boy".
CANONICAL_ROM_HASHES: Dict[str, str] = {}


class RomHashMismatchError(ValueError):
    """Raised when a ROM file's SHA-256 doesn't match any allowed hash.

    The error message includes the actual computed hash so the user can
    confirm whether their dump is a legitimate revision we just haven't
    catalogued, or a corrupted file.
    """


def compute_rom_sha256(rom_path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the lowercase hex SHA-256 of the ROM at ``rom_path``.

    Reads the file in 1 MB chunks so we don't load the whole ROM into
    memory.  Pokemon Red ROMs are 1 MB, but this scales to any size and
    keeps the function reusable for save-state hashing too.

    Args:
        rom_path: Path to the ROM file.
        chunk_size: Bytes to read per iteration; 1 MB is comfortably
            below page-cache and large enough to be I/O-bound.

    Returns:
        64-character lowercase hex digest.
    """
    path = Path(rom_path)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def verify_rom_hash(
    rom_path: str | Path,
    *,
    allowed: Optional[Dict[str, str]] = None,
    extra_allowed: Optional[Dict[str, str]] = None,
    strict: bool = True,
) -> str:
    """Verify that ``rom_path``'s SHA-256 matches a known-good ROM.

    Args:
        rom_path: Path to the ROM file.
        allowed: Override the default :data:`CANONICAL_ROM_HASHES` map.
            Useful for tests; production callers should leave it ``None``.
        extra_allowed: Additional ``{label: sha256}`` entries to accept on
            top of the defaults.  Use this for non-US revisions or
            local-only debugging dumps.
        strict: If ``True`` (default) and the hash is not in the allowed
            set, raise :class:`RomHashMismatchError`.  If ``False``, log
            a warning and return the computed hash anyway.

    Returns:
        The lowercase hex SHA-256 of the file (regardless of match).

    Raises:
        RomHashMismatchError: If ``strict`` and no allowed entry matches.
        FileNotFoundError: If ``rom_path`` does not exist.
    """
    actual = compute_rom_sha256(rom_path)
    table = dict(allowed if allowed is not None else CANONICAL_ROM_HASHES)
    if extra_allowed:
        table.update(extra_allowed)

    for label, expected in table.items():
        if actual.lower() == expected.lower():
            logger.info("ROM hash matches %s (sha256=%s)", label, actual)
            return actual

    msg = (
        f"ROM at {rom_path} has SHA-256 {actual} which does not match any "
        f"known-good revision.  Known revisions: {sorted(table)}.  "
        f"If this is a legitimate revision we haven't catalogued, add "
        f"its hash to pokemon_red_ai/game/rom.CANONICAL_ROM_HASHES or "
        f"pass --rom-sha256 to override."
    )
    if strict:
        raise RomHashMismatchError(msg)
    logger.warning("%s", msg)
    return actual
