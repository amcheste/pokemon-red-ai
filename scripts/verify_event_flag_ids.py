#!/usr/bin/env python3
"""
Verify that ``BOULDER_PATH_FLAGS`` IDs match the canonical ``pret/pokered``
disassembly.

Downloads ``constants/event_constants.asm`` from pret/pokered, parses it
by emulating the rgbds ``const_def`` / ``const`` / ``const_skip`` /
``const_next`` macros, and compares the resolved event IDs against the
values hard-coded in ``pokemon_red_ai/game/event_flags.py``.

Exits non-zero on any mismatch.

Usage::

    python scripts/verify_event_flag_ids.py
    python scripts/verify_event_flag_ids.py --offline path/to/event_constants.asm

The script is intentionally dependency-light (stdlib only) so it can run
in CI without installing the project's training stack.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Optional

_DEFAULT_URL = (
    "https://raw.githubusercontent.com/pret/pokered/master/"
    "constants/event_constants.asm"
)


def parse_event_constants(text: str) -> Dict[str, int]:
    """Parse pret/pokered's event_constants.asm into ``{name: id}``.

    Emulates the rgbds macros::

        const_def [N=0]     -> counter = N (default 0)
        const X             -> X = counter; counter += 1
        const_skip [N=1]    -> counter += N (default 1)
        const_next $X[ - K] -> counter = X (- K if specified)
    """
    events: Dict[str, int] = {}
    counter = 0

    for raw in text.splitlines():
        line = raw.split(";")[0].strip()
        if not line:
            continue
        toks = line.split()
        op = toks[0]
        if op == "const_def":
            counter = _parse_int(toks[1]) if len(toks) > 1 else 0
        elif op == "const":
            name = toks[1]
            events[name] = counter
            counter += 1
        elif op == "const_skip":
            counter += _parse_int(toks[1]) if len(toks) > 1 else 1
        elif op == "const_next":
            # Supports "$XXX" and "$XXX - N".
            rhs = " ".join(toks[1:]).replace("$", "0x")
            counter = int(eval(rhs, {"__builtins__": {}}, {}))
    return events


def _parse_int(tok: str) -> int:
    tok = tok.replace("$", "0x")
    return int(tok, 16) if tok.lower().startswith("0x") else int(tok)


def fetch_constants(url: str = _DEFAULT_URL) -> str:
    with urllib.request.urlopen(url, timeout=15) as resp:
        return resp.read().decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify event flag IDs against pret/pokered.",
    )
    parser.add_argument(
        "--offline",
        type=Path,
        default=None,
        help="Read event_constants.asm from a local path (skip network fetch).",
    )
    parser.add_argument(
        "--url",
        default=_DEFAULT_URL,
        help="URL to fetch event_constants.asm from (default: pret/pokered master).",
    )
    args = parser.parse_args()

    # Import the project's flag dict.  Done lazily so the script can also
    # be used to bootstrap a fresh check on a new flag list.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pokemon_red_ai.game.event_flags import BOULDER_PATH_FLAGS

    if args.offline is not None:
        text = args.offline.read_text(encoding="utf-8")
    else:
        text = fetch_constants(args.url)

    canonical = parse_event_constants(text)

    print(f"Parsed {len(canonical)} events from pret/pokered")
    print(f"Verifying {len(BOULDER_PATH_FLAGS)} pre-registered flags...")
    print()

    failures: Dict[str, Optional[int]] = {}
    for name, expected_id in BOULDER_PATH_FLAGS.items():
        actual_id = canonical.get(name)
        if actual_id is None:
            failures[name] = None
            mark = "MISSING"
        elif actual_id != expected_id:
            failures[name] = actual_id
            mark = "MISMATCH"
        else:
            mark = "OK"
        addr = 0xD747 + expected_id // 8
        bit = expected_id % 8
        print(
            f"  [{mark:8}] {name:<44} "
            f"code={expected_id:#06x}  pokered={actual_id if actual_id is not None else '----'}  "
            f"({hex(addr)}/{bit})"
        )

    print()
    if failures:
        print(f"FAIL: {len(failures)} flag(s) do not match pret/pokered.")
        for name, actual_id in failures.items():
            if actual_id is None:
                print(f"  - {name}: not found in pret/pokered")
            else:
                print(
                    f"  - {name}: code says {BOULDER_PATH_FLAGS[name]:#06x}, "
                    f"pokered says {actual_id:#06x}"
                )
        return 1

    print(f"OK: all {len(BOULDER_PATH_FLAGS)} flag IDs match pret/pokered.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
