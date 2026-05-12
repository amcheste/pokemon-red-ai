"""
Tests for ``pokemon_red_ai.game.rom``.

Covers the chunked SHA-256 reader, the verify_rom_hash dispatcher, and
the strict / non-strict behaviour modes.  All tests use synthetic byte
content — no real ROM file is needed.
"""

import hashlib

import pytest

from pokemon_red_ai.game.rom import (
    CANONICAL_ROM_HASHES,
    RomHashMismatchError,
    compute_rom_sha256,
    verify_rom_hash,
)


def _write_bytes(path, content: bytes) -> None:
    path.write_bytes(content)


class TestComputeRomSha256:

    def test_empty_file(self, tmp_path):
        rom = tmp_path / "empty.gb"
        _write_bytes(rom, b"")
        # SHA-256 of empty input is a well-known constant.
        assert compute_rom_sha256(rom) == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_short_payload(self, tmp_path):
        rom = tmp_path / "tiny.gb"
        _write_bytes(rom, b"hello world")
        # Verify against hashlib directly — same algorithm, so this just
        # confirms the chunking is bit-equivalent to a one-shot digest.
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert compute_rom_sha256(rom) == expected

    def test_large_payload_chunking(self, tmp_path):
        """Payload spans multiple read chunks — confirms the loop logic."""
        rom = tmp_path / "big.gb"
        # 3 MB of pseudo-random bytes; with chunk_size=1MB this is 3 reads.
        content = bytes(range(256)) * (3 * 1024 * 4)  # 3 MB
        _write_bytes(rom, content)
        expected = hashlib.sha256(content).hexdigest()
        assert compute_rom_sha256(rom) == expected

    def test_custom_chunk_size(self, tmp_path):
        rom = tmp_path / "small.gb"
        _write_bytes(rom, b"abcdefghij" * 100)  # 1000 bytes
        # 7-byte chunks force ~143 reads.  Same digest required.
        big_chunk = compute_rom_sha256(rom, chunk_size=1 << 20)
        small_chunk = compute_rom_sha256(rom, chunk_size=7)
        assert big_chunk == small_chunk

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compute_rom_sha256(tmp_path / "does_not_exist.gb")


class TestVerifyRomHash:

    def test_no_allowed_hashes_strict_raises(self, tmp_path):
        """Default CANONICAL_ROM_HASHES is empty; strict mode must fail
        loudly so users don't unknowingly skip the check."""
        rom = tmp_path / "rom.gb"
        _write_bytes(rom, b"any content")
        # Sanity: the module-level dict really is empty.
        assert CANONICAL_ROM_HASHES == {}
        with pytest.raises(RomHashMismatchError):
            verify_rom_hash(rom)

    def test_no_allowed_hashes_non_strict_warns_and_returns(self, tmp_path, caplog):
        import logging as stdlib_logging
        rom = tmp_path / "rom.gb"
        _write_bytes(rom, b"any content")
        # The autouse logging fixture in tests/conftest.py raises the level
        # to ERROR; locally drop it back so the warning is captured.
        with caplog.at_level(stdlib_logging.WARNING, logger="pokemon_red_ai.game.rom"):
            result = verify_rom_hash(rom, strict=False)
        assert result == hashlib.sha256(b"any content").hexdigest()
        warnings = [
            rec for rec in caplog.records if rec.levelname == "WARNING"
        ]
        assert any(str(rom) in rec.message for rec in warnings)

    def test_explicit_allowed_match(self, tmp_path):
        content = b"canonical rom bytes"
        rom = tmp_path / "rom.gb"
        _write_bytes(rom, content)
        sha = hashlib.sha256(content).hexdigest()
        result = verify_rom_hash(rom, allowed={"US_1.0": sha})
        assert result == sha

    def test_extra_allowed_adds_to_defaults(self, tmp_path):
        content = b"local debug dump"
        rom = tmp_path / "rom.gb"
        _write_bytes(rom, content)
        sha = hashlib.sha256(content).hexdigest()
        result = verify_rom_hash(rom, extra_allowed={"my_dump": sha})
        assert result == sha

    def test_case_insensitive_match(self, tmp_path):
        content = b"some rom"
        rom = tmp_path / "rom.gb"
        _write_bytes(rom, content)
        sha_upper = hashlib.sha256(content).hexdigest().upper()
        # Stored as uppercase, computed as lowercase — still matches.
        result = verify_rom_hash(rom, allowed={"x": sha_upper})
        assert result.lower() == sha_upper.lower()

    def test_strict_mismatch_raises(self, tmp_path):
        rom = tmp_path / "rom.gb"
        _write_bytes(rom, b"wrong content")
        with pytest.raises(RomHashMismatchError) as exc:
            verify_rom_hash(rom, allowed={"correct": "0" * 64})
        # Error message must include the actual hash so the user can
        # decide whether to whitelist their dump.
        actual = hashlib.sha256(b"wrong content").hexdigest()
        assert actual in str(exc.value)

    def test_non_strict_mismatch_returns_actual(self, tmp_path):
        rom = tmp_path / "rom.gb"
        _write_bytes(rom, b"another wrong content")
        result = verify_rom_hash(
            rom, allowed={"correct": "0" * 64}, strict=False,
        )
        assert result == hashlib.sha256(b"another wrong content").hexdigest()
