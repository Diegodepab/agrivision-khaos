"""Portable export assets, copied independently of the source dataset."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

EXPORT_ALGORITHM_VERSION = "export-v2"
COPY_BLOCK_SIZE = 1024 * 1024


def export_filename(sample_id: str, source: Path) -> str:
    return f"{sample_id}{source.suffix.lower()}"


def copy_verified_asset(source: Path, target: Path, expected_sha256: str | None = None):
    """Copy and hash in one bounded pass; reject changes since the visual audit."""
    digest = hashlib.sha256()
    size = 0
    created = False
    try:
        with source.open("rb") as reader, target.open("xb") as writer:
            created = True
            while block := reader.read(COPY_BLOCK_SIZE):
                writer.write(block)
                digest.update(block)
                size += len(block)
        checksum = digest.hexdigest()
        if expected_sha256 is not None and checksum != expected_sha256:
            raise RuntimeError(f"La imagen cambió después de la auditoría: {source}")
        if size == 0:
            raise RuntimeError(f"Imagen vacía durante la exportación: {source}")
        shutil.copystat(source, target)
    except Exception:
        if created:
            target.unlink(missing_ok=True)
        raise
    return checksum, size


def link_export_asset(source: Path, target: Path) -> str:
    """Share bytes only within the export; copy if hard links are unavailable."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "linked"
    except FileExistsError:
        raise
    except OSError:
        # Exclusive creation prevents overwriting an unrelated destination on fallback.
        copy_verified_asset(source, target)
        return "copied"
