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


def generate_palette(classes: list[str]) -> dict[str, Any]:
    """Generates standard palette mapping class names to integer IDs and RGB colors."""
    import colorsys

    palette = {
        "classes": {"background": 0},
        "colors": {"background": [0, 0, 0]},
    }
    n = len(classes)
    for idx, cls_name in enumerate(classes, start=1):
        palette["classes"][cls_name] = idx
        hue = (idx - 1) / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        palette["colors"][cls_name] = [int(r * 255), int(g * 255), int(b * 255)]
    return palette


def rasterize_detections_mask(
    height: int,
    width: int,
    detections: list[Any],
    class_to_id: dict[str, int],
    label_field: str = "label",
) -> Any | None:
    """Renders instance detection masks onto a single 2D semantic mask array (uint8)."""
    import cv2
    import numpy as np

    has_any_mask = False
    full_mask = np.zeros((height, width), dtype=np.uint8)

    for det in detections:
        det_mask = getattr(det, "mask", None)
        if det_mask is None:
            continue
        has_any_mask = True

        label = str(getattr(det, label_field, "") or "")
        class_id = class_to_id.get(label, 1)

        bbox = list(getattr(det, "bounding_box", None) or [])
        if len(bbox) != 4:
            continue
        x, y, w, h = bbox
        x0 = max(0, min(width - 1, int(round(x * width))))
        y0 = max(0, min(height - 1, int(round(y * height))))
        bw = max(1, min(width - x0, int(round(w * width))))
        bh = max(1, min(height - y0, int(round(h * height))))

        mask_np = np.asarray(det_mask, dtype=np.uint8)
        resized = cv2.resize(mask_np, (bw, bh), interpolation=cv2.INTER_NEAREST)
        full_mask[y0 : y0 + bh, x0 : x0 + bw][resized > 0] = class_id

    return full_mask if has_any_mask else None

