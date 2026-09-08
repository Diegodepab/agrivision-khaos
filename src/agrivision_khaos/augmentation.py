"""CPU descriptors and pair verification, independent of FiftyOne and model downloads."""

from __future__ import annotations

import base64
import hashlib
import json
import resource
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from agrivision_khaos.execution import atomic_write_json
from agrivision_khaos.geometry import affine_evidence
from agrivision_khaos.models import DeduplicationPolicy
from agrivision_khaos.retrieval import CandidateIndex

ALGORITHM_VERSION = "augmentation-v3"
THUMB_SIDE = 384


def _checksum(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _read_cache(path: Path) -> dict:
    envelope = json.loads(path.read_text())
    payload = envelope["payload"]
    if envelope["checksum"] != _checksum(payload):
        raise ValueError("Caché alterada o incompleta")
    return payload


def _write_cache(path: Path, payload: dict) -> None:
    atomic_write_json(path, {"payload": payload, "checksum": _checksum(payload)})


EXACT_LEVELS = {"exact_bytes", "exact_pixels"}
CONFIRMED_LEVELS = EXACT_LEVELS | {"verified_visual", "human_confirmed"}


def transforms(image: np.ndarray):
    """Eight discrete symmetries, without interpolation or loss of pixels."""
    for mirrored in (False, True):
        source = np.fliplr(image) if mirrored else image
        for quarter_turns in range(4):
            yield (
                f"{'mirror_' if mirrored else ''}r{quarter_turns * 90}",
                np.rot90(source, quarter_turns),
            )


def pixel_digest(image: np.ndarray) -> str:
    header = f"{ALGORITHM_VERSION}:{image.dtype}:{image.shape}:".encode()
    return hashlib.sha256(header + np.ascontiguousarray(image).tobytes()).hexdigest()


def perceptual_hash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].flatten()[1:]
    bits = low > np.median(low)
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def color_histogram(bgr: np.ndarray, alpha: np.ndarray | None = None) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 15, 240)
    if alpha is not None:
        mask[alpha < 250] = 0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256]).flatten()
    norm = float(np.linalg.norm(hist))
    return hist.astype(np.float32) / norm if norm else np.zeros(512, dtype=np.float32)


def corner_padding(gray: np.ndarray, alpha: np.ndarray | None = None) -> float:
    """Penalize triangular corner fill, not uniform white/black photographic backgrounds."""
    if alpha is not None and np.any(alpha < 250):
        return 0.0
    h, w = gray.shape
    band = max(1, min(h, w) // 10)
    extreme = (gray < 6) | (gray > 249)
    corners = [
        extreme[:band, :band],
        extreme[:band, -band:],
        extreme[-band:, :band],
        extreme[-band:, -band:],
    ]
    middles = [
        extreme[:band, w // 2 : w // 2 + band],
        extreme[-band:, w // 2 : w // 2 + band],
        extreme[h // 2 : h // 2 + band, :band],
        extreme[h // 2 : h // 2 + band, -band:],
    ]
    return max(
        0.0,
        float(np.mean([c.mean() for c in corners])) - float(np.mean([m.mean() for m in middles])),
    )


@dataclass
class Descriptor:
    asset: str
    pixels: dict[str, str]
    width: int
    height: int
    phashes: list[int]
    histogram: list[float] | np.ndarray
    thumbnail: str
    sharpness: float
    padding: float
    informative: bool
    has_alpha: bool
    version: str = ALGORITHM_VERSION
    thumbnail_path: str | None = None

    def __post_init__(self):
        self.histogram = np.asarray(self.histogram, dtype=np.float32)

    def validate(self) -> None:
        expected = {name for name, _ in transforms(np.zeros((1, 1), dtype=np.uint8))}
        if (
            self.thumbnail_path is not None
            or self.width < 1
            or self.height < 1
            or set(self.pixels) != expected
            or len(self.phashes) != 8
            or any(not isinstance(v, int) or not 0 <= v < 2**64 for v in self.phashes)
            or any(
                len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
                for v in self.pixels.values()
            )
        ):
            raise ValueError("Descriptor incompatible")
        hist = np.asarray(self.histogram)
        if (
            hist.shape != (512,)
            or not np.isfinite(hist).all()
            or (hist < 0).any()
            or not np.isfinite([self.sharpness, self.padding]).all()
            or self.sharpness < 0
            or not 0 <= self.padding <= 1
        ):
            raise ValueError("Métricas del descriptor inválidas")
        image = self.image()
        factor = min(1.0, THUMB_SIDE / max(self.width, self.height))
        expected_shape = (
            max(1, round(self.height * factor)),
            max(1, round(self.width * factor)),
            4 if self.has_alpha else 3,
        )
        if image.dtype != np.uint8 or image.shape != expected_shape:
            raise ValueError("Miniatura incompatible")

    def image(self) -> np.ndarray:
        thumbnail = self.thumbnail
        if self.thumbnail_path:
            thumbnail = _read_cache(Path(self.thumbnail_path))["thumbnail"]
        image = cv2.imdecode(
            np.frombuffer(base64.b64decode(thumbnail, validate=True), dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        if image is None:
            raise ValueError("Descriptor con miniatura inválida")
        return image


class DescriptorCache:
    def __init__(self, directory: Path | None = None):
        self.directory = directory
        self.hits = 0
        self.misses = 0
        self.pair_hits = 0

    def describe(self, path: str) -> Descriptor:
        # Hash the actual bytes even on cache hits: an old metadata hash is not identity proof.
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        target = (
            self.directory / ALGORITHM_VERSION / "descriptors" / f"{digest}.json"
            if self.directory
            else None
        )
        if target and target.exists():
            try:
                descriptor = Descriptor(**_read_cache(target))
                if descriptor.asset == digest and descriptor.version == ALGORITHM_VERSION:
                    descriptor.validate()
                    descriptor.thumbnail_path = str(target)
                    descriptor.thumbnail = ""
                    self.hits += 1
                    return descriptor
            except (ValueError, TypeError, KeyError, AttributeError, OSError, cv2.error):
                pass
        # IMREAD_UNCHANGED preserves alpha and does not apply EXIF orientation. Rotational
        # equivalence is handled explicitly by the eight symmetries, on the full raster.
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.uint8:
            raise ValueError("Imagen ilegible o profundidad no soportada para aumentaciones")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] not in (3, 4):
            raise ValueError("Número de canales no soportado")
        height, width = image.shape[:2]
        pixels = {name: pixel_digest(variant) for name, variant in transforms(image)}
        factor = min(1.0, THUMB_SIDE / max(height, width))
        thumb = cv2.resize(
            image,
            (max(1, round(width * factor)), max(1, round(height * factor))),
            interpolation=cv2.INTER_AREA,
        )
        bgr = thumb[:, :, :3]
        alpha = thumb[:, :, 3] if thumb.shape[2] == 4 else None
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        valid = np.ones(gray.shape, dtype=bool) if alpha is None else alpha >= 250
        values = gray[valid]
        informative = bool(values.size >= 256 and float(values.std()) >= 8)
        ok, encoded = cv2.imencode(".png", thumb)
        if not ok:
            raise ValueError("No se pudo codificar el descriptor")
        descriptor = Descriptor(
            digest,
            pixels,
            width,
            height,
            [perceptual_hash(variant) for _, variant in transforms(gray)],
            color_histogram(bgr, alpha).tolist(),
            base64.b64encode(encoded).decode(),
            float(cv2.Laplacian(gray, cv2.CV_64F)[valid].var()) if values.size else 0.0,
            corner_padding(gray, alpha),
            informative,
            alpha is not None,
        )
        self.misses += 1
        if target:
            _write_cache(target, {**asdict(descriptor), "histogram": descriptor.histogram.tolist()})
            descriptor.thumbnail_path = str(target)
            descriptor.thumbnail = ""
        return descriptor

    def verify(self, left: Descriptor, right: Descriptor, policy: DeduplicationPolicy) -> dict:
        payload = {
            "left": left.asset,
            "right": right.asset,
            "version": ALGORITHM_VERSION,
            "correlation": policy.verification_min_correlation,
            "error": policy.verification_max_error,
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        target = (
            self.directory / ALGORITHM_VERSION / "pairs" / f"{key}.json" if self.directory else None
        )
        if target and target.exists():
            try:
                cached = _read_cache(target)
                if cached.get("inputs") == payload and isinstance(cached.get("evidence"), dict):
                    self.pair_hits += 1
                    return cached["evidence"]
            except (ValueError, TypeError, KeyError, AttributeError, OSError):
                pass
        evidence = verify_pair(left, right, policy)
        if target:
            _write_cache(target, {"inputs": payload, "evidence": evidence})
        return evidence


def _comparison(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    a = cv2.cvtColor(left[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    b = cv2.cvtColor(right[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    # Require agreement across the image, not just a few keypoints or background pixels.
    correlations = []
    errors = []
    for rows in np.array_split(np.arange(a.shape[0]), 3):
        for cols in np.array_split(np.arange(a.shape[1]), 3):
            x, y = a[np.ix_(rows, cols)], b[np.ix_(rows, cols)]
            if not x.size:
                continue
            errors.append(float(np.mean(np.abs(x - y))))
            if min(float(x.std()), float(y.std())) > 0.025:
                correlations.append(float(np.corrcoef(x.ravel(), y.ravel())[0, 1]))
    return {
        "correlation": min(correlations, default=0.0),
        "error": max(errors, default=1.0),
        "textured_tiles": len(correlations),
        "color_error": float(
            np.mean(np.abs(left[:, :, :3].astype(np.float32) - right[:, :, :3].astype(np.float32)))
        )
        / 255,
    }


def verify_pair(left: Descriptor, right: Descriptor, policy: DeduplicationPolicy) -> dict:
    evidence = {
        "version": ALGORITHM_VERSION,
        "left_asset": left.asset,
        "right_asset": right.asset,
        "level": "candidate",
        "coverage": 0.0,
    }
    if left.asset == right.asset:
        return {
            **evidence,
            "level": "exact_bytes",
            "transform": "r0",
            "coverage": 1.0,
            "coverage_left": 1.0,
            "coverage_right": 1.0,
        }
    for transform, digest in right.pixels.items():
        if digest == left.pixels["r0"]:
            return {
                **evidence,
                "level": "exact_pixels",
                "transform": transform,
                "coverage": 1.0,
                "coverage_left": 1.0,
                "coverage_right": 1.0,
            }
    if not left.informative or not right.informative:
        return {**evidence, "reason": "low_information"}
    # Transparent pixels and alpha changes must not be hidden by a lossy comparison.
    if left.has_alpha or right.has_alpha:
        return {**evidence, "reason": "alpha_requires_review"}
    a, b = left.image(), right.image()
    best = None
    for transform, rotated in transforms(b):
        h, w = rotated.shape[:2]
        if abs((w / h) / (a.shape[1] / a.shape[0]) - 1) > 0.01:
            continue
        aligned = cv2.resize(rotated, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        scores = _comparison(a, aligned)
        candidate = {
            **evidence,
            **scores,
            "transform": transform,
            "coverage": 1.0,
            "coverage_left": 1.0,
            "coverage_right": 1.0,
        }
        if best is None or scores["error"] < best["error"]:
            best = candidate
        if (
            scores["textured_tiles"] >= 3
            and scores["correlation"] >= policy.verification_min_correlation
            and scores["error"] <= policy.verification_max_error
            and scores["color_error"] <= policy.verification_max_error
        ):
            return {**candidate, "level": "verified_visual"}
    # Geometry provides evidence for difficult candidates, never automatic removal.
    geometry = affine_evidence(a, b)
    plausible = bool(
        geometry.get("inliers", 0) >= 8 or (best is not None and best["correlation"] >= 0.80)
    )
    return {
        **(best or evidence),
        **geometry,
        "level": "candidate" if plausible else "rejected",
        "reason": "full_content_equivalence_unverified"
        if plausible
        else "no_support_for_tested_transforms",
    }


def candidate_pairs(
    descriptors: dict[str, Descriptor], policy: DeduplicationPolicy, metrics: dict | None = None
):
    """Linear-memory retrieval with bounded indexed work or an exhaustive recall oracle."""
    started = time.perf_counter()
    ids = sorted(descriptors, key=lambda key: (descriptors[key].asset, key))
    groups = defaultdict(list)
    for key in ids:
        groups[min(descriptors[key].pixels.values())].append(key)
    pairs = set()
    for members in groups.values():
        pairs.update(tuple(sorted((members[0], member))) for member in members[1:])
    active = [key for key in ids if descriptors[key].informative]
    histograms = np.asarray([descriptors[key].histogram for key in active], dtype=np.float32)
    hashes = np.asarray([descriptors[key].phashes for key in active], dtype=np.uint64)
    index = (
        CandidateIndex(hashes, histograms, policy.candidate_pool)
        if active and policy.candidate_retrieval == "indexed"
        else None
    )
    indexed_at = time.perf_counter()
    saturated = 0
    for i, key in enumerate(active):
        neighbors = index.query(i) if index else np.arange(len(active))
        neighbors = neighbors[neighbors != i]
        if not len(neighbors):
            continue
        distances = np.bitwise_count(hashes[neighbors] ^ hashes[i, 0]).min(axis=1)
        similarities = histograms[neighbors] @ histograms[i]
        candidates = np.flatnonzero(
            (distances <= policy.phash_distance) | (similarities >= policy.augmentation_similarity)
        ).tolist()
        saturated += int(len(candidates) > policy.candidate_neighbors)
        by_hash = sorted(candidates, key=lambda j: (int(distances[j]), int(neighbors[j])))
        by_color = sorted(candidates, key=lambda j: (-float(similarities[j]), int(neighbors[j])))
        selected = set(by_hash[: policy.candidate_neighbors]) | set(
            by_color[: policy.candidate_neighbors]
        )
        for j in selected:
            pairs.add(tuple(sorted((key, active[neighbors[j]]))))
    if metrics is not None:
        metrics.update(
            {
                "retrieval": policy.candidate_retrieval,
                "index_seconds": indexed_at - started,
                "retrieval_seconds": time.perf_counter() - indexed_at,
                "pool_truncated_images": index.truncated_queries if index else 0,
                "scored_neighbors": index.scored_neighbors
                if index
                else len(active) * max(0, len(active) - 1),
            }
        )
    return sorted(pairs), saturated


def analyze_images(
    paths: dict[str, str], policy: DeduplicationPolicy, cache: DescriptorCache, extra_pairs=()
) -> tuple[dict, list[dict], dict]:
    started = time.perf_counter()
    descriptors, invalid = {}, {}
    for key, path in paths.items():
        try:
            descriptors[key] = cache.describe(path)
        except (OSError, ValueError, cv2.error) as exc:
            invalid[key] = str(exc)
    retrieval_stats = {}
    candidates, saturated = candidate_pairs(descriptors, policy, retrieval_stats)
    candidates = sorted(
        set(candidates)
        | {tuple(sorted(pair)) for pair in extra_pairs if all(key in descriptors for key in pair)}
    )
    pairs = [
        {
            "left": left,
            "right": right,
            **cache.verify(descriptors[left], descriptors[right], policy),
            "phash_distance": min(
                (descriptors[left].phashes[0] ^ value).bit_count()
                for value in descriptors[right].phashes
            ),
            "histogram_similarity": float(
                np.dot(descriptors[left].histogram, descriptors[right].histogram)
            ),
        }
        for left, right in candidates
    ]
    stats = {
        "version": ALGORITHM_VERSION,
        "images": len(paths),
        "invalid": invalid,
        "candidates": len(pairs),
        "saturated_images": saturated,
        "descriptor_hits": cache.hits,
        "descriptor_misses": cache.misses,
        "pair_cache_hits": cache.pair_hits,
        "seconds": time.perf_counter() - started,
        **retrieval_stats,
        "low_information": sorted(key for key, item in descriptors.items() if not item.informative),
        "pair_levels": dict(Counter(pair["level"] for pair in pairs)),
        "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024 * 1024 if sys.platform == "darwin" else 1024),
    }
    stats["images_per_second"] = len(paths) / max(stats["seconds"], 1e-9)
    stats["comparisons_per_image"] = len(pairs) / max(len(paths), 1)
    return descriptors, pairs, stats
