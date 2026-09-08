"""Reproducible evaluation of augmentation families, without MongoDB or models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from agrivision_khaos.augmentation import (
    CONFIRMED_LEVELS,
    DescriptorCache,
    analyze_images,
    candidate_pairs,
    transforms,
)
from agrivision_khaos.execution import atomic_write_json
from agrivision_khaos.models import CurationPolicy, DeduplicationPolicy
from agrivision_khaos.synthetic import _require_empty_directory


def generate_fixture(directory: Path, families: int = 6, seed: int = 41) -> Path:
    _require_empty_directory(directory)
    rng = np.random.default_rng(seed)
    rows = []

    def save(family, transformation, image, extension=".png", parameters=()):
        key = f"{family}-{transformation}"
        filename = key + extension
        if not cv2.imwrite(str(directory / filename), image, list(parameters)):
            raise OSError(filename)
        rows.append(
            {"id": key, "filepath": filename, "family": family, "transformation": transformation}
        )

    for index in range(families):
        family = f"leaf-{index}"
        texture = rng.integers(20, 220, (192, 256, 3), dtype=np.uint8)
        image = cv2.GaussianBlur(texture, (5, 5), 0)
        for _ in range(18):
            point = (int(rng.integers(10, 246)), int(rng.integers(10, 182)))
            color = tuple(int(v) for v in rng.integers(20, 235, 3))
            cv2.circle(image, point, int(rng.integers(4, 20)), color, -1)
        for name, variant in transforms(image):
            save(family, name, variant)
        save(family, "jpeg", image, ".jpg", (cv2.IMWRITE_JPEG_QUALITY, 85))
        save(family, "scale", cv2.resize(image, (384, 288)))
        save(family, "brightness", cv2.convertScaleAbs(image, alpha=1, beta=20))
        save(family, "contrast", cv2.convertScaleAbs(image, alpha=1.25))
        matrix = cv2.getRotationMatrix2D((128, 96), 23, 0.8)
        save(family, "rotation_padding", cv2.warpAffine(image, matrix, (256, 192)))
        save(family, "crop", image[25:160, 35:225])
        # Same pixels and histogram, different spatial content: a hard negative.
        shuffled = image.reshape(-1, 3)[rng.permutation(image.shape[0] * image.shape[1])]
        save(f"negative-{index}", "shuffled", shuffled.reshape(image.shape))
    manifest = directory / "manifest.json"
    atomic_write_json(manifest, {"seed": seed, "synthetic": True, "samples": rows})
    return manifest


def read_manifest(manifest: Path) -> dict:
    payload = json.loads(manifest.read_text())
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("samples"), list)
        or not payload["samples"]
    ):
        raise ValueError("El manifiesto debe contener una lista samples no vacía")
    if "synthetic" not in payload or not isinstance(payload["synthetic"], bool):
        raise ValueError("Declara explícitamente synthetic: true/false")
    ids = set()
    for row in payload["samples"]:
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("id", "filepath", "family")
        ):
            raise ValueError("Cada muestra requiere id, filepath y family de texto no vacío")
        if row["id"] in ids:
            raise ValueError("IDs duplicados en el benchmark")
        ids.add(row["id"])
        for key in ("source", "transformation", "capture_group"):
            if key in row and (not isinstance(row[key], str) or not row[key].strip()):
                raise ValueError(f"{key} debe ser texto no vacío")
    return payload


def _wilson(successes: int, count: int) -> list[float] | None:
    """95% Wilson interval for independent family-level Bernoulli outcomes."""
    if not count:
        return None
    z = 1.959963984540054
    rate = successes / count
    center = (rate + z * z / (2 * count)) / (1 + z * z / count)
    radius = (
        z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def evaluate_manifest(
    manifest: Path,
    policy: DeduplicationPolicy,
    cache_dir: Path,
    *,
    compare_exhaustive: bool = False,
    baseline_limit: int = 2000,
    calibration_manifest: Path | None = None,
):
    started = time.perf_counter()
    payload = read_manifest(manifest)
    rows = payload["samples"]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("IDs duplicados en el benchmark")
    paths = {key: str((manifest.parent / row["filepath"]).resolve()) for key, row in by_id.items()}
    descriptors, pairs, stats = analyze_images(paths, policy, DescriptorCache(cache_dir))
    if stats["invalid"]:
        raise ValueError(
            f"Benchmark incompleto: {len(stats['invalid'])} imágenes inválidas: {stats['invalid']}"
        )
    split_validation = {"checked": False, "reason": "No se aportó manifiesto de calibración"}
    if calibration_manifest is not None:
        calibration = read_manifest(calibration_manifest)
        calibration_rows = calibration["samples"]

        def capture(row):
            return (row.get("source", "unknown"), row.get("capture_group", row["family"]))

        overlap = {capture(row) for row in rows} & {capture(row) for row in calibration_rows}
        calibration_cache = DescriptorCache(cache_dir)
        calibration_pixels = {
            min(
                calibration_cache.describe(
                    str((calibration_manifest.parent / row["filepath"]).resolve())
                ).pixels.values()
            )
            for row in calibration_rows
        }
        family_overlap = {row["family"] for row in rows} & {
            row["family"] for row in calibration_rows
        }
        if (
            overlap
            or family_overlap
            or calibration_pixels & {min(item.pixels.values()) for item in descriptors.values()}
        ):
            raise ValueError(
                "Calibración y evaluación comparten capturas o píxeles; reserva familias independientes"
            )
        split_validation = {
            "checked": True,
            "calibration_samples": len(calibration_rows),
            "note": "Comprueba IDs declarados y equivalencia exacta; revisar parentescos no declarados",
        }
    predictions = [pair for pair in pairs if pair["level"] in CONFIRMED_LEVELS]
    correct = [
        pair
        for pair in predictions
        if by_id[pair["left"]]["family"] == by_id[pair["right"]]["family"]
    ]
    recovered = {key for pair in correct for key in (pair["left"], pair["right"])}
    family_sizes = Counter(row["family"] for row in rows)
    totals, recovered_counts = Counter(), Counter()
    for key, row in by_id.items():
        if family_sizes[row["family"]] < 2:
            continue
        kind = row.get("transformation", "unknown")
        totals[kind] += 1
        recovered_counts[kind] += int(key in recovered)
    # Pairwise HSV threshold baseline, equivalent to the old signal, not to its
    # representative-selection heuristic. Evaluate against every labeled pair.
    baseline_total = baseline_correct = 0
    ids = sorted(descriptors)
    baseline_ids = ids if len(ids) <= baseline_limit else []
    for index, left in enumerate(baseline_ids):
        for right in baseline_ids[index + 1 :]:
            similarity = float(np.dot(descriptors[left].histogram, descriptors[right].histogram))
            if similarity >= policy.augmentation_similarity:
                baseline_total += 1
                baseline_correct += int(by_id[left]["family"] == by_id[right]["family"])
    predicted_families = {
        by_id[key]["family"] for pair in predictions for key in (pair["left"], pair["right"])
    }
    contaminated = {
        by_id[key]["family"]
        for pair in predictions
        if by_id[pair["left"]]["family"] != by_id[pair["right"]]["family"]
        for key in (pair["left"], pair["right"])
    }
    family_interval = _wilson(len(predicted_families) - len(contaminated), len(predicted_families))
    sources = {}
    for source in sorted({row.get("source", "unknown") for row in rows}):
        members = {key for key, row in by_id.items() if row.get("source", "unknown") == source}
        eligible = {key for key in members if family_sizes[by_id[key]["family"]] > 1}
        related = [pair for pair in predictions if {pair["left"], pair["right"]} & members]
        errors = sum(
            by_id[pair["left"]]["family"] != by_id[pair["right"]]["family"] for pair in related
        )
        sources[source] = {
            "images": len(members),
            "confirmed_pairs": len(related),
            "false_positive_pairs": errors,
            "pair_precision": 1 - errors / len(related) if related else None,
            "eligible_members": len(eligible),
            "member_recall": len(eligible & recovered) / len(eligible) if eligible else None,
        }
    comparison = {"evaluated": False}
    if compare_exhaustive:
        oracle_policy = policy.model_copy(update={"candidate_retrieval": "exhaustive"})
        oracle_pairs, _ = candidate_pairs(descriptors, oracle_policy)
        actual_pairs = {tuple(sorted((pair["left"], pair["right"]))) for pair in pairs}
        oracle_set = set(oracle_pairs)
        oracle_cache = DescriptorCache(cache_dir)
        oracle_confirmed = {
            pair
            for pair in oracle_pairs
            if oracle_cache.verify(descriptors[pair[0]], descriptors[pair[1]], policy)["level"]
            in CONFIRMED_LEVELS
        }
        comparison = {
            "evaluated": True,
            "oracle_candidates": len(oracle_set),
            "candidate_recall": len(actual_pairs & oracle_set) / len(oracle_set)
            if oracle_set
            else None,
            "confirmed_pair_recall": len(actual_pairs & oracle_confirmed) / len(oracle_confirmed)
            if oracle_confirmed
            else None,
            "missed_confirmed_pairs": [
                list(pair) for pair in sorted(oracle_confirmed - actual_pairs)
            ],
        }
    ambiguous_members = {
        key
        for pair in pairs
        if pair["level"] == "candidate"
        for key in (pair["left"], pair["right"])
    }
    return {
        "synthetic": bool(payload.get("synthetic", False)),
        "evaluation_seconds": time.perf_counter() - started,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "stats": stats,
        "holdout_validation": split_validation,
        "retrieval_comparison": comparison,
        "by_source": sources,
        "family_consistency": {
            "evaluated_families": len(predicted_families),
            "families_with_false_links": len(contaminated),
            "wilson_95_interval": family_interval,
            "assumption": "Capturas independientes; no es un intervalo de precisión de sustituciones",
        },
        "review_workload": {
            "ambiguous_pairs": sum(pair["level"] == "candidate" for pair in pairs),
            "ambiguous_images": len(ambiguous_members),
        },
        "activation": {
            "ready": False,
            "reason": "Este benchmark evalúa identidad, no certifica sustituciones ni preservación de anotaciones",
        },
        "pairs": pairs,
        "policy": policy.model_dump(),
        "confirmed_pairs": len(predictions),
        "false_positive_pairs": len(predictions) - len(correct),
        "pair_precision": len(correct) / len(predictions) if predictions else None,
        "heuristic_auto_removals_enabled": False,
        "member_recovery": {
            kind: {
                "recovered": recovered_counts[kind],
                "total": total,
                "recall": recovered_counts[kind] / total,
            }
            for kind, total in sorted(totals.items())
        },
        "hsv_pairwise_baseline": {
            "evaluated": bool(baseline_ids),
            "image_limit": baseline_limit,
            "pairs": baseline_total,
            "false_positive_pairs": baseline_total - baseline_correct,
            "precision": baseline_correct / baseline_total if baseline_total else None,
        },
        "note": "La recuperación mide miembros con al menos una pareja correcta. "
        "Los datos sintéticos no calibran la precisión en datasets agrícolas reales.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, help="JSON con samples: id, filepath, family, transformation"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families", type=int, default=6)
    parser.add_argument("--policy", type=Path, help="Política YAML del pipeline")
    parser.add_argument(
        "--compare-exhaustive",
        action="store_true",
        help="Mide pérdida de candidatos; coste cuadrático",
    )
    parser.add_argument(
        "--baseline-limit",
        type=int,
        default=2000,
        help="Omite referencia HSV por encima de este tamaño; 0 la desactiva",
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        help="Comprueba separación respecto al conjunto usado para ajustar parámetros",
    )
    args = parser.parse_args()
    if args.families < 1:
        parser.error("--families debe ser positivo")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.baseline_limit < 0:
        parser.error("--baseline-limit no puede ser negativo")
    try:
        policy = (
            CurationPolicy.model_validate(yaml.safe_load(args.policy.read_text())).deduplication
            if args.policy
            else DeduplicationPolicy()
        )
        manifest = args.manifest
        if manifest is None:
            manifest = args.output_dir / "fixture" / "manifest.json"
            if not manifest.exists():
                manifest = generate_fixture(args.output_dir / "fixture", args.families)
            elif (
                read_manifest(manifest).get("seed") != 41
                or len(read_manifest(manifest)["samples"]) != args.families * 15
            ):
                raise ValueError("El fixture existente no coincide; usa otro directorio de salida")
        report = evaluate_manifest(
            manifest,
            policy,
            args.output_dir / "cache",
            compare_exhaustive=args.compare_exhaustive,
            baseline_limit=args.baseline_limit,
            calibration_manifest=args.calibration_manifest,
        )
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    atomic_write_json(args.output_dir / "benchmark.json", report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "pairs"},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
