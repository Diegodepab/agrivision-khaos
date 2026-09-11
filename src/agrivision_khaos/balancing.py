"""Módulo de balanceo de clases mediante aumentación de datos sintéticos.

Diseñado para equilibrar clases minoritarias en datasets agrícolas (e.g., enfermedades
raras de hojas/frutos) aplicando transformaciones realistas del dominio agrícola
(rotaciones sutiles, variaciones de iluminación/HSV, simetrías) y confinando estrictamente
las muestras sintéticas al split 'train'.
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import fiftyone as fo
from agrivision_khaos.pipeline import PhaseResult

logger = logging.getLogger(__name__)


def compute_class_distribution(
    dataset: fo.Dataset,
    label_field: str = "ground_truth",
    split: str | None = "train",
) -> dict[str, int]:
    """Calcula el conteo de muestras por clase en el dataset (por defecto en split 'train')."""
    view = dataset
    if split is not None and "split" in dataset.get_field_schema():
        view = dataset.match(fo.ViewField("split") == split)

    # Filtrar solo muestras aceptadas (curation_kept)
    if "curation" in dataset.get_field_schema():
        view = view.match(fo.ViewField("curation.status") == "kept")

    counts: Counter[str] = Counter()
    for sample in view:
        label_val = None
        if hasattr(sample, label_field) and getattr(sample, label_field) is not None:
            field_obj = getattr(sample, label_field)
            if hasattr(field_obj, "label"):
                label_val = field_obj.label
            elif isinstance(field_obj, str):
                label_val = field_obj
        elif "normalized_label" in sample and sample["normalized_label"]:
            label_val = sample["normalized_label"]

        if label_val:
            counts[label_val] += 1

    return dict(counts)


def calculate_target_count(
    counts: dict[str, int],
    strategy: str = "median",
) -> int:
    """Calcula el número objetivo de muestras por clase según la estrategia dada."""
    if not counts:
        return 0

    values = list(counts.values())
    if strategy.isdigit():
        return int(strategy)
    elif strategy == "max":
        return max(values)
    elif strategy == "mean":
        return int(round(float(np.mean(values))))
    elif strategy == "median":
        return int(round(float(np.median(values))))
    else:
        try:
            val = int(strategy)
            return val
        except ValueError:
            logger.warning("Estrategia de balanceo desconocida '%s', usando 'median'.", strategy)
            return int(round(float(np.median(values))))


def apply_agronomic_augmentation(image: np.ndarray, seed: int | None = None) -> np.ndarray:
    """
    Aplica una combinación realista de transformaciones para imágenes agrícolas:
    - Rotación leve (-15° a 15°) o giro discreto de 90°/180°/270°.
    - Volteo horizontal o vertical.
    - Variación sutil de brillo y contraste (simula luz solar cambiante).
    - Ajuste leve de saturación y tono HSV (simula clorofila/sombra).
    - Desenfoque gaussiano muy leve opcional (simula desenfoque de cámara).
    """
    rng = np.random.default_rng(seed)
    augmented = image.copy()
    h, w = augmented.shape[:2]

    # 1. Flip horizontal aleatorio (50% probabilidad)
    if rng.random() > 0.5:
        augmented = cv2.flip(augmented, 1)

    # 2. Flip vertical aleatorio (30% probabilidad)
    if rng.random() > 0.7:
        augmented = cv2.flip(augmented, 0)

    # 3. Rotación sutil
    angle = float(rng.uniform(-15.0, 15.0))
    if rng.random() > 0.8:
        # En ocasiones giro de 90 o 180 grados
        quarter = int(rng.choice([1, 2, 3]))
        augmented = np.ascontiguousarray(np.rot90(augmented, quarter))
        h, w = augmented.shape[:2]
    else:
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        augmented = cv2.warpAffine(
            augmented,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    # 4. Ajuste fotométrico (Brillo y Contraste)
    alpha = float(rng.uniform(0.88, 1.14))  # Contraste
    beta = float(rng.uniform(-14.0, 14.0))   # Brillo
    augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)

    # 5. Ajuste de color en espacio HSV
    if rng.random() > 0.4:
        hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_shift = float(rng.uniform(-6.0, 6.0))
        s_scale = float(rng.uniform(0.85, 1.15))
        hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
        augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 6. Desenfoque leve ocasional (25% probabilidad)
    if rng.random() > 0.75:
        augmented = cv2.GaussianBlur(augmented, (3, 3), 0)

    return augmented


def balance_dataset_classes(
    dataset: fo.Dataset,
    label_field: str = "ground_truth",
    target_strategy: str = "median",
    output_dir: Path | None = None,
    max_augment_per_class: int = 2000,
) -> PhaseResult:
    """
    Equilibra las clases minoritarias del dataset mediante síntesis de imágenes aumentadas.
    Todas las muestras generadas se restringen estrictamente al split 'train' para evitar fugas.
    """
    result = PhaseResult(name="class_balancing")
    counts = compute_class_distribution(dataset, label_field=label_field, split="train")

    if not counts:
        # Si no hay split train asignado, contar sobre todo el dataset activo
        counts = compute_class_distribution(dataset, label_field=label_field, split=None)

    if not counts or len(counts) < 2:
        result.notes.append("No hay suficientes clases para balancear.")
        return result

    target_count = calculate_target_count(counts, strategy=target_strategy)
    logger.info(
        "Distribución actual de clases: %s. Objetivo de balanceo (%s): %d muestras/clase.",
        dict(counts),
        target_strategy,
        target_count,
    )

    if output_dir is None:
        output_dir = Path("/datasets/cache/augmented") / dataset.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Agrupar muestras activas por clase
    class_samples: dict[str, list[Any]] = {}
    for sample in dataset:
        if "curation" in dataset.get_field_schema() and sample.get_field("curation"):
            if getattr(sample.curation, "status", "kept") != "kept":
                continue
        # No aumentar sobre muestras ya sintéticas
        if sample.tags and "synthetic" in sample.tags:
            continue

        lbl = None
        if hasattr(sample, label_field) and getattr(sample, label_field) is not None:
            field_obj = getattr(sample, label_field)
            if hasattr(field_obj, "label"):
                lbl = field_obj.label
            elif isinstance(field_obj, str):
                lbl = field_obj
        elif "normalized_label" in sample and sample["normalized_label"]:
            lbl = sample["normalized_label"]

        if lbl:
            class_samples.setdefault(lbl, []).append(sample)

    new_samples: list[fo.Sample] = []
    total_generated = 0

    for lbl, current_num in counts.items():
        if current_num >= target_count:
            continue

        needed = min(target_count - current_num, max_augment_per_class)
        available = class_samples.get(lbl, [])
        if not available:
            continue

        logger.info("Clase '%s': %d muestras -> generando %d aumentaciones...", lbl, current_num, needed)
        generated_for_class = 0

        while generated_for_class < needed:
            parent_sample = random.choice(available)
            src_path = parent_sample.filepath
            img = cv2.imread(src_path)
            if img is None:
                continue

            aug_img = apply_agronomic_augmentation(img)
            aug_filename = f"synth_{lbl}_{parent_sample.id}_{generated_for_class:04d}.jpg"
            out_file = output_dir / aug_filename
            cv2.imwrite(str(out_file), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

            new_sample = fo.Sample(filepath=str(out_file))
            if hasattr(parent_sample, label_field) and getattr(parent_sample, label_field) is not None:
                parent_label = getattr(parent_sample, label_field)
                if isinstance(parent_label, fo.Classification):
                    new_sample[label_field] = fo.Classification(
                        label=parent_label.label,
                        confidence=1.0,
                    )
                elif hasattr(parent_label, "copy"):
                    new_sample[label_field] = parent_label.copy()

            if "normalized_label" in parent_sample and parent_sample["normalized_label"]:
                new_sample["normalized_label"] = parent_sample["normalized_label"]

            new_sample["split"] = "train"
            new_sample["is_synthetic"] = True
            new_sample["parent_sample_id"] = parent_sample.id
            new_sample["curation"] = fo.DynamicEmbeddedDocument(
                status="kept",
                reason="synthetic_balanced",
                phase="balancing",
            )
            new_sample.tags = ["synthetic", "augmented_balance", "curation_kept"]
            new_samples.append(new_sample)

            generated_for_class += 1
            total_generated += 1

    if new_samples:
        dataset.add_samples(new_samples)
        dataset.save()
        result.notes.append(f"Generadas {total_generated} muestras sintéticas en split train.")
        logger.info("Balanceo completado: %d muestras añadidas.", total_generated)
    else:
        result.notes.append("Dataset ya se encontraba balanceado según la estrategia.")

    result.kept = len(dataset)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditoría y balanceo de clases por aumentación.")
    parser.add_argument("--dataset", required=True, help="Nombre del dataset FiftyOne.")
    parser.add_argument("--strategy", default="median", help="Estrategia: median, max, mean, o entero.")
    parser.add_argument("--output-dir", default=None, help="Directorio para imágenes aumentadas.")
    args = parser.parse_args()

    if not fo.dataset_exists(args.dataset):
        raise SystemExit(f"Dataset '{args.dataset}' no encontrado en FiftyOne.")

    ds = fo.load_dataset(args.dataset)
    out_dir = Path(args.output_dir) if args.output_dir else None
    res = balance_dataset_classes(ds, target_strategy=args.strategy, output_dir=out_dir)
    print(json.dumps(asdict(res), indent=2))


if __name__ == "__main__":
    import json
    main()
