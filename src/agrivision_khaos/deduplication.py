from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import fiftyone as fo
import fiftyone.brain as fob
import numpy as np

# Configuración de logging orientado a los datos
from rich.logging import RichHandler

from agrivision_khaos.augmentation import (
    ALGORITHM_VERSION,
    DescriptorCache,
    analyze_images,
    corner_padding,
)
from agrivision_khaos.execution import PipelineLock
from agrivision_khaos.families import human_rejections, relation_groups, stable_identity
from agrivision_khaos.models import DeduplicationPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:  # Core ingestion/quality installations do not need embeddings
    torch = None

# Modelo de extracción de características para la detección semántica.
# Con GPU (CUDA) usamos ResNet50; en CPU, MobileNetV2 reduce el coste de inferencia.
SIMILARITY_MODEL = (
    "resnet50-imagenet-torch"
    if torch is not None and torch.cuda.is_available()
    else "mobilenet-v2-imagenet-torch"
)


def _require_embedding_backend() -> None:
    if torch is None:
        raise RuntimeError(
            "La deduplicación semántica requiere PyTorch: instala el extra de CPU con "
            "`uv sync --extra cpu` o el de NVIDIA CUDA 13.0 con `uv sync --extra cu130`."
        )


def load_dataset(dataset_name: str) -> fo.Dataset:
    """Carga el dataset de FiftyOne desde la base de datos local."""
    if not fo.dataset_exists(dataset_name):
        logger.error("El dataset '%s' no se encuentra en el sistema.", dataset_name)
        sys.exit(1)

    dataset = fo.load_dataset(dataset_name)
    logger.info("Dataset '%s' cargado. Muestras iniciales: %d", dataset_name, len(dataset))
    return dataset


def _get_padding_ratio(filepath: str) -> float:
    image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if image is None:
        return 1.0
    gray = image if image.ndim == 2 else cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    alpha = image[:, :, 3] if image.ndim == 3 and image.shape[2] == 4 else None
    return corner_padding(gray, alpha)


def _sample_value(sample: fo.Sample, name: str):
    try:
        if sample.has_field(name):
            return sample.get_field(name)
    except Exception:
        pass

    try:
        quality = sample.get_field("quality") if sample.has_field("quality") else None
    except Exception:
        quality = getattr(sample, "quality", None)

    if quality is None:
        return None
    if hasattr(quality, "get"):
        return quality.get(name)
    return getattr(quality, name, None)


def _metric(sample: fo.Sample, name: str, default: float = 0.0) -> float:
    value = _sample_value(sample, name)
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _flag(sample: fo.Sample, name: str) -> bool:
    return bool(_sample_value(sample, name))


def _is_probably_augmented_path(filepath: str) -> bool:
    normalized = filepath.lower()
    return ".rf." in normalized or "_aug" in normalized or "augmentation" in normalized


def _sample_keep_score(sample: fo.Sample) -> tuple[float, ...]:
    width = _metric(sample, "width")
    height = _metric(sample, "height")
    area = width * height
    blur = _metric(sample, "blur_variance", default=-1.0)
    padding = _metric(sample, "augmentation_padding", default=-1.0)
    if padding < 0:
        padding = _get_padding_ratio(sample.filepath)
    sharpness = _metric(sample, "augmentation_sharpness", default=blur)

    return (
        0.0 if _curation_status(sample) == "removed" else 1.0,
        0.0 if _curation_status(sample) == "review" else 1.0,
        0.0 if _flag(sample, "is_corrupted") or _flag(sample, "processing_error") else 1.0,
        0.0 if _sample_value(sample, "annotation_valid") is False else 1.0,
        0.0 if _flag(sample, "has_watermark") else 1.0,
        0.0 if _flag(sample, "has_smearing") else 1.0,
        0.0 if _flag(sample, "low_resolution") else 1.0,
        _annotation_richness(sample),
        sharpness,
        -padding,
        area,
        0.0 if _is_probably_augmented_path(sample.filepath) else 1.0,
    )


def _curation_status(sample: fo.Sample) -> str:
    curation = _sample_value(sample, "curation")
    if curation is None:
        return "kept"
    if hasattr(curation, "get"):
        return str(curation.get("status", "kept") or "kept")
    return str(getattr(curation, "status", "kept") or "kept")


def _label_signature(sample: fo.Sample) -> frozenset[str]:
    labels = _sample_value(sample, "normalized_labels") or []
    if labels:
        return frozenset(str(label) for label in labels if label)
    label = _sample_value(sample, "normalized_label")
    return frozenset([str(label)]) if label else frozenset()


def _connected_components(duplicates_dict: dict) -> list[list[str]]:
    """Converts overlapping neighbor groups into deterministic components."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for representative, duplicates in duplicates_dict.items():
        representative = str(representative)
        find(representative)
        for duplicate in duplicates:
            union(representative, str(duplicate))

    groups: dict[str, set[str]] = {}
    for sample_id in parent:
        groups.setdefault(find(sample_id), set()).add(sample_id)
    return sorted(
        (sorted(group) for group in groups.values() if len(group) > 1),
        key=lambda group: group[0],
    )


def _annotation_signature(sample, transform="r0"):
    task = _sample_value(sample, "task_type") or "unlabeled"
    if task == "unlabeled" and any(
        _sample_value(sample, name) is not None
        for name in ("original_label", "coco_detections", "yolo_detections", "voc_detections")
    ):
        return None  # Imported labels must be canonicalized before automatic substitution.
    if set(task.split(",")) & {"segmentation", "keypoints"}:
        return None  # No canonical geometry comparison is implemented for these tasks.
    detections = _sample_value(sample, "ground_truth_detections")
    boxes = []
    if detections is not None:
        for detection in detections.detections:
            if detection.mask is not None:
                return None  # Never discard an instance mask based on box equality.
            x, y, w, h = detection.bounding_box
            corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
            if transform.startswith("mirror_"):
                corners = [(1 - px, py) for px, py in corners]
            turns = int(transform.split("r")[-1]) // 90
            for _ in range(turns):
                corners = [(py, 1 - px) for px, py in corners]
            xs, ys = zip(*corners, strict=True)
            box = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            attributes = {
                key: item
                for key, item in detection.to_dict().items()
                if key
                not in {
                    "_id",
                    "_cls",
                    "label",
                    "bounding_box",
                    "blur_variance",
                    "source_label",
                    "source_field",
                }
            }
            boxes.append(
                (
                    detection.label,
                    tuple(round(v, 5) for v in box),
                    json.dumps(attributes, sort_keys=True, default=str),
                )
            )
    return task, _label_signature(sample), tuple(sorted(boxes))


def _annotation_richness(sample):
    signature = _annotation_signature(sample)
    if signature is None:
        return 0
    return sum(
        sum(item not in (None, {}, []) for item in json.loads(box[2]).values())
        for box in signature[2]
    )


def _compatible_annotations(left, right, evidence):
    transform = evidence.get("transform", "r0")
    left_signature = _annotation_signature(left)
    right_signature = _annotation_signature(right, transform)
    if (
        left_signature is None
        or right_signature is None
        or left_signature[:2] != right_signature[:2]
    ):
        return False
    if len(left_signature[2]) != len(right_signature[2]):
        return False
    available = list(left_signature[2])
    for candidate in right_signature[2]:
        attributes = json.loads(candidate[2])
        for index, representative in enumerate(available):
            represented = json.loads(representative[2])
            if representative[:2] == candidate[:2] and all(
                key in represented and represented[key] == item for key, item in attributes.items()
            ):
                available.pop(index)
                break
        else:
            return False
    return True


def rebuild_duplicate_families(dataset):
    samples = list(dataset)
    all_groups = relation_groups(samples)
    confirmed = relation_groups(samples, confirmed_only=True)
    for name in ("duplicate_cluster_id", "duplicate_family_id"):
        if name not in dataset.get_field_schema():
            dataset.add_sample_field(name, fo.StringField)
    dataset.set_values("duplicate_cluster_id", all_groups, key_field="id")
    dataset.set_values("duplicate_family_id", confirmed, key_field="id")


def process_duplicates(
    dataset: fo.Dataset,
    duplicates_dict: dict,
    tag: str,
    evidence: list[dict] | None = None,
    verifier=None,
    pixel_families: dict[str, str] | None = None,
):
    """Persist per-method edges and choose substitutes only with direct evidence."""
    conflict_tag = f"{tag}_label_conflict"
    dataset.untag_samples([tag, conflict_tag])
    samples = {sample.id: sample for sample in dataset}
    rejected_pairs = human_rejections(samples.values())
    records = (
        evidence
        if evidence is not None
        else [
            {
                "left": left,
                "right": right,
                "level": "exact_bytes" if tag == "redundant_exact" else "candidate",
                "transform": "r0",
                "version": ALGORITHM_VERSION,
            }
            for left, rights in duplicates_dict.items()
            for right in rights
        ]
    )
    edges = {}
    for record in records:
        left, right = record["left"], record["right"]
        if left in samples and right in samples and left != right:
            edges[frozenset((left, right))] = record
    adjacency = {}
    neighborhoods = {}
    for pair, record in edges.items():
        if record.get("level") == "rejected" or pair in rejected_pairs:
            continue
        left, right = sorted(pair)
        adjacency.setdefault(left, []).append(right)
        neighborhoods.setdefault(left, set()).add(right)
        neighborhoods.setdefault(right, set()).add(left)
    exact_members = {}
    for key, family in (pixel_families or {}).items():
        exact_members.setdefault(family, set()).add(key)
    components = _connected_components(adjacency)
    assignments, conflicts, pairs = {}, set(), []
    for members in components:
        ordered = sorted(members, key=lambda key: stable_identity(samples[key]))
        ordered.sort(key=lambda key: _sample_keep_score(samples[key]), reverse=True)
        rank = {key: index for index, key in enumerate(ordered)}
        remaining = set(members)
        for representative in ordered:
            if representative not in remaining:
                continue
            sample = samples[representative]
            if (
                _curation_status(sample) != "kept"
                or _flag(sample, "is_corrupted")
                or _flag(sample, "processing_error")
                or _sample_value(sample, "annotation_valid") is False
            ):
                continue
            remaining.remove(representative)
            neighbors = neighborhoods.get(representative, set())
            if pixel_families:
                neighbors = neighbors | exact_members.get(pixel_families.get(representative), set())
            available = remaining if tag == "redundant_exact" else remaining & neighbors
            for candidate in sorted(available, key=rank.__getitem__):
                if frozenset((representative, candidate)) in rejected_pairs:
                    continue
                record = edges.get(frozenset((representative, candidate)))
                if verifier:
                    record = {
                        "left": representative,
                        "right": candidate,
                        **verifier(representative, candidate),
                    }
                    # Only test additional representative edges for members of this component.
                    edges[frozenset((representative, candidate))] = record
                elif tag == "redundant_exact":
                    # Byte equality is transitive; these components come from exact hashing.
                    record = {
                        "left": representative,
                        "right": candidate,
                        "level": "exact_bytes",
                        "transform": "r0",
                        "version": ALGORITHM_VERSION,
                    }
                    edges[frozenset((representative, candidate))] = record
                if record is None:
                    continue
                if record.get("level") == "rejected":
                    continue
                # Approximate geometry cannot safely transform detection annotations.
                if record.get("left") != representative and record.get("transform", "r0") != "r0":
                    conflicts.update((representative, candidate))
                    continue
                if not _compatible_annotations(sample, samples[candidate], record):
                    conflicts.update((representative, candidate))
                    continue
                assignments[candidate] = {**record, "representative_id": representative}
                pairs.append((representative, candidate))
                remaining.remove(candidate)
        # No valid representative: preserve the family for review, never force a kept image.
        if remaining:
            conflicts.update(remaining)
    # A conflict anywhere must not hide or discard an annotation from this method.
    for candidate in list(assignments):
        if candidate in conflicts or assignments[candidate]["representative_id"] in conflicts:
            del assignments[candidate]
    pairs = [(a, b) for a, b in pairs if b in assignments]
    links = {
        key: [
            dict(link)
            for link in (_sample_value(sample, "duplicate_links") or [])
            if link.get("method") != tag
        ]
        for key, sample in samples.items()
    }
    for record in edges.values():
        left, right = record["left"], record["right"]
        record = {
            **record,
            "left_asset": _sample_value(samples[left], "asset_sha256") or "",
            "right_asset": _sample_value(samples[right], "asset_sha256") or "",
        }
        for key, other in ((left, right), (right, left)):
            links[key].append(
                {
                    **record,
                    "other_id": other,
                    "method": tag,
                    "rejected": record.get("level") == "rejected",
                }
            )
    for name, field_type in (
        ("duplicate_links", fo.ListField),
        (f"{tag}_evidence", fo.DictField),
        ("duplicate_schema_version", fo.StringField),
        ("duplicate_representative_id", fo.StringField),
    ):
        if name not in dataset.get_field_schema():
            kwargs = {"subfield": fo.DictField} if name == "duplicate_links" else {}
            dataset.add_sample_field(name, field_type, **kwargs)
    dataset.set_values("duplicate_links", links, key_field="id")
    dataset.set_values(
        f"{tag}_evidence", {key: assignments.get(key, {}) for key in samples}, key_field="id"
    )
    dataset.set_values(
        "duplicate_schema_version", {key: ALGORITHM_VERSION for key in samples}, key_field="id"
    )
    # Legacy convenience field; decisions read method-specific evidence instead.
    dataset.set_values(
        "duplicate_representative_id",
        {key: assignments.get(key, {}).get("representative_id", "") for key in samples},
        key_field="id",
    )
    rebuild_duplicate_families(dataset)
    if conflicts:
        dataset.select(sorted(conflicts)).tag_samples(conflict_tag)
    if assignments:
        dataset.select(sorted(assignments)).tag_samples(tag)
    return len(assignments), pairs


def detect_exact_duplicates(dataset: fo.Dataset, tag: str) -> tuple[int, list[tuple[str, str]]]:
    """Estrategia 1: Detección a nivel de bytes mediante hashing."""
    logger.info("Iniciando detección de duplicados exactos (Hashing)...")
    groups = {}
    hashes = {}
    for sample in dataset:
        if _flag(sample, "is_corrupted") or _flag(sample, "processing_error"):
            continue
        with open(sample.filepath, "rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        hashes[sample.id] = digest
        groups.setdefault(digest, []).append(sample.id)
    if "asset_sha256" not in dataset.get_field_schema():
        dataset.add_sample_field("asset_sha256", fo.StringField)
    dataset.set_values("asset_sha256", hashes, key_field="id")
    duplicates_dict = {ids[0]: ids[1:] for ids in groups.values() if len(ids) > 1}
    return process_duplicates(dataset, duplicates_dict, tag)


def _compute_similarity(dataset: fo.Dataset, brain_key: str) -> "fob.SimilarityIndex":
    """Wrapper centralizado para compute_similarity. Un único lugar para configurar modelo y parámetros."""
    _require_embedding_backend()
    # Nota: FiftyOne lanzará un warning de "Model does not support batching" porque
    # El modelo activo procesa imágenes con tamaños distintos (ragged_batches=True).
    # Esto es vital para no distorsionar las texturas de las hojas.
    return fob.compute_similarity(
        dataset,
        model=SIMILARITY_MODEL,
        brain_key=brain_key,
        metric="cosine",
        batch_size=16,
        num_workers=0,  # CRÍTICO: Previene el colapso de /dev/shm en Docker al usar el hilo principal
    )


def detect_semantic_duplicates(
    dataset: fo.Dataset, tag: str, threshold: float
) -> tuple[int, list[tuple[str, str]]]:
    """
    Estrategia 2: Detección a nivel visual mediante embeddings.
    Captura imágenes recortadas, redimensionadas o con distinta compresión.
    """
    logger.info("Iniciando detección semántica (modelo: %s)...", SIMILARITY_MODEL)
    brain_key = "semantic_similarity"
    eligible = [
        sample
        for sample in dataset
        if _curation_status(sample) != "removed"
        and not _flag(sample, "is_corrupted")
        and not _flag(sample, "processing_error")
    ]
    if len(eligible) < 2:
        return process_duplicates(dataset, {}, tag)
    signature = hashlib.sha256(
        json.dumps(
            {
                "model": SIMILARITY_MODEL,
                "version": ALGORITHM_VERSION,
                "assets": sorted(
                    (
                        sample.id,
                        _sample_value(sample, "asset_sha256") or "",
                        Path(sample.filepath).stat().st_size,
                        Path(sample.filepath).stat().st_mtime_ns,
                    )
                    for sample in eligible
                ),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    view = dataset.select([sample.id for sample in eligible])
    if (
        brain_key in dataset.list_brain_runs()
        and dataset.info.get("semantic_similarity_signature") != signature
    ):
        dataset.delete_brain_run(brain_key)

    # 1. Calcular o reutilizar el índice de similitud
    if brain_key in dataset.list_brain_runs():
        logger.info("Índice '%s' encontrado. Intentando cargarlo...", brain_key)
        results = dataset.load_brain_results(brain_key)
        if results is None:
            logger.warning("Índice corrupto o incompleto. Limpiando y recomputando...")
            dataset.delete_brain_run(brain_key)
            results = _compute_similarity(view, brain_key)
        else:
            logger.info("Índice cargado correctamente. Saltando extracción de embeddings.")
    else:
        logger.info("Extrayendo características visuales. Esto puede tardar (ejecución en CPU)...")
        results = _compute_similarity(view, brain_key)
    dataset.info["semantic_similarity_signature"] = signature
    dataset.save()

    # 2. Agrupar clústeres según el umbral de distancia (convertimos similitud a distancia coseno)
    dist_thresh = 1.0 - threshold
    logger.info(
        "Agrupando muestras con una similitud superior a %s (distancia <= %.3f)...",
        threshold,
        dist_thresh,
    )
    results.find_duplicates(thresh=dist_thresh)

    # Convertir neighbors_map {id: [(dup_id, dist), ...]} al formato
    # plano {id: [dup_id, ...]} que espera process_duplicates.
    duplicates_dict = {
        rep_id: [dup_id for dup_id, _dist in dup_list]
        for rep_id, dup_list in results.neighbors_map.items()
    }
    evidence = [
        {
            "left": left,
            "right": right,
            "level": "candidate",
            "distance": float(distance),
            "version": ALGORITHM_VERSION,
        }
        for left, neighbors in results.neighbors_map.items()
        for right, distance in neighbors
    ]
    return process_duplicates(dataset, duplicates_dict, tag, evidence=evidence)


def apply_stored_decisions(dataset, tag, policy, cache_dir):
    """Apply stored pairs after revalidating content and annotations; never delete raw/DB records."""
    from agrivision_khaos.pipeline import (
        Decision,
        apply_second_opinion,
        decision_for_tagged_duplicates,
        decisions_for_label_conflicts,
        reconcile_duplicate_representatives,
        write_decisions,
    )

    if f"{tag}_evidence" not in dataset.get_field_schema():
        raise ValueError(
            "No hay detección persistida para este método; ejecuta primero la detección"
        )
    cache = DescriptorCache(cache_dir)
    phase = (
        "augmentation_duplicates"
        if tag == "redundant_augmented"
        else f"{tag.removeprefix('redundant_')}_duplicates"
    )
    decisions = decision_for_tagged_duplicates(
        dataset,
        tag,
        phase,
        "Duplicado verificado",
        status="removed"
        if tag == "redundant_exact"
        or (
            tag == "redundant_augmented"
            and policy.remove_exact_transforms
            and policy.augmentation_action == "remove"
        )
        else "review",
    )
    # Validate everything before committing any curation decisions.
    sample_ids = list(decisions.keys())
    sample_map = {s.id: s for s in dataset.select(sample_ids)}
    rep_ids_needed = set()
    for key, decision in decisions.items():
        sample = sample_map.get(key)
        if sample is not None and decision.status == "removed":
            evidence = _sample_value(sample, f"{tag}_evidence") or {}
            rep_id = evidence.get("representative_id")
            if rep_id and rep_id not in sample_map:
                rep_ids_needed.add(rep_id)
    if rep_ids_needed:
        for s in dataset.select(list(rep_ids_needed)):
            sample_map[s.id] = s

    for key, decision in decisions.items():
        sample = sample_map[key]
        evidence = _sample_value(sample, f"{tag}_evidence") or {}
        if decision.status == "removed":
            representative = sample_map[evidence["representative_id"]]
            left = cache.describe(representative.filepath)
            right = cache.describe(sample.filepath)
            current = cache.verify(left, right, policy)
            if (
                current["level"] not in {"exact_bytes", "exact_pixels"}
                or not _compatible_annotations(representative, sample, current)
                or _curation_status(representative) != "kept"
            ):
                raise ValueError(
                    "El contenido, las anotaciones o el representante han cambiado; repite la detección"
                )
    decisions.update(decisions_for_label_conflicts(dataset, tag, "annotation_duplicates"))
    for sample in dataset.match_tags(f"{tag}_invalid"):
        decisions[sample.id] = Decision(
            "review", "augmentation_duplicates", "Descriptor no disponible o insuficiente", 0.0
        )
    apply_second_opinion(list(dataset), decisions, 0.40, 0.65)
    result = write_decisions(dataset, decisions, "stored_duplicates")
    reconcile_duplicate_representatives(dataset)
    return result


def review_pairs(dataset, instructions):
    """Human relation decisions are reversible and independent of sample acceptance."""
    from agrivision_khaos.pipeline import Decision, write_decisions

    if not isinstance(instructions, list):
        raise ValueError("La revisión debe ser una lista de parejas")
    samples = {sample.id: sample for sample in dataset}
    updates = {}
    for row in instructions:
        if not isinstance(row, dict):
            raise ValueError("Cada decisión de revisión debe ser un objeto")
        left, right, action = row.get("left"), row.get("right"), row.get("decision")
        if (
            left not in samples
            or right not in samples
            or left == right
            or action not in {"confirm", "reject"}
        ):
            raise ValueError("Pareja o decisión de revisión inválida")
        pair = frozenset((left, right))
        if pair in updates and updates[pair] != action:
            raise ValueError("Decisiones contradictorias para una pareja")
        updates[pair] = action
    if "duplicate_links" not in dataset.get_field_schema():
        dataset.add_sample_field("duplicate_links", fo.ListField, subfield=fo.DictField)
    if "duplicate_review_history" not in dataset.get_field_schema():
        dataset.add_sample_field("duplicate_review_history", fo.ListField, subfield=fo.DictField)
    links = {
        key: [
            dict(link)
            for link in (_sample_value(sample, "duplicate_links") or [])
            if not (
                link.get("method") == "human_review"
                and frozenset((key, link["other_id"])) in updates
            )
        ]
        for key, sample in samples.items()
    }
    decisions = {}
    review_history = {
        key: list(_sample_value(sample, "duplicate_review_history") or [])
        for key, sample in samples.items()
    }
    for pair, action in updates.items():
        left, right = sorted(pair)
        for key, other in ((left, right), (right, left)):
            previous = next(
                (
                    link
                    for link in (_sample_value(samples[key], "duplicate_links") or [])
                    if link.get("method") == "human_review" and link.get("other_id") == other
                ),
                None,
            )
            old_action = ("reject" if previous.get("rejected") else "confirm") if previous else None
            if old_action != action:
                review_history[key].append(
                    {
                        "at": datetime.now(UTC).isoformat(),
                        "other_id": other,
                        "before": old_action,
                        "after": action,
                    }
                )
            links[key].append(
                {
                    "left": left,
                    "right": right,
                    "other_id": other,
                    "method": "human_review",
                    "level": "human_confirmed",
                    "rejected": action == "reject",
                    "version": ALGORITHM_VERSION,
                }
            )
            curation = _sample_value(samples[key], "curation")
            if action == "reject" and getattr(curation, "representative_id", "") == other:
                decisions[key] = Decision(
                    "review", "human_review", "Relación de duplicidad rechazada", 1.0
                )
    dataset.set_values("duplicate_links", links, key_field="id")
    dataset.set_values("duplicate_review_history", review_history, key_field="id")
    for key in decisions:
        sample = samples[key]
        sample.tags = [tag for tag in sample.tags if not tag.startswith("redundant_")]
        sample.save()
    write_decisions(dataset, decisions, "human_review")
    rebuild_duplicate_families(dataset)


def detect_augmentation_duplicates(
    dataset: fo.Dataset,
    tag: str,
    threshold: float,
    policy: DeduplicationPolicy | None = None,
    cache_dir: Path | None = None,
):
    policy = policy or DeduplicationPolicy(augmentation_similarity=threshold)
    cache = DescriptorCache(cache_dir)
    samples = list(dataset)
    paths = {
        sample.id: sample.filepath
        for sample in samples
        if not _flag(sample, "is_corrupted") and not _flag(sample, "processing_error")
    }
    semantic_pairs = {
        (sample.id, link["other_id"])
        for sample in samples
        for link in (_sample_value(sample, "duplicate_links") or [])
        if link.get("method") == "redundant_semantic" and not link.get("rejected")
    }
    descriptors, evidence, stats = analyze_images(paths, policy, cache, semantic_pairs)
    if "asset_sha256" not in dataset.get_field_schema():
        dataset.add_sample_field("asset_sha256", fo.StringField)
    dataset.set_values(
        "asset_sha256", {key: item.asset for key, item in descriptors.items()}, key_field="id"
    )
    for name in ("augmentation_sharpness", "augmentation_padding"):
        if name not in dataset.get_field_schema():
            dataset.add_sample_field(name, fo.FloatField)
    for name, metric_name in (
        ("augmentation_sharpness", "sharpness"),
        ("augmentation_padding", "padding"),
    ):
        dataset.set_values(
            name,
            {key: getattr(descriptor, metric_name) for key, descriptor in descriptors.items()},
            key_field="id",
        )
    adjacency = {}
    for record in evidence:
        adjacency.setdefault(record["left"], []).append(record["right"])
    result = process_duplicates(
        dataset,
        adjacency,
        tag,
        evidence=evidence,
        verifier=lambda left, right: cache.verify(descriptors[left], descriptors[right], policy),
        pixel_families={key: min(item.pixels.values()) for key, item in descriptors.items()},
    )
    dataset.untag_samples(f"{tag}_invalid")
    unavailable = set(stats["invalid"]) | set(stats["low_information"])
    if unavailable:
        dataset.select(sorted(unavailable)).tag_samples(f"{tag}_invalid")
    stats["pair_cache_hits"] = cache.pair_hits
    dataset.info["augmentation_analysis"] = stats
    dataset.save()
    return result


def detect_mislabeled_samples(
    dataset: fo.Dataset, tag: str = "review", threshold: float = 0.7
) -> int:
    """
    Detecta posibles errores de etiquetado a partir de los embeddings activos.
    Aplica normalización L2 obligatoria antes de calcular centroides.
    """
    logger.info("Iniciando auditoría de etiquetado (Mislabeling Detection)...")
    _require_embedding_backend()

    try:
        embeddings = dataset.compute_embeddings(model=SIMILARITY_MODEL)
    except Exception as exc:
        logger.error("Error extrayendo embeddings para mislabeling: %s", exc)
        return 0

    # Normalización L2 requerida para evaluar similitud semántica sin sesgo de magnitud
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings_l2 = embeddings / norms

    labels = np.array(dataset.values("normalized_label"))
    sample_ids = np.array(dataset.values("id"))

    unique_labels = np.unique([label for label in labels if label])
    if len(unique_labels) < 2:
        logger.info("Se necesitan al menos 2 clases para comparar mislabeling.")
        return 0

    # Calcular centroides normalizados
    centroids = {}
    for lbl in unique_labels:
        mask = labels == lbl
        if np.sum(mask) == 0:
            continue
        class_embs = embeddings_l2[mask]
        centroid = np.mean(class_embs, axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 0:
            centroid /= c_norm
        centroids[lbl] = centroid

    suspicious_ids = []

    for i, lbl in enumerate(labels):
        if not lbl or lbl not in centroids:
            continue

        emb = embeddings_l2[i]
        own_sim = np.dot(emb, centroids[lbl])
        own_dist = 1.0 - own_sim

        min_other_dist = float("inf")
        for other_lbl, c_emb in centroids.items():
            if other_lbl == lbl:
                continue
            sim = np.dot(emb, c_emb)
            dist = 1.0 - sim
            if dist < min_other_dist:
                min_other_dist = dist

        # Heurística: Si está más lejos de su clase que de otra clase (con un margen)
        if own_dist > min_other_dist + 0.05:
            suspicious_ids.append(sample_ids[i])

    if suspicious_ids:
        dataset.select(suspicious_ids).tag_samples(tag)
        for s in dataset.select(suspicious_ids):
            if not s.has_field("curation"):
                s.curation = fo.DynamicEmbeddedDocument()
            s.curation.status = "review"
            s.curation.reason = "possible_mislabel"
            s.save()

        logger.warning("Detectados %d posibles errores de etiquetado.", len(suspicious_ids))
    else:
        logger.info("No se detectaron outliers de etiquetado.")

    return len(suspicious_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Detecta, revisa y aplica duplicados sin borrar originales."
    )
    parser.add_argument("--method", choices=["exact", "semantic", "augmented", "mislabel"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--dataset", default="agrivision-dataset")
    parser.add_argument("--policy", default="configs/quality-first.yaml")
    parser.add_argument(
        "--lock-dir",
        type=Path,
        default=Path("/datasets/cache/locks"),
        help="Directorio de locks compartido con el pipeline",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("/datasets/cache/augmentation"))
    parser.add_argument(
        "--apply",
        "--delete",
        dest="apply",
        action="store_true",
        help="Aplica decisiones lógicas sobre parejas persistidas; no borra registros ni archivos",
    )
    parser.add_argument(
        "--inspect", action="store_true", help="Inspecciona la detección persistida"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Revierte decisiones automáticas de duplicidad y conserva revisiones humanas",
    )
    parser.add_argument(
        "--review-file", type=Path, help="JSON de parejas con decision confirm/reject"
    )
    args = parser.parse_args()
    from agrivision_khaos.pipeline import (
        Decision,
        decision_for_tagged_duplicates,
        decisions_for_label_conflicts,
        load_policy,
        prepare_duplicate_run,
        slugify,
        write_decisions,
    )

    if args.reset and (
        args.method or args.review_file or args.inspect or args.apply or args.threshold is not None
    ):
        parser.error("--reset se usa solo, junto a --dataset y opciones de rutas")
    if not args.method and not args.review_file and not args.reset:
        parser.error("Indica --method o --review-file")
    if args.threshold is not None and (args.apply or args.inspect):
        parser.error(
            "--threshold solo modifica una detección nueva; la revisión usa evidencias persistidas"
        )
    if args.apply and args.inspect:
        parser.error("Usa --inspect y --apply por separado")
    try:
        policy = load_policy(args.policy).deduplication
        if args.threshold is not None:
            values = policy.model_dump()
            values[f"{'augmentation' if args.method == 'augmented' else 'semantic'}_similarity"] = (
                args.threshold
            )
            policy = DeduplicationPolicy.model_validate(values)
    except ValueError as exc:
        parser.error(str(exc))
    with PipelineLock(args.lock_dir / f"{slugify(args.dataset)}.lock"):
        dataset = load_dataset(args.dataset)
        if args.reset:
            prepare_duplicate_run(dataset)
            rebuild_duplicate_families(dataset)
            logger.info("Decisiones automáticas restablecidas; las revisiones humanas se conservan")
            return
        if args.review_file:
            try:
                review_pairs(dataset, json.loads(args.review_file.read_text()))
            except (OSError, ValueError) as exc:
                parser.error(str(exc))
            return
        tag = f"redundant_{args.method}"
        if args.apply:
            if args.method == "mislabel":
                parser.error("Los errores de etiqueta requieren revisión manual")
            try:
                result = apply_stored_decisions(dataset, tag, policy, args.cache_dir)
            except ValueError as exc:
                parser.error(str(exc))
            logger.info(
                "Decisiones aplicadas: %d descartadas, %d en revisión",
                result.removed,
                result.review,
            )
            return
        if args.method == "mislabel":
            detect_mislabeled_samples(dataset, threshold=args.threshold or 0.7)
            return
        if not args.inspect or f"{tag}_evidence" not in dataset.get_field_schema():
            prepare_duplicate_run(dataset, method=tag)
            if args.method == "exact":
                detect_exact_duplicates(dataset, tag)
            elif args.method == "semantic":
                detect_semantic_duplicates(dataset, tag, policy.semantic_similarity)
            else:
                detect_augmentation_duplicates(
                    dataset, tag, policy.augmentation_similarity, policy, args.cache_dir
                )
            # Detection marks candidates for review. Applying is a separate, explicit operation.
            phase = (
                "augmentation_duplicates"
                if args.method == "augmented"
                else f"{args.method}_duplicates"
            )
            decisions = decision_for_tagged_duplicates(
                dataset, tag, phase, "Duplicado verificado", status="review"
            )
            decisions.update(decisions_for_label_conflicts(dataset, tag, phase))
            for sample in dataset.match_tags(f"{tag}_invalid"):
                decisions[sample.id] = Decision(
                    "review", phase, "Descriptor no disponible o insuficiente", 0.0
                )
            write_decisions(dataset, decisions, phase)
    if args.inspect:
        ids = [
            sample.id
            for sample in dataset
            if any(
                link.get("method") == tag
                and not link.get("rejected")
                and link.get("level") != "rejected"
                for link in (_sample_value(sample, "duplicate_links") or [])
            )
            or f"{tag}_invalid" in sample.tags
        ]
        view = dataset.select(ids).sort_by("duplicate_cluster_id")
        session = fo.launch_app(view=view, address="0.0.0.0", port=5151)
        session.wait()
    logger.info("Detección persistida. Revisa las parejas y usa --apply o la exportación manual.")


if __name__ == "__main__":
    main()
