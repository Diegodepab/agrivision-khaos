"""Ontology management and hierarchical class mapping for agricultural vision datasets.

Supports both flat and hierarchical ontologies:
- Fine (granular) labels: e.g. 'repilo', 'antracnosis', 'tuberculosis', 'sano'.
- Coarse (derived) labels: e.g. 'enfermo' vs 'sano' (or 'diseased' vs 'healthy').

Enables seamless unification of granular disease datasets with binary (healthy/diseased) datasets.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml

HEALTHY_TOKENS: frozenset[str] = frozenset({
    "sano",
    "sana",
    "sanos",
    "sanas",
    "healthy",
    "health",
    "saglam",
    "normal",
    "clean",
    "control",
    "vigorous",
    "saludable",
    "sin_sintomas",
    "sin_danos",
    "intacto",
    "good",
})

DISEASE_TOKENS: frozenset[str] = frozenset({
    "repilo",
    "antracnosis",
    "anthracnose",
    "tuberculosis",
    "escudete",
    "blight",
    "rot",
    "spot",
    "rust",
    "curl",
    "mildew",
    "virus",
    "pest",
    "damage",
    "damaged",
    "disease",
    "diseased",
    "enfermo",
    "enferma",
    "enfermos",
    "enfermas",
    "defect",
    "defective",
    "podredumbre",
    "mancha",
    "scab",
    "canker",
    "necrosis",
    "wilt",
    "bacterial",
    "fungal",
    "mite",
    "aphid",
    "decay",
    "mold",
    "leaf_spot",
    "leaf_scorch",
    "hastalikli",
    "hastalıklı",
    "cycloconium",
    "virosis",
    "fumagina",
})


def normalize_label(label: str) -> str:
    """Normalizes label text to lowercase alphanumeric with underscores."""
    raw = str(label).strip()
    if not raw:
        return "unlabeled"
    normalized = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii").lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return normalized or "unlabeled"


def infer_coarse_label(fine_label: str, known_classes: list[str] | None = None) -> str:
    """Infers the coarse/derived category (e.g. 'sano' vs 'enfermo') from a fine label.

    If the fine label matches healthy terminology, returns 'sano'.
    If it matches disease terminology or if the dataset contains healthy classes, returns 'enfermo'.
    If the dataset does not have a healthy/diseased domain (e.g. fruit varieties), returns fine_label.
    """
    normalized = normalize_label(fine_label)
    tokens = set(re.split(r"[_\-\s]+", normalized.lower()))

    # Direct match or token intersection
    if tokens & HEALTHY_TOKENS or any(t in normalized for t in ("healthy", "sano", "saglam")):
        return "sano"

    if tokens & DISEASE_TOKENS or any(
        t in normalized for t in ("enferm", "diseas", "rot", "spot", "blight", "rust")
    ):
        return "enfermo"

    # Contextual inference: If known classes in this dataset include healthy classes,
    # then any non-healthy class in this domain is assumed to be diseased/abnormal.
    if known_classes:
        has_healthy_peers = any(
            bool(set(re.split(r"[_\-\s]+", normalize_label(c))) & HEALTHY_TOKENS)
            for c in known_classes
        )
        if has_healthy_peers:
            return "enfermo"

    # Default fallback: class retains its identity if neither health nor disease applies
    return normalized


def load_hierarchical_ontology(
    path: str | Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Loads an ontology file supporting both flat and hierarchical structures.

    Returns:
        tuple[fine_mapping, coarse_mapping]:
        - fine_mapping: maps source string -> canonical fine label (e.g. 'Olive Repilo' -> 'repilo')
        - coarse_mapping: maps canonical fine label -> coarse label (e.g. 'repilo' -> 'enfermo')
    """
    ontology_path = Path(path)
    if not ontology_path.is_file():
        raise ValueError(f"No existe el archivo de ontología: {ontology_path}")
    try:
        payload = yaml.safe_load(ontology_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Ontología YAML inválida {ontology_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("La ontología debe ser un mapa YAML con formato plano o jerárquico")

    fine_mapping: dict[str, str] = {}
    coarse_mapping: dict[str, str] = {}

    # Check if hierarchical structure is present
    is_hierarchical = (
        "canonical_mapping" in payload
        or "classes" in payload
        or "fine" in payload
        or "coarse_mapping" in payload
        or "hierarchy" in payload
    )

    classes_section: dict[str, Any]
    if is_hierarchical:
        classes_section = (
            payload.get("canonical_mapping")
            or payload.get("classes")
            or payload.get("fine")
            or {}
        )
        coarse_section = payload.get("coarse_mapping") or payload.get("hierarchy") or {}
        if not isinstance(classes_section, dict):
            raise ValueError("La sección de clases debe ser un diccionario de etiqueta: [sinónimos]")
        if not isinstance(coarse_section, dict):
            raise ValueError("La sección 'coarse_mapping' debe ser un diccionario de grupo: [clases]")
    else:
        classes_section = payload
        coarse_section = {}

    for target, originals in classes_section.items():
        canonical = normalize_label(str(target))
        if not canonical or not isinstance(originals, list) or not originals:
            raise ValueError(
                f"Entrada de ontología inválida para {target!r}: se esperaba una lista no vacía"
            )
        for original in originals:
            if not isinstance(original, str) or not original.strip():
                raise ValueError(f"Etiqueta de origen inválida bajo {target!r}: {original!r}")
            previous = fine_mapping.get(original)
            if previous is not None and previous != canonical:
                raise ValueError(
                    f"La etiqueta {original!r} se asigna a dos clases: {previous!r} y {canonical!r}"
                )
            fine_mapping[original] = canonical

    all_fine_classes = list(classes_section.keys())

    # Build coarse mapping
    if coarse_section:
        for group, members in coarse_section.items():
            if isinstance(members, list):
                canonical_group = normalize_label(str(group))
                for member in members:
                    norm_member = normalize_label(str(member))
                    coarse_mapping[norm_member] = canonical_group
            elif isinstance(members, str):
                norm_member = normalize_label(str(group))
                canonical_group = normalize_label(str(members))
                coarse_mapping[norm_member] = canonical_group
            else:
                raise ValueError(
                    f"Entrada de jerarquía inválida para {group!r}: se esperaba una lista de miembros o una cadena"
                )

    # Fill in any fine classes missing from explicit coarse mapping using domain heuristics
    for target in classes_section.keys():
        norm_target = normalize_label(str(target))
        if norm_target not in coarse_mapping:
            coarse_mapping[norm_target] = infer_coarse_label(norm_target, all_fine_classes)

    return fine_mapping, coarse_mapping


def load_ontology_mapping(path: str | Path) -> dict[str, str]:
    """Backward-compatible loader returning the fine label mapping."""
    fine_map, _ = load_hierarchical_ontology(path)
    return fine_map

