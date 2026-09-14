from __future__ import annotations

import argparse
import base64
import copy
import csv
import difflib
import hashlib
import html
import importlib.util
import json
import logging
import math
import os
import re
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import repeat
from pathlib import Path
from typing import Any

import cv2
import fiftyone as fo
import numpy as np
import yaml
from pydantic import ValidationError
from rich.logging import RichHandler

from agrivision_khaos.augmentation import (
    ALGORITHM_VERSION,
    CONFIRMED_LEVELS,
    EXACT_LEVELS,
    DescriptorCache,
    candidate_pairs,
    transforms,
)
from agrivision_khaos.curation_history import ensure_history, record_transition, snapshot
from agrivision_khaos.dry_run import audit_raw_datasets
from agrivision_khaos.execution import (
    PipelineLock,
    RunCheckpoint,
    default_workers,
    source_fingerprint,
)
from agrivision_khaos.export_assets import (
    EXPORT_ALGORITHM_VERSION,
    copy_verified_asset,
    export_filename,
    link_export_asset,
)
from agrivision_khaos.families import (
    audit_assignments,
    human_rejections,
    relation_groups,
    stable_identity,
)
from agrivision_khaos.models import (
    CurationPolicy,
    DeduplicationPolicy,
    QualityPolicy,
    SourceManifest,
)
from agrivision_khaos.preflight import run_preflight
from agrivision_khaos.split_audit import audit_visual_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger(__name__)

SPLIT_NAMES = {"train", "test", "valid", "val", "validation"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SOURCE_MANIFEST_NAMES = ("source.yaml", "source.yml", "source.json")
HARD_CONFIDENCE = 0.98
CANONICAL_SCHEMA_VERSION = "1.0"
SOURCE_DETECTION_FIELDS = ("coco_detections", "yolo_detections", "voc_detections")
SUPPORTED_OUTPUT_FORMATS = {"classification", "coco", "yolo", "datumaro", "fiftyone"}

SAFE_LABEL_ALIASES = {
    "healthy": "healthy",
    "saglam": "healthy",
    "sağlam": "healthy",
    "hastalıklı": "diseased",
    "hastalikli": "diseased",
    "olive peacock spot": "olive_peacock_spot",
    "olive_peacock_spot": "olive_peacock_spot",
    "peacock spot": "olive_peacock_spot",
    "aculus olearius": "aculus_olearius",
    "aculus_olearius": "aculus_olearius",
    "knot disease": "knot_disease",
}


@dataclass
class Decision:
    status: str
    phase: str
    reason: str
    confidence: float
    keep_reason: str = ""
    cluster_id: str = ""
    representative_id: str = ""
    evidence_level: str = ""
    review_reasons: list[str] = field(default_factory=list)


@dataclass
class PhaseResult:
    name: str
    removed: int = 0
    review: int = 0
    kept: int = 0
    notes: list[str] = field(default_factory=list)
    duplicate_pairs: list[tuple[str, str]] = field(default_factory=list)


def optional_tool_status() -> dict[str, dict[str, str]]:
    statuses = {}
    for package in ("datumaro", "cleanlab"):
        statuses[package] = {
            "available": bool(importlib.util.find_spec(package)),
            "status": "available" if importlib.util.find_spec(package) else "not_installed",
        }
    return statuses


def slugify(value: str, default: str = "unknown") -> str:
    normalized = normalize_text(value)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return normalized or default


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(value))
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", ascii_text.replace("_", " ").replace("-", " ")).strip().lower()


def normalize_label(label: str | None) -> str:
    normalized = normalize_text(label)
    if not normalized:
        return ""
    return SAFE_LABEL_ALIASES.get(normalized, normalized.replace(" ", "_"))


def discover_source_format(path: Path) -> str:
    lower_names = {p.name.lower() for p in path.rglob("*") if p.is_file()}
    if any(name.endswith(".json") and ("coco" in name or "annotation" in name) for name in lower_names):
        return "coco"
    if "data.yaml" in lower_names or "dataset.yaml" in lower_names:
        return "yolo"
    if any(name.endswith(".xml") for name in lower_names):
        return "voc"
    return "classification_tree"


def discover_sources(raw_dir: Path) -> list[dict[str, Any]]:
    if not raw_dir.exists():
        return []
    sources = []
    for child in sorted(raw_dir.iterdir()):
        if not child.is_dir():
            continue
        metadata_path = next((child / name for name in SOURCE_MANIFEST_NAMES if (child / name).is_file()), None)
        metadata = SourceManifest(name=child.name)
        if metadata_path is not None:
            try:
                raw_metadata = (
                    json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata_path.suffix == ".json"
                    else yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
                )
                metadata = SourceManifest.model_validate(raw_metadata or {})
            except (OSError, ValueError, ValidationError) as exc:
                raise ValueError(f"Manifiesto de fuente inválido {metadata_path}: {exc}") from exc
        sources.append({
            "path": str(child),
            "format": discover_source_format(child),
            "manifest_path": str(metadata_path) if metadata_path else "",
            **metadata.model_dump(mode="json"),
            "display_name": metadata.name or child.name,
            "name": child.name,
        })
    return sources


def ensure_pipeline_schema(dataset: fo.Dataset) -> None:
    schema = dataset.get_field_schema(flat=True)
    for field_name in (
        "source_dataset",
        "source_split",
        "source_format",
        "source_path",
        "source_label",
        "normalized_label",
        "task_type",
        "schema_version",
        "source_version",
        "source_license",
        "source_url",
        "source_citation",
        "source_sensor",
        "source_geography",
    ):
        if field_name not in schema:
            dataset.add_sample_field(field_name, fo.StringField)

    for field_name in ("source_labels", "normalized_labels"):
        if field_name not in schema:
            dataset.add_sample_field(field_name, fo.ListField, subfield=fo.StringField)

    if "annotation_issues" not in schema:
        dataset.add_sample_field("annotation_issues", fo.ListField, subfield=fo.StringField)
    if "annotation_valid" not in schema:
        dataset.add_sample_field("annotation_valid", fo.BooleanField)

    if "curation" not in schema:
        dataset.add_sample_field(
            "curation",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.DynamicEmbeddedDocument,
        )

    if "final_label" not in schema:
        dataset.add_sample_field("final_label", fo.EmbeddedDocumentField, embedded_doc_type=fo.Classification)

    if "ground_truth_classification" not in schema:
        dataset.add_sample_field(
            "ground_truth_classification",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Classification,
        )

    if "ground_truth_detections" not in schema:
        dataset.add_sample_field(
            "ground_truth_detections",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )


def canonicalize_annotations(sample: fo.Sample) -> dict[str, object]:
    """Copies source annotations into stable task-specific canonical fields."""
    task_types: list[str] = []
    source_labels: set[str] = set()
    normalized_labels: set[str] = set()
    annotation_issues: list[str] = []

    classification = sample_field(sample, "original_label")
    source_class = str(getattr(classification, "label", "") or "")
    if source_class:
        normalized_class = normalize_label(source_class)
        canonical_class = fo.Classification(
            label=normalized_class,
            source_label=source_class,
        )
        sample["ground_truth_classification"] = canonical_class
        sample["final_label"] = copy.deepcopy(canonical_class)
        sample["source_label"] = source_class
        sample["normalized_label"] = normalized_class
        task_types.append("classification")
        source_labels.add(source_class)
        normalized_labels.add(normalized_class)

    canonical_detections: list[fo.Detection] = []
    for field_name in SOURCE_DETECTION_FIELDS:
        detections = sample_field(sample, field_name)
        for detection in getattr(detections, "detections", None) or []:
            canonical = copy.deepcopy(detection)
            source_label = str(getattr(canonical, "label", "") or "")
            if not source_label:
                continue
            canonical["source_label"] = source_label
            canonical.label = normalize_label(source_label)
            canonical["source_field"] = field_name
            bounding_box = list(getattr(canonical, "bounding_box", None) or [])
            valid_box = (
                len(bounding_box) == 4
                and all(isinstance(value, (int, float)) and math.isfinite(value) for value in bounding_box)
                and bounding_box[0] >= 0
                and bounding_box[1] >= 0
                and bounding_box[2] > 0
                and bounding_box[3] > 0
                and bounding_box[0] + bounding_box[2] <= 1.0 + 1e-9
                and bounding_box[1] + bounding_box[3] <= 1.0 + 1e-9
            )
            if not valid_box:
                issue = f"invalid_bbox:{field_name}:{len(canonical_detections)}"
                canonical["validation_error"] = issue
                annotation_issues.append(issue)
            canonical_detections.append(canonical)
            source_labels.add(source_label)
            normalized_labels.add(canonical.label)

    if canonical_detections:
        sample["ground_truth_detections"] = fo.Detections(detections=canonical_detections)
        task_types.append("detection")

    sample["task_type"] = ",".join(task_types) if task_types else "unlabeled"
    sample["schema_version"] = CANONICAL_SCHEMA_VERSION
    sample["source_labels"] = sorted(source_labels)
    sample["normalized_labels"] = sorted(normalized_labels)
    sample["annotation_issues"] = annotation_issues
    sample["annotation_valid"] = not annotation_issues
    return {
        "task_type": sample["task_type"],
        "source_labels": sample["source_labels"],
        "normalized_labels": sample["normalized_labels"],
        "annotation_issues": annotation_issues,
    }


def infer_split(parts: tuple[str, ...]) -> str:
    for part in parts:
        normalized = normalize_text(part)
        if normalized in SPLIT_NAMES:
            return "valid" if normalized == "validation" else normalized
    return ""


def infer_label_from_path(parts: tuple[str, ...]) -> str:
    for index, part in enumerate(parts):
        if normalize_text(part) in SPLIT_NAMES and index + 1 < len(parts) - 1:
            return parts[index + 1]
    if len(parts) >= 3:
        return parts[-2]
    return ""


def extract_sample_label(sample: fo.Sample, raw_dir: Path) -> str:
    original_label = sample_field(sample, "original_label")
    if original_label is not None and getattr(original_label, "label", None):
        label = str(original_label.label)
        if label.lower() not in {"dataset", "data", "images", "img", "train", "test", "val", "valid"}:
            return label

    try:
        parts = Path(sample.filepath).resolve().relative_to(raw_dir.resolve()).parts
    except ValueError:
        parts = Path(sample.filepath).parts
    return infer_label_from_path(parts)


def infer_source_metadata(
    sample: fo.Sample,
    raw_dir: Path,
    source_details: dict[str, dict[str, Any]],
) -> dict[str, str]:
    try:
        relative = Path(sample.filepath).resolve().relative_to(raw_dir.resolve())
        parts = relative.parts
    except ValueError:
        relative = Path(sample.filepath)
        parts = relative.parts

    source_dataset = parts[0] if parts else "unknown"
    details = source_details.get(source_dataset, {})
    source_format = str(details.get("format", "unknown"))
    source_label = extract_sample_label(sample, raw_dir) if source_format == "classification_tree" else ""
    return {
        "source_dataset": source_dataset,
        "source_split": infer_split(parts),
        "source_format": source_format,
        "source_path": str(relative),
        "source_label": source_label,
        "normalized_label": normalize_label(source_label),
        "source_version": str(details.get("version", "unknown")),
        "source_license": str(details.get("license", "unknown")),
        "source_url": str(details.get("homepage", "") or ""),
        "source_citation": str(details.get("citation", "") or ""),
        "source_sensor": str(details.get("sensor", "RGB")),
        "source_geography": str(details.get("geography", "") or ""),
    }


def annotate_sources(dataset: fo.Dataset, raw_dir: Path, sources: list[dict[str, Any]]) -> None:
    logger.info("Registrando procedencia y etiquetas normalizadas...")
    ensure_pipeline_schema(dataset)
    source_details = {str(source["name"]): source for source in sources}
    initial_decision = fo.DynamicEmbeddedDocument(
        status="kept",
        phase="ingestion",
        reason="accepted_on_ingest",
        confidence=1.0,
        keep_reason="initial_candidate",
        cluster_id="",
    )

    for sample in dataset.iter_samples(progress=True, autosave=True):
        metadata = infer_source_metadata(sample, raw_dir, source_details)
        for field_name, value in metadata.items():
            sample[field_name] = value
        canonicalize_annotations(sample)
        sample["curation"] = initial_decision


def evaluate_quality(sample: fo.Sample, policy: QualityPolicy | None = None) -> Decision:
    policy = policy or QualityPolicy()
    if flag(sample, "processing_error"):
        return Decision("review", "quality", "Error de Procesamiento", 1.0)
    if sample_field(sample, "annotation_valid", True) is False:
        return Decision("review", "annotations", "Anotación Geométrica Inválida", 1.0)
    if flag(sample, "is_corrupted"):
        return Decision("removed", "quality", "Corrupta", 1.0)
    if flag(sample, "has_watermark"):
        return Decision("removed", "quality", "Marca de Agua", 0.99)
    if flag(sample, "low_resolution"):
        return Decision("removed", "quality", "Baja Resolución", 0.98)
    if flag(sample, "has_smearing"):
        return Decision("removed", "quality", "Bordes Estirados (Smearing)", 0.92)

    blur = metric(sample, "blur_variance")
    if blur is not None and blur < policy.severe_blur:
        return Decision("removed", "quality", "Desenfoque Severo", 0.94)
    if blur is not None and blur < policy.review_blur:
        return Decision("review", "quality", "Desenfoque Leve", 0.72)

    brightness = metric(sample, "brightness_mean")
    p5 = metric(sample, "brightness_p5")
    p95 = metric(sample, "brightness_p95")
    if brightness is not None and (
        brightness < policy.min_brightness or brightness > policy.max_brightness
    ):
        return Decision("removed", "quality", "Brillo Extremo", 0.90)
    if (p95 is not None and p95 < policy.dark_p95) or (
        p5 is not None and p5 > policy.bright_p5
    ):
        return Decision("review", "quality", "Brillo Anómalo", 0.70)

    return Decision("kept", "quality", "quality_passed", 1.0, keep_reason="quality_passed")


def render_example_card(example: dict[str, object], caption: str | None = None) -> str:
    import os
    b64 = get_base64_image(str(example["filepath"]))
    if not b64:
        return ""
    
    source = str(example.get('source_dataset', ''))
    if len(source) > 15:
        source = source[:15] + "..."
        
    filename = os.path.basename(str(example["filepath"]))
    if len(filename) > 15:
        filename = filename[:15] + "..."
        
    label = str(example.get('label', ''))
    base_file = os.path.basename(str(example.get('filepath', '')))
    
    subtitle = f"<strong>{html.escape(source)}</strong> | {html.escape(label)}<br><small title='{html.escape(base_file)}'>File: {html.escape(filename)}</small>"
    
    details = [
        f"fase {format_report_value(example.get('phase'))}",
        f"conf {format_report_value(example.get('confidence'))}",
    ]
    metrics = (
        f"blur {format_report_value(example.get('blur_variance'))}",
        f"res {format_report_value(example.get('width'))}x{format_report_value(example.get('height'))}",
        f"low_res {format_report_value(example.get('low_resolution'))}",
        f"wm {format_report_value(example.get('has_watermark'))}",
        f"smear {format_report_value(example.get('has_smearing'))}",
    )
    metric_badges = "".join(f'<span class="badge">{html.escape(m)}</span> ' for m in metrics if m and not m.endswith('None'))
    
    return (
        '<article class="example">'
        f'<img src="data:image/jpeg;base64,{b64}" alt="">'
        f"<span>{subtitle}</span>"
        f"<small>{html.escape(' | '.join(details))}</small>"
        f"<div style='margin-top: 6px;'>{metric_badges}</div>"
        "</article>"
    )


def render_reason_sections(title: str, sections: list[dict[str, object]], empty_message: str) -> str:
    if not sections:
        return f"<h2>{html.escape(title)}</h2><p>{html.escape(empty_message)}</p>"
    cards = []
    for section in sections:
        reason = str(section.get("reason", "unknown"))
        examples = section.get("examples", [])
        rendered_examples = "".join(render_example_card(example, caption=reason) for example in examples)
        cards.append(
            "<details open class=\"reason-group\">"
            f"<summary><strong>{html.escape(reason)}</strong> · {section.get('count', 0)} muestras</summary>"
            f"<div class=\"grid\">{rendered_examples}</div>"
            "</details>"
        )
    return f"<h2>{html.escape(title)}</h2>{''.join(cards)}"


def render_duplicate_sections(title: str, sections: list[dict[str, object]]) -> str:
    if not sections:
        return f"<h2>{html.escape(title)}</h2><p>No hay pares de ejemplo.</p>"
    blocks = []
    for section in sections:
        pair_cards = []
        for pair in section.get("pairs", []):
            kept = pair.get("kept", {})
            removed = pair.get("removed", {})
            evidence = pair.get("evidence", {})
            kept_b64 = get_base64_image(str(kept.get("filepath", "")))
            removed_b64 = get_base64_image(str(removed.get("filepath", "")),
                                          transform=str(evidence.get("transform", "r0")),
                                          affine=evidence.get("affine"),
                                          reference=str(kept.get("filepath", "")))
            if not kept_b64 or not removed_b64:
                continue
            import os
            kept_filename = os.path.basename(str(kept.get("filepath", "")))
            removed_filename = os.path.basename(str(removed.get("filepath", "")))
            
            kept_metrics = f"<span class='badge'>blur {format_report_value(kept.get('blur_variance'))}</span> <span class='badge'>res {format_report_value(kept.get('width'))}x{format_report_value(kept.get('height'))}</span>"
            status_names = {"kept": "Conservada", "removed": "Descartada", "review": "En revisión"}
            kept_state = status_names.get(str(kept.get("status", "")), "Representante")
            removed_state = status_names.get(str(removed.get("status", "")), "Candidata")
            full_candidate = ""
            if evidence.get("affine"):
                raw_b64 = get_base64_image(str(removed.get("filepath", "")))
                full_candidate = (
                    '<details><summary>Ver candidata completa sin alinear</summary>'
                    f'<img src="data:image/jpeg;base64,{raw_b64}" alt="Candidata completa" style="width:100%;object-fit:contain"></details>'
                )
            comparison_note = html.escape(
                f"Alineación: {'afín estimada' if evidence.get('affine') else evidence.get('transform', 'sin verificar')} · "
                f"Evidencia: {evidence.get('level', 'candidate')} · "
                f"Cobertura: {format_report_value(evidence.get('coverage'))}"
            )
            removed_metrics = f"<span class='badge'>blur {format_report_value(removed.get('blur_variance'))}</span> <span class='badge'>res {format_report_value(removed.get('width'))}x{format_report_value(removed.get('height'))}</span>"
            pair_cards.append(
                '<div class="example-pair" style="display:flex;gap:12px;border:1px solid #e2e8f0;border-radius:12px;padding:12px;background:#ffffff;box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);">'
                f'<div style="flex:1;overflow:hidden;"><img src="data:image/jpeg;base64,{kept_b64}" alt="" style="width:100%;aspect-ratio:1/1;object-fit:contain;border-radius:8px;">'
                f'<span class="status-kept" style="display:block;margin-top:8px;font-size:14px;">{kept_state}</span>'
                f'<small style="display:block;color:#64748b;margin-top:2px;">{html.escape(str(kept.get("source_dataset", "")))} | {html.escape(str(kept.get("label", "")))}</small>'
                f'<code style="display:block;margin-top:6px;font-size:11px;word-break:break-all;line-height:1.3;background:#f1f5f9;color:#334155;border:1px solid #e2e8f0;padding:4px 6px;border-radius:6px;" title="{html.escape(kept_filename)}">{html.escape(kept_filename)}</code>'
                f'<div style="margin-top:6px;">{kept_metrics}</div></div>'
                f'<div style="flex:1;overflow:hidden;"><img src="data:image/jpeg;base64,{removed_b64}" alt="" style="width:100%;aspect-ratio:1/1;object-fit:contain;border-radius:8px;">'
                f'<span class="status-removed" style="display:block;margin-top:8px;font-size:14px;">{removed_state}</span>'
                f'<small style="display:block;color:#64748b;margin-top:2px;">{html.escape(str(removed.get("source_dataset", "")))} | {html.escape(str(removed.get("label", "")))}</small>'
                f'<code style="display:block;margin-top:6px;font-size:11px;word-break:break-all;line-height:1.3;background:#f1f5f9;color:#334155;border:1px solid #e2e8f0;padding:4px 6px;border-radius:6px;" title="{html.escape(removed_filename)}">{html.escape(removed_filename)}</code>'
                f'<div style="margin-top:6px;">{removed_metrics}</div><small>{comparison_note}</small>{full_candidate}</div>'
                "</div>"
            )
        blocks.append(
            "<details open class=\"reason-group\">"
            f"<summary><strong>{html.escape(str(section.get('phase', 'duplicates')))}</strong> · {section.get('removed', 0)} descartadas, {section.get('review', 0)} en revisión</summary>"
            f"<div class=\"grid-pairs\" style=\"display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px;\">{''.join(pair_cards)}</div>"
            "</details>"
        )
    return f"<h2>{html.escape(title)}</h2>{''.join(blocks)}"


def render_count_table(title: str, rows: dict[str, dict[str, int]], show_drop_rates: bool = False) -> str:
    body = []
    for name, counts in rows.items():
        total = sum(counts.values())
        kept = counts.get('kept', 0)
        review = counts.get('review', 0)
        removed_q = counts.get('removed_quality', counts.get('removed', 0))
        removed_d = counts.get('removed_duplicates', 0)
        total_removed = removed_q + removed_d
        
        row_html = (
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td class='number-cell'>{total}</td>"
            f"<td class='number-cell'>{kept}</td>"
            f"<td class='number-cell'>{review}</td>"
            f"<td class='number-cell removed-col'>{total_removed}</td>"
        )
        
        if show_drop_rates:
            pct_q = (removed_q / total * 100) if total > 0 else 0.0
            pct_d = (removed_d / total * 100) if total > 0 else 0.0
            q_color = "#991b1b" if pct_q > 30 else "#475569"
            q_weight = "bold" if pct_q > 30 else "normal"
            
            row_html += (
                f"<td class='number-cell' style='color:{q_color}; font-weight:{q_weight};'>{pct_q:.1f}%</td>"
                f"<td class='number-cell' style='color:#64748b;'>{pct_d:.1f}%</td>"
            )
        
        row_html += "</tr>"
        body.append(row_html)
        
    header = (
        "<table><thead><tr><th>Grupo</th><th class='number-cell'>Total</th><th class='number-cell'>Kept</th>"
        "<th class='number-cell'>Review</th><th class='number-cell removed-col'>Removed</th>"
    )
    if show_drop_rates:
        header += "<th class='number-cell'>% Descarte (Calidad)</th><th class='number-cell'>% Descarte (Duplicados)</th>"
    header += "</tr></thead><tbody>"
    
    return (
        f"<h2>{html.escape(title)}</h2>"
        + header
        + "".join(body)
        + "</tbody></table>"
    )

def render_contamination_matrix(cross_contamination: list[dict[str, object]]) -> str:
    if not cross_contamination:
        return ""
        
    items = []
    for entry in cross_contamination:
        ds = entry.get("datasets", [])
        if len(ds) == 2:
            items.append(
                f"<li style='margin-bottom: 8px;'>"
                f"<strong>{html.escape(ds[0])}</strong> &harr; <strong>{html.escape(ds[1])}</strong> : "
                f"<span style='color: #ef4444; font-weight: bold;'>{entry.get('count')}</span> imágenes idénticas compartidas"
                f"</li>"
            )
            
    if not items:
        return ""
        
    return (
        "<div style='max-width: 1280px; margin: 0 auto 32px; background: #ffffff; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);'>"
        "<h3 style='margin-top: 0; color: #0f172a;'>Fuga de Datos / Contaminación Cruzada</h3>"
        "<p style='color: #64748b; font-size: 0.9em; margin-bottom: 16px;'>"
        "Esta lista muestra pares de datasets que contienen imágenes idénticas. "
        "<strong>Atención:</strong> Si vas a particionar la red neuronal por dataset, evita cruzar estos datasets entre *train* y *test* para no inflar las métricas de evaluación."
        "</p>"
        "<ul style='list-style-type: none; padding-left: 0; margin: 0; font-size: 0.95em; color: #334155;'>"
        + "".join(items) +
        "</ul>"
        "</div>"
    )


def render_ontology_harmonization_section(label_mapping: dict[str, Any]) -> str:
    if not label_mapping:
        return ""

    typos = label_mapping.get("typos_and_similar", [])
    matrix = label_mapping.get("dataset_matrix", [])
    yaml_text = label_mapping.get("proposed_yaml", "")
    applied = label_mapping.get("applied", {})
    
    total_raw = sum(len(m.get("original_labels", [])) for m in matrix) if matrix else len(applied)
    total_canonical = len(matrix) if matrix else len(set(applied.values()))
    cross_dataset_typos = [t for t in typos if t.get("cross_dataset") or t.get("recommended_merge")]

    alerts_html = ""
    if cross_dataset_typos:
        rows = []
        for t in cross_dataset_typos[:25]:
            left_ds = ", ".join(t.get("left_datasets", [])) or "desconocido"
            right_ds = ", ".join(t.get("right_datasets", [])) or "desconocido"
            sim = int(t.get("similarity", 0) * 100)
            badge_color = "#dc2626" if sim >= 85 else "#d97706"
            badge_bg = "#fef2f2" if sim >= 85 else "#fffbeb"
            canon = t.get("suggested_canonical") or "-"
            
            rows.append(
                "<tr>"
                f"<td><strong>{html.escape(str(t['left']))}</strong><br><small style='color:#64748b;'>{html.escape(str(left_ds))} ({t.get('left_count', 0)} imgs)</small></td>"
                f"<td><strong>{html.escape(str(t['right']))}</strong><br><small style='color:#64748b;'>{html.escape(str(right_ds))} ({t.get('right_count', 0)} imgs)</small></td>"
                f"<td><span class='badge' style='background:{badge_bg};color:{badge_color};border-color:{badge_color};font-weight:700;'>{sim}% Similitud</span><br><small style='color:#475569;'>{html.escape(str(t.get('diagnosis', '')))}</small></td>"
                f"<td><code style='font-weight:700;color:#0369a1;background:#e0f2fe;'>{html.escape(str(canon))}</code></td>"
                "</tr>"
            )
        alerts_html = (
            "<div style='margin-top: 24px;'>"
            "<h4 style='margin: 0 0 10px; color: #0f172a; display: flex; align-items: center; gap: 8px;'>"
            "Posible unificación de etiquetas"
            "</h4>"
            "<p style='color: #64748b; font-size: 0.9em; margin-bottom: 12px;'>"
            "Los datasets combinados utilizan nombres ligeramente distintos para la misma afección o fruto. "
            "Revisa las sugerencias de unificación a continuación:</p>"
            "<table><thead><tr>"
            "<th>Etiqueta A (Dataset)</th><th>Etiqueta B (Dataset)</th><th>Diagnóstico</th><th>Fusión Recomendada</th>"
            "</tr></thead><tbody>"
            + "".join(rows) +
            "</tbody></table></div>"
        )

    matrix_html = ""
    if matrix:
        matrix_rows = []
        for m in matrix:
            ds_badges = " ".join(
                f"<span class='badge' style='background:#f8fafc;border:1px solid #cbd5e1;color:#334155;'><strong>{html.escape(str(ds))}:</strong> {cnt}</span>"
                for ds, cnt in m.get("datasets", {}).items()
            )
            orig_labels = ", ".join(
                f"<code>{html.escape(str(label))}</code>"
                for label in m.get("original_labels", [])
            )
            matrix_rows.append(
                "<tr>"
                f"<td><strong style='color:#0f172a;'>{html.escape(str(m['canonical']))}</strong></td>"
                f"<td class='number-cell' style='font-weight:700;'>{m['total']:,}</td>"
                f"<td>{ds_badges}</td>"
                f"<td>{orig_labels}</td>"
                "</tr>"
            )

        matrix_html = (
            "<div style='margin-top: 28px;'>"
            "<h4 style='margin: 0 0 10px; color: #0f172a;'>Distribución de Clases Canónicas por Dataset de Origen</h4>"
            "<p style='color: #64748b; font-size: 0.9em; margin-bottom: 12px;'>"
            "Muestra qué datasets crudos aportan imágenes a cada categoría unificada:</p>"
            "<table><thead><tr>"
            "<th>Clase Canónica</th><th class='number-cell'>Total Imágenes</th><th>Datasets de Origen (Muestras)</th><th>Etiquetas Crudas Incluidas</th>"
            "</tr></thead><tbody>"
            + "".join(matrix_rows) +
            "</tbody></table></div>"
        )

    yaml_html = ""
    if yaml_text:
        yaml_html = (
            "<details style='margin-top: 24px; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; background: #ffffff;'>"
            "<summary style='cursor: pointer; font-weight: 600; color: #0369a1;'>Ver / Copiar archivo de ontología sugerido (<code>proposed_ontology.yaml</code>)</summary>"
            "<p style='color: #64748b; font-size: 0.9em; margin: 12px 0 8px;'>"
            "Puedes guardar este contenido en un archivo (ej. <code>configs/ontology.yaml</code>) y pasarlo al pipeline con <code>ONTOLOGY=configs/ontology.yaml</code> para forzar la estandarización exacta de clases."
            "</p>"
            f"<pre style='background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; overflow-x: auto; font-size: 0.85em; color: #1e293b; max-height: 400px;'><code>{html.escape(yaml_text)}</code></pre>"
            "</details>"
        )

    return (
        "<div style='max-width: 1280px; margin: 0 auto 32px; background: #ffffff; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.05);'>"
        "<h3 style='margin-top: 0; color: #0f172a; display: flex; align-items: center; gap: 8px;'>"
        "Armonización de Etiquetas y Ontología Multi-Dataset"
        "</h3>"
        "<p style='color: #64748b; font-size: 0.95em; margin-bottom: 20px; line-height: 1.5;'>"
        "Al integrar múltiples datasets agronómicos (YOLO, COCO, Classification), cada autor introduce sus propias convenciones o erratas tipográficas (como <em>Anthracnose</em> vs <em>Athracnose</em>). "
        "AgriVision Khaos analiza la coherencia inter-dataset para unificar clases y proponer una taxonomía limpia."
        "</p>"
        "<div class='stats' style='margin-top: 0; margin-bottom: 20px;'>"
        f"<div class='stat'><strong>{total_raw}</strong><span>Etiquetas Originales</span></div>"
        f"<div class='stat'><strong style='color: #0369a1;'>{total_canonical}</strong><span>Clases Canónicas</span></div>"
        f"<div class='stat'><strong style='color: #d97706;'>{len(cross_dataset_typos)}</strong><span>Alertas Inter-Dataset</span></div>"
        f"<div class='stat'><strong style='color: #10b981;'>{len(label_mapping.get('automatic', {}))}</strong><span>Agrupadas Automát.</span></div>"
        "</div>"
        + alerts_html
        + matrix_html
        + yaml_html +
        "</div>"
    )


def write_reports(
    report_dir: Path,
    dataset_name: str,
    run_id: str,
    summary: dict[str, Any],
    evidence: dict[str, object],
) -> None:
    write_json(report_dir / "summary.json", summary)
    write_dataset_card(report_dir / "DATASET_CARD.md", summary)
    copy_examples_by_reason(list(evidence.get("removed_by_reason", [])), report_dir / "discarded_examples")
    copy_examples_by_reason(list(evidence.get("review_by_reason", [])), report_dir / "review_examples")
    copy_duplicate_pair_gallery(list(evidence.get("duplicate_phases", [])), report_dir / "duplicate_examples")

    html_doc = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Pipeline report - {html.escape(dataset_name)}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    body {{ margin: 0; font-family: 'Inter', system-ui, sans-serif; color: #1e293b; background: #f8fafc; }}
    main {{ max-width: 1280px; margin: 40px auto; padding: 0 24px; }}
    .hero {{ background: #ffffff; color: #0f172a; padding: 36px; border-radius: 16px; margin-bottom: 32px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05); }}
    .hero h1 {{ margin: 0 0 8px; font-size: 2.2em; letter-spacing: -0.02em; color: #0f172a; }}
    .hero p {{ margin: 0; color: #64748b; font-size: 1.05em; }}
    h2 {{ margin-top: 40px; padding-bottom: 12px; border-bottom: 1px solid #e2e8f0; color: #0f172a; font-weight: 600; letter-spacing: -0.01em; }}
    details.reason-group {{ margin: 16px 0; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; background: #ffffff; box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.05); transition: all 0.2s; }}
    details.reason-group:hover {{ box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.08); }}
    details.reason-group > summary {{ cursor: pointer; list-style: none; font-weight: 600; color: #334155; }}
    
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin-top: 24px; }}
    .stat {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; }}
    .stat strong {{ display: block; font-size: 30px; font-weight: 700; color: #0f172a; }}
    .stat span {{ color: #64748b; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; display: block; font-weight: 600; }}
    
    .progress-bar {{ display: flex; height: 10px; border-radius: 6px; overflow: hidden; margin-top: 24px; background: #e2e8f0; }}
    .progress-kept {{ background: #10b981; }}
    .progress-review {{ background: #f59e0b; }}
    .progress-removed {{ background: #ef4444; }}

    table {{ width: 100%; border-collapse: separate; border-spacing: 0; margin-top: 16px; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.05); border: 1px solid #e2e8f0; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 12px 16px; text-align: left; }}
    th {{ background: #f8fafc; font-weight: 600; color: #475569; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.05em; }}
    tr:last-child td {{ border-bottom: none; }}
    
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; margin-top: 16px; }}
    .grid-pairs {{ display: flex; flex-wrap: wrap; gap: 16px; margin-top: 16px; }}
    .example {{ border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px; background: #ffffff; box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05); }}
    .example img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 8px; margin-bottom: 8px; }}
    .example span, .example small {{ display: block; color: #64748b; margin-top: 4px; }}
    .example strong {{ color: #0f172a; }}
    
    code {{ background: #f1f5f9; padding: 3px 6px; border-radius: 4px; font-family: monospace; color: #1e293b; font-size: 0.85em; border: 1px solid #e2e8f0; }}
    .status-kept {{ color: #10b981; font-weight: 600; }}
    .status-removed {{ color: #ef4444; font-weight: 600; }}
    .badge {{ background: #f1f5f9; color: #475569; padding: 2px 8px; border-radius: 12px; font-size: 0.75em; display: inline-block; margin-top: 4px; border: 1px solid #e2e8f0; font-weight: 600; }}
    .number-cell {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .removed-col {{ background-color: #fef2f2; color: #991b1b; font-weight: 600; }}
  </style>
</head>
<body>
<main>
  <div class="hero">
    <h1>AgriVision Khaos &middot; Reporte de Curación</h1>
    <p><strong>Dataset:</strong> {html.escape(dataset_name)} &nbsp;|&nbsp; <strong>Run:</strong> <code>{html.escape(run_id)}</code></p>
    
    <div class="progress-bar">
      <div class="progress-kept" style="width: {max(0, summary['counts']['kept'] / max(1, summary['counts']['initial']) * 100)}%" title="Conservadas"></div>
      <div class="progress-review" style="width: {max(0, summary['counts']['review'] / max(1, summary['counts']['initial']) * 100)}%" title="En Revisión"></div>
      <div class="progress-removed" style="width: {max(0, summary['counts']['removed'] / max(1, summary['counts']['initial']) * 100)}%" title="Eliminadas"></div>
    </div>
    
    <section class="stats">
      <div class="stat"><strong>{summary['counts']['initial']:,}</strong><span>Iniciales</span></div>
      <div class="stat"><strong style="color: #059669;">{summary['counts']['kept']:,}</strong><span>Exportadas</span></div>
      <div class="stat"><strong style="color: #d97706;">{summary['counts']['review']:,}</strong><span>En Revisión</span></div>
      <div class="stat"><strong style="color: #dc2626;">{summary['counts']['removed']:,}</strong><span>Descartadas</span></div>
    </section>
    
    <div style="margin-top: 24px; padding: 18px; background: #f8fafc; border-radius: 12px; border: 1px solid #e2e8f0;">
      <h3 style="margin: 0 0 12px; font-size: 1em; color: #0f172a;">Resumen de Fases</h3>
      <ul style="margin: 0; padding-left: 20px; color: #334155; font-size: 0.9em; line-height: 1.6;">
        {''.join(
            f"<li><strong>{html.escape(p['name'])}</strong>"
            f"{': ' + str(p['removed']) + ' descartadas | ' + str(p['review']) + ' en revisión | ' + str(p['kept']) + ' conservadas' if (p['removed'] > 0 or p['review'] > 0 or p['kept'] > 0) else ''}"
            f"<br><span style='color:#64748b;'>{'<br>'.join(html.escape(n) for n in p['notes'])}</span></li>"
            for p in summary["phases"]
        )}
      </ul>
    </div>
  </div>
  
  
  <div style="max-width: 1280px; margin: 0 auto 32px; background: #ffffff; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.05);">
    <h3 style="margin-top: 0; color: #0f172a;">Sugerencia de Balanceo (Class Weights)</h3>
    <p style="color: #64748b; font-size: 0.9em; margin-bottom: 16px;">Copia y pega este fragmento en tu código de entrenamiento para penalizar matemáticamente las clases mayoritarias y ayudar a la red a converger de forma equitativa.</p>
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 16px; border-radius: 8px; overflow-x: auto;">
      <code style="color: #0f172a; background: transparent; border: none; font-size: 0.95em; padding: 0;">
class_weights = {{<br>
{''.join(f"    '{html.escape(cls)}': {weight},<br>" for cls, weight in summary.get('class_weights', {}).items())}
}}<br><br>
# PyTorch:<br>
# weights_tensor = torch.tensor([class_weights[c] for c in classes], dtype=torch.float32)<br>
# criterion = nn.CrossEntropyLoss(weight=weights_tensor)
      </code>
    </div>
  </div>

  {render_contamination_matrix(list(evidence.get('cross_contamination', [])))}

  {render_ontology_harmonization_section(summary.get('label_mapping', {}))}

  {render_count_table("Evolucion por dataset origen", summary["groups"]["by_source"], show_drop_rates=True)}
  {render_count_table("Evolucion por etiqueta", summary["groups"]["by_label"])}
  {render_reason_sections("Ejemplos descartados por razon", list(evidence.get('removed_by_reason', [])), "No hay ejemplos descartados.")}
  {render_duplicate_sections("Duplicados y mantenimiento del original", list(evidence.get('duplicate_phases', [])))}
  {render_reason_sections("Ejemplos en revision", list(evidence.get('review_by_reason', [])), "No hay ejemplos en revision.")}
</main>
</body>
</html>
"""
    (report_dir / "report.html").write_text(html_doc, encoding="utf-8")


def write_dataset_card(path: Path, summary: dict[str, Any]) -> None:
    """Writes a portable dataset card with provenance and known limitations."""
    counts = summary.get("counts", {})
    sources = summary.get("sources", [])
    unknown_licenses = [
        str(source.get("name", "unknown"))
        for source in sources
        if str(source.get("license", "unknown")).lower() == "unknown"
    ]
    source_rows = [
        "| Fuente | Versión | Licencia | Formato |",
        "|---|---:|---|---|",
    ]
    source_rows.extend(
        f"| {source.get('name', 'unknown')} | {source.get('version', 'unknown')} | "
        f"{source.get('license', 'unknown')} | {source.get('format', 'unknown')} |"
        for source in sources
    )
    limitations = [
        "Los umbrales de calidad son heurísticos y deben calibrarse para cada dominio de captura.",
        "Los casos en estado `review` no se incluyen en los exports de entrenamiento.",
        "La similitud visual no demuestra identidad biológica de plantas o lesiones.",
    ]
    if unknown_licenses:
        limitations.append(
            "No publicar hasta resolver licencias desconocidas: " + ", ".join(unknown_licenses) + "."
        )
    card = "\n".join(
        [
            f"# Dataset Card: {summary.get('dataset', 'unknown')}",
            "",
            f"- Run: `{summary.get('run_id', 'unknown')}`",
            f"- Generado: `{summary.get('created_at', 'unknown')}`",
            f"- Muestras iniciales: {counts.get('initial', 0)}",
            f"- Conservadas: {counts.get('kept', 0)}",
            f"- Revisión pendiente: {counts.get('review', 0)}",
            f"- Eliminadas: {counts.get('removed', 0)}",
            "",
            "## Fuentes",
            "",
            *source_rows,
            "",
            "## Limitaciones y uso responsable",
            "",
            *(f"- {limitation}" for limitation in limitations),
            "",
            "La configuración exacta está incluida en `curation_summary.json`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(card, encoding="utf-8")


def current_status(sample: fo.Sample) -> str:
    curation = sample_field(sample, "curation")
    return str(getattr(curation, "status", "kept") or "kept")


def source_label_key(sample: fo.Sample) -> tuple[str, str]:
    return (
        str(sample_field(sample, "source_dataset", "unknown") or "unknown"),
        str(sample_field(sample, "normalized_label", "") or ""),
    )


def apply_second_opinion(
    samples: list[fo.Sample],
    decisions: dict[str, Decision],
    max_phase_drop: float,
    max_total_drop: float,
) -> list[str]:
    notes = []
    group_totals: Counter[tuple[str, str]] = Counter()
    group_phase_removed: Counter[tuple[str, str]] = Counter()
    group_already_removed: Counter[tuple[str, str]] = Counter()

    for sample in samples:
        key = source_label_key(sample)
        group_totals[key] += 1
        if current_status(sample) == "removed":
            group_already_removed[key] += 1
        decision = decisions.get(sample.id)
        if decision is not None and decision.status == "removed":
            group_phase_removed[key] += 1

    for key, total in group_totals.items():
        if not total:
            continue
        phase_ratio = group_phase_removed[key] / total
        total_ratio = (group_already_removed[key] + group_phase_removed[key]) / total
        if phase_ratio <= max_phase_drop and total_ratio <= max_total_drop:
            continue

        converted = 0
        for sample in samples:
            if source_label_key(sample) != key:
                continue
            decision = decisions.get(sample.id)
            if (
                decision is not None
                and decision.status == "removed"
                and (decision.confidence < HARD_CONFIDENCE
                     or (decision.phase.endswith("duplicates")
                         and decision.evidence_level not in EXACT_LEVELS))
            ):
                decision.status = "review"
                decision.reason = f"second_opinion_{decision.reason}"
                converted += 1

        if converted:
            source, label = key
            notes.append(
                f"{converted} descartes fronterizos pasan a revision en source={source}, "
                f"label={label or 'unknown'} por caida anomala."
            )

    return notes


def write_decisions(dataset: fo.Dataset, decisions: dict[str, Decision], tag_prefix: str) -> PhaseResult:
    result = PhaseResult(name=tag_prefix)
    if not decisions:
        result.notes.append("No hubo decisiones nuevas.")
        return result

    ensure_history(dataset)
    for sample in dataset.select(list(decisions)).iter_samples(progress=True, autosave=True):
        before = snapshot(sample)
        decision = decisions[sample.id]
        previous = sample_field(sample, "curation")
        if current_status(sample) == "review":
            reasons = list(getattr(previous, "review_reasons", []) or [])
            if getattr(previous, "reason", ""):
                reasons.append(previous.reason)
            if (getattr(previous, "phase", "") == decision.phase
                    and decision.evidence_level in EXACT_LEVELS and decision.status == "removed"):
                resolved = {decision.reason, getattr(previous, "reason", "")}
                reasons = [reason for reason in reasons if reason not in resolved]
            decision.review_reasons = sorted(set(decision.review_reasons + reasons))
            # A new duplicate decision cannot silently resolve a previous quality/label issue.
            if decision.status != "review" and decision.review_reasons:
                decision.status = "review"
        if decision.status == "review":
            decision.review_reasons = sorted(set(decision.review_reasons + [decision.reason]))
        sample["curation"] = fo.DynamicEmbeddedDocument(**asdict(decision))
        state_tags = {"curation_kept", "curation_review", "curation_removed"}
        retained_tags = [tag for tag in sample.tags if tag not in state_tags]
        sample.tags = sorted(
            set(retained_tags + [f"curation_{decision.status}", f"{tag_prefix}_{decision.reason}"])
        )
        record_transition(sample, before, tag_prefix)
        result.removed += int(decision.status == "removed")
        result.review += int(decision.status == "review")
        result.kept += int(decision.status == "kept")

    return result


def run_quality_phase(
    dataset: fo.Dataset,
    max_phase_drop: float,
    max_total_drop: float,
    policy: QualityPolicy | None = None,
) -> PhaseResult:
    active_samples = [sample for sample in dataset if current_status(sample) != "removed"]
    decisions = {sample.id: evaluate_quality(sample, policy) for sample in active_samples}
    notes = apply_second_opinion(list(dataset), decisions, max_phase_drop, max_total_drop)
    result = write_decisions(dataset, decisions, "quality")
    result.notes.extend(notes)
    return result


def decision_for_tagged_duplicates(
    dataset: fo.Dataset,
    tag: str,
    phase: str,
    reason_text: str,
    status: str = "removed",
    confidence: float = 1.0,
) -> dict[str, Decision]:
    decisions = {}
    for sample in dataset.match_tags(tag):
        if current_status(sample) == "removed":
            continue
        if getattr(sample_field(sample, "curation"), "phase", "") == "human_review":
            continue
        evidence = sample_field(sample, f"{tag}_evidence", {}) or {}
        representative_id = str(evidence.get("representative_id", ""))
        level = evidence.get("level", "candidate")
        effective_status = status
        if status == "removed" and (level not in EXACT_LEVELS or not representative_id):
            effective_status = "review"
        if sample_field(sample, "annotation_valid", True) is False:
            effective_status = "review"
        decisions[sample.id] = Decision(
            status=effective_status,
            phase=phase,
            reason=reason_text,
            confidence=1.0 if level in EXACT_LEVELS else min(confidence, 0.85),
            evidence_level=level,
            keep_reason=(
                f"duplicate_of:{representative_id}"
                if effective_status == "removed" and representative_id
                else "potential_duplicate_requires_human_review"
            ),
            cluster_id=str(sample_field(sample, "duplicate_cluster_id", "") or ""),
            representative_id=representative_id,
        )
    return decisions


def decisions_for_label_conflicts(dataset: fo.Dataset, tag: str, phase: str) -> dict[str, Decision]:
    decisions = {}
    for sample in dataset.match_tags(f"{tag}_label_conflict"):
        if current_status(sample) == "removed":
            continue
        if getattr(sample_field(sample, "curation"), "phase", "") == "human_review":
            continue
        decisions[sample.id] = Decision(
            status="review",
            phase=phase,
            reason="Conflicto de Etiquetas entre Duplicados",
            confidence=1.0,
            keep_reason="annotation_conflict_requires_human_review",
            cluster_id=str(sample_field(sample, "duplicate_cluster_id", "") or ""),
            representative_id=str(
                sample_field(sample, "duplicate_representative_id", "") or ""
            ),
        )
    return decisions



def prepare_duplicate_run(dataset: fo.Dataset, method: str | None = None) -> None:
    """Restore the pre-dedup state before retrying an incomplete or changed analysis."""
    baseline_field = f"{method}_baseline" if method else "pre_duplicate_curation"
    target_phase = ("augmentation_duplicates" if method == "redundant_augmented"
                    else f"{method.removeprefix('redundant_')}_duplicates") if method else None
    if baseline_field not in dataset.get_field_schema():
        dataset.add_sample_field(baseline_field, fo.DictField)
    ensure_history(dataset)
    if method and "pre_duplicate_curation" not in dataset.get_field_schema():
        dataset.add_sample_field("pre_duplicate_curation", fo.DictField)
    prefixes = ("exact_duplicates_", "semantic_duplicates_", "augmentation_duplicates_",
                "redundant_exact", "redundant_semantic", "redundant_augmented", "curation_",
                "consistency_duplicates_")
    for sample in dataset.iter_samples(autosave=True):
        before = snapshot(sample)
        previous = sample_field(sample, "curation")
        phase = getattr(previous, "phase", "")
        if phase == "human_review":
            continue
        if method and not phase.endswith("duplicates"):
            sample["pre_duplicate_curation"] = (before["curation"] or asdict(Decision("kept", "ingestion", "", 1.0)))
        baseline = sample_field(sample, baseline_field)
        if (phase == target_phase if method else phase.endswith("duplicates")) and baseline:
            sample["curation"] = fo.DynamicEmbeddedDocument(**dict(baseline))
        else:
            sample[baseline_field] = (
                {key: item for key, item in previous.to_dict().items() if key != "_cls"}
                if previous is not None else asdict(Decision("kept", "ingestion", "", 1.0))
            )
        cleared = (method, target_phase + "_", "curation_") if method else prefixes
        sample.tags = [tag for tag in sample.tags if not tag.startswith(cleared)]
        sample.tags.append(f"curation_{current_status(sample)}")
        record_transition(sample, before, "reset_" + (method or "duplicates"))
    if method in (None, "redundant_augmented"):
        dataset.info.pop("augmentation_analysis", None)
        dataset.save()
    # Clear all automated methods, including ones now disabled. Human overrides survive.
    schema = dataset.get_field_schema()
    if "duplicate_links" in schema:
        dataset.set_values("duplicate_links", {
            sample.id: [dict(link) for link in (sample_field(sample, "duplicate_links", []) or [])
                        if (link.get("method") != method if method else link.get("method") == "human_review")] for sample in dataset
        }, key_field="id")
    for name in schema:
        if name in {"duplicate_cluster_id", "duplicate_family_id", "duplicate_representative_id"}:
            dataset.set_values(name, [""] * len(dataset))
        elif name in ({f"{method}_evidence"} if method else {"redundant_exact_evidence", "redundant_semantic_evidence", "redundant_augmented_evidence"}):
            dataset.set_values(name, [{} for _ in dataset])


def reconcile_duplicate_representatives(dataset: fo.Dataset) -> PhaseResult:
    status_map = dict(zip(dataset.values("id"), dataset.values("curation.status")))
    rep_map = dict(zip(dataset.values("id"), dataset.values("curation.representative_id")))
    decisions = {}
    for sample_id, status in status_map.items():
        representative_id = rep_map.get(sample_id) or ""
        if status != "removed" or not representative_id:
            continue
        rep_status = status_map.get(representative_id)
        if rep_status != "kept":
            decisions[sample_id] = Decision(
                "review", "consistency_duplicates", "Representante pendiente de revisión", 0.0,
                representative_id=representative_id,
            )
    return write_decisions(dataset, decisions, "consistency_duplicates")


def run_duplicate_phases(
    dataset: fo.Dataset,
    work_dir: Path,
    max_phase_drop: float,
    max_total_drop: float,
    policy: CurationPolicy | None = None,
) -> list[PhaseResult]:
    from agrivision_khaos.deduplication import (
        detect_augmentation_duplicates,
        detect_exact_duplicates,
        detect_semantic_duplicates,
    )

    policy = policy or CurationPolicy()

    if isinstance(dataset, fo.Dataset):
        prepare_duplicate_run(dataset)

    results = []
    if policy.deduplication.exact_enabled:
        logger.info("Detectando duplicados exactos con FiftyOne...")
        _, pairs = detect_exact_duplicates(dataset, "redundant_exact")
        decisions = decision_for_tagged_duplicates(dataset, "redundant_exact", "exact_duplicates", "Redundante (Exacta)")
        decisions.update(decisions_for_label_conflicts(dataset, "redundant_exact", "exact_duplicates"))
        notes = apply_second_opinion(list(dataset), decisions, max_phase_drop, max_total_drop)
        exact_result = write_decisions(dataset, decisions, "exact_duplicates")
        exact_result.notes.extend(notes)
        exact_result.duplicate_pairs = pairs
        results.append(exact_result)
    else:
        results.append(PhaseResult(name="exact_duplicates", notes=["Desactivada por política."]))

    if policy.deduplication.semantic_enabled:
        logger.info("Detectando near-duplicates semanticos con FiftyOne...")
        try:
            _, pairs = detect_semantic_duplicates(
                dataset,
                "redundant_semantic",
                threshold=policy.deduplication.semantic_similarity,
            )
            decisions = decision_for_tagged_duplicates(
                dataset,
                "redundant_semantic",
                "semantic_duplicates",
                "Redundante (Semántica)",
                status="review",
                confidence=0.85,
            )
            decisions.update(
                decisions_for_label_conflicts(
                    dataset, "redundant_semantic", "semantic_duplicates"
                )
            )
            notes = apply_second_opinion(
                list(dataset), decisions, max_phase_drop, max_total_drop
            )
            if policy.deduplication.augmentation_enabled:
                semantic_result = PhaseResult(
                    name="semantic_duplicates",
                    notes=["Candidatos delegados a la verificación de aumentaciones antes de decidir."],
                )
            else:
                semantic_result = write_decisions(dataset, decisions, "semantic_duplicates")
            semantic_result.notes.extend(notes)
            semantic_result.duplicate_pairs = pairs
            results.append(semantic_result)
        except Exception as exc:
            raise RuntimeError(
                "La deduplicación semántica estaba habilitada y falló; "
                "se cancela la publicación del dataset"
            ) from exc
    else:
        results.append(PhaseResult(name="semantic_duplicates", notes=["Desactivada por política."]))

    if policy.deduplication.augmentation_enabled:
        logger.info("Detectando y verificando variantes de una misma captura...")
        try:
            _, pairs = detect_augmentation_duplicates(
                dataset,
                "redundant_augmented",
                threshold=policy.deduplication.augmentation_similarity,
                policy=policy.deduplication,
                cache_dir=work_dir / "augmentation-cache",
            )
            decisions = decision_for_tagged_duplicates(
                dataset,
                "redundant_augmented",
                "augmentation_duplicates",
                "Redundante (Aumentación)",
                status=("removed" if policy.deduplication.remove_exact_transforms
                        and policy.deduplication.augmentation_action == "remove" else "review"),
                confidence=0.75,
            )
            decisions.update(
                decisions_for_label_conflicts(
                    dataset, "redundant_augmented", "augmentation_duplicates"
                )
            )
            for sample in dataset.match_tags("redundant_augmented_invalid"):
                if current_status(sample) != "removed":
                    decisions[sample.id] = Decision(
                        "review", "augmentation_duplicates", "Descriptor visual no disponible", 0.0
                    )
            notes = apply_second_opinion(
                list(dataset), decisions, max_phase_drop, max_total_drop
            )
            augmented_result = write_decisions(
                dataset, decisions, "augmentation_duplicates"
            )
            augmented_result.notes.extend(notes)
            augmented_result.duplicate_pairs = pairs
            results.append(augmented_result)
        except Exception as exc:
            raise RuntimeError(
                "La detección de aumentaciones estaba habilitada y falló; "
                "se cancela la publicación del dataset"
            ) from exc
    else:
        results.append(PhaseResult(name="augmentation_duplicates", notes=["Desactivada por política."]))

    consistency = reconcile_duplicate_representatives(dataset)
    if consistency.review:
        results.append(consistency)
    return results


def suggest_label_mappings(
    labels: list[str],
    label_to_sources: dict[str, set[str]] | None = None,
    label_counts: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    normalized_to_sources: dict[str, set[str]] = defaultdict(set)
    for label in labels:
        normalized_to_sources[normalize_label(label)].add(label)

    automatic = {
        normalized: sorted(values)
        for normalized, values in normalized_to_sources.items()
        if normalized and len(values) > 1
    }
    candidates = []
    typos_and_similar = []
    normalized_labels = sorted(label for label in normalized_to_sources if label)
    
    antonym_prefixes = ("un_", "non_", "partially_", "pre_", "un", "non", "partially")

    known_standard_terms = {
        "anthracnose", "healthy", "avocado", "powdery_mildew", "leaf_spot",
        "bacterial_canker", "fruit_rot", "apple", "banana", "orange", "mango",
        "papaya", "tomato", "chirimoya", "black_spot", "ring_spot", "phytophthora"
    }

    def _clean_stem(s: str) -> str:
        res = re.sub(r"(_diease|_disease|_diseases|_dataset|_sin_fondo|_evaluacion|_evaluación|_entrenamiento)$", "", s)
        return res or s

    for index, left in enumerate(normalized_labels):
        for right in normalized_labels[index + 1 :]:
            if left == right:
                continue

            stem_left = _clean_stem(left)
            stem_right = _clean_stem(right)

            raw_ratio = difflib.SequenceMatcher(None, left, right).ratio()
            stem_ratio = difflib.SequenceMatcher(None, stem_left, stem_right).ratio()
            sm_ratio = max(raw_ratio, stem_ratio)

            left_tokens = set(left.split("_"))
            right_tokens = set(right.split("_"))
            overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)

            is_lifecycle = any(
                (left.startswith(p) and right == left[len(p):]) or
                (right.startswith(p) and left == right[len(p):])
                for p in antonym_prefixes
            )

            if overlap >= 0.5:
                candidates.append({"left": left, "right": right, "token_overlap": round(overlap, 3)})

            if sm_ratio >= 0.75 or overlap >= 0.5:
                left_origs = normalized_to_sources[left]
                right_origs = normalized_to_sources[right]
                left_ds: set[str] = set()
                right_ds: set[str] = set()
                left_count = 0
                right_count = 0

                if label_to_sources:
                    for lo in left_origs:
                        left_ds.update(label_to_sources.get(lo, set()))
                    for ro in right_origs:
                        right_ds.update(label_to_sources.get(ro, set()))

                if label_counts:
                    for lo in left_origs:
                        left_count += sum(label_counts.get(lo, {}).values())
                    for ro in right_origs:
                        right_count += sum(label_counts.get(ro, {}).values())

                cross_dataset = bool(left_ds and right_ds and (left_ds != right_ds))

                if is_lifecycle:
                    cat = "lifecycle_stage"
                    diag = "Fases de maduración / Estados opuestos"
                    rec_merge = False
                elif sm_ratio >= 0.85 and overlap < 0.5:
                    cat = "typo_variant"
                    diag = f"Errata tipográfica inter-dataset ({int(sm_ratio*100)}% similitud)"
                    rec_merge = True
                elif sm_ratio >= 0.85:
                    cat = "exact_variant"
                    diag = f"Variante cercana ({int(sm_ratio*100)}% similitud)"
                    rec_merge = True
                else:
                    cat = "shared_tokens"
                    diag = f"Concepto relacionado ({int(overlap*100)}% palabras compartidas)"
                    rec_merge = False

                if rec_merge:
                    # Prefer known canonical terms or standard word spelling
                    if stem_left in known_standard_terms:
                        suggested_canonical = stem_left
                    elif stem_right in known_standard_terms:
                        suggested_canonical = stem_right
                    elif left in known_standard_terms:
                        suggested_canonical = left
                    elif right in known_standard_terms:
                        suggested_canonical = right
                    else:
                        suggested_canonical = left if left_count >= right_count else right
                else:
                    suggested_canonical = None

                typos_and_similar.append({
                    "left": left,
                    "right": right,
                    "left_origs": sorted(left_origs),
                    "right_origs": sorted(right_origs),
                    "left_datasets": sorted(left_ds),
                    "right_datasets": sorted(right_ds),
                    "left_count": left_count,
                    "right_count": right_count,
                    "similarity": round(sm_ratio, 2),
                    "token_overlap": round(overlap, 2),
                    "category": cat,
                    "diagnosis": diag,
                    "recommended_merge": rec_merge,
                    "suggested_canonical": suggested_canonical,
                    "cross_dataset": cross_dataset,
                })

    typos_and_similar.sort(key=lambda x: (-x["similarity"], -x["token_overlap"]))

    return {
        "automatic": automatic,
        "candidates": candidates,
        "typos_and_similar": typos_and_similar,
    }


def load_ontology_mapping(path: str | Path) -> dict[str, str]:
    ontology_path = Path(path)
    if not ontology_path.is_file():
        raise ValueError(f"No existe el archivo de ontología: {ontology_path}")
    try:
        payload = yaml.safe_load(ontology_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Ontología YAML inválida {ontology_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("La ontología debe ser un mapa canonical_label: [source_labels]")

    mapping: dict[str, str] = {}
    for target, originals in payload.items():
        canonical = normalize_label(str(target))
        if not canonical or not isinstance(originals, list) or not originals:
            raise ValueError(
                f"Entrada de ontología inválida para {target!r}: se esperaba una lista no vacía"
            )
        for original in originals:
            if not isinstance(original, str) or not original.strip():
                raise ValueError(f"Etiqueta de origen inválida bajo {target!r}: {original!r}")
            previous = mapping.get(original)
            if previous is not None and previous != canonical:
                raise ValueError(
                    f"La etiqueta {original!r} se asigna a dos clases: "
                    f"{previous!r} y {canonical!r}"
                )
            mapping[original] = canonical
    return mapping


def run_label_phase(dataset: fo.Dataset, cleanlab_mode: str, ontology_map: str | None, report_dir: Path) -> tuple[PhaseResult, dict[str, Any]]:
    if cleanlab_mode == "on":
        if importlib.util.find_spec("cleanlab") is None:
            raise RuntimeError("--cleanlab-mode on requiere instalar Cleanlab")
        raise RuntimeError(
            "--cleanlab-mode on requiere predicciones out-of-sample, que este pipeline "
            "todavía no recibe; usa auto u off"
        )

    # Vectorized fast collection of all labels and dataset associations
    src_datasets = dataset.values("source_dataset") if hasattr(dataset, "has_sample_field") and dataset.has_sample_field("source_dataset") else []
    src_labels_list = dataset.values("source_labels") if hasattr(dataset, "has_sample_field") and dataset.has_sample_field("source_labels") else []
    src_single_labels = dataset.values("source_label") if hasattr(dataset, "has_sample_field") and dataset.has_sample_field("source_label") else []
    det_labels_list = dataset.values("ground_truth_detections.detections.label") if hasattr(dataset, "has_sample_field") and dataset.has_sample_field("ground_truth_detections") else []

    label_to_sources: dict[str, set[str]] = defaultdict(set)
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    all_raw_labels: set[str] = set()

    n_samples = len(dataset) if hasattr(dataset, "__len__") else 0
    for idx in range(n_samples):
        src = str(src_datasets[idx] if idx < len(src_datasets) and src_datasets[idx] else "unknown")
        sample_labels: set[str] = set()
        if idx < len(src_single_labels) and src_single_labels[idx]:
            sample_labels.add(str(src_single_labels[idx]).strip())
        if idx < len(src_labels_list) and src_labels_list[idx]:
            for label in src_labels_list[idx]:
                if label:
                    sample_labels.add(str(label).strip())
        if idx < len(det_labels_list) and det_labels_list[idx]:
            for label in det_labels_list[idx]:
                if label:
                    sample_labels.add(str(label).strip())
        for label in sample_labels:
            all_raw_labels.add(label)
            label_to_sources[label].add(src)
            label_counts[label][src] += 1

    labels = sorted(all_raw_labels)
    mapping = suggest_label_mappings(labels, label_to_sources, label_counts)
    result = PhaseResult(name="labels")
    
    automatic_map = {}
    if ontology_map:
        automatic_map = load_ontology_mapping(ontology_map)
        result.notes.append(f"Ontología de usuario aplicada desde {ontology_map}.")
        mapping["applied"] = dict(sorted(automatic_map.items()))
    else:
        # High confidence typo unification
        merge_aliases: dict[str, str] = {}
        for item in mapping.get("typos_and_similar", []):
            if item.get("recommended_merge") and item.get("suggested_canonical"):
                canon = item["suggested_canonical"]
                other = item["right"] if canon == item["left"] else item["left"]
                if other not in merge_aliases:
                    merge_aliases[other] = canon

        proposed: dict[str, list[str]] = defaultdict(list)
        for norm, origs in mapping["automatic"].items():
            target_key = merge_aliases.get(norm, norm)
            for o in origs:
                if o not in proposed[target_key]:
                    proposed[target_key].append(o)

        for label in labels:
            norm = normalize_label(label)
            target_key = merge_aliases.get(norm, norm)
            if label not in proposed[target_key]:
                proposed[target_key].append(label)

        yaml_lines = [
            "# ==============================================================================",
            "# AGRIVISION-KHAOS: ONTOLOGÍA MULTI-DATASET SUGERIDA",
            "# ==============================================================================",
            "# Este archivo unifica las etiquetas de los diferentes datasets integrados,",
            "# resolviendo diferencias de nombrado, mayúsculas/minúsculas y erratas detectadas.",
            "#",
            "# Puedes editar este archivo según tus necesidades y aplicarlo en el pipeline con:",
            "#   make pipeline ONTOLOGY=reports/pipeline/.../proposed_ontology.yaml",
            "# o configurando ONTOLOGY=configs/ontology.yaml en tu archivo .env",
            "# ==============================================================================",
            "",
        ]
        for canonical, orig_list in sorted(proposed.items()):
            yaml_lines.append(f"{canonical}:")
            for orig in orig_list:
                src_info = []
                for ds, cnt in sorted(label_counts[orig].items()):
                    src_info.append(f"{ds}: {cnt}")
                info_str = ", ".join(src_info)
                norm_o = normalize_label(orig)
                note = " [Errata/Variante unificada]" if norm_o in merge_aliases else ""
                yaml_lines.append(f"  - {orig!r}  # {info_str}{note}")
            yaml_lines.append("")

        proposed_yaml_text = "\n".join(yaml_lines)
        proposed_path = report_dir / "proposed_ontology.yaml"
        with open(proposed_path, "w", encoding="utf-8") as f:
            f.write(proposed_yaml_text)
        result.notes.append(f"Archivo de ontología sugerido generado en {proposed_path}.")
        mapping["proposed_yaml"] = proposed_yaml_text

        # Build dataset class matrix
        dataset_matrix = []
        for canonical, orig_list in sorted(proposed.items()):
            total_samples = 0
            ds_breakdown = Counter()
            for orig in orig_list:
                for ds, cnt in label_counts[orig].items():
                    ds_breakdown[ds] += cnt
                    total_samples += cnt
            dataset_matrix.append({
                "canonical": canonical,
                "total": total_samples,
                "datasets": dict(ds_breakdown),
                "original_labels": orig_list,
            })
        dataset_matrix.sort(key=lambda x: -x["total"])
        mapping["dataset_matrix"] = dataset_matrix

        # Use heuristics
        for canonical, orig_list in proposed.items():
            for orig in orig_list:
                automatic_map[orig] = canonical

    mapping["applied"] = dict(sorted(automatic_map.items()))

    if not dataset.has_sample_field("normalized_label"):
        dataset.add_sample_field("normalized_label", fo.StringField)

    for sample in dataset.iter_samples(autosave=True):
        normalized_labels: set[str] = set()
        original = sample_field(sample, "source_label")
        if original:
            normalized = automatic_map.get(original, normalize_label(original))
            sample["normalized_label"] = normalized
            normalized_labels.add(normalized)
            canonical_class = sample_field(sample, "ground_truth_classification")
            if canonical_class is not None:
                canonical_class.label = normalized
                sample["ground_truth_classification"] = canonical_class
                sample["final_label"] = copy.deepcopy(canonical_class)

        detections = sample_field(sample, "ground_truth_detections")
        for detection in getattr(detections, "detections", None) or []:
            try:
                source_label_value = detection.get_field("source_label")
            except Exception:
                source_label_value = getattr(detection, "source_label", None)
            source_label = str(
                source_label_value or getattr(detection, "label", "") or ""
            )
            if not source_label:
                continue
            detection.label = automatic_map.get(source_label, normalize_label(source_label))
            normalized_labels.add(detection.label)
        if detections is not None:
            sample["ground_truth_detections"] = detections
        sample["normalized_labels"] = sorted(normalized_labels)

    if cleanlab_mode == "off":
        result.notes.append("Cleanlab desactivado.")
    elif importlib.util.find_spec("cleanlab") is None:
        result.notes.append("Cleanlab no esta instalado; se generaron sugerencias heuristicas.")
    else:
        result.notes.append(
            "Cleanlab disponible, pero no se ejecuta sin predicciones out-of-sample; "
            "se conserva como fase condicional."
        )

    result.notes.append(
        f"{len(mapping['automatic'])} grupos de etiquetas normalizados automaticamente y "
        f"{len(mapping['candidates'])} candidatos dudosos."
    )
    return result, mapping


def status_counts(dataset: fo.Dataset) -> Counter[str]:
    counts: Counter[str] = Counter()
    for sample in dataset:
        counts[current_status(sample)] += 1
    return counts


def grouped_counts(dataset: fo.Dataset) -> dict[str, dict[str, dict[str, int]]]:
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_label: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in dataset:
        status = current_status(sample)
        if status == "removed":
            curation = sample_field(sample, "curation")
            if curation and curation.phase == "quality":
                status = "removed_quality"
            elif curation:
                status = "removed_duplicates"
                
        by_source[str(sample_field(sample, "source_dataset", "unknown") or "unknown")][status] += 1
        by_label[str(sample_field(sample, "normalized_label", "unknown") or "unknown")][status] += 1
    return {
        "by_source": {key: dict(value) for key, value in sorted(by_source.items())},
        "by_label": {key: dict(value) for key, value in sorted(by_label.items())},
    }


def get_base64_image(filepath: str, max_size: int = 220, transform: str = "r0",
                     affine=None, reference: str = "") -> str:
    try:
        if affine is not None and reference:
            cache = DescriptorCache()
            image = cache.describe(filepath).image()
            target = cache.describe(reference).image()
            matrix = np.asarray(affine, dtype=np.float64)
            if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
                return ""
            image = cv2.warpAffine(image, matrix, (target.shape[1], target.shape[0]))
        else:
            image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if image is None:
                return ""
            image = next((variant for name, variant in transforms(image) if name == transform), image)
        if image.ndim == 3 and image.shape[2] == 4:
            alpha = image[:, :, 3:4].astype(np.float32) / 255
            yy, xx = np.indices(image.shape[:2])
            checker = (180 + ((xx // 8 + yy // 8) % 2) * 50)[:, :, None]
            image = (image[:, :, :3] * alpha + checker * (1-alpha)).astype(np.uint8)
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return base64.b64encode(buffer).decode("utf-8")
    except Exception:
        return ""


def sample_field(sample: fo.Sample, field_name: str, default: Any = None) -> Any:
    try:
        return sample.get_field(field_name)
    except Exception:
        return default


def metric(sample: fo.Sample, field_name: str, default: float | None = None) -> float | None:
    value = sample_field(sample, field_name)
    if value is None:
        quality = sample_field(sample, "quality")
        value = getattr(quality, field_name, None) if quality is not None else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def flag(sample: fo.Sample, field_name: str) -> bool:
    value = sample_field(sample, field_name)
    if value is None:
        quality = sample_field(sample, "quality")
        value = getattr(quality, field_name, None) if quality is not None else None
    return bool(value)


def format_report_value(value: object) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, bool):
        return "sí" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def sample_curation_payload(sample: fo.Sample) -> dict[str, object]:
    curation = sample_field(sample, "curation")
    return {
        "id": sample.id,
        "filepath": sample.filepath,
        "source_dataset": str(sample_field(sample, "source_dataset", "")),
        "source_split": str(sample_field(sample, "source_split", "")),
        "source_label": str(sample_field(sample, "source_label", "")),
        "label": str(sample_field(sample, "normalized_label", "")),
        "status": str(getattr(curation, "status", "")),
        "phase": str(getattr(curation, "phase", "")),
        "reason": str(getattr(curation, "reason", "")),
        "confidence": getattr(curation, "confidence", None),
        "keep_reason": str(getattr(curation, "keep_reason", "")),
        "cluster_id": str(getattr(curation, "cluster_id", "")),
        "representative_id": str(getattr(curation, "representative_id", "")),
        "evidence_level": str(getattr(curation, "evidence_level", "")),
        "review_reasons": list(getattr(curation, "review_reasons", []) or []),
        "family_id": str(sample_field(sample, "duplicate_family_id", "")),
        "blur_variance": metric(sample, "blur_variance"),
        "brightness_mean": metric(sample, "brightness_mean"),
        "brightness_p5": metric(sample, "brightness_p5"),
        "brightness_p95": metric(sample, "brightness_p95"),
        "width": sample_field(sample, "width"),
        "height": sample_field(sample, "height"),
        "low_resolution": flag(sample, "low_resolution"),
        "has_watermark": flag(sample, "has_watermark"),
        "has_smearing": flag(sample, "has_smearing"),
        "processing_error": flag(sample, "processing_error"),
        "error_message": str(sample_field(sample, "error_message", "") or ""),
        "asset_sha256": str(sample_field(sample, "asset_sha256", "") or ""),
    }


def collect_examples(dataset: fo.Dataset, status: str, limit: int = 24) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for sample in dataset:
        if current_status(sample) != status:
            continue
        examples.append(sample_curation_payload(sample))
        if len(examples) >= limit:
            break
    return examples


def collect_examples_by_reason(
    dataset: fo.Dataset,
    status: str,
    per_reason_limit: int = 3,
    max_reasons: int = 10,
) -> list[dict[str, object]]:
    counts: Counter[str] = Counter()
    samples_by_reason: dict[str, list[dict[str, object]]] = defaultdict(list)

    for sample in dataset:
        if current_status(sample) != status:
            continue
        curation = sample_field(sample, "curation")
        reason = str(getattr(curation, "reason", "") or "unknown")
        counts[reason] += 1
        bucket = samples_by_reason[reason]
        if len(bucket) < per_reason_limit:
            bucket.append(sample_curation_payload(sample))

    ordered_reasons = sorted(counts, key=lambda reason: (-counts[reason], reason))[:max_reasons]
    return [
        {
            "reason": reason,
            "count": counts[reason],
            "examples": samples_by_reason[reason],
        }
        for reason in ordered_reasons
    ]


def copy_example_gallery(examples: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, example in enumerate(examples, start=1):
        source = Path(str(example["filepath"]))
        if not source.exists():
            continue
        target = output_dir / f"{index:03d}_{slugify(str(example.get('reason', 'sample')))}{source.suffix.lower()}"
        shutil.copy2(source, target)


def copy_examples_by_reason(sections: list[dict[str, object]], output_dir: Path) -> None:
    for section in sections:
        reason_dir = output_dir / slugify(str(section.get("reason", "unknown")))
        copy_example_gallery(list(section.get("examples", [])), reason_dir)


def copy_duplicate_pair_gallery(sections: list[dict[str, object]], output_dir: Path) -> None:
    for section in sections:
        phase_dir = output_dir / slugify(str(section.get("phase", "duplicates")))
        phase_dir.mkdir(parents=True, exist_ok=True)
        for index, pair in enumerate(section.get("pairs", []), start=1):
            pair_dir = phase_dir / f"{index:03d}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            for role in ("kept", "removed"):
                example = pair.get(role, {})
                source = Path(str(example.get("filepath", "")))
                if not source.exists():
                    continue
                suffix = source.suffix.lower() or ".jpg"
                target = pair_dir / f"{role}_{slugify(str(example.get('reason', role)))}{suffix}"
                shutil.copy2(source, target)


def build_curation_evidence(
    dataset: fo.Dataset,
    duplicate_results: list[PhaseResult],
) -> dict[str, object]:
    removed_by_reason = collect_examples_by_reason(dataset, "removed")
    review_by_reason = collect_examples_by_reason(dataset, "review")

    duplicate_sections: list[dict[str, object]] = []
    cross_contamination_counts = Counter()
    sample_to_source = dict(
        zip(
            dataset.values("id"),
            dataset.values("source_dataset") if dataset.has_sample_field("source_dataset") else repeat("unknown"),
        )
    )

    for result in duplicate_results:
        # Calculate cross contamination for all pairs without per-sample MongoDB queries
        for kept_id, removed_id in result.duplicate_pairs:
            kept_ds = str(sample_to_source.get(kept_id, "unknown"))
            rem_ds = str(sample_to_source.get(removed_id, "unknown"))
            if kept_ds != rem_ds:
                pair_key = tuple(sorted([kept_ds, rem_ds]))
                cross_contamination_counts[pair_key] += 1

        top_ids = [pid for pair in result.duplicate_pairs[:8] for pid in pair]
        preview_samples = {s.id: s for s in dataset.select(top_ids)} if top_ids else {}
        pairs: list[dict[str, dict[str, object]]] = []
        for kept_id, removed_id in result.duplicate_pairs[:8]:
            kept_sample = preview_samples.get(kept_id)
            removed_sample = preview_samples.get(removed_id)
            if kept_sample is None or removed_sample is None:
                continue
            pairs.append(
                {
                    "kept": sample_curation_payload(kept_sample),
                    "removed": sample_curation_payload(removed_sample),
                    "evidence": dict(sample_field(
                        removed_sample,
                        {"exact_duplicates": "redundant_exact_evidence",
                         "semantic_duplicates": "redundant_semantic_evidence",
                         "augmentation_duplicates": "redundant_augmented_evidence"}.get(result.name, ""),
                        {},
                    ) or {}),
                }
            )

        duplicate_sections.append(
            {
                "phase": result.name,
                "removed": result.removed,
                "review": result.review,
                "notes": result.notes,
                "pairs": pairs,
            }
        )

    # Format cross contamination from Counter to a sorted list of dicts
    cross_contamination = [
        {"datasets": list(pair), "count": count}
        for pair, count in cross_contamination_counts.most_common()
    ]

    return {
        "removed_by_reason": removed_by_reason,
        "review_by_reason": review_by_reason,
        "duplicate_phases": duplicate_sections,
        "cross_contamination": cross_contamination,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _split_labels(sample: fo.Sample) -> list[str]:
    labels = list(sample_field(sample, "normalized_labels", []) or [])
    return sorted(set(labels)) or ["__unlabeled__"]


def allocate_group_splits(
    groups: dict[str, list[str]],
    proportions: dict[str, float],
    sizes: dict[str, int] | None = None,
) -> dict[str, str]:
    """Greedily minimizes global class and size error without breaking groups."""
    if set(proportions) != {"train", "val", "test"}:
        raise ValueError("Las proporciones deben definir train, val y test")
    if any(value < 0 for value in proportions.values()) or abs(sum(proportions.values()) - 1.0) > 1e-9:
        raise ValueError("Las proporciones de split deben ser no negativas y sumar 1")

    group_counts = {group: Counter(labels) for group, labels in groups.items()}
    group_sizes = sizes or {group: max(sum(counts.values()), 1) for group, counts in group_counts.items()}
    if set(group_sizes) != set(groups) or any(size < 1 for size in group_sizes.values()):
        raise ValueError("Cada grupo debe tener un tamaño positivo")
    total_size = sum(group_sizes.values())
    label_totals = Counter(label for labels in groups.values() for label in labels)
    target_sizes = {split: total_size * proportion for split, proportion in proportions.items()}
    target_labels = {
        split: {label: total * proportions[split] for label, total in label_totals.items()}
        for split in proportions
    }
    current_sizes = Counter()
    current_labels = {split: Counter() for split in proportions}

    def global_error(candidate_group: str, candidate_split: str) -> float:
        error = 0.0
        for split in proportions:
            size = current_sizes[split]
            if split == candidate_split:
                size += group_sizes[candidate_group]
            error += ((size - target_sizes[split]) ** 2) / max(target_sizes[split], 1.0)
            for label, target in target_labels[split].items():
                count = current_labels[split][label]
                if split == candidate_split:
                    count += group_counts[candidate_group][label]
                error += ((count - target) ** 2) / max(target, 1.0)
        return error

    rarity = {
        group: min(label_totals[label] for label in counts) if counts else total_size
        for group, counts in group_counts.items()
    }
    ordered_groups = sorted(
        groups,
        key=lambda group: (
            rarity[group],
            -group_sizes[group],
            hashlib.sha256(group.encode("utf-8")).hexdigest(),
        ),
    )
    assignments: dict[str, str] = {}
    for group in ordered_groups:
        split = min(
            proportions,
            key=lambda name: (
                global_error(group, name),
                hashlib.sha256(f"{group}:{name}".encode("utf-8")).hexdigest(),
            ),
        )
        assignments[group] = split
        current_sizes[split] += group_sizes[group]
        current_labels[split].update(group_counts[group])
    return assignments


def partition_dataset(
    dataset: fo.Dataset,
    train_p: float = 0.8,
    val_p: float = 0.1,
    test_p: float = 0.1,
    group_by_location: bool = False,
) -> dict[str, int]:
    """Creates deterministic group-aware and approximately stratified splits."""
    logger.info("Generando partición estratificada por grupos (Train/Val/Test)...")
    all_samples = list(dataset)
    groups = relation_groups(all_samples, captures=True, locations=group_by_location)
    eligible = [sample for sample in all_samples if current_status(sample) == "kept"]
    samples_by_group: dict[str, list[fo.Sample]] = defaultdict(list)
    labels_by_group: dict[str, list[str]] = defaultdict(list)
    for sample in eligible:
        group = groups[sample.id]
        samples_by_group[group].append(sample)
        labels_by_group[group].extend(_split_labels(sample))

    assignments = allocate_group_splits(
        dict(labels_by_group),
        {"train": train_p, "val": val_p, "test": test_p},
        sizes={group: len(samples) for group, samples in samples_by_group.items()},
    )
    sample_assignments = {
        sample.id: assignments[group]
        for group, samples in samples_by_group.items()
        for sample in samples
    }
    audit_assignments(all_samples, sample_assignments, locations=group_by_location)
    dataset.untag_samples(["train", "val", "test"])
    if "split_group_id" not in dataset.get_field_schema():
        dataset.add_sample_field("split_group_id", fo.StringField)
    dataset.set_values("split_group_id", groups, key_field="id")
    for sample in dataset.select(list(sample_assignments)).iter_samples(autosave=True):
        split = sample_assignments[sample.id]
        sample.tags = sorted(set(sample.tags + [split]))
    return dict(Counter(sample_assignments.values()))

def audit_duplicate_representatives(dataset, cache: DescriptorCache | None = None):
    samples = {sample.id: sample for sample in dataset}
    for sample in samples.values():
        curation = sample_field(sample, "curation")
        if current_status(sample) != "removed":
            continue
        if not getattr(curation, "representative_id", ""):
            if getattr(curation, "phase", "").endswith("duplicates"):
                raise RuntimeError(f"Descarte automático sin representante para {sample.id}")
            continue
        representative = samples.get(curation.representative_id)
        if representative is None or current_status(representative) != "kept":
            raise RuntimeError(f"Representante ausente o no conservado para {sample.id}")
        if cache is not None:
            from agrivision_khaos.deduplication import _compatible_annotations
            left = cache.describe(representative.filepath)
            right = cache.describe(sample.filepath)
            evidence = cache.verify(left, right, CurationPolicy().deduplication)
            if evidence["level"] not in EXACT_LEVELS or not _compatible_annotations(representative, sample, evidence):
                raise RuntimeError(f"Sustitución obsoleta o incompatible para {sample.id}; repite la detección")


def reconcile_visual_splits(
    paths: dict[str, str],
    assignments: dict[str, str],
    dataset: fo.Dataset,
    policy: DeduplicationPolicy,
    cache: DescriptorCache,
    rejected_pairs: Any = (),
) -> int:
    """Detects confirmed visual variants crossing splits and unifies their split to prevent data leakage."""
    audit_policy = policy.model_copy(
        update={
            "candidate_neighbors": min(200, policy.candidate_neighbors * 2),
            "candidate_pool": min(8192, policy.candidate_pool * 2),
        }
    )
    descriptors = {key: cache.describe(path) for key, path in paths.items()}
    metrics: dict[str, Any] = {}
    candidates, _ = candidate_pairs(descriptors, audit_policy, metrics)
    rejected = set(rejected_pairs)

    sample_to_group = {}
    for sample in dataset.select(list(assignments)):
        sample_to_group[sample.id] = sample_field(sample, "split_group_id", sample.id)

    group_to_samples = defaultdict(list)
    for sample_id, group_id in sample_to_group.items():
        group_to_samples[group_id].append(sample_id)

    split_priority = {"train": 0, "val": 1, "test": 2}
    reconciled = 0

    for left, right in candidates:
        if assignments.get(left) == assignments.get(right) or frozenset((left, right)) in rejected:
            continue
        if left not in assignments or right not in assignments:
            continue
        evidence = cache.verify(descriptors[left], descriptors[right], audit_policy)
        if evidence["level"] in CONFIRMED_LEVELS:
            left_group = sample_to_group.get(left, left)
            right_group = sample_to_group.get(right, right)
            left_split = assignments[left]
            right_split = assignments[right]
            target_split = left_split if split_priority.get(left_split, 9) <= split_priority.get(right_split, 9) else right_split

            all_members = set(group_to_samples[left_group] + group_to_samples[right_group])
            for member in all_members:
                assignments[member] = target_split

            merged_members = list(all_members)
            group_to_samples[left_group] = merged_members
            group_to_samples[right_group] = merged_members

            reconciled += 1
            logger.warning(
                "Fuga visual prevenida: unificando partición para pareja confirmada %s (%s) y %s (%s) en '%s'",
                left, left_split, right, right_split, target_split
            )

    if reconciled > 0:
        for sample in dataset.select(list(assignments)).iter_samples(autosave=True):
            if sample.id in assignments:
                split = assignments[sample.id]
                clean_tags = [t for t in sample.tags if t not in {"train", "val", "test"}]
                sample.tags = sorted(set(clean_tags + [split]))

    return reconciled


def _snapshot_export_view(view, images_dir: Path):
    """Point an unsaved view at the copied images without a large ID-to-path mapping."""
    parts = fo.ViewField("filepath").split("/")[-1].split(".")
    has_suffix = (
        (parts.length() > 1)
        & (parts[-1] != "")
        & ((parts.length() > 2) | (parts[0] != ""))
    )
    suffix = has_suffix.if_else(fo.ViewExpression(".").concat(parts[-1].lower()), "")
    return view.set_field(
        "filepath",
        fo.ViewExpression(str(images_dir.resolve()) + "/").concat(
            fo.ViewField("_id").to_string(), suffix
        ),
    )


def _link_view_media(view, directory: Path, statistics: dict):
    directory.mkdir(parents=True, exist_ok=True)
    for sample in view.select_fields("filepath"):
        source = Path(sample.filepath)
        mode = link_export_asset(source, directory / source.name)
        statistics[mode] += 1


def export_clean_dataset(
    dataset: fo.Dataset,
    export_dir: Path,
    output_formats: list[str],
    label_mapping: dict[str, Any],
    summary: dict[str, Any],
    policy: CurationPolicy | None = None,
    balance_classes: bool = False,
    balance_target: str = "median",
    cache_dir: Path | None = None,
) -> dict[str, str]:
    policy = policy or CurationPolicy()
    unknown_formats = sorted(set(output_formats) - SUPPORTED_OUTPUT_FORMATS)
    if not output_formats or unknown_formats:
        raise ValueError(
            "Formatos de salida inválidos: "
            + (", ".join(unknown_formats) if unknown_formats else "lista vacía")
        )
    if export_dir.exists() and any(export_dir.iterdir()):
        raise ValueError(f"La exportación requiere un directorio vacío: {export_dir}")
    export_dir.mkdir(parents=True, exist_ok=True)
    images_dir = export_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Aplicar partición zero-leak
    split_counts = partition_dataset(
        dataset,
        train_p=policy.splits.train,
        val_p=policy.splits.val,
        test_p=policy.splits.test,
        group_by_location=policy.splits.group_by_location,
    )
    summary["splits"] = split_counts
    audited_samples = list(dataset)
    assigned = {sample.id: next(tag for tag in sample.tags if tag in {"train", "val", "test"})
                for sample in audited_samples if current_status(sample) == "kept"}
    summary["split_audit"] = audit_assignments(
        audited_samples, assigned, locations=policy.splits.group_by_location
    )
    # DescriptorCache hashes the current bytes even on hits. Persistent reuse avoids
    # decoding/recomputing descriptors while keeping the content audit fresh.
    audit_cache_context = (
        nullcontext(cache_dir / "deduplication" / dataset.name / "augmentation-cache")
        if cache_dir is not None
        else tempfile.TemporaryDirectory(prefix="agrivision-split-audit-")
    )
    audited_assets = {}
    with audit_cache_context as audit_cache:
        content_cache = DescriptorCache(Path(audit_cache))
        audit_duplicate_representatives(dataset, content_cache)
        reconciled = reconcile_visual_splits(
            {sample.id: sample.filepath for sample in audited_samples if sample.id in assigned},
            assigned,
            dataset,
            policy.deduplication,
            content_cache,
            human_rejections(audited_samples),
        )
        if reconciled > 0:
            summary["splits"] = dict(Counter(assigned.values()))
            summary["split_audit"] = audit_assignments(
                audited_samples, assigned, locations=policy.splits.group_by_location
            )
        summary["visual_split_audit"] = audit_visual_splits(
            {sample.id: sample.filepath for sample in audited_samples if sample.id in assigned},
            assigned, policy.deduplication, content_cache,
            human_rejections(audited_samples),
            asset_digests=audited_assets,
        )
        summary["visual_split_audit"]["cache"] = {
            "descriptor_hits": content_cache.hits,
            "descriptor_misses": content_cache.misses,
            "pair_hits": content_cache.pair_hits,
        }
    if balance_classes:
        from agrivision_khaos.balancing import balance_dataset_classes
        logger.info("Aplicando balanceo de clases mediante aumentación agronómica...")
        aug_dir = (cache_dir / "augmented" / dataset.name) if cache_dir else (export_dir / "augmented")
        balance_res = balance_dataset_classes(
            dataset,
            target_strategy=balance_target,
            output_dir=aug_dir,
        )
        summary["class_balancing"] = asdict(balance_res)
        summary["splits"] = dict(Counter(
            next((tag for tag in s.tags if tag in {"train", "val", "test"}), "train")
            for s in dataset if current_status(s) == "kept"
        ))
    write_json(export_dir / "duplicate_evidence.json", {
        "version": ALGORITHM_VERSION,
        "samples": [{"id": sample.id, "stable_id": stable_identity(sample),
                     "source_path": sample_field(sample, "source_path", sample.filepath),
                     "links": list(sample_field(sample, "duplicate_links", []) or []),
                     "annotations": {
                         name: json.loads(label.to_json()) for name in
                         ("ground_truth_classification", "ground_truth_detections")
                         if (label := sample_field(sample, name)) is not None
                     },
                     "curation": sample_curation_payload(sample),
                     "history": sample_field(sample, "curation_history", []),
                     "review_history": sample_field(sample, "duplicate_review_history", [])} for sample in audited_samples],
        "analysis": dataset.info.get("augmentation_analysis", {}),
        "policy": policy.model_dump(mode="json"),
    })
    
    clean_view = dataset.match(fo.ViewField("curation.status") == "kept")
    if not len(clean_view):
        raise RuntimeError("No quedan muestras aprobadas; no se publicará un dataset vacío")

    # Unknown blur is not evidence that an annotation is unusable.
    has_detections = dataset.has_sample_field("ground_truth_detections")
    excluded_detection_ids = []
    if has_detections:
        logger.info(
            "Filtrando bounding boxes con blur_variance < %.2f...",
            policy.quality.min_box_blur,
        )
        keep_box = (
            (fo.ViewField("blur_variance") == None)  # noqa: E711 - FiftyOne expression
            | (fo.ViewField("blur_variance") >= policy.quality.min_box_blur)
        )
        box_count = fo.ViewField("ground_truth_detections.detections").if_null([]).length()
        excluded_detection_ids = (
            clean_view.match(box_count > 0)
            .filter_labels("ground_truth_detections", keep_box, only_matches=False)
            .match(box_count == 0)
            .values("id")
        )
        clean_view = clean_view.filter_labels(
            "ground_truth_detections",
            keep_box,
            only_matches=False,
        )
    exports: dict[str, str] = {}
    statistics = {
        "version": EXPORT_ALGORITHM_VERSION,
        "manifest_paths": "relative_to_export_root",
        "images": 0,
        "image_bytes": 0,
        "linked": 0,
        "copied": 0,
        "classification_images": 0,
        "detection_images_excluded_after_box_filter": len(excluded_detection_ids),
        "skipped_formats": {},
    }
    summary["export"] = statistics
    excluded_detection_ids = set(excluded_detection_ids)
    # Stream the manifests instead of retaining one large dictionary per image.
    with (
        (export_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as csv_file,
        (export_dir / "manifest.jsonl").open("w", encoding="utf-8") as jsonl_file,
        (export_dir / "checksums.sha256").open("w", encoding="utf-8") as checksums,
    ):
        writer = None
        for sample in clean_view:
            source_path = Path(sample.filepath)
            target_path = images_dir / export_filename(sample.id, source_path)
            digest, size = copy_verified_asset(source_path, target_path, audited_assets.get(sample.id))
            splits = set(sample.tags) & {"train", "val", "test"}
            if not splits and sample_field(sample, "is_synthetic", False):
                splits = {"train"} if sample_field(sample, "split") == "train" else set()
            if len(splits) != 1:
                raise RuntimeError(f"La muestra {sample.id} no tiene una partición única")
            row = {
                "id": sample.id,
                "filepath": target_path.relative_to(export_dir).as_posix(),
                "source_path": str(sample_field(sample, "source_path", "")),
                "source_dataset": str(sample_field(sample, "source_dataset", "")),
                "source_split": str(sample_field(sample, "source_split", "")),
                "assigned_split": next(iter(splits)),
                "source_label": str(sample_field(sample, "source_label", "")),
                "normalized_label": str(sample_field(sample, "normalized_label", "")),
                "task_type": str(sample_field(sample, "task_type", "unlabeled")),
                "normalized_labels": ",".join(sample_field(sample, "normalized_labels", []) or []),
                "asset_sha256": digest,
                "size_bytes": size,
                "detection_exported": (
                    sample_field(sample, "ground_truth_detections") is not None
                    and sample.id not in excluded_detection_ids
                    and bool(set(output_formats) & {"coco", "yolo"})
                ),
                "family_id": str(sample_field(sample, "duplicate_family_id", "")),
                "split_group_id": str(sample_field(sample, "split_group_id", "")),
                "representative_id": str(getattr(sample_field(sample, "curation"), "representative_id", "")),
                "curation_reason": str(getattr(sample_field(sample, "curation"), "reason", "")),
                "deduplication_version": ALGORITHM_VERSION,
                "source_version": str(sample_field(sample, "source_version", "unknown")),
                "source_license": str(sample_field(sample, "source_license", "unknown")),
            }
            if writer is None:
                writer = csv.DictWriter(csv_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            checksums.write(f"{digest}  {row['filepath']}\n")
            statistics["images"] += 1
            statistics["image_bytes"] += size
            if "classification" in output_formats and "classification" in row["task_type"].split(","):
                label = slugify(row["normalized_label"] or "unlabeled")
                label_dir = export_dir / "classification" / row["assigned_split"] / label
                mode = link_export_asset(target_path, label_dir / target_path.name)
                statistics[mode] += 1
                statistics["classification_images"] += 1

    write_json(export_dir / "label_mapping.json", label_mapping)
    exports["manifest"] = str(export_dir)
    # All format exporters read this fixed copy; the original dataset is unchanged.
    snapshot_view = _snapshot_export_view(clean_view, images_dir)
    if "fiftyone" in output_formats:
        try:
            snapshot_view.export(
                export_dir=str(export_dir / "fiftyone"),
                dataset_type=fo.types.FiftyOneDataset,
                export_media=True,
            )
            exports["fiftyone"] = str(export_dir / "fiftyone")
        except Exception as exc:
            exports["fiftyone_error"] = str(exc)

    if "classification" in output_formats:
        if statistics["classification_images"]:
            exports["classification"] = str(export_dir / "classification")
        else:
            statistics["skipped_formats"]["classification"] = "No hay muestras de clasificación"

    detection_view = (
        snapshot_view.exists("ground_truth_detections").exclude(list(excluded_detection_ids))
        if has_detections else None
    )
    detection_count = len(detection_view) if detection_view is not None else 0
    classes = (
        sorted(detection_view.distinct("ground_truth_detections.detections.label"))
        if detection_count else []
    )
    for format_name in ("coco", "yolo"):
        if format_name in output_formats and not detection_count:
            statistics["skipped_formats"][format_name] = "No hay muestras de detección exportables"

    if "coco" in output_formats and detection_count:
        try:
            for split in ("train", "val", "test"):
                split_view = detection_view.match_tags(split)
                if not len(split_view):
                    continue
                split_view.export(
                    export_dir=str(export_dir / "coco" / split),
                    dataset_type=fo.types.COCODetectionDataset,
                    label_field="ground_truth_detections",
                    export_media=False,
                    classes=classes,
                )
                _link_view_media(split_view, export_dir / "coco" / split / "data", statistics)
            exports["coco"] = str(export_dir / "coco")
        except Exception as exc:
            exports["coco_error"] = str(exc)

    if "yolo" in output_formats and detection_count:
        try:
            yolo_dir = export_dir / "yolo"
            yolo_config = {"names": dict(enumerate(classes))}
            for split in ("train", "val", "test"):
                split_view = detection_view.match_tags(split)
                if not len(split_view):
                    continue
                split_view.export(
                    export_dir=str(yolo_dir),
                    dataset_type=fo.types.YOLOv5Dataset,
                    label_field="ground_truth_detections",
                    export_media=False,
                    split=split,
                    classes=classes,
                )
                _link_view_media(split_view, yolo_dir / "images" / split, statistics)
                yolo_config[split] = f"./images/{split}/"
            (yolo_dir / "dataset.yaml").write_text(
                yaml.safe_dump(yolo_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
            exports["yolo"] = str(yolo_dir)
        except Exception as exc:
            exports["yolo_error"] = str(exc)

    if "datumaro" in output_formats:
        datumaro_type = getattr(fo.types, "DatumaroDataset", None)
        if datumaro_type is None:
            exports["datumaro_error"] = "FiftyOne no expone DatumaroDataset en esta version."
        else:
            try:
                snapshot_view.export(
                    export_dir=str(export_dir / "datumaro"),
                    dataset_type=datumaro_type,
                    export_media=True,
                )
                exports["datumaro"] = str(export_dir / "datumaro")
            except Exception as exc:
                exports["datumaro_error"] = str(exc)

    write_json(export_dir / "curation_summary.json", summary)
    if not any(format_name in exports for format_name in output_formats):
        errors = {key: value for key, value in exports.items() if key.endswith("_error")}
        raise RuntimeError(
            f"No se generó ningún formato solicitado: {errors or statistics['skipped_formats']}"
        )
    return exports


def compute_class_weights(dataset: fo.Dataset) -> dict[str, float]:
    counts = Counter(
        sample_field(sample, "normalized_label")
        for sample in dataset
        if current_status(sample) == "kept" and sample_field(sample, "normalized_label")
    )
    total_valid = sum(counts.values())
    n_classes = len(counts)
    if n_classes == 0:
        return {}
        
    weights = {}
    for cls, count in counts.items():
        if count > 0:
            weights[cls] = round(total_valid / (n_classes * count), 4)
    
    # Ordenar por label
    return dict(sorted(weights.items()))


def build_summary(
    dataset: fo.Dataset,
    dataset_name: str,
    run_id: str,
    raw_dir: Path,
    sources: list[dict[str, Any]],
    tools: dict[str, dict[str, str]],
    phases: list[PhaseResult],
    label_mapping: dict[str, Any],
) -> dict[str, Any]:
    counts = status_counts(dataset)
    class_weights = compute_class_weights(dataset)
    phase_payloads = []
    for phase in phases:
        payload = asdict(phase)
        pairs = payload.pop("duplicate_pairs", [])
        payload["duplicate_pair_count"] = len(pairs)
        phase_payloads.append(payload)

    return {
        "dataset": dataset_name,
        "run_id": run_id,
        "raw_dir": str(raw_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "counts": {
            "initial": len(dataset),
            "kept": counts.get("kept", 0),
            "review": counts.get("review", 0),
            "removed": counts.get("removed", 0),
        },
        "sources": sources,
        "tools": tools,
        "phases": phase_payloads,
        "groups": grouped_counts(dataset),
        "label_mapping": label_mapping,
        "class_weights": class_weights,
    }


def stage_datumaro_inventory(interim_dir: Path, sources: list[dict[str, Any]], tools: dict[str, dict[str, str]]) -> None:
    datumaro_dir = interim_dir / "datumaro"
    datumaro_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        datumaro_dir / "source_inventory.json",
        {
            "status": tools["datumaro"]["status"],
            "note": (
                "Datumaro esta disponible para conversion/export posterior."
                if tools["datumaro"]["available"]
                else "Datumaro no esta instalado; FiftyOne actua como staging canonico."
            ),
            "sources": sources,
        },
    )


def parse_output_formats(value: str) -> list[str]:
    formats = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = sorted(set(formats) - SUPPORTED_OUTPUT_FORMATS)
    if not formats or unknown:
        raise ValueError(
            "Formatos de salida inválidos: "
            + (", ".join(unknown) if unknown else "lista vacía")
            + f". Permitidos: {', '.join(sorted(SUPPORTED_OUTPUT_FORMATS))}"
        )
    return formats


def load_policy(path: str | Path) -> CurationPolicy:
    policy_path = Path(path)
    try:
        payload = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
        return CurationPolicy.model_validate(payload)
    except (OSError, ValueError, ValidationError) as exc:
        raise ValueError(f"Política de curación inválida {policy_path}: {exc}") from exc


def _restore_phase(payload: dict[str, Any]) -> PhaseResult:
    return PhaseResult(**payload)


def _pipeline_fingerprint(args, policy: CurationPolicy, raw_dir: Path) -> str:
    """Identify result-affecting settings, including CLI overrides and ontology content."""
    from agrivision_khaos.quality import QUALITY_ALGORITHM_VERSION

    configuration = policy.model_dump(mode="json")
    enable_ocr = getattr(args, "enable_ocr", None)
    if enable_ocr is not None:
        configuration["quality"]["ocr_enabled"] = enable_ocr
    ontology = getattr(args, "ontology_map", None)
    ontology_identity = None
    if ontology:
        ontology_path = Path(ontology).resolve()
        ontology_identity = {
            "path": str(ontology_path),
            "sha256": hashlib.sha256(ontology_path.read_bytes()).hexdigest(),
        }
    return source_fingerprint(raw_dir, {
        **configuration,
        "quality_version": QUALITY_ALGORITHM_VERSION,
        "deduplication_version": ALGORITHM_VERSION,
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "export_version": EXPORT_ALGORITHM_VERSION,
        "ontology_map": ontology_identity,
        "execution": {
            "skip_quality": bool(getattr(args, "skip_quality", False)),
            "skip_duplicates": bool(getattr(args, "skip_duplicates", False)),
            "skip_labels": bool(getattr(args, "skip_labels", False)),
            "cleanlab_mode": getattr(args, "cleanlab_mode", "auto"),
            "max_phase_drop": getattr(args, "max_phase_drop", 0.40),
            "max_total_drop": getattr(args, "max_total_drop", 0.65),
            "output_formats": sorted(parse_output_formats(args.output_formats)),
            "export_dir": str(Path(args.export_dir).resolve()),
            "report_dir": str(Path(args.report_dir).resolve()),
            "require_gpu": bool(getattr(args, "require_gpu", False)),
            "require_read_only": bool(getattr(args, "require_read_only", False)),
            "minimum_free_gb": getattr(args, "minimum_free_gb", 5.0),
            "balance_classes": bool(getattr(args, "balance_classes", False)),
            "balance_target": str(getattr(args, "balance_target", "median") or "median"),
        },
    })


def _execute_pipeline(
    args: argparse.Namespace,
    policy: CurationPolicy,
    raw_dir: Path,
    dataset_name: str,
    requested_run_id: str,
) -> None:
    from agrivision_khaos.ingest import create_unified_dataset
    from agrivision_khaos.quality import compute_dataset_quality

    cache_dir = Path(args.cache_dir)
    fingerprint = _pipeline_fingerprint(args, policy, raw_dir)
    effective_fingerprint = fingerprint
    if not args.resume:
        effective_fingerprint = hashlib.sha256(
            f"{fingerprint}:{requested_run_id}".encode()
        ).hexdigest()
    checkpoint_path = cache_dir / "runs" / dataset_name / "checkpoint.json"

    with PipelineLock(cache_dir / "locks" / f"{slugify(dataset_name)}.lock"):
        checkpoint = RunCheckpoint(
            checkpoint_path,
            effective_fingerprint,
            requested_run_id,
        )
        run_id = checkpoint.run_id
        report_dir = Path(args.report_dir) / dataset_name / run_id
        export_dir = Path(args.export_dir) / dataset_name / run_id
        interim_dir = cache_dir / "interim" / dataset_name / run_id

        if args.resume and checkpoint.data.get("status") == "completed":
            result = checkpoint.data.get("result", {})
            success_marker = Path(str(result.get("success_marker", "")))
            if success_marker.is_file():
                logger.info(
                    "[bold yellow]El pipeline detectó que este dataset ya fue procesado con esta misma configuración y huella de datos (Run: %s).[/bold yellow]",
                    run_id,
                )
                logger.info("  • Resultado previo: %s", success_marker.parent)
                logger.info("  • Para forzar una re-ejecución limpia: ejecuta con [bold green]RESUME=0[/bold green] (ej. make pipeline RESUME=0)")
                logger.info("  • Para aplicar ontología o balanceo: define ONTOLOGY y BALANCE_CLASSES en .env o pásalos en el comando:")
                logger.info("      make pipeline RESUME=0 ONTOLOGY=\"reports/.../proposed_ontology.yaml\" BALANCE_CLASSES=1")
                logger.info("  • Para explorar las imágenes en FiftyOne: [bold green]make app DATASET=\"%s\"[/bold green]", dataset_name)
                return

        report_dir.mkdir(parents=True, exist_ok=True)
        interim_dir.mkdir(parents=True, exist_ok=True)

        preflight = run_preflight(
            raw_dir,
            Path(args.export_dir),
            cache_dir,
            database_uri=os.environ.get("FIFTYONE_DATABASE_URI"),
            require_read_only=args.require_read_only,
            require_gpu=args.require_gpu,
            minimum_free_gb=args.minimum_free_gb,
        )
        write_json(report_dir / "preflight.json", preflight)
        if not preflight["valid"]:
            raise RuntimeError(
                "Preflight fallido: " + ", ".join(preflight["failed_checks"])
            )

        logger.info("=== INICIANDO PIPELINE DESATENDIDO ===")
        logger.info(
            "Dataset=%s RAW_DIR=%s RUN_ID=%s FINGERPRINT=%s",
            dataset_name,
            raw_dir,
            run_id,
            fingerprint[:16],
        )

        tools = optional_tool_status()
        sources = discover_sources(raw_dir)
        stage_datumaro_inventory(interim_dir, sources, tools)
        phases: list[PhaseResult] = []
        ingestion_payload = checkpoint.phase_payload("ingestion") if args.resume else None
        can_resume_dataset = (
            ingestion_payload is not None
            and fo.dataset_exists(dataset_name)
            and len(fo.load_dataset(dataset_name)) == ingestion_payload.get("sample_count")
            and fo.load_dataset(dataset_name).info.get("run_fingerprint")
            == effective_fingerprint
        )
        if can_resume_dataset:
            dataset = fo.load_dataset(dataset_name)
            phases.append(_restore_phase(ingestion_payload["phase"]))
            logger.info("Checkpoint de ingesta reutilizado (%d muestras).", len(dataset))
        else:
            dataset = create_unified_dataset(
                dataset_name,
                raw_dir,
                staging_dir=interim_dir / "ingestion",
                strict=True,
            )
            if len(dataset) == 0:
                raise RuntimeError(f"No se ingestaron imagenes desde {raw_dir}")
            annotate_sources(dataset, raw_dir, sources)
            dataset.info["run_fingerprint"] = effective_fingerprint
            dataset.save()
            phase = PhaseResult(
                name="ingestion",
                kept=len(dataset),
                notes=[f"{len(sources)} fuentes detectadas."],
            )
            phases.append(phase)
            checkpoint.mark_phase(
                "ingestion", {"sample_count": len(dataset), "phase": asdict(phase)}
            )

        effective_enable_ocr = (
            args.enable_ocr if getattr(args, "enable_ocr", None) is not None
            else policy.quality.ocr_enabled
        )
        if getattr(args, "skip_quality", False):
            logger.info("Fase de calidad omitida por configuración (--skip-quality).")
            phase = PhaseResult(name="quality", kept=len(dataset), notes=["Omitida por configuración."])
            phases.append(phase)
        else:
            quality_payload = checkpoint.phase_payload("quality") if args.resume else None
            if quality_payload is not None and can_resume_dataset:
                phases.append(_restore_phase(quality_payload["phase"]))
                logger.info("Checkpoint de métricas de calidad reutilizado.")
            else:
                logger.info("Calculando metricas visuales...")
                compute_dataset_quality(
                    dataset_name=dataset_name,
                    workers=args.workers,
                    enable_ocr=effective_enable_ocr,
                    ocr_confidence=policy.quality.ocr_confidence,
                    ocr_timeout_seconds=policy.quality.ocr_timeout_seconds,
                    min_valid_size=policy.quality.min_resolution,
                )
                dataset = fo.load_dataset(dataset_name)
                phase = run_quality_phase(
                    dataset,
                    args.max_phase_drop,
                    args.max_total_drop,
                    policy=policy.quality,
                )
                phases.append(phase)
                checkpoint.mark_phase("quality", {"phase": asdict(phase)})

        if getattr(args, "skip_duplicates", False):
            logger.info("Fase de deduplicación omitida por configuración (--skip-duplicates).")
            duplicate_results = [
                PhaseResult(name="duplicates", kept=len(dataset), notes=["Omitida por configuración."])
            ]
        else:
            duplicates_payload = checkpoint.phase_payload("duplicates") if args.resume else None
            if duplicates_payload is not None and can_resume_dataset:
                duplicate_results = [
                    _restore_phase(phase) for phase in duplicates_payload["phases"]
                ]
                logger.info("Checkpoint de deduplicación reutilizado.")
            else:
                duplicate_results = run_duplicate_phases(
                    dataset,
                    work_dir=cache_dir / "deduplication" / dataset_name,
                    max_phase_drop=args.max_phase_drop,
                    max_total_drop=args.max_total_drop,
                    policy=policy,
                )
                checkpoint.mark_phase(
                    "duplicates", {"phases": [asdict(phase) for phase in duplicate_results]}
                )
        phases.extend(duplicate_results)

        if getattr(args, "skip_labels", False):
            logger.info("Fase de ontología/etiquetas omitida por configuración (--skip-labels).")
            label_phase = PhaseResult(name="labels", kept=len(dataset), notes=["Omitida por configuración."])
            label_mapping = {}
        else:
            labels_payload = checkpoint.phase_payload("labels") if args.resume else None
            if labels_payload is not None and can_resume_dataset:
                label_phase = _restore_phase(labels_payload["phase"])
                label_mapping = labels_payload["mapping"]
                logger.info("Checkpoint de ontología/etiquetas reutilizado.")
            else:
                label_phase, label_mapping = run_label_phase(
                    dataset, args.cleanlab_mode, args.ontology_map, report_dir
                )
                checkpoint.mark_phase(
                    "labels", {"phase": asdict(label_phase), "mapping": label_mapping}
                )
        phases.append(label_phase)
        phases.append(reconcile_duplicate_representatives(dataset))

        evidence = build_curation_evidence(dataset, duplicate_results)
        summary = build_summary(
            dataset=dataset,
            dataset_name=dataset_name,
            run_id=run_id,
            raw_dir=raw_dir,
            sources=sources,
            tools=tools,
            phases=phases,
            label_mapping=label_mapping,
        )
        summary["evidence"] = evidence
        summary["policy"] = policy.model_dump(mode="json")
        summary["source_fingerprint"] = fingerprint
        summary["preflight"] = preflight

        export_dir.parent.mkdir(parents=True, exist_ok=True)
        if export_dir.exists():
            raise RuntimeError(
                f"El destino final ya existe sin checkpoint completo: {export_dir}"
            )
        attempt_dir = Path(
            tempfile.mkdtemp(prefix=f".{run_id}.incomplete-", dir=export_dir.parent)
        )
        # mkdtemp intentionally creates mode 0700.  Exported datasets commonly
        # cross the container/host boundary, so keep the staging directory
        # private only until its name is allocated and then make the eventual
        # atomic result traversable by non-root host users.
        attempt_dir.chmod(0o755)
        exports = export_clean_dataset(
            dataset=dataset,
            export_dir=attempt_dir,
            output_formats=parse_output_formats(args.output_formats),
            label_mapping=label_mapping,
            summary=summary,
            policy=policy,
            balance_classes=getattr(args, "balance_classes", False),
            balance_target=getattr(args, "balance_target", "median"),
            cache_dir=cache_dir,
        )
        export_errors = {
            key: value for key, value in exports.items() if key.endswith("_error")
        }
        if export_errors:
            raise RuntimeError(f"Falló la exportación transaccional: {export_errors}")
        exports = {
            key: str(export_dir / Path(value).relative_to(attempt_dir))
            for key, value in exports.items()
        }
        summary["exports"] = exports
        write_reports(report_dir, dataset_name, run_id, summary, evidence)
        write_json(attempt_dir / "curation_summary.json", summary)
        write_dataset_card(attempt_dir / "DATASET_CARD.md", summary)
        success_payload = {
            "dataset": dataset_name,
            "run_id": run_id,
            "source_fingerprint": fingerprint,
        }
        write_json(attempt_dir / "_SUCCESS", success_payload)
        os.replace(attempt_dir, export_dir)
        success_marker = export_dir / "_SUCCESS"
        checkpoint.mark_phase("export", {"path": str(export_dir)})
        checkpoint.mark_complete(
            {"export_dir": str(export_dir), "success_marker": str(success_marker)}
        )

        # Registrar vistas guardadas estándar en FiftyOne para facilitar la navegación
        try:
            from agrivision_khaos.export import update_curation_views
            update_curation_views(dataset)
        except Exception as exc:
            logger.warning("No se pudieron registrar las vistas guardadas en FiftyOne: %s", exc)

        n_kept = len(dataset.match(fo.ViewField("curation.status") == "kept"))
        n_review = len(dataset.match(fo.ViewField("curation.status") == "review"))
        n_removed = len(dataset.match(fo.ViewField("curation.status") == "removed"))

        logger.info("")
        logger.info("[bold green]================================================================================[/bold green]")
        logger.info("[bold green]                       PIPELINE FINALIZADO CON ÉXITO                            [/bold green]")
        logger.info("[bold green]================================================================================[/bold green]")
        logger.info("  • Reporte HTML : [bold cyan]%s[/bold cyan]", report_dir / "report.html")
        logger.info("  • Exportación  : [bold cyan]%s[/bold cyan]", export_dir)
        logger.info("")
        logger.info("[bold cyan]--------------------------------------------------------------------------------[/bold cyan]")
        logger.info("[bold cyan]               ETAPA SIGUIENTE: EXPLORACIÓN Y AUDITORÍA EN FIFTYONE             [/bold cyan]")
        logger.info("[bold cyan]--------------------------------------------------------------------------------[/bold cyan]")
        logger.info("1. [bold white]Iniciar la aplicación visual:[/bold white]")
        logger.info("   $ [bold green]make app DATASET=\"%s\"[/bold green]  -> Abre tu navegador en [bold cyan]http://localhost:5151[/bold cyan]", dataset_name)
        logger.info("")
        logger.info("2. [bold white]Filtrar mediante las Vistas Guardadas (menú superior 'Saved Views'):[/bold white]")
        logger.info("   • [bold green]01_Exportadas_Kept[/bold green]    : %d imágenes limpias (superaron filtros y fueron exportadas)", n_kept)
        logger.info("   • [bold yellow]02_En_Revision_Review[/bold yellow]  : %d imágenes dudosas (pendientes de confirmación humana)", n_review)
        logger.info("   • [bold red]03_Descartadas_Removed[/bold red] : %d imágenes descartadas (duplicados, borrosas, baja calidad)", n_removed)
        logger.info("")
        logger.info("3. [bold white]Auditar y tomar decisiones (Human-in-the-Loop):[/bold white]")
        logger.info("   • Selecciona muestras en FiftyOne y pulsa la tecla [bold cyan]'t'[/bold cyan] (o icono etiqueta 🏷️):")
        logger.info("     - Tag [bold green]'kept'[/bold green]    -> Aprobar caso dudoso O RECUPERAR falso positivo descartado.")
        logger.info("     - Tag [bold red]'removed'[/bold red] -> Confirmar descarte de caso dudoso.")
        logger.info("")
        logger.info("4. [bold white]Persistir decisiones tomadas en FiftyOne:[/bold white]")
        logger.info("   • Para guardar en FiftyOne DB sin re-exportar: $ [bold green]make sync-reviews DATASET=\"%s\"[/bold green]", dataset_name)
        logger.info("   • Para exportar el dataset limpio a disco:     $ [bold green]make export DATASET=\"%s\"[/bold green]", dataset_name)
        logger.info("")
        logger.info("  => Guía completa documentada: [bold underline]docs/workflow/4_manual_review.md[/bold underline]")
        logger.info("[bold green]================================================================================[/bold green]")
        logger.info("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline desatendido de curacion y unificacion.")
    parser.add_argument("--dataset", required=True, help="Nombre del dataset unificado.")
    parser.add_argument("--raw-dir", required=True, help="Directorio con datasets crudos.")
    parser.add_argument("--profile", default="quality-first", choices=["quality-first"])
    parser.add_argument("--policy", default="configs/quality-first.yaml")
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--output-formats", default="coco,yolo,classification")
    parser.add_argument("--cleanlab-mode", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--export-dir", default="/datasets/processed")
    parser.add_argument("--report-dir", default="reports/pipeline")
    parser.add_argument("--cache-dir", default="/datasets/cache")
    parser.add_argument("--ontology-map", default=None, help="Ruta al archivo YAML de ontologia.")
    parser.add_argument("--max-phase-drop", type=float, default=0.40)
    parser.add_argument("--max-total-drop", type=float, default=0.65)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reutiliza checkpoints compatibles; usa --no-resume para forzar un run nuevo.",
    )
    parser.add_argument("--require-read-only", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--minimum-free-gb", type=float, default=5.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida fuentes y simula el plan sin MongoDB, modelos ni exportaciones.",
    )
    parser.add_argument(
        "--dry-run-report",
        default=None,
        help="Ruta opcional del JSON; por defecto se guarda dentro de --report-dir.",
    )
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="Omite la fase de evaluación de calidad visual.",
    )
    parser.add_argument(
        "--skip-duplicates",
        action="store_true",
        help="Omite la fase de detección y resolución de duplicados.",
    )
    parser.add_argument(
        "--skip-labels",
        action="store_true",
        help="Omite la fase de auditoría de etiquetas y Cleanlab.",
    )
    parser.add_argument(
        "--enable-ocr",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Activa (--enable-ocr) o desactiva (--no-enable-ocr) explícitamente el filtro OCR.",
    )
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="Equilibra clases minoritarias mediante aumentación agronómica sintética antes de exportar.",
    )
    parser.add_argument(
        "--balance-target",
        default="median",
        help="Estrategia de balanceo de clases: 'median', 'max', 'mean' o número entero de muestras.",
    )
    args = parser.parse_args()
    policy = load_policy(args.policy)

    raw_dir = Path(args.raw_dir)
    dataset_name = args.dataset
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    try:
        parse_output_formats(args.output_formats)
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        logger.info("=== DRY-RUN ESTÁTICO: NO SE EJECUTARÁN MODELOS NI MONGODB ===")
        report = audit_raw_datasets(raw_dir)
        report.update(
            {
                "dataset": dataset_name,
                "run_id": run_id,
                "policy": policy.model_dump(mode="json"),
                "planned_actions": [
                    "ingestar y canonicalizar las fuentes detectadas",
                    "calcular métricas visuales (omitido en dry-run)",
                    "deduplicar por hash y similitud (solo hash simulado)",
                    "aplicar revisión de etiquetas (omitido en dry-run)",
                    f"exportar: {', '.join(parse_output_formats(args.output_formats))}",
                ],
            }
        )
        report_path = (
            Path(args.dry_run_report)
            if args.dry_run_report
            else Path(args.report_dir) / dataset_name / run_id / "dry_run.json"
        )
        write_json(report_path, report)
        counts = report["counts"]
        logger.info(
            "Fuentes=%s Imágenes=%s Anotaciones=%s Errores=%s Avisos=%s",
            counts["sources"], counts["images"], counts["annotations"],
            counts["errors"], counts["warnings"],
        )
        for issue in report["issues"]:
            log_method = logger.error if issue["severity"] == "error" else logger.warning
            log_method("[%s] %s: %s", issue["code"], issue["path"], issue["message"])
        logger.info("Reporte dry-run: %s", report_path)
        if not report["valid"]:
            raise SystemExit(2)
        return

    _execute_pipeline(args, policy, raw_dir, dataset_name, run_id)


if __name__ == "__main__":
    main()
