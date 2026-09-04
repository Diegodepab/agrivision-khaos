from __future__ import annotations

import argparse
import base64
import copy
import csv
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
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import fiftyone as fo
import yaml
from pydantic import ValidationError
from rich.logging import RichHandler

from agrivision_khaos.dry_run import audit_raw_datasets
from agrivision_khaos.execution import PipelineLock, RunCheckpoint, source_fingerprint
from agrivision_khaos.models import CurationPolicy, QualityPolicy, SourceManifest
from agrivision_khaos.preflight import run_preflight

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
            kept_b64 = get_base64_image(str(kept.get("filepath", "")))
            removed_b64 = get_base64_image(str(removed.get("filepath", "")))
            if not kept_b64 or not removed_b64:
                continue
            import os
            kept_filename = os.path.basename(str(kept.get("filepath", "")))
            removed_filename = os.path.basename(str(removed.get("filepath", "")))
            
            kept_metrics = f"<span class='badge'>blur {format_report_value(kept.get('blur_variance'))}</span> <span class='badge'>res {format_report_value(kept.get('width'))}x{format_report_value(kept.get('height'))}</span>"
            removed_metrics = f"<span class='badge'>blur {format_report_value(removed.get('blur_variance'))}</span> <span class='badge'>res {format_report_value(removed.get('width'))}x{format_report_value(removed.get('height'))}</span>"
            pair_cards.append(
                '<div class="example-pair" style="display:flex;gap:10px;border:1px solid #e2e8f0;border-radius:12px;padding:12px;background:#ffffff;box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);">'
                f'<div style="flex:1;overflow:hidden;"><img src="data:image/jpeg;base64,{kept_b64}" alt="" style="width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:8px;">'
                f'<span class="status-kept" style="display:block;margin-top:8px;font-size:14px;">Conservada</span>'
                f'<small style="display:block;color:#64748b;margin-top:2px;">{html.escape(str(kept.get("source_dataset", "")))} | {html.escape(str(kept.get("label", "")))}</small>'
                f'<code style="display:block;margin-top:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="{html.escape(kept_filename)}">{html.escape(kept_filename)}</code>'
                f'<div style="margin-top:6px;">{kept_metrics}</div></div>'
                f'<div style="flex:1;overflow:hidden;"><img src="data:image/jpeg;base64,{removed_b64}" alt="" style="width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:8px;">'
                f'<span class="status-removed" style="display:block;margin-top:8px;font-size:14px;">Eliminada</span>'
                f'<small style="display:block;color:#64748b;margin-top:2px;">{html.escape(str(removed.get("source_dataset", "")))} | {html.escape(str(removed.get("label", "")))}</small>'
                f'<code style="display:block;margin-top:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="{html.escape(removed_filename)}">{html.escape(removed_filename)}</code>'
                f'<div style="margin-top:6px;">{removed_metrics}</div></div>'
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
    .hero {{ background: #0f172a; color: white; padding: 40px; border-radius: 16px; margin-bottom: 32px; box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1); }}
    .hero h1 {{ margin: 0 0 8px; font-size: 2.5em; letter-spacing: -0.02em; }}
    .hero p {{ margin: 0; color: #94a3b8; font-size: 1.1em; }}
    h2 {{ margin-top: 40px; padding-bottom: 12px; border-bottom: 1px solid #e2e8f0; color: #0f172a; font-weight: 600; letter-spacing: -0.01em; }}
    details.reason-group {{ margin: 16px 0; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; background: #ffffff; box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1); transition: all 0.2s; }}
    details.reason-group:hover {{ box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }}
    details.reason-group > summary {{ cursor: pointer; list-style: none; font-weight: 600; color: #334155; }}
    
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin-top: 24px; }}
    .stat {{ background: rgba(255, 255, 255, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 12px; padding: 20px; }}
    .stat strong {{ display: block; font-size: 32px; font-weight: 700; color: white; }}
    .stat span {{ color: #94a3b8; font-size: 0.9em; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; display: block; }}
    
    .progress-bar {{ display: flex; height: 8px; border-radius: 4px; overflow: hidden; margin-top: 24px; background: #334155; }}
    .progress-kept {{ background: #10b981; }}
    .progress-review {{ background: #f59e0b; }}
    .progress-removed {{ background: #ef4444; }}

    table {{ width: 100%; border-collapse: separate; border-spacing: 0; margin-top: 16px; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1); border: 1px solid #e2e8f0; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 12px 16px; text-align: left; }}
    th {{ background: #f8fafc; font-weight: 600; color: #475569; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.05em; }}
    tr:last-child td {{ border-bottom: none; }}
    
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; margin-top: 16px; }}
    .grid-pairs {{ display: flex; flex-wrap: wrap; gap: 16px; margin-top: 16px; }}
    .example {{ border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px; background: #ffffff; box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05); }}
    .example img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 8px; margin-bottom: 8px; }}
    .example span, .example small {{ display: block; color: #64748b; margin-top: 4px; }}
    .example strong {{ color: #0f172a; }}
    
    code {{ background: rgba(0, 0, 0, 0.3); padding: 4px 8px; border-radius: 6px; font-family: monospace; color: #e2e8f0; font-size: 0.9em; }}
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
      <div class="stat"><strong style="color: #34d399;">{summary['counts']['kept']:,}</strong><span>Exportadas</span></div>
      <div class="stat"><strong style="color: #fbbf24;">{summary['counts']['review']:,}</strong><span>En Revisión</span></div>
      <div class="stat"><strong style="color: #f87171;">{summary['counts']['removed']:,}</strong><span>Descartadas</span></div>
    </section>
    
    <div style="margin-top: 24px; padding: 16px; background: rgba(255, 255, 255, 0.05); border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.1);">
      <h3 style="margin: 0 0 12px; font-size: 1em; color: white;">Resumen de Fases</h3>
      <ul style="margin: 0; padding-left: 20px; color: #cbd5e1; font-size: 0.9em; line-height: 1.5;">
        {''.join(
            f"<li><strong>{html.escape(p['name'])}</strong>"
            f"{': ' + str(p['removed']) + ' descartadas | ' + str(p['review']) + ' en revisión | ' + str(p['kept']) + ' conservadas' if (p['removed'] > 0 or p['review'] > 0 or p['kept'] > 0) else ''}"
            f"<br><span style='color:#94a3b8;'>{'<br>'.join(html.escape(n) for n in p['notes'])}</span></li>"
            for p in summary["phases"]
        )}
      </ul>
    </div>
  </div>
  
  <div style="max-width: 1280px; margin: 0 auto 32px; background: #ffffff; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);">
    <h3 style="margin-top: 0;">Sugerencia de Balanceo (Class Weights)</h3>
    <p style="color: #64748b; font-size: 0.9em; margin-bottom: 16px;">Copia y pega este fragmento en tu código de entrenamiento para penalizar matemáticamente las clases mayoritarias y ayudar a la red a converger de forma equitativa.</p>
    <div style="background: #0f172a; padding: 16px; border-radius: 8px; overflow-x: auto;">
      <code style="color: #e2e8f0; background: transparent; font-size: 1em; padding: 0;">
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
                and decision.confidence < HARD_CONFIDENCE
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

    for sample in dataset.select(list(decisions)).iter_samples(progress=True, autosave=True):
        decision = decisions[sample.id]
        sample["curation"] = fo.DynamicEmbeddedDocument(**asdict(decision))
        state_tags = {"curation_kept", "curation_review", "curation_removed"}
        retained_tags = [tag for tag in sample.tags if tag not in state_tags]
        sample.tags = sorted(
            set(retained_tags + [f"curation_{decision.status}", f"{tag_prefix}_{decision.reason}"])
        )
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
        representative_id = str(sample_field(sample, "duplicate_representative_id", "") or "")
        decisions[sample.id] = Decision(
            status=status,
            phase=phase,
            reason=reason_text,
            confidence=confidence,
            keep_reason=(
                f"duplicate_of:{representative_id}"
                if status == "removed" and representative_id
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
                status=policy.deduplication.semantic_action,
                confidence=(
                    0.99 if policy.deduplication.semantic_action == "remove" else 0.85
                ),
            )
            decisions.update(
                decisions_for_label_conflicts(
                    dataset, "redundant_semantic", "semantic_duplicates"
                )
            )
            notes = apply_second_opinion(
                list(dataset), decisions, max_phase_drop, max_total_drop
            )
            semantic_result = write_decisions(
                dataset, decisions, "semantic_duplicates"
            )
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
        logger.info("Detectando aumentaciones por huella de color...")
        try:
            _, pairs = detect_augmentation_duplicates(
                dataset,
                "redundant_augmented",
                threshold=policy.deduplication.augmentation_similarity,
            )
            decisions = decision_for_tagged_duplicates(
                dataset,
                "redundant_augmented",
                "augmentation_duplicates",
                "Redundante (Aumentación)",
                status=policy.deduplication.augmentation_action,
                confidence=(
                    0.99 if policy.deduplication.augmentation_action == "remove" else 0.75
                ),
            )
            decisions.update(
                decisions_for_label_conflicts(
                    dataset, "redundant_augmented", "augmentation_duplicates"
                )
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

    return results


def suggest_label_mappings(labels: list[str]) -> dict[str, Any]:
    normalized_to_sources: dict[str, set[str]] = defaultdict(set)
    for label in labels:
        normalized_to_sources[normalize_label(label)].add(label)

    automatic = {
        normalized: sorted(values)
        for normalized, values in normalized_to_sources.items()
        if normalized and len(values) > 1
    }
    candidates = []
    normalized_labels = sorted(label for label in normalized_to_sources if label)
    for index, left in enumerate(normalized_labels):
        for right in normalized_labels[index + 1 :]:
            if left == right:
                continue
            left_tokens = set(left.split("_"))
            right_tokens = set(right.split("_"))
            overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
            if overlap >= 0.5:
                candidates.append({"left": left, "right": right, "token_overlap": round(overlap, 3)})

    return {"automatic": automatic, "candidates": candidates}


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

    labels = sorted(
        {
            label
            for sample_labels in dataset.values("source_labels")
            for label in (sample_labels or [])
            if label
        }
    )
    mapping = suggest_label_mappings(labels)
    result = PhaseResult(name="labels")
    
    automatic_map = {}
    if ontology_map:
        automatic_map = load_ontology_mapping(ontology_map)
        result.notes.append(f"Ontología de usuario aplicada desde {ontology_map}.")
        
    else:
        # Generar proposed ontology
        proposed = {norm: origs for norm, origs in mapping["automatic"].items()}
        for label in set(labels):
            bucket = proposed.setdefault(normalize_label(label), [])
            if label not in bucket:
                bucket.append(label)
                
        proposed_path = report_dir / "proposed_ontology.yaml"
        with open(proposed_path, "w", encoding="utf-8") as f:
            f.write("# Sugerencias heurísticas de mapeo (puedes editar y pasar con --ontology-map)\n")
            yaml.dump(proposed, f, allow_unicode=True, default_flow_style=False)
        result.notes.append(f"Archivo de ontología sugerido generado en {proposed_path}.")
        
        # Use heuristics
        for normalized, original_list in mapping["automatic"].items():
            for original in original_list:
                automatic_map[original] = normalized

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


def get_base64_image(filepath: str, max_size: int = 220) -> str:
    try:
        image = cv2.imread(filepath)
        if image is None:
            return ""
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
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
    
    for result in duplicate_results:
        # Calculate cross contamination for all pairs
        for kept_id, removed_id in result.duplicate_pairs:
            try:
                kept_ds = str(sample_field(dataset[kept_id], "source_dataset", "unknown"))
                rem_ds = str(sample_field(dataset[removed_id], "source_dataset", "unknown"))
                if kept_ds != rem_ds:
                    pair_key = tuple(sorted([kept_ds, rem_ds]))
                    cross_contamination_counts[pair_key] += 1
            except Exception:
                pass
                
        pairs: list[dict[str, dict[str, object]]] = []
        for kept_id, removed_id in result.duplicate_pairs[:8]:
            try:
                kept_sample = dataset[kept_id]
                removed_sample = dataset[removed_id]
            except Exception:
                continue
            pairs.append(
                {
                    "kept": sample_curation_payload(kept_sample),
                    "removed": sample_curation_payload(removed_sample),
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


def _split_group_key(sample: fo.Sample) -> str:
    for field_name in ("duplicate_cluster_id", "capture_group", "video_id", "location"):
        value = sample_field(sample, field_name)
        if value:
            return f"{field_name}:{value}"
    return f"asset:{sample.id}"


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
) -> dict[str, int]:
    """Creates deterministic group-aware and approximately stratified splits."""
    logger.info("Generando partición estratificada por grupos (Train/Val/Test)...")
    dataset.untag_samples(["train", "val", "test"])
    eligible = [sample for sample in dataset if current_status(sample) == "kept"]
    samples_by_group: dict[str, list[fo.Sample]] = defaultdict(list)
    labels_by_group: dict[str, list[str]] = defaultdict(list)
    for sample in eligible:
        group = _split_group_key(sample)
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
    for sample in dataset.select(list(sample_assignments)).iter_samples(autosave=True):
        split = sample_assignments[sample.id]
        sample.tags = sorted(set(sample.tags + [split]))
    return dict(Counter(sample_assignments.values()))

def export_clean_dataset(
    dataset: fo.Dataset,
    export_dir: Path,
    output_formats: list[str],
    label_mapping: dict[str, Any],
    summary: dict[str, Any],
    policy: CurationPolicy | None = None,
) -> dict[str, str]:
    policy = policy or CurationPolicy()
    unknown_formats = sorted(set(output_formats) - SUPPORTED_OUTPUT_FORMATS)
    if not output_formats or unknown_formats:
        raise ValueError(
            "Formatos de salida inválidos: "
            + (", ".join(unknown_formats) if unknown_formats else "lista vacía")
        )
    export_dir.mkdir(parents=True, exist_ok=True)
    images_dir = export_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Aplicar partición zero-leak
    split_counts = partition_dataset(
        dataset,
        train_p=policy.splits.train,
        val_p=policy.splits.val,
        test_p=policy.splits.test,
    )
    summary["splits"] = split_counts
    
    clean_view = dataset.match(fo.ViewField("curation.status") == "kept")
    if not len(clean_view):
        raise RuntimeError("No quedan muestras aprobadas; no se publicará un dataset vacío")

    # Filtrar cajas delimitadoras borrosas en la vista exportada, sin mutar el snapshot.
    has_detections = bool(
        clean_view.count_values("ground_truth_detections.detections.label")
    )
    if has_detections:
        logger.info(
            "Filtrando bounding boxes con blur_variance < %.2f...",
            policy.quality.min_box_blur,
        )
        clean_view = clean_view.filter_labels(
            "ground_truth_detections",
            fo.ViewField("blur_variance") >= policy.quality.min_box_blur,
            only_matches=False,
        )
    exports: dict[str, str] = {}

    manifest_rows = []
    for sample in clean_view:
        source_path = Path(sample.filepath)
        short_src = slugify(sample_field(sample, 'source_dataset', 'source'))[:12]
        safe_name = f"{short_src}_{sample.id}{source_path.suffix.lower()}"
        target_path = images_dir / safe_name
        if source_path.exists():
            shutil.copy2(source_path, target_path)
        manifest_rows.append(
            {
                "id": sample.id,
                "filepath": str(target_path),
                "source_path": str(sample_field(sample, "source_path", "")),
                "source_dataset": str(sample_field(sample, "source_dataset", "")),
                "source_split": str(sample_field(sample, "source_split", "")),
                "assigned_split": "train" if "train" in sample.tags else ("val" if "val" in sample.tags else "test"),
                "source_label": str(sample_field(sample, "source_label", "")),
                "normalized_label": str(sample_field(sample, "normalized_label", "")),
                "task_type": str(sample_field(sample, "task_type", "unlabeled")),
                "normalized_labels": ",".join(sample_field(sample, "normalized_labels", []) or []),
                "asset_sha256": str(sample_field(sample, "asset_sha256", "") or ""),
                "source_version": str(sample_field(sample, "source_version", "unknown")),
                "source_license": str(sample_field(sample, "source_license", "unknown")),
            }
        )

    with (export_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(manifest_rows[0]) if manifest_rows else ["id"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (export_dir / "manifest.jsonl").open("w", encoding="utf-8") as jsonl_file:
        for row in manifest_rows:
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_json(export_dir / "label_mapping.json", label_mapping)
    write_json(export_dir / "curation_summary.json", summary)
    exports["manifest"] = str(export_dir)

    try:
        clean_view.export(
            export_dir=str(export_dir / "fiftyone"),
            dataset_type=fo.types.FiftyOneDataset,
            export_media=True,
        )
        exports["fiftyone"] = str(export_dir / "fiftyone")
    except Exception as exc:
        exports["fiftyone_error"] = str(exc)

    if "classification" in output_formats:
        try:
            class_dir = export_dir / "classification"
            class_dir.mkdir(parents=True, exist_ok=True)
            import os

            for row in manifest_rows:
                if "classification" not in row["task_type"].split(","):
                    continue
                label = slugify(row["normalized_label"] or "unlabeled")
                label_dir = class_dir / row["assigned_split"] / label
                label_dir.mkdir(parents=True, exist_ok=True)
                target_path = label_dir / Path(row["filepath"]).name
                if not target_path.exists():
                    try:
                        os.link(row["filepath"], target_path)
                    except OSError:
                        shutil.copy2(row["filepath"], target_path)
            exports["classification"] = str(class_dir)
        except Exception as exc:
            exports["classification_error"] = str(exc)

    if "coco" in output_formats and dataset.has_sample_field("ground_truth_detections"):
        try:
            detection_view = clean_view.exists("ground_truth_detections")
            for split in ("train", "val", "test"):
                split_view = detection_view.match_tags(split)
                if not len(split_view):
                    continue
                split_view.export(
                    export_dir=str(export_dir / "coco" / split),
                    dataset_type=fo.types.COCODetectionDataset,
                    label_field="ground_truth_detections",
                    export_media=True,
                )
            exports["coco"] = str(export_dir / "coco")
        except Exception as exc:
            exports["coco_error"] = str(exc)

    if "yolo" in output_formats and dataset.has_sample_field("ground_truth_detections") and hasattr(fo.types, "YOLOv5Dataset"):
        try:
            detection_view = clean_view.exists("ground_truth_detections")
            classes = sorted(detection_view.distinct("ground_truth_detections.detections.label"))
            for split in ("train", "val", "test"):
                split_view = detection_view.match_tags(split)
                if not len(split_view):
                    continue
                split_view.export(
                    export_dir=str(export_dir / "yolo" / split),
                    dataset_type=fo.types.YOLOv5Dataset,
                    label_field="ground_truth_detections",
                    export_media=True,
                    split=split,
                    classes=classes,
                )
            exports["yolo"] = str(export_dir / "yolo")
        except Exception as exc:
            exports["yolo_error"] = str(exc)

    if "datumaro" in output_formats:
        datumaro_type = getattr(fo.types, "DatumaroDataset", None)
        if datumaro_type is None:
            exports["datumaro_error"] = "FiftyOne no expone DatumaroDataset en esta version."
        else:
            try:
                clean_view.export(
                    export_dir=str(export_dir / "datumaro"),
                    dataset_type=datumaro_type,
                    export_media=True,
                )
                exports["datumaro"] = str(export_dir / "datumaro")
            except Exception as exc:
                exports["datumaro_error"] = str(exc)

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
    fingerprint = source_fingerprint(raw_dir, policy.model_dump(mode="json"))
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
                logger.info("Run ya completado para estas fuentes y política: %s", run_id)
                logger.info("Resultado: %s", success_marker.parent)
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

        quality_payload = checkpoint.phase_payload("quality") if args.resume else None
        if quality_payload is not None and can_resume_dataset:
            phases.append(_restore_phase(quality_payload["phase"]))
            logger.info("Checkpoint de métricas de calidad reutilizado.")
        else:
            logger.info("Calculando metricas visuales...")
            compute_dataset_quality(
                dataset_name=dataset_name,
                workers=args.workers,
                enable_ocr=policy.quality.ocr_enabled,
                ocr_confidence=policy.quality.ocr_confidence,
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

        duplicates_payload = checkpoint.phase_payload("duplicates") if args.resume else None
        if duplicates_payload is not None and can_resume_dataset:
            duplicate_results = [
                _restore_phase(phase) for phase in duplicates_payload["phases"]
            ]
            logger.info("Checkpoint de deduplicación reutilizado.")
        else:
            duplicate_results = run_duplicate_phases(
                dataset,
                work_dir=interim_dir,
                max_phase_drop=args.max_phase_drop,
                max_total_drop=args.max_total_drop,
                policy=policy,
            )
            checkpoint.mark_phase(
                "duplicates", {"phases": [asdict(phase) for phase in duplicate_results]}
            )
        phases.extend(duplicate_results)

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

        logger.info("=== PIPELINE FINALIZADO ===")
        logger.info("Reporte HTML: %s", report_dir / "report.html")
        logger.info("Dataset exportado: %s", export_dir)
        logger.info("")
        logger.info("[bold cyan]¡Para explorar visualmente el dataset resultante, ejecuta:[/bold cyan]")
        logger.info(f"    [bold green]make app DATASET=\"{dataset_name}\"[/bold green]")
        logger.info("[bold cyan]Y abre el puerto configurado por FIFTYONE_PORT (5151 por defecto).[/bold cyan]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline desatendido de curacion y unificacion.")
    parser.add_argument("--dataset", required=True, help="Nombre del dataset unificado.")
    parser.add_argument("--raw-dir", required=True, help="Directorio con datasets crudos.")
    parser.add_argument("--profile", default="quality-first", choices=["quality-first"])
    parser.add_argument("--policy", default="configs/quality-first.yaml")
    parser.add_argument("--workers", type=int, default=4)
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
