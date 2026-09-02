from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import importlib.util
import json
import logging
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import fiftyone as fo

from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_NAMES = {"train", "test", "valid", "val", "validation"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
HARD_CONFIDENCE = 0.98

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


@dataclass
class PhaseResult:
    name: str
    removed: int = 0
    review: int = 0
    kept: int = 0
    notes: list[str] = field(default_factory=list)
    duplicate_pairs: list[tuple[str, str]] = field(default_factory=list)


def load_symbol(relative_path: str, symbol_name: str) -> Any:
    module_path = REPO_ROOT / relative_path
    module_name = "agrivision_dynamic_" + hashlib.sha1(str(module_path).encode()).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, symbol_name)


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


def discover_sources(raw_dir: Path) -> list[dict[str, str]]:
    if not raw_dir.exists():
        return []
    return [
        {
            "name": child.name,
            "path": str(child),
            "format": discover_source_format(child),
        }
        for child in sorted(raw_dir.iterdir())
        if child.is_dir()
    ]


def ensure_pipeline_schema(dataset: fo.Dataset) -> None:
    schema = dataset.get_field_schema(flat=True)
    for field_name in (
        "source_dataset",
        "source_split",
        "source_format",
        "source_path",
        "source_label",
        "normalized_label",
    ):
        if field_name not in schema:
            dataset.add_sample_field(field_name, fo.StringField)

    if "curation" not in schema:
        dataset.add_sample_field(
            "curation",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.DynamicEmbeddedDocument,
        )

    if "final_label" not in schema:
        dataset.add_sample_field("final_label", fo.EmbeddedDocumentField, embedded_doc_type=fo.Classification)


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

    detections = sample_field(sample, "coco_detections")
    if detections is not None and getattr(detections, "detections", None):
        labels = sorted({det.label for det in detections.detections if getattr(det, "label", None)})
        if labels:
            return ",".join(labels)

    try:
        parts = Path(sample.filepath).resolve().relative_to(raw_dir.resolve()).parts
    except ValueError:
        parts = Path(sample.filepath).parts
    return infer_label_from_path(parts)


def infer_source_metadata(sample: fo.Sample, raw_dir: Path, source_formats: dict[str, str]) -> dict[str, str]:
    try:
        relative = Path(sample.filepath).resolve().relative_to(raw_dir.resolve())
        parts = relative.parts
    except ValueError:
        relative = Path(sample.filepath)
        parts = relative.parts

    source_dataset = parts[0] if parts else "unknown"
    source_label = extract_sample_label(sample, raw_dir)
    return {
        "source_dataset": source_dataset,
        "source_split": infer_split(parts),
        "source_format": source_formats.get(source_dataset, "unknown"),
        "source_path": str(relative),
        "source_label": source_label,
        "normalized_label": normalize_label(source_label),
    }


def annotate_sources(dataset: fo.Dataset, raw_dir: Path, sources: list[dict[str, str]]) -> None:
    logger.info("Registrando procedencia y etiquetas normalizadas...")
    ensure_pipeline_schema(dataset)
    source_formats = {source["name"]: source["format"] for source in sources}
    initial_decision = fo.DynamicEmbeddedDocument(
        status="kept",
        phase="ingestion",
        reason="accepted_on_ingest",
        confidence=1.0,
        keep_reason="initial_candidate",
        cluster_id="",
    )

    for sample in dataset.iter_samples(progress=True, autosave=True):
        metadata = infer_source_metadata(sample, raw_dir, source_formats)
        for field_name, value in metadata.items():
            sample[field_name] = value
        sample["curation"] = initial_decision
        if metadata["normalized_label"]:
            sample["final_label"] = fo.Classification(label=metadata["normalized_label"])


def evaluate_quality(sample: fo.Sample) -> Decision:
    if flag(sample, "is_corrupted"):
        return Decision("removed", "quality", "corrupted_or_unreadable", 1.0)
    if flag(sample, "has_watermark"):
        return Decision("removed", "quality", "watermark_detected", 0.99)
    if flag(sample, "low_resolution"):
        return Decision("removed", "quality", "low_resolution", 0.98)
    if flag(sample, "has_smearing"):
        return Decision("removed", "quality", "border_smearing_detected", 0.92)

    blur = metric(sample, "blur_variance")
    if blur is not None and blur < 30.0:
        return Decision("removed", "quality", "severe_blur", 0.94)
    if blur is not None and blur < 70.0:
        return Decision("review", "quality", "borderline_blur", 0.72)

    brightness = metric(sample, "brightness_mean")
    p5 = metric(sample, "brightness_p5")
    p95 = metric(sample, "brightness_p95")
    if brightness is not None and (brightness < 18.0 or brightness > 245.0):
        return Decision("removed", "quality", "extreme_brightness", 0.90)
    if (p95 is not None and p95 < 45.0) or (p5 is not None and p5 > 210.0):
        return Decision("review", "quality", "borderline_brightness", 0.70)

    return Decision("kept", "quality", "quality_passed", 1.0, keep_reason="quality_passed")


def render_example_card(example: dict[str, object], caption: str | None = None) -> str:
    b64 = get_base64_image(str(example["filepath"]))
    if not b64:
        return ""
    subtitle = caption or f"{example.get('source_dataset', '')} | {example.get('label', '')}"
    details = [
        f"fase {format_report_value(example.get('phase'))}",
        f"conf {format_report_value(example.get('confidence'))}",
    ]
    if example.get("reason"):
        details.insert(0, str(example.get("reason")))
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
        f"<strong>{html.escape(str(example.get('reason', 'sample')))}</strong>"
        f"<span>{html.escape(subtitle)}</span>"
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


def render_count_table(title: str, rows: dict[str, dict[str, int]]) -> str:
    body = []
    for name, counts in rows.items():
        total = sum(counts.values())
        body.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td class='number-cell'>{total}</td>"
            f"<td class='number-cell'>{counts.get('kept', 0)}</td>"
            f"<td class='number-cell'>{counts.get('review', 0)}</td>"
            f"<td class='number-cell removed-col'>{counts.get('removed', 0)}</td>"
            "</tr>"
        )
    return (
        f"<h2>{html.escape(title)}</h2>"
        "<table><thead><tr><th>Grupo</th><th class='number-cell'>Total</th><th class='number-cell'>Kept</th>"
        "<th class='number-cell'>Review</th><th class='number-cell removed-col'>Removed</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def write_reports(
    report_dir: Path,
    dataset_name: str,
    run_id: str,
    summary: dict[str, Any],
    evidence: dict[str, object],
) -> None:
    write_json(report_dir / "summary.json", summary)
    copy_examples_by_reason(list(evidence.get("removed_by_reason", [])), report_dir / "discarded_examples")
    copy_examples_by_reason(list(evidence.get("review_by_reason", [])), report_dir / "review_examples")
    copy_duplicate_pair_gallery(list(evidence.get("duplicate_phases", [])), report_dir / "duplicate_examples")

    md_lines = [
        f"# Pipeline report: {dataset_name}",
        "",
        f"- Run: `{run_id}`",
        f"- Initial samples: {summary['counts']['initial']}",
        f"- Kept/exported: {summary['counts']['kept']}",
        f"- Review: {summary['counts']['review']}",
        f"- Removed: {summary['counts']['removed']}",
        "",
        "## Phase notes",
    ]
    for phase in summary["phases"]:
        md_lines.append(f"- {phase['name']}: removed={phase['removed']}, review={phase['review']}, kept={phase['kept']}")
        for note in phase["notes"]:
            md_lines.append(f"  - {note}")
    md_lines.extend(["", "## Razones de descarte"])
    for section in evidence.get("removed_by_reason", []):
        md_lines.append(f"- {section['reason']}: {section['count']} ejemplos")
    md_lines.extend(["", "## Razones en revisión"])
    for section in evidence.get("review_by_reason", []):
        md_lines.append(f"- {section['reason']}: {section['count']} ejemplos")
    md_lines.extend(["", "## Pares de duplicados"])
    for section in evidence.get("duplicate_phases", []):
        md_lines.append(f"- {section['phase']}: {section['removed']} descartadas, {section['review']} en revisión")
        for note in section.get("notes", []):
            md_lines.append(f"  - {note}")
    (report_dir / "report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

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
  </div>
  {render_count_table("Evolucion por dataset origen", summary["groups"]["by_source"])}
  {render_count_table("Evolucion por etiqueta", summary["groups"]["by_label"])}
  {render_reason_sections("Ejemplos descartados por razon", list(evidence.get('removed_by_reason', [])), "No hay ejemplos descartados.")}
  {render_duplicate_sections("Duplicados y mantenimiento del original", list(evidence.get('duplicate_phases', [])))}
  {render_reason_sections("Ejemplos en revision", list(evidence.get('review_by_reason', [])), "No hay ejemplos en revision.")}
</main>
</body>
</html>
"""
    (report_dir / "report.html").write_text(html_doc, encoding="utf-8")


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
        sample.tags = sorted(set(sample.tags + [f"curation_{decision.status}", f"{tag_prefix}_{decision.reason}"]))
        result.removed += int(decision.status == "removed")
        result.review += int(decision.status == "review")
        result.kept += int(decision.status == "kept")

    return result


def run_quality_phase(
    dataset: fo.Dataset,
    max_phase_drop: float,
    max_total_drop: float,
) -> PhaseResult:
    active_samples = [sample for sample in dataset if current_status(sample) != "removed"]
    decisions = {sample.id: evaluate_quality(sample) for sample in active_samples}
    notes = apply_second_opinion(active_samples, decisions, max_phase_drop, max_total_drop)
    result = write_decisions(dataset, decisions, "quality")
    result.notes.extend(notes)
    return result


def decision_for_tagged_duplicates(dataset: fo.Dataset, tag: str, phase: str) -> dict[str, Decision]:
    decisions = {}
    for sample in dataset.match_tags(tag):
        if current_status(sample) == "removed":
            continue
        decisions[sample.id] = Decision(
            status="removed",
            phase=phase,
            reason=tag,
            confidence=1.0,
            keep_reason="redundant_visual_cluster",
            cluster_id="",
        )
    return decisions



def run_duplicate_phases(
    dataset: fo.Dataset,
    work_dir: Path,
    max_phase_drop: float,
    max_total_drop: float,
) -> list[PhaseResult]:
    detect_exact_duplicates = load_symbol("src/02_curation/deduplicate_dataset.py", "detect_exact_duplicates")
    detect_semantic_duplicates = load_symbol("src/02_curation/deduplicate_dataset.py", "detect_semantic_duplicates")
    detect_augmentation_duplicates = load_symbol("src/02_curation/deduplicate_dataset.py", "detect_augmentation_duplicates")

    results = []
    logger.info("Detectando duplicados exactos con FiftyOne...")
    _, pairs = detect_exact_duplicates(dataset, "redundant_exact")
    decisions = decision_for_tagged_duplicates(dataset, "redundant_exact", "exact_duplicates")
    active_samples = [sample for sample in dataset if current_status(sample) != "removed"]
    notes = apply_second_opinion(active_samples, decisions, max_phase_drop, max_total_drop)
    exact_result = write_decisions(dataset, decisions, "exact_duplicates")
    exact_result.notes.extend(notes)
    exact_result.duplicate_pairs = pairs
    results.append(exact_result)

    logger.info("Detectando near-duplicates semanticos con FiftyOne...")
    try:
        _, pairs = detect_semantic_duplicates(dataset, "redundant_semantic", threshold=0.96)
        decisions = decision_for_tagged_duplicates(dataset, "redundant_semantic", "semantic_duplicates")
        active_samples = [sample for sample in dataset if current_status(sample) != "removed"]
        notes = apply_second_opinion(active_samples, decisions, max_phase_drop, max_total_drop)
        semantic_result = write_decisions(dataset, decisions, "semantic_duplicates")
        semantic_result.notes.extend(notes)
        semantic_result.duplicate_pairs = pairs
        results.append(semantic_result)
    except Exception as exc:
        results.append(PhaseResult(name="semantic_duplicates", notes=[f"Fase omitida por error: {exc}"]))

    logger.info("Detectando aumentaciones por huella de color...")
    try:
        _, pairs = detect_augmentation_duplicates(dataset, "redundant_augmented", threshold=0.99)
        decisions = decision_for_tagged_duplicates(dataset, "redundant_augmented", "augmentation_duplicates")
        active_samples = [sample for sample in dataset if current_status(sample) != "removed"]
        notes = apply_second_opinion(active_samples, decisions, max_phase_drop, max_total_drop)
        augmented_result = write_decisions(dataset, decisions, "augmentation_duplicates")
        augmented_result.notes.extend(notes)
        augmented_result.duplicate_pairs = pairs
        results.append(augmented_result)
    except Exception as exc:
        results.append(PhaseResult(name="augmentation_duplicates", notes=[f"Fase omitida por error: {exc}"]))

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


def run_label_phase(dataset: fo.Dataset, cleanlab_mode: str) -> tuple[PhaseResult, dict[str, Any]]:
    labels = [label for label in dataset.values("source_label") if label]
    mapping = suggest_label_mappings(labels)
    result = PhaseResult(name="labels")

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
        "blur_variance": metric(sample, "blur_variance"),
        "brightness_mean": metric(sample, "brightness_mean"),
        "brightness_p5": metric(sample, "brightness_p5"),
        "brightness_p95": metric(sample, "brightness_p95"),
        "width": sample_field(sample, "width"),
        "height": sample_field(sample, "height"),
        "low_resolution": flag(sample, "low_resolution"),
        "has_watermark": flag(sample, "has_watermark"),
        "has_smearing": flag(sample, "has_smearing"),
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
    for result in duplicate_results:
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

    return {
        "removed_by_reason": removed_by_reason,
        "review_by_reason": review_by_reason,
        "duplicate_phases": duplicate_sections,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def partition_dataset(dataset: fo.Dataset, train_p: float = 0.8, val_p: float = 0.1, test_p: float = 0.1) -> None:
    """
    Partición estratificada sin fugas (Zero-Leak Splitting).
    Agrupa por `video_id`, `location` o la carpeta padre (proxy para ráfagas de fotos).
    """
    logger.info("Generando partición Zero-Leak (Train/Val/Test)...")
    import hashlib
    
    # Limpiar tags anteriores
    dataset.untag_samples(["train", "val", "test"])
    
    for sample in dataset.iter_samples(autosave=True):
        # Determinar el grupo (Granularidad menor que el dataset entero)
        group_key = ""
        if sample.has_field("video_id") and sample.video_id:
            group_key = str(sample.video_id)
        elif sample.has_field("location") and sample.location:
            group_key = str(sample.location)
        else:
            # Fallback a la carpeta origen (suele contener ráfagas de la misma toma)
            group_key = str(Path(sample.filepath).parent)
            
        # Hash estable para asegurar que el grupo caiga siempre en el mismo split
        hash_val = int(hashlib.md5(group_key.encode('utf-8')).hexdigest(), 16)
        normalized_hash = (hash_val % 10000) / 10000.0
        
        if normalized_hash < train_p:
            sample.tags.append("train")
        elif normalized_hash < train_p + val_p:
            sample.tags.append("val")
        else:
            sample.tags.append("test")

def export_clean_dataset(
    dataset: fo.Dataset,
    export_dir: Path,
    output_formats: list[str],
    label_mapping: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, str]:
    export_dir.mkdir(parents=True, exist_ok=True)
    images_dir = export_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Aplicar partición zero-leak
    partition_dataset(dataset)
    
    # Filtrar cajas delimitadoras borrosas a nivel de Crop (Blur Variance < 150)
    # Conservamos la imagen original, pero eliminamos las anotaciones inservibles.
    if dataset.has_sample_field("coco_detections"):
        logger.info("Filtrando bounding boxes borrosos (blur_variance < 150)...")
        dataset.filter_labels(
            "coco_detections",
            fo.ViewField("blur_variance") >= 150.0,
            only_matches=True
        )

    clean_view = dataset.match(fo.ViewField("curation.status") == "kept")
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
            export_media="copy",
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
                label = row["normalized_label"] or "unlabeled"
                label_dir = class_dir / label
                label_dir.mkdir(exist_ok=True)
                target_path = label_dir / Path(row["filepath"]).name
                if not target_path.exists():
                    try:
                        os.link(row["filepath"], target_path)
                    except OSError:
                        shutil.copy2(row["filepath"], target_path)
            exports["classification"] = str(class_dir)
        except Exception as exc:
            exports["classification_error"] = str(exc)

    if "coco" in output_formats and dataset.has_sample_field("coco_detections"):
        try:
            clean_view.export(
                export_dir=str(export_dir / "coco"),
                dataset_type=fo.types.COCODetectionDataset,
                label_field="coco_detections",
                export_media="copy",
            )
            exports["coco"] = str(export_dir / "coco")
        except Exception as exc:
            exports["coco_error"] = str(exc)

    if "yolo" in output_formats and dataset.has_sample_field("coco_detections") and hasattr(fo.types, "YOLOv5Dataset"):
        try:
            clean_view.export(
                export_dir=str(export_dir / "yolo"),
                dataset_type=fo.types.YOLOv5Dataset,
                label_field="coco_detections",
                export_media="copy",
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
                    export_media="copy",
                )
                exports["datumaro"] = str(export_dir / "datumaro")
            except Exception as exc:
                exports["datumaro_error"] = str(exc)

    return exports


def build_summary(
    dataset: fo.Dataset,
    dataset_name: str,
    run_id: str,
    raw_dir: Path,
    sources: list[dict[str, str]],
    tools: dict[str, dict[str, str]],
    phases: list[PhaseResult],
    label_mapping: dict[str, Any],
) -> dict[str, Any]:
    counts = status_counts(dataset)
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
        "phases": [asdict(phase) for phase in phases],
        "groups": grouped_counts(dataset),
        "label_mapping": label_mapping,
    }


def stage_datumaro_inventory(interim_dir: Path, sources: list[dict[str, str]], tools: dict[str, dict[str, str]]) -> None:
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
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline desatendido de curacion y unificacion.")
    parser.add_argument("--dataset", required=True, help="Nombre del dataset unificado.")
    parser.add_argument("--raw-dir", required=True, help="Directorio con datasets crudos.")
    parser.add_argument("--profile", default="quality-first", choices=["quality-first"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-formats", default="datumaro,coco,yolo,classification")
    parser.add_argument("--cleanlab-mode", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--export-dir", default="data/processed")
    parser.add_argument("--report-dir", default="reports/pipeline")
    parser.add_argument("--max-phase-drop", type=float, default=0.40)
    parser.add_argument("--max-total-drop", type=float, default=0.65)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    dataset_name = args.dataset
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir) / dataset_name / run_id
    export_dir = Path(args.export_dir) / dataset_name / run_id
    interim_dir = Path("data/interim") / dataset_name / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== INICIANDO PIPELINE DESATENDIDO ===")
    logger.info("Dataset=%s RAW_DIR=%s RUN_ID=%s", dataset_name, raw_dir, run_id)

    tools = optional_tool_status()
    sources = discover_sources(raw_dir)
    stage_datumaro_inventory(interim_dir, sources, tools)

    create_unified_dataset = load_symbol("src/01_acquisition/make_dataset.py", "create_unified_dataset")
    compute_dataset_quality = load_symbol("src/02_curation/compute_quality_metrics.py", "compute_dataset_quality")

    phases: list[PhaseResult] = []
    dataset = create_unified_dataset(dataset_name, raw_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No se ingestaron imagenes desde {raw_dir}")

    annotate_sources(dataset, raw_dir, sources)
    phases.append(PhaseResult(name="ingestion", kept=len(dataset), notes=[f"{len(sources)} fuentes detectadas."]))

    logger.info("Calculando metricas visuales...")
    compute_dataset_quality(
        dataset_name=dataset_name,
        workers=args.workers,
        enable_ocr=True,
        ocr_confidence=60.0,
    )
    dataset = fo.load_dataset(dataset_name)
    phases.append(run_quality_phase(dataset, args.max_phase_drop, args.max_total_drop))
    duplicate_results = run_duplicate_phases(
        dataset,
        work_dir=interim_dir,
        max_phase_drop=args.max_phase_drop,
        max_total_drop=args.max_total_drop,
    )
    phases.extend(duplicate_results)

    label_phase, label_mapping = run_label_phase(dataset, args.cleanlab_mode)
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
    exports = export_clean_dataset(
        dataset=dataset,
        export_dir=export_dir,
        output_formats=parse_output_formats(args.output_formats),
        label_mapping=label_mapping,
        summary=summary,
    )
    summary["exports"] = exports
    write_reports(report_dir, dataset_name, run_id, summary, evidence)
    write_json(export_dir / "curation_summary.json", summary)

    logger.info("=== PIPELINE FINALIZADO ===")
    logger.info("Reporte HTML: %s", report_dir / "report.html")
    logger.info("Dataset exportado: %s", export_dir)
    logger.info("")
    logger.info("[bold cyan]¡Para explorar visualmente el dataset resultante, ejecuta:[/bold cyan]")
    logger.info(f"    [bold green]make app DATASET=\"{dataset_name}\"[/bold green]")
    logger.info("[bold cyan]Y abre http://localhost:5152 en tu navegador.[/bold cyan]")


if __name__ == "__main__":
    main()
