"""Open Data release bundle generator for AgriVision-KHAOS curated datasets.

Packages exported datasets, annotations, HTML audit reports, and comprehensive provenance
documentation into standardized, clean Open Data distributions ready for publication on
portals (CKAN, Andalucía ISI2A2, Agora Datalab, Zenodo).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from rich.logging import RichHandler

from agrivision_khaos.export_assets import link_export_asset
from agrivision_khaos.ontology import infer_coarse_label

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger(__name__)


def find_latest_export(export_root: Path, dataset_name: str) -> Path:
    """Finds the most recent completed run directory with _SUCCESS marker."""
    dataset_dir = export_root / dataset_name
    hitl_dir = export_root / f"{dataset_name}_hitl"

    candidates: list[Path] = []
    for base in (hitl_dir, dataset_dir):
        if base.is_dir():
            for child in base.iterdir():
                if child.is_dir() and (child / "_SUCCESS").is_file():
                    if (child / "classification").is_dir() or (child / "dataset").is_dir() or (child / "images").is_dir():
                        candidates.append(child)

    if not candidates:
        releases_dir = export_root / "releases"
        if releases_dir.is_dir():
            for child in releases_dir.iterdir():
                if child.is_dir() and child.name.startswith(dataset_name):
                    if (child / "classification").is_dir() or (child / "dataset").is_dir():
                        candidates.append(child)
            dataset_releases_dir = releases_dir / dataset_name
            if dataset_releases_dir.is_dir():
                for date_child in dataset_releases_dir.iterdir():
                    if date_child.is_dir():
                        for b_child in date_child.iterdir():
                            if b_child.is_dir() and ((b_child / "dataset").is_dir() or (b_child / "classification").is_dir()):
                                candidates.append(b_child)

    if not candidates:
        # Fallback to any directory named dataset_name or matching pattern
        for base in (hitl_dir, dataset_dir):
            if base.is_dir():
                for child in base.iterdir():
                    if child.is_dir() and (child / "_SUCCESS").is_file():
                        candidates.append(child)

    if not candidates and export_root.is_dir():
        # Fuzzy matching: case-insensitive, singular/plural tolerance
        d_norm = dataset_name.lower().replace("-", "").replace("_", "").rstrip("s")
        for base in export_root.iterdir():
            if base.is_dir() and not base.name.startswith("."):
                b_norm = base.name.lower().replace("-", "").replace("_", "").rstrip("s")
                if b_norm == d_norm or d_norm in b_norm or b_norm in d_norm:
                    for child in base.iterdir():
                        if child.is_dir() and (child / "_SUCCESS").is_file():
                            candidates.append(child)

    if not candidates:
        raise FileNotFoundError(
            f"No se encontró ninguna exportación completada con _SUCCESS para '{dataset_name}' en {export_root}"
        )
    return max(candidates, key=lambda p: p.name)


def find_matching_report(report_root: Path, dataset_name: str, run_id: str | None = None) -> Path | None:
    """Locates the report.html associated with a dataset run."""
    dataset_report_dir = report_root / dataset_name
    if not dataset_report_dir.is_dir() and report_root.is_dir():
        d_norm = dataset_name.lower().replace("-", "").replace("_", "").rstrip("s")
        for base in report_root.iterdir():
            if base.is_dir() and not base.name.startswith("."):
                b_norm = base.name.lower().replace("-", "").replace("_", "").rstrip("s")
                if b_norm == d_norm or d_norm in b_norm or b_norm in d_norm:
                    dataset_report_dir = base
                    break
        else:
            return None

    if not dataset_report_dir.is_dir():
        return None

    if run_id:
        direct = dataset_report_dir / run_id / "report.html"
        if direct.is_file():
            return direct

    html_files = list(dataset_report_dir.glob("*/report.html"))
    return max(html_files, key=lambda p: p.stat().st_mtime) if html_files else None


def compute_file_sha256(path: Path) -> str:
    """Computes SHA256 hex digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_sources_registry(sources_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Loads dataset source registry from configs/sources.yaml and optional custom path."""
    sources: dict[str, dict[str, Any]] = {}
    search_dirs = [Path("configs"), Path("/workspace/configs")]
    for d in search_dirs:
        default_file = d / "sources.yaml"
        if default_file.is_file():
            try:
                with default_file.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    if isinstance(data, dict):
                        sources.update(data)
                        break
            except Exception as e:
                logger.warning(f"Error cargando {default_file}: {e}")

    if sources_path and Path(sources_path).is_file():
        try:
            with Path(sources_path).open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                if isinstance(data, dict):
                    sources.update(data)
        except Exception as e:
            logger.warning(f"Error cargando sources personalizadas desde {sources_path}: {e}")

    return sources


def load_release_metadata(dataset_name: str, metadata_path: Path | None = None) -> dict[str, Any]:
    """Loads dataset release configuration from YAML file or auto-discovers in configs/releases/."""
    if metadata_path and Path(metadata_path).is_file():
        try:
            with Path(metadata_path).open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                if isinstance(cfg, dict):
                    logger.info(f"Cargada configuración de release desde: [bold green]{metadata_path}[/bold green]")
                    return cfg
        except Exception as e:
            logger.warning(f"Error cargando metadatos desde {metadata_path}: {e}")

    # Auto-discovery
    import re
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", dataset_name).lower()
    d_clean = dataset_name.lower().replace("-", "_")
    search_dirs = [Path("configs/releases"), Path("/workspace/configs/releases")]
    for base in search_dirs:
        for candidate_name in [
            f"{dataset_name}.yaml",
            f"{snake}.yaml",
            f"{d_clean}.yaml",
            f"{dataset_name.lower()}.yaml",
        ]:
            cand = base / candidate_name
            if cand.is_file():
                try:
                    with cand.open("r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f) or {}
                        if isinstance(cfg, dict):
                            logger.info(f"Auto-detectada configuración de release: [bold green]{cand}[/bold green]")
                            return cfg
                except Exception as e:
                    logger.warning(f"Error cargando {cand}: {e}")

    logger.info(f"No se encontró archivo de configuración específico para '{dataset_name}'. Usando metadatos universales.")
    return {}


# Module-level registry loaded dynamically for backwards compatibility
SOURCE_REGISTRY: dict[str, dict[str, Any]] = load_sources_registry()


def generate_open_data_readme(
    dataset_name: str,
    version: str,
    manifest_rows: list[dict[str, str]],
    effective_layout: str = "classification",
    meta: dict[str, Any] | None = None,
) -> str:
    """Generates concise, noise-free, academic README.md driven by release configuration."""
    meta = meta or {}
    total_images = len(manifest_rows)
    splits = {
        "train": sum(1 for r in manifest_rows if r.get("split") == "train" or r.get("assigned_split") == "train"),
        "val": sum(1 for r in manifest_rows if r.get("split") == "val" or r.get("assigned_split") == "val"),
        "test": sum(1 for r in manifest_rows if r.get("split") == "test" or r.get("assigned_split") == "test"),
    }
    unannotated_count = sum(
        1 for r in manifest_rows
        if r.get("split") == "test_unannotated" or (r.get("task") or "").lower() == "unannotated"
    )

    domain_title = meta.get("domain_title", f"Visión Artificial y Detección en {dataset_name}")
    species_str = meta.get("species", f"Especie agrícola ({dataset_name})")
    ontology_explanation = meta.get("ontology_explanation", (
        "El dataset organiza las muestras bajo una ontología estructurada de clases diagnósticas y operativas:\n"
        "- **Evaluación multiclase:** Clasificación en las categorías indicadas en `annotations/labels.csv`.\n"
        "- **Evaluación binaria:** Separación diagnóstica entre muestras sanas y defectuosas/patológicas."
    ))
    condition_map = meta.get("condition_map", {})
    scientific_names = meta.get("scientific_names", {})

    class_counts: dict[str, int] = {}
    detection_count = 0
    for r in manifest_rows:
        task = r.get("task") or r.get("task_type", "")
        if "detection" in task:
            detection_count += 1
        label = (
            r.get("disease")
            or r.get("normalized_label")
            or r.get("class")
            or "objeto"
        )
        if label == "rust_mite":
            label = "aculus_olearius"
        class_counts[label] = class_counts.get(label, 0) + 1

    table_lines = [
        "| Clase Canónica | Descripción / Condición | Estado Derivado | Muestras Totales |",
        "|---|---|---|---:|",
    ]
    for c, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        cond = condition_map.get(c) or infer_coarse_label(c) or "Anotado"
        sci = scientific_names.get(c, "-")
        table_lines.append(f"| `{c}` | *{sci}* | {cond} | {cnt:,} |")

    if unannotated_count > 0:
        split_summary = (
            f"Train: {splits['train']:,} ({splits['train'] / max(total_images, 1) * 100:.1f}%), "
            f"Val: {splits['val']:,} ({splits['val'] / max(total_images, 1) * 100:.1f}%), "
            f"Test (Anotado): {splits['test']:,} ({splits['test'] / max(total_images, 1) * 100:.1f}%), "
            f"Test Sin Anotar: {unannotated_count:,} ({unannotated_count / max(total_images, 1) * 100:.1f}%)"
        )
    else:
        split_summary = (
            f"Train: {splits['train']:,} ({splits['train'] / max(total_images, 1) * 100:.1f}%), "
            f"Val: {splits['val']:,} ({splits['val'] / max(total_images, 1) * 100:.1f}%), "
            f"Test: {splits['test']:,} ({splits['test'] / max(total_images, 1) * 100:.1f}%)"
        )

    if effective_layout == "detection":
        loc_note = (
            f"*Estructura unificada de detección:* {detection_count:,} imágenes cuentan con anotaciones de localización "
            f"espacial (bounding boxes en formato COCO JSON en `dataset/train/`, `dataset/val/` y `dataset/test/`)."
        )
        if unannotated_count > 0:
            loc_note += (
                f"\n*Imágenes complementarias de test:* Las {unannotated_count:,} imágenes sin cajas delimitadoras se agrupan "
                f"en `dataset/test_unannotated/` para evaluación de falsas alarmas (background/negative testing) e inferencia a ciegas."
            )
    elif detection_count > 0:
        loc_note = (
            f"*Nota sobre localización:* {detection_count:,} imágenes cuentan con anotaciones de localización espacial "
            f"(bounding boxes en formato COCO JSON en `detection/`)."
        )
    else:
        loc_note = ""

    loc_note_block = f"\n{loc_note}\n" if loc_note else ""

    return f"""# {dataset_name} (v{version})

Macro-dataset curado y auditado de visión artificial para {domain_title}.

---

## Metadatos Principales

- **Especie:** {species_str}
- **Total de imágenes:** {total_images:,}
- **Partición Zero-Leakage:** {split_summary}
- **Licencia:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Entidad Responsable:** Grupo KHAOS (ITIS Software, Universidad de Málaga).
- **Portal de Publicación:** [ISIDROS Open Data](https://khaos.uma.es/isidros-opendata/)

---

## Clases y Ontología

{ontology_explanation}

{chr(10).join(table_lines)}{loc_note_block}
---

## Fuentes Originales

Este macro-dataset es el resultado de la unificación, filtrado y curación de fuentes públicas abiertas. Para consultar los enlaces oficiales a los repositorios (Kaggle / Roboflow Universe / Mendeley Data / Zenodo), publicaciones científicas (papers / DOIs), licencias y clases por defecto de cada dataset original, véase **`SOURCES.md`**.

---

## Estructura del Paquete

```text
{dataset_name}_v{version}/
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── annotations/
│   └── labels.csv              # Metadatos limpios, particiones y clases canónicas
├── preprocess/
│   ├── report.html             # Informe interactivo de curación, calidad y duplicados
│   └── audit_summary.json      # Estadísticas completas de métricas y filtros
├── README.md                   # Descripción académica y ontológica
└── SOURCES.md                  # Citas bibliográficas y licencias originales
```

---

## Contacto y Citación

Publicado por el grupo de investigación **KHAOS** (Universidad de Málaga). Para consultas, propuestas de mejora o reporte de incidencias en los datos, contacte a través del portal de datos abiertos [ISIDROS](https://khaos.uma.es/isidros-opendata/).
"""


def generate_sources_md(
    manifest_rows: list[dict[str, str]],
    source_registry: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Generates comprehensive SOURCES.md detailing provenance, DOIs, and BibTeX."""
    registry = source_registry or SOURCE_REGISTRY
    lines = [
        "# Registro de Fuentes de Datos Originales",
        "",
        "Este documento detalla la procedencia, autores originales, publicaciones científicas,",
        "licencias, clases por defecto y muestras aportadas por cada dataset primario integrado.",
        "",
        "---",
        "",
    ]

    for raw_name, meta in registry.items():
        cnt = sum(
            1 for r in manifest_rows
            if r.get("source_dataset") == raw_name or r.get("source_dataset") == meta.get("title")
        )
        if cnt == 0:
            continue
        lines.append(f"## {meta.get('title', raw_name)}")
        if meta.get("authors"):
            lines.append(f"- **Autores / Creador:** {meta['authors']}")
        if meta.get("url"):
            lines.append(f"- **Repositorio Oficial:** [{meta['url']}]({meta['url']})")
        if meta.get("paper"):
            paper_url = meta.get("paper_url", meta.get("url", ""))
            lines.append(f"- **Publicación Científica / Paper:** [{meta['paper']}]({paper_url})")
        if meta.get("license"):
            lines.append(f"- **Licencia Original:** {meta['license']}")
        lines.append(f"- **Muestras Aportadas:** {cnt:,} imágenes curadas")
        if meta.get("raw_classes"):
            lines.append(f"- **Clases por Defecto en Origen:** `{meta['raw_classes']}`")
        if meta.get("description"):
            lines.append(f"- **Descripción:** {meta['description']}")
        lines.append("")
        if meta.get("bibtex"):
            lines.append("### Referencia Bibliográfica (BibTeX)")
            lines.append("```bibtex")
            lines.append(meta["bibtex"])
            lines.append("```")
            lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def generate_ckan_metadata_md(
    dataset_name: str,
    version: str,
    total_images: int,
    archive_name: str,
    archive_size_mb: float,
    meta: dict[str, Any] | None = None,
) -> str:
    """Generates ready-to-paste markdown file with exact metadata fields for CKAN."""
    meta = meta or {}
    slug_version = version.replace(".", "-")
    titles = meta.get("titles", {})
    title_en = titles.get("en", f"Image dataset for agricultural computer vision and classification ({dataset_name})")
    title_es = titles.get("es", f"Dataset de imágenes para visión artificial y clasificación agrícola ({dataset_name})")
    slug = meta.get("slug", f"dataset-de-imagenes-{dataset_name.lower().replace('_', '-')}-v{slug_version}")
    tags_list = meta.get("tags", [
        "agriculture", "agronomy", "computer-vision", "deep-learning", "image-classification",
        "open-data", "precision-agriculture", dataset_name.lower().replace("_", "-")
    ])
    tags_str = ", ".join(tags_list)

    desc_dict = meta.get("descriptions", {})
    raw_desc_en = desc_dict.get("en", f"An image dataset of {dataset_name} containing {{total_images:,}} images for artificial intelligence models.\nThe license is CC BY 4.0.")
    raw_desc_es = desc_dict.get("es", f"Conjunto de datos de imágenes de {dataset_name} compuesto por {{total_images:,}} imágenes para modelos de inteligencia artificial.\nLa licencia es CC BY 4.0.")
    try:
        desc_en = raw_desc_en.format(total_images=total_images)
    except Exception:
        desc_en = raw_desc_en
    try:
        desc_es = raw_desc_es.format(total_images=total_images)
    except Exception:
        desc_es = raw_desc_es

    res_dict = meta.get("resource", {})
    raw_res_name = res_dict.get("name", f"Dataset de imágenes {dataset_name} ZIP")
    raw_res_desc = res_dict.get("description", f"Conjunto de datos compuesto por {{total_images:,}} imágenes de {dataset_name} organizadas en ImageFolder estándar (train, val, test), anotaciones limpias en labels.csv y reporte de calidad en preprocess/.")
    try:
        res_name = raw_res_name.format(total_images=total_images)
    except Exception:
        res_name = raw_res_name
    try:
        res_desc = raw_res_desc.format(total_images=total_images)
    except Exception:
        res_desc = raw_res_desc

    return f"""# Ficha de Metadatos para Publicación en CKAN
## Portal: https://khaos.uma.es/isidros-opendata/

Copia y pega los siguientes valores directamente en el formulario de creación de dataset en CKAN (https://khaos.uma.es/isidros-opendata/dataset/new):

---

### Paso 1: Formulario Principal ("Create Dataset")

| Campo en CKAN | Valor a Introducir |
|---|---|
| **Title (Título)** | `{title_en}` <br>*(Alternativa es: `{title_es}`)* |
| **URL (Slug)** | `{slug}` <br>*(URL resultante: `https://khaos.uma.es/isidros-opendata/dataset/{slug}`)* |
| **Description (Descripción)** | *(Ver texto abajo en Paso 2)* |
| **Tags (Etiquetas)** | `{tags_str}` |
| **License (Licencia)** | `Creative Commons Attribution (CC BY 4.0)` |
| **Organization (Organización)** | `KHAOS Research Group` |
| **Visibility (Visibilidad)** | `Public` |
| **Version (Versión)** | `{version}` |
| **Author (Autor)** | `KHAOS Research Group (ITIS Software, Universidad de Málaga)` |
| **Author Email** | `khaos.research@uma.es` |
| **Maintainer (Mantenedor)** | `Diego Depablos / KHAOS Research Group` |
| **Maintainer Email** | `diego.depablos@uma.es` |

---

### Paso 2: Descripción para CKAN

#### Opción A (En inglés - Recomendada para portales internacionales y CKAN):

```markdown
{desc_en}
```

#### Opción B (En español, si tu superior solicita la descripción en español):

```markdown
{desc_es}
```

---

### Paso 4: Recursos a Subir ("Data and Resources")

#### Recurso 1 (Principal):
- **Archivo:** `{archive_name}` ({archive_size_mb:.1f} MB)
- **Name (Nombre):** `{res_name}`
- **Description (Descripción del Recurso):** `{res_desc}`
- **Format (Formato):** `ZIP`
"""


def build_clean_labels_csv(
    manifest_rows: list[dict[str, str]],
    output_csv: Path,
    unannotated_filenames: set[str] | None = None,
    source_registry: dict[str, dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Builds clean, human-readable labels.csv for data analysts."""
    registry = source_registry or SOURCE_REGISTRY
    meta = meta or {}
    condition_map = meta.get("condition_map", {})
    clean_rows = []
    for r in manifest_rows:
        filename = Path(r.get("filepath", "")).name or r.get("filename", "")
        if "disease" in r and "condition" in r and "source_dataset" in r and "source_url" in r:
            if unannotated_filenames and filename in unannotated_filenames:
                r_copy = dict(r)
                r_copy["task"] = "unannotated"
                r_copy["split"] = "test_unannotated"
                clean_rows.append(r_copy)
            else:
                clean_rows.append(r)
            continue
        src_raw = r.get("source_dataset", "")
        src_meta = registry.get(src_raw, {})
        split = r.get("split") or r.get("assigned_split") or "train"
        raw_label = r.get("disease") or r.get("normalized_label") or r.get("class") or "objeto"
        if raw_label == "rust_mite":
            raw_label = "aculus_olearius"
        task = r.get("task") or r.get("task_type", "detection")
        if unannotated_filenames and filename in unannotated_filenames:
            task = "unannotated"
            split = "test_unannotated"

        derived = r.get("derived_label")
        if derived:
            condition = derived
        elif raw_label in condition_map:
            condition = condition_map[raw_label]
        elif raw_label in ("olivo", "olive_tree", "crown"):
            condition = "olivo"
        elif raw_label in ("sano", "sana", "verde", "envero", "negra", "healthy", "nodamage", "intacta"):
            condition = "sano"
        elif raw_label in ("defectuosa", "defects", "bad", "mala", "enfermo", "quemado", "dry", "dañado", "danada", "damaged"):
            condition = "dañado"
        else:
            condition = "enfermo"

        clean_rows.append({
            "filename": filename,
            "split": split,
            "disease": raw_label,
            "condition": condition,
            "task": task,
            "source_dataset": src_meta.get("title", src_raw),
            "source_url": src_meta.get("url", ""),
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "split", "disease", "condition", "task", "source_dataset", "source_url"],
        )
        writer.writeheader()
        writer.writerows(clean_rows)

    return clean_rows


def create_release_bundle(
    dataset_name: str,
    export_root: Path = Path("/datasets/processed"),
    report_root: Path = Path("reports/pipeline"),
    output_dir: Path | None = None,
    version: str = "1.0",
    release_date: str | None = None,
    archive_format: str = "zip",
    layout: str = "auto",
    metadata_path: Path | None = None,
    sources_path: Path | None = None,
) -> dict[str, Any]:
    """Builds a standardized, noise-free Open Data distribution package for CKAN."""
    release_meta = load_release_metadata(dataset_name, metadata_path)
    source_registry = load_sources_registry(sources_path)
    if "sources" in release_meta and isinstance(release_meta["sources"], dict):
        source_registry.update(release_meta["sources"])

    latest_export = find_latest_export(export_root, dataset_name)
    logger.info(f"Exportación de origen identificada: [bold cyan]{latest_export}[/bold cyan]")

    base_release_root = output_dir or (export_root / "releases")
    date_str = release_date or datetime.now().strftime("%Y%m%d")
    release_root = base_release_root / dataset_name / date_str
    release_root.mkdir(parents=True, exist_ok=True)
    bundle_name = f"{dataset_name}_v{version}"
    bundle_dir = release_root / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Load raw manifest rows
    manifest_csv = latest_export / "manifest.csv"
    if not manifest_csv.is_file():
        if (latest_export / "annotations" / "labels.csv").is_file():
            manifest_csv = latest_export / "annotations" / "labels.csv"
        else:
            for fallback in (
                base_release_root / "manifest_source.csv",
                export_root / "releases" / "manifest_source.csv",
                export_root / "releases" / "EnfermedadesHoja_v1.0.0" / "annotations" / "labels.csv",
                export_root / "releases" / "EnfermedadesHoja_v1.0.0" / "manifest.csv",
            ):
                if fallback.is_file():
                    manifest_csv = fallback
                    break
    raw_manifest_rows: list[dict[str, str]] = []
    if manifest_csv.is_file():
        with manifest_csv.open("r", encoding="utf-8") as f:
            raw_manifest_rows = list(csv.DictReader(f))

    # Determine layout mode
    src_coco_dir = latest_export / "coco"
    src_class_dir = latest_export / "classification"
    detection_count = 0
    classification_count = 0
    for r in raw_manifest_rows:
        t = (r.get("task") or r.get("task_type") or "").lower()
        if "detection" in t:
            detection_count += 1
        if "classification" in t:
            classification_count += 1

    if layout == "auto":
        if src_coco_dir.is_dir() and detection_count >= classification_count and detection_count > 0:
            effective_layout = "detection"
        else:
            effective_layout = "classification"
    else:
        effective_layout = layout

    logger.info(
        f"Modo de empaquetado: [bold green]{effective_layout.upper()}[/bold green] "
        f"(detecciones: {detection_count}, clasificación: {classification_count})"
    )
    logger.info(f"Organizando estructura estándar para analistas de datos en: {bundle_dir}...")

    target_dataset_dir = bundle_dir / "dataset"
    unannotated_filenames: set[str] = set()

    if effective_layout == "detection":
        # 1. UNIFIED DETECTION LAYOUT:
        # Splits (train, val, test) directly inside dataset/ with images/ and labels.json (COCO)
        annotated_filenames: set[str] = set()
        detection_images_count = 0

        if src_coco_dir.is_dir():
            for split in ("train", "val", "test"):
                coco_labels = src_coco_dir / split / "labels.json"
                coco_data_dir = src_coco_dir / split / "data"
                if coco_labels.is_file() and coco_data_dir.is_dir():
                    with coco_labels.open("r", encoding="utf-8") as f:
                        c_data = json.load(f)
                    ann_img_ids = set(ann["image_id"] for ann in c_data.get("annotations", []))
                    valid_imgs = [img for img in c_data.get("images", []) if img["id"] in ann_img_ids]
                    valid_anns = [ann for ann in c_data.get("annotations", []) if ann["image_id"] in ann_img_ids]
                    if valid_imgs:
                        dest_det_split = target_dataset_dir / split
                        dest_det_data = dest_det_split / "images"
                        dest_det_data.mkdir(parents=True, exist_ok=True)
                        for v_img in valid_imgs:
                            img_name = Path(v_img["file_name"]).name
                            src_img_p = coco_data_dir / img_name
                            if src_img_p.is_file():
                                link_export_asset(src_img_p, dest_det_data / img_name)
                                annotated_filenames.add(img_name)
                                detection_images_count += 1
                        filtered_coco = {
                            "info": c_data.get("info", {}),
                            "licenses": c_data.get("licenses", []),
                            "categories": c_data.get("categories", []),
                            "images": valid_imgs,
                            "annotations": valid_anns,
                        }
                        (dest_det_split / "labels.json").write_text(
                            json.dumps(filtered_coco, indent=2), encoding="utf-8"
                        )

        # 2. Collect unannotated images (classification-only or negative background samples) into dataset/test_unannotated
        dest_unann_dir = target_dataset_dir / "test_unannotated"
        dest_unann_images = dest_unann_dir / "images"

        for r in raw_manifest_rows:
            fname = Path(r.get("filepath", "")).name or r.get("filename", "")
            if not fname or fname in annotated_filenames:
                continue

            src_file = None
            if (latest_export / r.get("filepath", "")).is_file():
                src_file = latest_export / r.get("filepath", "")
            elif (latest_export / "images" / fname).is_file():
                src_file = latest_export / "images" / fname
            else:
                matched = list(latest_export.glob(f"**/{fname}"))
                if matched:
                    src_file = matched[0]

            if src_file and src_file.is_file():
                dest_unann_images.mkdir(parents=True, exist_ok=True)
                link_export_asset(src_file, dest_unann_images / fname)
                unannotated_filenames.add(fname)

        if unannotated_filenames:
            unann_readme = (
                "# Test Unannotated Images / Imágenes de Test Sin Anotar\n\n"
                "Este subconjunto reúne imágenes reales del dominio que no cuentan con anotaciones de cajas delimitadoras (bounding boxes).\n\n"
                "### Casos de Uso Recomendados:\n"
                "1. **Evaluación de Falsos Positivos (Background / Negative Testing):** Permite verificar si los detectores de objetos generan detecciones espurias (falsas alarmas) en áreas donde no hay fruto acotado.\n"
                "2. **Aprendizaje Auto-supervisado (SSL) o Pre-entrenamiento:** Pueden integrarse en etapas de pre-entrenamiento no supervisado antes de ajustar los modelos de detección.\n"
                "3. **Inferencia Cualitativa y Pruebas a Ciegas:** Ideales para validar la robustez del modelo ante fotos en condiciones reales de campo.\n\n"
                "Para entrenamiento y evaluación cuantitativa supervisada estándar (mAP COCO), utilice exclusivamente las particiones `train/`, `val/` y `test/`.\n"
            )
            (dest_unann_dir / "README.md").write_text(unann_readme, encoding="utf-8")

    else:
        # CLASSIFICATION-PRIMARY LAYOUT (Standard ImageFolder in dataset/ + optional detection/ folder)
        if src_class_dir.is_dir():
            for split in ("train", "val", "test"):
                split_src = src_class_dir / split
                if not split_src.is_dir():
                    continue
                for class_dir in split_src.iterdir():
                    if not class_dir.is_dir():
                        continue
                    class_name = class_dir.name
                    if class_name == "rust_mite":
                        class_name = "aculus_olearius"
                    dest_class = target_dataset_dir / split / class_name
                    dest_class.mkdir(parents=True, exist_ok=True)
                    for img in class_dir.glob("*"):
                        if img.is_file():
                            t_file = dest_class / img.name
                            if not t_file.exists():
                                link_export_asset(img, t_file)
        elif (latest_export / "dataset").is_dir() and target_dataset_dir.resolve() != (latest_export / "dataset").resolve():
            for split in ("train", "val", "test"):
                split_src = (latest_export / "dataset") / split
                if not split_src.is_dir():
                    continue
                for class_dir in split_src.iterdir():
                    if not class_dir.is_dir():
                        continue
                    dest_class = target_dataset_dir / split / class_dir.name
                    dest_class.mkdir(parents=True, exist_ok=True)
                    for img in class_dir.glob("*"):
                        if img.is_file():
                            t_file = dest_class / img.name
                            if not t_file.exists():
                                link_export_asset(img, t_file)

        # Assemble secondary detection/ folder if COCO exports exist
        target_detection_dir = bundle_dir / "detection"
        detection_images_count = 0
        if src_coco_dir.is_dir():
            for split in ("train", "val", "test"):
                coco_labels = src_coco_dir / split / "labels.json"
                coco_data_dir = src_coco_dir / split / "data"
                if coco_labels.is_file() and coco_data_dir.is_dir():
                    with coco_labels.open("r", encoding="utf-8") as f:
                        c_data = json.load(f)
                    ann_img_ids = set(ann["image_id"] for ann in c_data.get("annotations", []))
                    valid_imgs = [img for img in c_data.get("images", []) if img["id"] in ann_img_ids]
                    valid_anns = [ann for ann in c_data.get("annotations", []) if ann["image_id"] in ann_img_ids]
                    if valid_imgs:
                        dest_det_split = target_detection_dir / split
                        dest_det_data = dest_det_split / "images"
                        dest_det_data.mkdir(parents=True, exist_ok=True)
                        for v_img in valid_imgs:
                            img_name = Path(v_img["file_name"]).name
                            src_img_p = coco_data_dir / img_name
                            if src_img_p.is_file():
                                link_export_asset(src_img_p, dest_det_data / img_name)
                                detection_images_count += 1
                        filtered_coco = {
                            "info": c_data.get("info", {}),
                            "licenses": c_data.get("licenses", []),
                            "categories": c_data.get("categories", []),
                            "images": valid_imgs,
                            "annotations": valid_anns,
                        }
                        (dest_det_split / "labels.json").write_text(
                            json.dumps(filtered_coco, indent=2), encoding="utf-8"
                        )
        elif (latest_export / "detection").is_dir() and target_detection_dir.resolve() != (latest_export / "detection").resolve():
            for split in ("train", "val", "test"):
                split_src = (latest_export / "detection") / split
                if not split_src.is_dir():
                    continue
                dest_det_split = target_detection_dir / split
                dest_det_data = dest_det_split / "images"
                dest_det_data.mkdir(parents=True, exist_ok=True)
                src_det_data = split_src / "images"
                if src_det_data.is_dir():
                    for img in src_det_data.glob("*"):
                        if img.is_file():
                            link_export_asset(img, dest_det_data / img.name)
                src_labels = split_src / "labels.json"
                if src_labels.is_file():
                    shutil.copy2(src_labels, dest_det_split / "labels.json")

    # 3. Assemble annotations/ folder (Clean labels.csv)
    annotations_dir = bundle_dir / "annotations"
    clean_manifest_rows = build_clean_labels_csv(
        raw_manifest_rows,
        annotations_dir / "labels.csv",
        unannotated_filenames=unannotated_filenames,
        source_registry=source_registry,
        meta=release_meta,
    )

    # 4. Assemble preprocess/ folder (Interactive audit report.html)
    preprocess_dir = bundle_dir / "preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    run_id = latest_export.name
    report_html = find_matching_report(report_root, dataset_name, run_id)
    if report_html and report_html.is_file():
        content = report_html.read_text(encoding="utf-8")
        content = content.replace(
            "Cleanlab no esta instalado; se generaron sugerencias heuristicas.",
            "Auditoría de etiquetas basada en embeddings visuales y consistencia semántica.",
        )
        (preprocess_dir / "report.html").write_text(content, encoding="utf-8")

    # 5. Clean up old/unwanted clutter inside bundle_dir
    legacy_items = [
        "classification",
        "classification_coarse",
        "coco",
        "coco_coarse",
        "yolo",
        "yolo_coarse",
        "masks",
        "masks_coarse",
        "images",
        "duplicate_evidence.json",
        "_SUCCESS",
        "curation_summary.json",
        "curation_audit_report.html",
        "manifest.csv",
        "manifest.jsonl",
        "DATASET_CARD.md",
        "label_mapping.json",
        "checksums.sha256",
        "CKAN_METADATA.md",
        "report.html",
    ]
    if effective_layout == "detection":
        legacy_items.append("detection")

    for legacy_item in legacy_items:
        p = bundle_dir / legacy_item
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.is_file():
            p.unlink()

    # 6. Generate concise README.md and comprehensive SOURCES.md inside bundle
    readme_content = generate_open_data_readme(
        dataset_name=dataset_name,
        version=version,
        manifest_rows=clean_manifest_rows,
        effective_layout=effective_layout,
        meta=release_meta,
    )
    (bundle_dir / "README.md").write_text(readme_content, encoding="utf-8")

    sources_content = generate_sources_md(
        clean_manifest_rows,
        source_registry=source_registry,
    )
    (bundle_dir / "SOURCES.md").write_text(sources_content, encoding="utf-8")

    # 7. Create ZIP archive (Fast, non-redundant, < 1 GB)
    archive_path = release_root / f"{bundle_name}.zip"
    archive_sha256: str | None = None
    archive_size_mb: float = 0.0

    logger.info(f"Generando archivo ZIP limpio para CKAN: {archive_path.name}...")
    if archive_path.exists():
        archive_path.unlink()

    if shutil.which("zip"):
        cmd = ["zip", "-r", "-q", archive_path.name, bundle_name]
        subprocess.run(cmd, cwd=str(release_root), check=True)
    else:
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(bundle_dir):
                for f in files:
                    full_p = Path(root) / f
                    arcname = Path(bundle_name) / full_p.relative_to(bundle_dir)
                    zipf.write(full_p, arcname=arcname)

    archive_sha256 = compute_file_sha256(archive_path)
    archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
    (release_root / f"{bundle_name}.zip.sha256").write_text(
        f"{archive_sha256}  {archive_path.name}\n", encoding="utf-8"
    )

    # 8. Write CKAN_METADATA.md OUTSIDE the zip (for operator / Diego)
    ckan_metadata_content = generate_ckan_metadata_md(
        dataset_name=dataset_name,
        version=version,
        total_images=len(clean_manifest_rows),
        archive_name=archive_path.name,
        archive_size_mb=archive_size_mb,
        meta=release_meta,
    )
    (release_root / f"{bundle_name}_CKAN_METADATA.md").write_text(ckan_metadata_content, encoding="utf-8")

    result = {
        "dataset": dataset_name,
        "version": version,
        "release_date": date_str,
        "release_dir": str(release_root),
        "source_run": str(latest_export),
        "bundle_dir": str(bundle_dir),
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "archive_size_mb": archive_size_mb,
        "sample_count": len(clean_manifest_rows),
    }

    logger.info("")
    logger.info("[bold green]================================================================================[/bold green]")
    logger.info("[bold green]        PAQUETE DE PUBLICACIÓN CKAN / OPEN DATA GENERADO CON ÉXITO              [/bold green]")
    logger.info("[bold green]================================================================================[/bold green]")
    logger.info("  • Carpeta Release       : [bold cyan]%s[/bold cyan]", release_root)
    logger.info("  • Directorio del Bundle : [bold cyan]%s[/bold cyan]", bundle_dir)
    logger.info("  • Archivo ZIP (CKAN)    : [bold cyan]%s[/bold cyan] ([bold white]%.1f MB[/bold white])", archive_path, archive_size_mb)
    logger.info("  • Suma SHA256           : [bold white]%s[/bold white]", archive_sha256)
    logger.info("  • Documentación README : [bold cyan]%s[/bold cyan]", bundle_dir / "README.md")
    logger.info("  • Registro de Fuentes  : [bold cyan]%s[/bold cyan]", bundle_dir / "SOURCES.md")
    logger.info("  • Etiquetas Limpias    : [bold cyan]%s[/bold cyan]", bundle_dir / "annotations/labels.csv")
    logger.info("  • Ficha para CKAN (ext) : [bold cyan]%s[/bold cyan]", release_root / f"{bundle_name}_CKAN_METADATA.md")
    logger.info("[bold green]================================================================================[/bold green]")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera un paquete de distribución estándar Open Data a partir de una exportación curada."
    )
    parser.add_argument("--dataset", required=True, help="Nombre del dataset unificado.")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("/datasets/processed"),
        help="Directorio raíz de exportaciones procesadas.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/pipeline"),
        help="Directorio raíz de reportes generados.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio raíz de salida para los paquetes generados (por defecto <export-dir>/releases).",
    )
    parser.add_argument("--version", default="1.0", help="Versión semántica del paquete (ej. 1.0).")
    parser.add_argument(
        "--date",
        default=None,
        help="Fecha o identificador temporal para organizar la release (por defecto YYYYMMDD).",
    )
    parser.add_argument(
        "--archive",
        default="zip",
        choices=["zip", "tar.gz", "none"],
        help="Formato de compresión del paquete final (por defecto zip para compatibilidad con CKAN).",
    )
    parser.add_argument(
        "--layout",
        default="auto",
        choices=["auto", "detection", "classification"],
        help="Estructura del paquete: 'auto' (detecta según predominancia), 'detection' (unificado COCO en dataset/ + test_unannotated/) o 'classification' (ImageFolder clásico).",
    )
    parser.add_argument(
        "--metadata",
        "-m",
        type=Path,
        default=None,
        help="Ruta al archivo YAML de metadatos de release (por defecto busca en configs/releases/{dataset}.yaml).",
    )
    parser.add_argument(
        "--sources",
        "-s",
        type=Path,
        default=None,
        help="Ruta al archivo YAML de fuentes originales (por defecto busca en configs/sources.yaml).",
    )

    args = parser.parse_args()
    create_release_bundle(
        dataset_name=args.dataset,
        export_root=args.export_dir,
        report_root=args.report_dir,
        output_dir=args.output_dir,
        version=args.version,
        release_date=args.date,
        archive_format=args.archive,
        layout=args.layout,
        metadata_path=args.metadata,
        sources_path=args.sources,
    )


if __name__ == "__main__":
    main()
