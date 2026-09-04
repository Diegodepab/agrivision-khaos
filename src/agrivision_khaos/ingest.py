from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import fiftyone as fo
import fiftyone.types as fot
from pydantic import BaseModel
from rich.logging import RichHandler

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger(__name__)
COCO_JSON_HINTS = ("annotation", "coco", "instances")


class IngestionError(RuntimeError):
    """Raised when a dataset cannot be assembled without partial data loss."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_staged_text(path: Path, content: str) -> None:
    """Atomically writes generated metadata outside of the immutable raw tree."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            Path(temporary_name).unlink(missing_ok=True)
        finally:
            raise


class DatasetConfig(BaseModel):
    name: str
    path: Path
    dataset_type: type
    splits: Optional[List[str]] = None
    label_field: str = "ground_truth"
    data_path: Optional[str] = None
    labels_path: Optional[str] = None


def _coco_json_candidates(directory: Path, recursive: bool = False) -> list[Path]:
    paths = directory.rglob("*.json") if recursive else directory.glob("*.json")
    return sorted(
        path for path in paths if any(hint in path.name.lower() for hint in COCO_JSON_HINTS)
    )


def _select_single_coco_json(directory: Path) -> Path:
    candidates = _coco_json_candidates(directory)
    if not candidates:
        raise IngestionError(f"No se encontró un JSON COCO en {directory}")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise IngestionError(
            f"Hay varios JSON COCO candidatos en {directory}: {names}. "
            "Deja un descriptor inequívoco por split."
        )
    return candidates[0]


def _find_file_early_exit(root_dir: Path, target_exts: set[str], target_names: set[str] = None, max_depth: int = 3) -> Path | None:
    """
    Realiza una búsqueda en anchura (BFS) limitando la profundidad y retorna el primer archivo
    que coincida con las extensiones o nombres, abortando la búsqueda tempranamente para no
    comprometer el rendimiento.
    """
    stack = [(root_dir, 0)]
    while stack:
        current_dir, depth = stack.pop(0)
        if depth > max_depth:
            continue
            
        try:
            for item in current_dir.iterdir():
                if item.is_file():
                    if item.suffix.lower() in target_exts or (target_names and item.name.lower() in target_names):
                        return item
                elif item.is_dir():
                    stack.append((item, depth + 1))
        except Exception:
            pass
            
    return None


def discover_datasets(raw_data_dir: Path) -> List[DatasetConfig]:
    """
    Descubre automáticamente los datasets en el directorio de datos crudos
    y utiliza heurísticas para determinar el formato de cada uno.
    """
    if not raw_data_dir.exists():
        logger.warning(f"El directorio de datos crudos no existe: {raw_data_dir}")
        return []

    configs = []
    
    for item in sorted(raw_data_dir.iterdir()):
        if not item.is_dir():
            continue
            
        dataset_name = item.name
        
        # Heurística 1: ¿Es COCO? (Busca archivos _annotations.coco.json)
        coco_files = _coco_json_candidates(item, recursive=True)
        if coco_files:
            # Detectar splits si existen carpetas train/test/valid
            splits = sorted(
                s.name
                for s in item.iterdir()
                if s.is_dir() and s.name in ("train", "test", "valid", "val")
            )
            
            configs.append(
                DatasetConfig(
                    name=dataset_name,
                    path=item,
                    dataset_type=fot.COCODetectionDataset,
                    splits=splits if splits else None,
                    # COCO usa label_field como prefijo y añade "_detections".
                    label_field="coco",
                )
            )
            continue
            
        # Heurística YOLO: Busca data.yaml o dataset.yaml
        yolo_file = _find_file_early_exit(item, set(), {"data.yaml", "dataset.yaml"})
        if yolo_file:
            configs.append(
                DatasetConfig(
                    name=dataset_name,
                    path=yolo_file.parent,
                    dataset_type=fot.YOLOv5Dataset,
                    label_field="yolo_detections",
                )
            )
            continue
            
        # Heurística PASCAL VOC: Busca un archivo .xml
        voc_file = _find_file_early_exit(item, {".xml"})
        if voc_file:
            annotations_dir = voc_file.parent
            base_dir = annotations_dir.parent
            data_path = None
            # Intentar encontrar la carpeta de imágenes asumiendo que está junto a Annotations
            for child in base_dir.iterdir():
                if child.is_dir() and child.name.lower() in ("jpegimages", "images", "original images"):
                    data_path = str(child.resolve())
                    break
                    
            configs.append(
                DatasetConfig(
                    name=dataset_name,
                    path=base_dir,
                    dataset_type=fot.VOCDetectionDataset,
                    label_field="voc_detections",
                    data_path=data_path,
                    labels_path=str(annotations_dir.resolve()),
                )
            )
            continue
            
        # Heurística Fallback: Clasificación de Imágenes (Carpetas)
        # Buscamos la primera imagen (hasta 5 niveles de profundidad) para deducir la ruta real de las clases
        image_file = _find_file_early_exit(item, {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}, max_depth=5)
        
        if image_file:
            # image_file.parent es la carpeta de la clase (ej. 'Anthracnose')
            # image_file.parent.parent es donde están todas las carpetas de clases
            base_dir = image_file.parent.parent
            
            # Si el abuelo es una carpeta de split (train, val, test), entonces el bisabuelo es la raíz del dataset
            if base_dir.name.lower() in ("train", "test", "valid", "val"):
                base_dir = base_dir.parent
                
            splits = [s.name for s in base_dir.iterdir() if s.is_dir() and s.name.lower() in ("train", "test", "valid", "val")]
            
            configs.append(
                DatasetConfig(
                    name=dataset_name,
                    path=base_dir,
                    dataset_type=fot.ImageClassificationDirectoryTree,
                    splits=splits if splits else None,
                    label_field="original_label",
                )
            )
        else:
            logger.warning(f"No se detectaron imágenes en {item.name}, omitiendo dataset.")
        
    return configs


def _stage_coco_json(
    json_path: Path,
    data_dir: Path,
    staging_dir: Path,
    strict: bool = True,
) -> tuple[Path, dict[str, object]]:
    """
    Creates a validated COCO metadata copy without modifying the source JSON.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    images = data.get("images")
    if not isinstance(images, list):
        raise IngestionError(f"COCO inválido, falta la lista 'images': {json_path}")

    data_root = data_dir.resolve()
    valid_images = []
    valid_image_ids = set()
    for image in images:
        file_name = image.get("file_name") if isinstance(image, dict) else None
        if not file_name:
            continue
        candidate = (data_root / str(file_name)).resolve()
        try:
            candidate.relative_to(data_root)
        except ValueError:
            continue
        if candidate.is_file():
            valid_images.append(image)
            valid_image_ids.add(image.get("id"))

    removed_images = len(images) - len(valid_images)
    annotations = data.get("annotations", [])
    if not isinstance(annotations, list):
        raise IngestionError(f"COCO inválido, 'annotations' no es una lista: {json_path}")
    valid_annotations = [
        annotation
        for annotation in annotations
        if isinstance(annotation, dict) and annotation.get("image_id") in valid_image_ids
    ]
    removed_annotations = len(annotations) - len(valid_annotations)
    if strict and (removed_images or removed_annotations):
        raise IngestionError(
            f"COCO referencia {removed_images} imágenes ausentes/no seguras y "
            f"{removed_annotations} anotaciones huérfanas: {json_path}"
        )
    data["images"] = valid_images
    if "annotations" in data:
        data["annotations"] = valid_annotations

    staged_path = staging_dir / json_path.name
    _write_staged_text(staged_path, json.dumps(data, ensure_ascii=False))
    return staged_path, {
        "source": str(json_path),
        "source_sha256": _sha256(json_path),
        "staged": str(staged_path),
        "images": len(valid_images),
        "removed_missing_images": removed_images,
        "removed_orphan_annotations": removed_annotations,
    }


def _stage_yolo_yaml(yaml_path: Path, dataset_root: Path, staging_dir: Path) -> tuple[Path, dict[str, object]]:
    """
    Creates a portable YOLO descriptor that points at the local immutable media.
    """
    import yaml

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise IngestionError(f"YAML de YOLO inválido: {yaml_path}")

    data["path"] = str(dataset_root.resolve())
    resolved_splits: dict[str, str] = {}
    for key in ("train", "val", "test"):
        value = data.get(key)
        if not isinstance(value, str):
            continue

        original = Path(value.replace("\\", "/"))
        candidates = [] if original.is_absolute() else [dataset_root / original]
        candidates.extend(
            dataset_root / candidate
            for candidate in (
                Path(key),
                Path("images") / key,
                Path("images") / original.name,
                Path(original.name),
            )
        )
        match = next((candidate for candidate in candidates if candidate.exists()), None)
        if match is None:
            raise IngestionError(f"No se pudo resolver el split YOLO '{key}={value}' en {dataset_root}")

        relative = match.resolve().relative_to(dataset_root.resolve())
        data[key] = relative.as_posix()
        resolved_splits[key] = data[key]

    if not resolved_splits:
        raise IngestionError(f"El YAML de YOLO no define ningún split utilizable: {yaml_path}")

    staged_path = staging_dir / yaml_path.name
    _write_staged_text(staged_path, yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    return staged_path, {
        "source": str(yaml_path),
        "source_sha256": _sha256(yaml_path),
        "staged": str(staged_path),
        "splits": resolved_splits,
    }


def _promote_dataset(candidate: fo.Dataset, target_name: str, run_token: str) -> fo.Dataset:
    """Atomically promotes a completed dataset while preserving rollback capability."""
    backup_name = f"{target_name}__backup__{run_token}"
    previous = fo.load_dataset(target_name) if fo.dataset_exists(target_name) else None
    if previous is not None:
        previous.name = backup_name

    try:
        candidate.name = target_name
        candidate.persistent = True
    except Exception:
        if previous is not None:
            previous.name = target_name
        raise

    if previous is not None and fo.dataset_exists(backup_name):
        fo.delete_dataset(backup_name)
    return candidate


def create_unified_dataset(
    dataset_name: str,
    raw_data_dir: Path,
    staging_dir: Path | None = None,
    strict: bool = True,
) -> fo.Dataset:
    """
    Builds a candidate dataset and only replaces the current dataset after success.
    """
    run_token = uuid4().hex[:12]
    candidate_name = f"{dataset_name}__building__{run_token}"
    staging_dir = staging_dir or Path("data/interim/ingestion") / dataset_name / run_token
    staging_dir.mkdir(parents=True, exist_ok=True)

    dataset = fo.Dataset(candidate_name)
    dataset.persistent = False
    dataset.media_type = "image"
    source_records: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    configs = discover_datasets(raw_data_dir)
    
    if not configs:
        fo.delete_dataset(candidate_name)
        raise IngestionError(f"No se encontraron datasets para ingestar en {raw_data_dir}")

    for config in configs:
        logger.info(f"Ingestando {config.name} como {config.dataset_type.__name__}...")
        source_start_count = len(dataset)
        source_record: dict[str, object] = {
            "name": config.name,
            "path": str(config.path),
            "format": config.dataset_type.__name__,
            "status": "pending",
        }
        
        try:
            if config.dataset_type == fot.COCODetectionDataset:
                if config.splits:
                    for split in config.splits:
                        split_path = config.path / split
                        # Asume el estandar de Roboflow para COCO json
                        labels_source = _select_single_coco_json(split_path)
                        labels_path, metadata = _stage_coco_json(
                            labels_source,
                            split_path,
                            staging_dir / config.name / split,
                            strict=strict,
                        )
                        source_record.setdefault("metadata", []).append(metadata)
                            
                        dataset.add_dir(
                            dataset_type=config.dataset_type,
                            data_path=str(split_path),
                            labels_path=str(labels_path),
                            label_field=config.label_field,
                            tags=[config.name, split],
                        )
                else:
                    labels_source = _select_single_coco_json(config.path)
                    labels_path, metadata = _stage_coco_json(
                        labels_source,
                        config.path,
                        staging_dir / config.name,
                        strict=strict,
                    )
                    source_record.setdefault("metadata", []).append(metadata)
                    dataset.add_dir(
                        dataset_type=config.dataset_type,
                        data_path=str(config.path),
                        labels_path=str(labels_path),
                        label_field=config.label_field,
                        tags=[config.name],
                    )
            
            elif config.dataset_type == fot.YOLOv5Dataset:
                yaml_files = list(config.path.glob("*.yaml"))
                yaml_path = None
                resolved_splits: list[str] = []
                if yaml_files:
                    staged_yaml, metadata = _stage_yolo_yaml(
                        yaml_files[0], config.path, staging_dir / config.name
                    )
                    yaml_path = str(staged_yaml.resolve())
                    source_record["metadata"] = [metadata]
                    resolved_splits = list(metadata["splits"])
                
                if not yaml_path or not resolved_splits:
                    raise IngestionError(f"No se encontró un descriptor YOLO válido en {config.path}")
                for split in resolved_splits:
                    dataset.add_dir(
                        dataset_dir=str(config.path),
                        dataset_type=config.dataset_type,
                        label_field=config.label_field,
                        tags=[config.name, split],
                        yaml_path=yaml_path,
                        split=split,
                    )
                
            elif config.dataset_type == fot.VOCDetectionDataset:
                kwargs = {
                    "dataset_dir": str(config.path),
                    "dataset_type": config.dataset_type,
                    "label_field": config.label_field,
                    "tags": [config.name],
                }
                if config.data_path:
                    kwargs["data_path"] = config.data_path
                if config.labels_path:
                    kwargs["labels_path"] = config.labels_path
                    
                dataset.add_dir(**kwargs)
            
            elif config.dataset_type == fot.ImageClassificationDirectoryTree:
                # Ingestión manual robusta para filtrar archivos basura (.docx, .txt, etc.)
                valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
                samples = []
                
                splits_to_process = config.splits if config.splits else [None]
                
                for split in splits_to_process:
                    split_path = config.path / split if split else config.path
                    if not split_path.exists():
                        continue
                        
                    for cls_dir in split_path.iterdir():
                        if not cls_dir.is_dir():
                            continue
                            
                        label = cls_dir.name
                        for img_path in cls_dir.iterdir():
                            if img_path.is_file() and img_path.suffix.lower() in valid_exts:
                                sample = fo.Sample(filepath=str(img_path))
                                sample[config.label_field] = fo.Classification(label=label)
                                if split:
                                    sample.tags.extend([config.name, split])
                                else:
                                    sample.tags.append(config.name)
                                samples.append(sample)
                                
                if samples:
                    dataset.add_samples(samples)
                else:
                    logger.warning(f"No se encontraron imágenes válidas en {config.name}")
                    raise IngestionError(f"No se encontraron imágenes válidas en {config.name}")

            added_samples = len(dataset) - source_start_count
            if added_samples <= 0:
                raise IngestionError(f"La fuente {config.name} no añadió ninguna muestra")
            source_record["status"] = "ingested"
            source_record["added_samples"] = added_samples
            source_record["sample_count_after"] = len(dataset)
        except Exception as e:
            logger.error(f"Error al ingestar {config.name}: {str(e)}")
            source_record["status"] = "failed"
            source_record["error"] = str(e)
            errors.append({"source": config.name, "error": str(e)})
        finally:
            source_records.append(source_record)

    report = {
        "dataset": dataset_name,
        "candidate_dataset": candidate_name,
        "raw_dir": str(raw_data_dir),
        "strict": strict,
        "sources": source_records,
        "errors": errors,
        "sample_count": len(dataset),
    }
    _write_staged_text(
        staging_dir / "ingestion_report.json",
        json.dumps(report, ensure_ascii=False, indent=2),
    )

    if errors and strict:
        fo.delete_dataset(candidate_name)
        failed = ", ".join(error["source"] for error in errors)
        raise IngestionError(
            f"Ingesta cancelada: fallaron {len(errors)} fuente(s): {failed}. "
            f"Consulta {staging_dir / 'ingestion_report.json'}"
        )
    if len(dataset) == 0:
        fo.delete_dataset(candidate_name)
        raise IngestionError("La ingesta no produjo ninguna muestra válida")

    dataset.info["ingestion"] = report
    dataset.save()
    dataset = _promote_dataset(dataset, dataset_name, run_token)

    logger.info(f"Ingesta completada. Dataset guardado en FiftyOne DB con {len(dataset)} muestras.")
    return dataset


def main() -> None:
    """CLI entry point for dataset discovery and ingestion."""
    parser = argparse.ArgumentParser(description="Ingesta dinámica de datasets en FiftyOne.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="agrivision-dataset",
        help="Nombre del dataset final unificado en FiftyOne.",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directorio donde se encuentran las carpetas crudas a ingestar.",
    )
    parser.add_argument(
        "--staging-dir",
        type=str,
        default=None,
        help="Directorio para metadatos validados; nunca se escribe en raw.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Promueve fuentes válidas aunque otras fallen; desactivado por defecto.",
    )
    
    args = parser.parse_args()
    
    raw_path = Path(args.raw_dir)
    unified_dataset = create_unified_dataset(
        dataset_name=args.dataset_name,
        raw_data_dir=raw_path,
        staging_dir=Path(args.staging_dir) if args.staging_dir else None,
        strict=not args.allow_partial,
    )
    
    print("\nResumen del dataset:")
    print(unified_dataset)
    print(f"\nPara visualizar el dataset, corre:\nmake app DATASET=\"{args.dataset_name}\"")


if __name__ == "__main__":
    main()
