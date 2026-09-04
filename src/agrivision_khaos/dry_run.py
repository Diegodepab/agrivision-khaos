from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import cv2
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    source: str
    path: str
    message: str


@dataclass
class AuditedAsset:
    path: Path
    source: str
    labels: set[str]
    sha256: str | None = None


class StaticDatasetAuditor:
    """Validates common vision formats without FiftyOne, MongoDB, or ML models."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir
        self.issues: list[AuditIssue] = []
        self.assets: dict[Path, AuditedAsset] = {}
        self.source_reports: list[dict[str, Any]] = []
        self.annotation_count = 0

    def issue(self, severity: str, code: str, source: str, path: Path, message: str) -> None:
        self.issues.append(
            AuditIssue(severity, code, source, str(path), message)
        )

    def asset(self, path: Path, source: str, labels: set[str]) -> None:
        resolved = path.resolve()
        try:
            resolved.relative_to(self.raw_dir.resolve())
        except ValueError:
            self.issue(
                "error",
                "external_media_path",
                source,
                path,
                "La anotación referencia contenido fuera del directorio raw",
            )
            return
        current = self.assets.get(resolved)
        if current is None:
            current = AuditedAsset(resolved, source, set())
            self.assets[resolved] = current
        current.labels.update(label for label in labels if label)

    def validate_asset(self, asset: AuditedAsset) -> None:
        if not asset.path.is_file():
            self.issue("error", "missing_image", asset.source, asset.path, "La imagen no existe")
            return
        image = cv2.imread(str(asset.path), cv2.IMREAD_UNCHANGED)
        if image is None:
            self.issue(
                "error", "corrupt_image", asset.source, asset.path, "OpenCV no puede decodificar la imagen"
            )
            return
        digest = hashlib.sha256()
        with asset.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        asset.sha256 = digest.hexdigest()

    @staticmethod
    def _finite_numbers(values: Any, expected: int) -> list[float] | None:
        if not isinstance(values, list) or len(values) != expected:
            return None
        try:
            numbers = [float(value) for value in values]
        except (TypeError, ValueError):
            return None
        return numbers if all(math.isfinite(number) for number in numbers) else None

    def audit_coco(self, source: Path, json_paths: list[Path]) -> None:
        source_images = 0
        source_annotations = 0
        for json_path in json_paths:
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                self.issue("error", "invalid_coco_json", source.name, json_path, str(exc))
                continue
            images = payload.get("images")
            annotations = payload.get("annotations", [])
            categories = payload.get("categories", [])
            if not isinstance(images, list) or not isinstance(annotations, list):
                self.issue(
                    "error", "invalid_coco_schema", source.name, json_path,
                    "'images' y 'annotations' deben ser listas",
                )
                continue
            category_names = {
                category.get("id"): str(category.get("name", ""))
                for category in categories
                if isinstance(category, dict)
            }
            image_records = {
                image.get("id"): image
                for image in images
                if isinstance(image, dict) and image.get("id") is not None
            }
            labels_by_image: dict[Any, set[str]] = defaultdict(set)
            for index, annotation in enumerate(annotations):
                if not isinstance(annotation, dict):
                    self.issue("error", "invalid_coco_annotation", source.name, json_path, f"Anotación {index} no es un objeto")
                    continue
                image_record = image_records.get(annotation.get("image_id"))
                label = category_names.get(annotation.get("category_id"), "")
                if image_record is None:
                    self.issue("error", "orphan_annotation", source.name, json_path, f"Anotación {index} referencia image_id inexistente")
                if not label:
                    self.issue("error", "unknown_category", source.name, json_path, f"Anotación {index} referencia una categoría inexistente")
                bbox = self._finite_numbers(annotation.get("bbox"), 4)
                if bbox is None:
                    self.issue("error", "invalid_bbox", source.name, json_path, f"Anotación {index}: bbox debe contener cuatro números finitos")
                elif image_record is not None:
                    x, y, width, height = bbox
                    image_width = float(image_record.get("width", 0) or 0)
                    image_height = float(image_record.get("height", 0) or 0)
                    if width <= 0 or height <= 0 or x < 0 or y < 0 or (
                        image_width > 0 and x + width > image_width
                    ) or (image_height > 0 and y + height > image_height):
                        self.issue("error", "bbox_out_of_bounds", source.name, json_path, f"Anotación {index}: bbox fuera de los límites declarados")
                if image_record is not None:
                    labels_by_image[annotation.get("image_id")].add(label)
                source_annotations += 1
            for image_id, image in image_records.items():
                filename = image.get("file_name")
                if not isinstance(filename, str) or not filename:
                    self.issue("error", "missing_file_name", source.name, json_path, f"Imagen {image_id} sin file_name")
                    continue
                self.asset(json_path.parent / filename, source.name, labels_by_image[image_id])
                source_images += 1
        self.annotation_count += source_annotations
        self.source_reports.append(
            {"name": source.name, "format": "coco", "images": source_images, "annotations": source_annotations}
        )

    @staticmethod
    def _yolo_names(payload: dict[str, Any]) -> dict[int, str]:
        names = payload.get("names", [])
        if isinstance(names, list):
            return {index: str(name) for index, name in enumerate(names)}
        if isinstance(names, dict):
            result: dict[int, str] = {}
            for key, value in names.items():
                try:
                    result[int(key)] = str(value)
                except (TypeError, ValueError):
                    continue
            return result
        return {}

    def audit_yolo(self, source: Path, yaml_path: Path) -> None:
        try:
            payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            self.issue("error", "invalid_yolo_yaml", source.name, yaml_path, str(exc))
            self.source_reports.append({"name": source.name, "format": "yolo", "images": 0, "annotations": 0})
            return
        if not isinstance(payload, dict):
            self.issue("error", "invalid_yolo_schema", source.name, yaml_path, "El YAML debe ser un objeto")
            return
        names = self._yolo_names(payload)
        if not names:
            self.issue("error", "missing_yolo_names", source.name, yaml_path, "No se definieron clases")
        declared_root = payload.get("path", ".")
        dataset_root = source
        if isinstance(declared_root, str):
            candidate_root = Path(declared_root)
            candidate_root = candidate_root if candidate_root.is_absolute() else source / candidate_root
            if candidate_root.is_dir():
                dataset_root = candidate_root.resolve()
        source_images = 0
        source_annotations = 0
        for split in ("train", "val", "test"):
            split_value = payload.get(split)
            if not isinstance(split_value, str):
                continue
            split_path = Path(split_value)
            image_dir = split_path if split_path.is_absolute() else dataset_root / split_path
            if not image_dir.is_dir():
                self.issue("error", "missing_yolo_split", source.name, image_dir, f"No existe el split {split}")
                continue
            for image_path in sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
                relative = image_path.relative_to(image_dir)
                label_candidates = [
                    dataset_root / "labels" / split / relative.with_suffix(".txt"),
                    image_dir.parent / "labels" / relative.with_suffix(".txt"),
                ]
                try:
                    relative_to_images = image_path.relative_to(dataset_root / "images")
                    label_candidates.insert(
                        0,
                        dataset_root / "labels" / relative_to_images.with_suffix(".txt"),
                    )
                except ValueError:
                    pass
                label_path = next(
                    (candidate for candidate in label_candidates if candidate.is_file()),
                    label_candidates[0],
                )
                labels: set[str] = set()
                if label_path.is_file():
                    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                        parts = line.split()
                        if len(parts) != 5:
                            self.issue("error", "invalid_yolo_row", source.name, label_path, f"Línea {line_number}: se esperaban 5 columnas")
                            continue
                        try:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:])
                        except ValueError:
                            self.issue("error", "invalid_yolo_row", source.name, label_path, f"Línea {line_number}: valores no numéricos")
                            continue
                        label = names.get(class_id, "")
                        if not label:
                            self.issue("error", "unknown_yolo_class", source.name, label_path, f"Línea {line_number}: clase {class_id} inexistente")
                        labels.add(label)
                        if not all(math.isfinite(value) for value in (x_center, y_center, width, height)) or width <= 0 or height <= 0 or x_center - width / 2 < 0 or y_center - height / 2 < 0 or x_center + width / 2 > 1 or y_center + height / 2 > 1:
                            self.issue("error", "bbox_out_of_bounds", source.name, label_path, f"Línea {line_number}: caja YOLO fuera de [0, 1]")
                        source_annotations += 1
                else:
                    self.issue("warning", "missing_yolo_label", source.name, label_path, "Imagen sin archivo de etiquetas")
                self.asset(image_path, source.name, labels)
                source_images += 1
        self.annotation_count += source_annotations
        self.source_reports.append(
            {"name": source.name, "format": "yolo", "images": source_images, "annotations": source_annotations}
        )

    def audit_voc(self, source: Path, xml_paths: list[Path]) -> None:
        image_dirs = [path for path in source.iterdir() if path.is_dir() and path.name.lower() in {"jpegimages", "images", "original images"}]
        image_dir = image_dirs[0] if image_dirs else source
        source_annotations = 0
        for xml_path in xml_paths:
            try:
                root = ET.parse(xml_path).getroot()
            except (OSError, ET.ParseError) as exc:
                self.issue("error", "invalid_voc_xml", source.name, xml_path, str(exc))
                continue
            filename = (root.findtext("filename") or "").strip()
            labels: set[str] = set()
            try:
                declared_width = float(root.findtext("size/width") or 0)
                declared_height = float(root.findtext("size/height") or 0)
            except ValueError:
                declared_width = declared_height = 0
            for index, obj in enumerate(root.findall("object")):
                label = (obj.findtext("name") or "").strip()
                labels.add(label)
                try:
                    xmin = float(obj.findtext("bndbox/xmin") or "nan")
                    ymin = float(obj.findtext("bndbox/ymin") or "nan")
                    xmax = float(obj.findtext("bndbox/xmax") or "nan")
                    ymax = float(obj.findtext("bndbox/ymax") or "nan")
                except ValueError:
                    xmin = ymin = xmax = ymax = math.nan
                if not label:
                    self.issue("error", "missing_voc_label", source.name, xml_path, f"Objeto {index} sin etiqueta")
                if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)) or xmin < 0 or ymin < 0 or xmax <= xmin or ymax <= ymin or (declared_width > 0 and xmax > declared_width) or (declared_height > 0 and ymax > declared_height):
                    self.issue("error", "bbox_out_of_bounds", source.name, xml_path, f"Objeto {index}: caja VOC inválida")
                source_annotations += 1
            if not filename:
                self.issue("error", "missing_file_name", source.name, xml_path, "VOC sin filename")
            else:
                self.asset(image_dir / filename, source.name, labels)
        self.annotation_count += source_annotations
        self.source_reports.append(
            {"name": source.name, "format": "voc", "images": len(xml_paths), "annotations": source_annotations}
        )

    def audit(self) -> dict[str, Any]:
        if not self.raw_dir.is_dir():
            self.issue("error", "missing_raw_dir", "", self.raw_dir, "El directorio raw no existe")
        else:
            for source in sorted(path for path in self.raw_dir.iterdir() if path.is_dir()):
                yolo_path = next((path for path in (source / "data.yaml", source / "dataset.yaml") if path.is_file()), None)
                coco_paths = sorted(
                    path for path in source.rglob("*.json")
                    if any(
                        hint in path.name.lower()
                        for hint in ("coco", "annotation", "instances")
                    )
                )
                xml_paths = sorted(source.rglob("*.xml"))
                if yolo_path is not None:
                    self.audit_yolo(source, yolo_path)
                elif coco_paths:
                    self.audit_coco(source, coco_paths)
                elif xml_paths:
                    self.audit_voc(source, xml_paths)
                else:
                    images = sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
                    if not images:
                        self.issue("warning", "unsupported_source", source.name, source, "No se detectó un formato compatible")
                        continue
                    for image in images:
                        self.asset(image, source.name, {image.parent.name})
                    self.source_reports.append({"name": source.name, "format": "classification_tree", "images": len(images), "annotations": len(images)})
                    self.annotation_count += len(images)

        for asset in self.assets.values():
            self.validate_asset(asset)

        hash_groups: dict[str, list[AuditedAsset]] = defaultdict(list)
        for asset in self.assets.values():
            if asset.sha256:
                hash_groups[asset.sha256].append(asset)
        duplicates = []
        conflicts = []
        for digest, assets in sorted(hash_groups.items()):
            if len(assets) < 2:
                continue
            record = {
                "sha256": digest,
                "paths": [str(asset.path) for asset in assets],
                "labels": sorted({label for asset in assets for label in asset.labels}),
            }
            duplicates.append(record)
            label_sets = {frozenset(asset.labels) for asset in assets if asset.labels}
            if len(label_sets) > 1:
                conflicts.append(record)
                self.issue(
                    "error", "conflicting_duplicate_labels", "multiple", assets[0].path,
                    f"El mismo contenido tiene etiquetas incompatibles: {record['labels']}",
                )

        errors = sum(issue.severity == "error" for issue in self.issues)
        warnings = sum(issue.severity == "warning" for issue in self.issues)
        return {
            "mode": "dry-run",
            "raw_dir": str(self.raw_dir.resolve()),
            "valid": errors == 0,
            "counts": {
                "sources": len(self.source_reports),
                "images": len(self.assets),
                "annotations": self.annotation_count,
                "errors": errors,
                "warnings": warnings,
                "duplicate_groups": len(duplicates),
                "conflicting_duplicate_groups": len(conflicts),
            },
            "sources": self.source_reports,
            "issues": [asdict(issue) for issue in self.issues],
            "duplicate_groups": duplicates,
            "conflicting_duplicate_groups": conflicts,
        }


def audit_raw_datasets(raw_dir: Path) -> dict[str, Any]:
    return StaticDatasetAuditor(raw_dir).audit()
