from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import yaml

IMAGE_SIZE = 96
CLASS_NAMES = ("healthy", "diseased")


def _require_empty_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"El directorio de salida no está vacío: {path}. "
            "Usa una ruta nueva para no sobrescribir datos."
        )
    path.mkdir(parents=True, exist_ok=True)


def _image(rng: np.random.Generator, index: int) -> np.ndarray:
    image = rng.integers(0, 150, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    color = (40 + (index * 29) % 180, 180, 60 + (index * 17) % 160)
    cv2.rectangle(image, (10, 12), (45, 42), color, thickness=-1)
    cv2.circle(image, (68, 65), 12 + index % 5, (210, 90, 40), thickness=-1)
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"No se pudo escribir la imagen sintética: {path}")


def _write_manifest(path: Path, task: str) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "version": "synthetic-1",
                "license": "CC0-1.0",
                "tasks": [task],
                "sensor": "synthetic-rgb",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _generate_coco(root: Path, images: list[np.ndarray], include_edge_cases: bool) -> Path:
    source = root / "mock_coco"
    split = source / "train"
    split.mkdir(parents=True)
    categories = [{"id": index + 1, "name": name} for index, name in enumerate(CLASS_NAMES)]
    coco_images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    for index, image in enumerate(images):
        filename = f"coco-{index:02d}.jpg"
        _write_image(split / filename, image)
        coco_images.append(
            {"id": index + 1, "file_name": filename, "width": IMAGE_SIZE, "height": IMAGE_SIZE}
        )
        bbox = [-4, 12, 35, 30] if include_edge_cases and index == 1 else [10, 12, 35, 30]
        annotations.append(
            {
                "id": index + 1,
                "image_id": index + 1,
                "category_id": index % len(CLASS_NAMES) + 1,
                "bbox": bbox,
                "area": 1050,
                "iscrowd": 0,
            }
        )
    (split / "_annotations.coco.json").write_text(
        json.dumps(
            {"images": coco_images, "annotations": annotations, "categories": categories},
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_manifest(source / "source.yaml", "detection")
    return split / "coco-00.jpg"


def _generate_yolo(
    root: Path,
    images: list[np.ndarray],
    include_edge_cases: bool,
    conflict_source: Path,
) -> None:
    source = root / "mock_yolo"
    image_dir = source / "images" / "train"
    label_dir = source / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index, image in enumerate(images):
        stem = f"yolo-{index:02d}"
        _write_image(image_dir / f"{stem}.jpg", image)
        (label_dir / f"{stem}.txt").write_text(
            f"{index % len(CLASS_NAMES)} 0.286458 0.281250 0.364583 0.312500\n",
            encoding="utf-8",
        )
    if include_edge_cases:
        shutil.copyfile(conflict_source, image_dir / "conflicting-duplicate.jpg")
        (label_dir / "conflicting-duplicate.txt").write_text(
            "1 0.286458 0.281250 0.364583 0.312500\n",
            encoding="utf-8",
        )
    (source / "data.yaml").write_text(
        yaml.safe_dump(
            {"path": ".", "train": "images/train", "names": list(CLASS_NAMES)},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_manifest(source / "source.yaml", "detection")


def _voc_annotation(filename: str, label: str) -> ET.Element:
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = filename
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(IMAGE_SIZE)
    ET.SubElement(size, "height").text = str(IMAGE_SIZE)
    ET.SubElement(size, "depth").text = "3"
    obj = ET.SubElement(annotation, "object")
    ET.SubElement(obj, "name").text = label
    box = ET.SubElement(obj, "bndbox")
    for key, value in (("xmin", 10), ("ymin", 12), ("xmax", 45), ("ymax", 42)):
        ET.SubElement(box, key).text = str(value)
    return annotation


def _generate_voc(root: Path, images: list[np.ndarray], include_edge_cases: bool) -> None:
    source = root / "mock_voc"
    image_dir = source / "JPEGImages"
    annotation_dir = source / "Annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    for index, image in enumerate(images):
        filename = f"voc-{index:02d}.jpg"
        _write_image(image_dir / filename, image)
        tree = ET.ElementTree(_voc_annotation(filename, CLASS_NAMES[index % len(CLASS_NAMES)]))
        tree.write(annotation_dir / f"voc-{index:02d}.xml", encoding="utf-8", xml_declaration=True)
    if include_edge_cases:
        corrupt_name = "corrupt.jpg"
        (image_dir / corrupt_name).write_bytes(b"this-is-not-a-decodable-image")
        ET.ElementTree(_voc_annotation(corrupt_name, "healthy")).write(
            annotation_dir / "corrupt.xml", encoding="utf-8", xml_declaration=True
        )
    _write_manifest(source / "source.yaml", "detection")


def generate_mock_dataset(
    output_dir: Path,
    samples_per_format: int = 10,
    seed: int = 42,
    include_edge_cases: bool = False,
) -> dict[str, object]:
    if samples_per_format < 1:
        raise ValueError("samples_per_format debe ser al menos 1")
    _require_empty_directory(output_dir)
    rng = np.random.default_rng(seed)
    images = [_image(rng, index) for index in range(samples_per_format)]
    conflict_source = _generate_coco(output_dir, images, include_edge_cases)
    _generate_yolo(output_dir, images, include_edge_cases, conflict_source)
    _generate_voc(output_dir, images, include_edge_cases)
    summary = {
        "output_dir": str(output_dir.resolve()),
        "seed": seed,
        "samples_per_format": samples_per_format,
        "formats": ["coco", "yolo", "voc"],
        "edge_cases": include_edge_cases,
    }
    (output_dir / "mock_dataset.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera fixtures pequeñas COCO, YOLO y VOC.")
    parser.add_argument("--output-dir", default="data/mock", help="Debe estar vacío o no existir.")
    parser.add_argument("--samples-per-format", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-edge-cases", action="store_true")
    args = parser.parse_args()
    summary = generate_mock_dataset(
        Path(args.output_dir),
        samples_per_format=args.samples_per_format,
        seed=args.seed,
        include_edge_cases=args.include_edge_cases,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
