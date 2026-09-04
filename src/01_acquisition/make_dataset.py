import argparse
import logging
from pathlib import Path
from typing import List, Optional

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


class DatasetConfig(BaseModel):
    name: str
    path: Path
    dataset_type: type
    splits: Optional[List[str]] = None
    label_field: str = "ground_truth"
    data_path: Optional[str] = None
    labels_path: Optional[str] = None


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
    
    for item in raw_data_dir.iterdir():
        if not item.is_dir():
            continue
            
        dataset_name = item.name
        
        # Heurística 1: ¿Es COCO? (Busca archivos _annotations.coco.json)
        coco_files = list(item.rglob("*.json"))
        if any("coco" in f.name.lower() or "annotations" in f.name.lower() for f in coco_files):
            # Detectar splits si existen carpetas train/test/valid
            splits = [s.name for s in item.iterdir() if s.is_dir() and s.name in ("train", "test", "valid", "val")]
            
            configs.append(
                DatasetConfig(
                    name=dataset_name,
                    path=item,
                    dataset_type=fot.COCODetectionDataset,
                    splits=splits if splits else None,
                    label_field="coco_detections",
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


def _sanitize_coco_json(json_path: Path, data_dir: Path) -> None:
    """
    Purga las referencias a imágenes (y sus anotaciones) que no existen físicamente en disco.
    Previene KeyError fatales en el importador de FiftyOne causados por datasets corrompidos.
    """
    try:
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if "images" not in data:
            return
            
        valid_images = []
        valid_image_ids = set()
        
        for img in data["images"]:
            file_name = img.get("file_name")
            if file_name and (data_dir / file_name).exists():
                valid_images.append(img)
                valid_image_ids.add(img.get("id"))
                
        if len(valid_images) == len(data["images"]):
            return
            
        logger.warning(f"Sanitizando JSON: Eliminadas {len(data['images']) - len(valid_images)} imágenes fantasma en {json_path.name}")
        data["images"] = valid_images
        
        if "annotations" in data:
            data["annotations"] = [ann for ann in data["annotations"] if ann.get("image_id") in valid_image_ids]
            
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
            
    except Exception as e:
        logger.error(f"Fallo al intentar sanitizar el JSON COCO {json_path.name}: {e}")


def _sanitize_yolo_yaml(yaml_path: Path) -> None:
    """
    Lee el archivo YAML de YOLO y corrige rutas absolutas que apuntan a máquinas locales (ej. Kaggle).
    Asegura que 'train', 'val' y 'test' usen rutas relativas compatibles con el entorno actual.
    """
    try:
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        if not data:
            return
            
        modified = False
        for key in ["train", "val", "test"]:
            if key in data and isinstance(data[key], str):
                path_str = data[key]
                # Si es ruta absoluta o estilo windows
                if ":" in path_str or path_str.startswith("/") or "\\" in path_str:
                    local_dir = yaml_path.parent
                    folder_name = Path(path_str).name
                    
                    possible_paths = [
                        Path(key),
                        Path("images") / key,
                        Path("images") / folder_name,
                        Path("images"),
                        Path(folder_name)
                    ]
                    
                    for p in possible_paths:
                        if (local_dir / p).exists():
                            logger.warning(f"Sanitizando YAML: Ruta absoluta '{path_str}' en '{key}' cambiada a './{p}'")
                            data[key] = f"./{p}"
                            modified = True
                            break
                            
        if modified:
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False)
                
    except Exception as e:
        logger.error(f"Fallo al intentar sanitizar el YAML YOLO {yaml_path.name}: {e}")


def create_unified_dataset(dataset_name: str, raw_data_dir: Path) -> fo.Dataset:
    """
    Crea o carga el dataset unificado y orquesta la ingesta de las fuentes descubiertas.
    """
    if fo.dataset_exists(dataset_name):
        logger.info(f"El dataset '{dataset_name}' ya existe. Eliminándolo para recrear...")
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True
    dataset.media_type = "image"

    configs = discover_datasets(raw_data_dir)
    
    if not configs:
        logger.warning("No se encontraron datasets para ingestar en data/raw/")
        return dataset

    for config in configs:
        logger.info(f"Ingestando {config.name} como {config.dataset_type.__name__}...")
        
        try:
            if config.dataset_type == fot.COCODetectionDataset:
                if config.splits:
                    for split in config.splits:
                        split_path = config.path / split
                        # Asume el estandar de Roboflow para COCO json
                        json_files = list(split_path.glob("*.json"))
                        if not json_files:
                            continue
                            
                        _sanitize_coco_json(json_files[0], split_path)
                            
                        dataset.add_dir(
                            dataset_type=config.dataset_type,
                            data_path=str(split_path),
                            labels_path=str(json_files[0]),
                            label_field=config.label_field,
                            tags=[config.name, split],
                        )
                else:
                    json_files = list(config.path.glob("*.json"))
                    if json_files:
                        _sanitize_coco_json(json_files[0], config.path)
                        dataset.add_dir(
                            dataset_type=config.dataset_type,
                            data_path=str(config.path),
                            labels_path=str(json_files[0]),
                            label_field=config.label_field,
                            tags=[config.name],
                        )
            
            elif config.dataset_type == fot.YOLOv5Dataset:
                yaml_files = list(config.path.glob("*.yaml"))
                yaml_path = None
                if yaml_files:
                    _sanitize_yolo_yaml(yaml_files[0])
                    yaml_path = str(yaml_files[0].resolve())
                
                kwargs = {
                    "dataset_dir": str(config.path),
                    "dataset_type": config.dataset_type,
                    "label_field": config.label_field,
                    "tags": [config.name],
                }
                if yaml_path:
                    kwargs["yaml_path"] = yaml_path
                    
                dataset.add_dir(**kwargs)
                
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
                    
        except Exception as e:
            logger.error(f"Error al ingestar {config.name}: {str(e)}")
            continue

    logger.info(f"Ingesta completada. Dataset guardado en FiftyOne DB con {len(dataset)} muestras.")
    return dataset


if __name__ == "__main__":
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
    
    args = parser.parse_args()
    
    raw_path = Path(args.raw_dir)
    unified_dataset = create_unified_dataset(dataset_name=args.dataset_name, raw_data_dir=raw_path)
    
    print("\nResumen del dataset:")
    print(unified_dataset)
    print(f"\nPara visualizar el dataset, corre:\nmake app DATASET=\"{args.dataset_name}\"")
