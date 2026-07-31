import argparse
import logging
from pathlib import Path
from typing import List, Optional

import fiftyone as fo
import fiftyone.types as fot
from pydantic import BaseModel

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetConfig(BaseModel):
    name: str
    path: Path
    dataset_type: type
    splits: Optional[List[str]] = None
    label_field: str = "ground_truth"


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
            
        # Heurística 2: Clasificación de Imágenes (Carpetas)
        splits = [s.name for s in item.iterdir() if s.is_dir() and s.name in ("train", "test", "valid", "val")]
        
        configs.append(
            DatasetConfig(
                name=dataset_name,
                path=item,
                dataset_type=fot.ImageClassificationDirectoryTree,
                splits=splits if splits else None,
                label_field="original_label",
            )
        )
        
    return configs


def create_unified_dataset(dataset_name: str, raw_data_dir: Path) -> fo.Dataset:
    """
    Crea o carga el dataset unificado y orquesta la ingesta de las fuentes descubiertas.
    """
    if fo.dataset_exists(dataset_name):
        logger.info(f"El dataset '{dataset_name}' ya existe. Eliminándolo para recrear...")
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True

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
                        dataset.add_dir(
                            dataset_type=config.dataset_type,
                            data_path=str(config.path),
                            labels_path=str(json_files[0]),
                            label_field=config.label_field,
                            tags=[config.name],
                        )
            
            elif config.dataset_type == fot.ImageClassificationDirectoryTree:
                if config.splits:
                    for split in config.splits:
                        split_path = config.path / split
                        dataset.add_dir(
                            dataset_dir=str(split_path),
                            dataset_type=config.dataset_type,
                            label_field=config.label_field,
                            tags=[config.name, split],
                        )
                else:
                    dataset.add_dir(
                        dataset_dir=str(config.path),
                        dataset_type=config.dataset_type,
                        label_field=config.label_field,
                        tags=[config.name],
                    )
                    
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
