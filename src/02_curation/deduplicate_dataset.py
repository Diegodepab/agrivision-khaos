import argparse
import logging
import sys
from itertools import chain

import fiftyone as fo
import fiftyone.brain as fob

# Configuración de logging orientado a los datos
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Modelo de extracción de características para la detección semántica.
# ResNet50 ofrece mayor profundidad y discriminación de texturas que MobileNetV2,
# lo cual es relevante para distinguir lesiones foliares reales de variaciones lumínicas.
SIMILARITY_MODEL = "resnet50-imagenet-torch"


def load_dataset(dataset_name: str) -> fo.Dataset:
    """Carga el dataset de FiftyOne desde la base de datos local."""
    if not fo.dataset_exists(dataset_name):
        logger.error("El dataset '%s' no se encuentra en el sistema.", dataset_name)
        sys.exit(1)
        
    dataset = fo.load_dataset(dataset_name)
    logger.info("Dataset '%s' cargado. Muestras iniciales: %d", dataset_name, len(dataset))
    return dataset


def process_duplicates(dataset: fo.Dataset, duplicates_dict: dict, tag: str) -> int:
    """
    Función de utilidad para consolidar el etiquetado de clústeres.
    Recibe un diccionario de duplicados, extrae los IDs redundantes y los etiqueta.
    """
    if not duplicates_dict:
        logger.info("Análisis completado: No se detectaron muestras redundantes.")
        return 0

    num_redundant = sum(len(dup_list) for dup_list in duplicates_dict.values())
    redundant_ids = list(chain.from_iterable(duplicates_dict.values()))
    
    if redundant_ids:
        dataset.untag_samples(tag)
        dataset.select(redundant_ids).tag_samples(tag)
        
        percentage = (num_redundant / len(dataset)) * 100
        logger.info(
            "Se detectaron %d clústeres con %d muestras redundantes (%.2f %%). Etiquetadas como '%s'.",
            len(duplicates_dict), num_redundant, percentage, tag
        )
        
    return num_redundant


def detect_exact_duplicates(dataset: fo.Dataset, tag: str) -> int:
    """Estrategia 1: Detección a nivel de bytes mediante hashing."""
    logger.info("Iniciando detección de duplicados exactos (Hashing)...")
    duplicates_dict = fob.compute_exact_duplicates(dataset)
    return process_duplicates(dataset, duplicates_dict, tag)


def _compute_similarity(dataset: fo.Dataset, brain_key: str) -> "fob.SimilarityIndex":
    """Wrapper centralizado para compute_similarity. Un único lugar para configurar modelo y parámetros."""
    # Nota: FiftyOne lanzará un warning de "Model does not support batching" porque
    # ResNet50 procesa las imágenes manteniendo su aspect ratio original (ragged_batches=True).
    # Esto es vital para no distorsionar las texturas de las hojas.
    return fob.compute_similarity(
        dataset,
        model=SIMILARITY_MODEL,
        brain_key=brain_key,
        batch_size=16,
        num_workers=2,  # Paraleliza la lectura de imágenes desde el disco
    )


def detect_semantic_duplicates(dataset: fo.Dataset, tag: str, threshold: float) -> int:
    """
    Estrategia 2: Detección a nivel visual mediante embeddings.
    Captura imágenes recortadas, redimensionadas o con distinta compresión.
    """
    logger.info("Iniciando detección semántica (modelo: %s)...", SIMILARITY_MODEL)
    brain_key = "semantic_similarity"

    # 1. Calcular o reutilizar el índice de similitud
    if brain_key in dataset.list_brain_runs():
        logger.info("Índice '%s' encontrado. Intentando cargarlo...", brain_key)
        results = dataset.load_brain_results(brain_key)
        if results is None:
            logger.warning("Índice corrupto o incompleto. Limpiando y recomputando...")
            dataset.delete_brain_run(brain_key)
            results = _compute_similarity(dataset, brain_key)
        else:
            logger.info("Índice cargado correctamente. Saltando extracción de embeddings.")
    else:
        logger.info("Extrayendo características visuales. Esto puede tardar (ejecución en CPU)...")
        results = _compute_similarity(dataset, brain_key)

    # 2. Agrupar clústeres según el umbral de distancia
    logger.info("Agrupando muestras con una similitud superior a %s...", threshold)
    results.find_duplicates(thresh=threshold)

    # Convertir neighbors_map {id: [(dup_id, dist), ...]} al formato
    # plano {id: [dup_id, ...]} que espera process_duplicates.
    duplicates_dict = {
        rep_id: [dup_id for dup_id, _dist in dup_list]
        for rep_id, dup_list in results.neighbors_map.items()
    }
    return process_duplicates(dataset, duplicates_dict, tag)


def remove_redundant_samples(dataset: fo.Dataset, tag: str) -> None:
    """Elimina permanentemente las muestras etiquetadas del dataset."""
    redundant_view = dataset.match_tags(tag)
    count = len(redundant_view)
    
    if count > 0:
        logger.info("Ejecutando purga destructiva de %d muestras etiquetadas...", count)
        dataset.delete_samples(redundant_view)
        logger.info("Purga completada.")
    else:
        logger.info("No hay muestras pendientes de eliminación.")


def inspect_duplicates_view(dataset: fo.Dataset, threshold: float) -> None:
    """
    Carga el índice visual precomputado, calcula duplicados con umbral estricto
    y abre la UI de FiftyOne ordenando las muestras en clústeres emparejados.
    """
    brain_key = "semantic_similarity"
    if brain_key not in dataset.list_brain_runs():
        logger.error("No existe el índice '%s'. Ejecuta primero la detección semántica.", brain_key)
        sys.exit(1)

    logger.info("Cargando resultados de similitud '%s'...", brain_key)
    results = dataset.load_brain_results(brain_key)

    logger.info("Agrupando clústeres con umbral estricto: %s...", threshold)
    results.find_duplicates(thresh=threshold)

    # Genera la vista donde las imágenes redundantes aparecen pegadas a su original
    dup_view = results.duplicates_view()

    logger.info("Muestras en vista de clústeres: %d. Abriendo UI...", len(dup_view))
    session = fo.launch_app(view=dup_view)
    session.wait()


def main():
    parser = argparse.ArgumentParser(description="Herramienta modular de deduplicación de datasets.")
    parser.add_argument(
        "--method", 
        type=str, 
        choices=["exact", "semantic"], 
        required=True,
        help="Estrategia de deduplicación: 'exact' (hash) o 'semantic' (visual)."
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.98,
        help="Umbral de similitud para el método semántico (ej. 0.98). Ignorado en método exacto."
    )
    parser.add_argument(
        "--delete", 
        action="store_true", 
        help="Si se incluye, elimina automáticamente los duplicados detectados."
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Abre la UI con una vista ordenada de clústeres emparejados (solo para semántico)."
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="agrivision-dataset",
        help="Nombre del dataset en FiftyOne.",
    )
    
    args = parser.parse_args()
    dataset = load_dataset(args.dataset)
    
    tag_name = f"redundant_{args.method}"
    
    # Enrutamiento de estrategia
    if args.method == "exact":
        redundant_count = detect_exact_duplicates(dataset, tag_name)
    elif args.method == "semantic":
        if args.inspect:
            inspect_duplicates_view(dataset, args.threshold)
            return
        redundant_count = detect_semantic_duplicates(dataset, tag_name, args.threshold)
        
    # Fase de resolución
    if redundant_count > 0:
        if args.delete:
            remove_redundant_samples(dataset, tag_name)
        else:
            logger.info(
                "Eliminación automática desactivada. Valida en la UI filtrando por el tag '%s'.", 
                tag_name
            )
            
    logger.info("Pipeline finalizado. Muestras actuales: %d", len(dataset))


if __name__ == "__main__":
    main()
