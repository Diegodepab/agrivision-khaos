from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from itertools import chain

import cv2
import fiftyone as fo
import fiftyone.brain as fob
import numpy as np

# Configuración de logging orientado a los datos
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:  # Core ingestion/quality installations do not need embeddings
    torch = None

# Modelo de extracción de características para la detección semántica.
# Con GPU (CUDA) usamos ResNet50; en CPU, MobileNetV2 reduce el coste de inferencia.
SIMILARITY_MODEL = (
    "resnet50-imagenet-torch"
    if torch is not None and torch.cuda.is_available()
    else "mobilenet-v2-imagenet-torch"
)


def _require_embedding_backend() -> None:
    if torch is None:
        raise RuntimeError(
            "La deduplicación semántica requiere PyTorch: instala el extra de CPU con "
            "`uv sync --extra cpu` o el de NVIDIA CUDA 13.0 con `uv sync --extra cu130`."
        )


def load_dataset(dataset_name: str) -> fo.Dataset:
    """Carga el dataset de FiftyOne desde la base de datos local."""
    if not fo.dataset_exists(dataset_name):
        logger.error("El dataset '%s' no se encuentra en el sistema.", dataset_name)
        sys.exit(1)
        
    dataset = fo.load_dataset(dataset_name)
    logger.info("Dataset '%s' cargado. Muestras iniciales: %d", dataset_name, len(dataset))
    return dataset


def _get_padding_ratio(filepath: str) -> float:
    """Calcula el porcentaje de píxeles que son puramente negros o blancos (padding artificial)."""
    try:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 1.0
        padding_mask = (img <= 5) | (img >= 250)
        return float(np.mean(padding_mask))
    except Exception:
        return 1.0


def _sample_value(sample: fo.Sample, name: str):
    try:
        if sample.has_field(name):
            return sample.get_field(name)
    except Exception:
        pass

    try:
        quality = sample.get_field("quality") if sample.has_field("quality") else None
    except Exception:
        quality = getattr(sample, "quality", None)

    if quality is None:
        return None
    if hasattr(quality, "get"):
        return quality.get(name)
    return getattr(quality, name, None)


def _metric(sample: fo.Sample, name: str, default: float = 0.0) -> float:
    value = _sample_value(sample, name)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _flag(sample: fo.Sample, name: str) -> bool:
    return bool(_sample_value(sample, name))


def _is_probably_augmented_path(filepath: str) -> bool:
    normalized = filepath.lower()
    return ".rf." in normalized or "_aug" in normalized or "augmentation" in normalized


def _sample_keep_score(sample: fo.Sample) -> tuple[float, ...]:
    width = _metric(sample, "width")
    height = _metric(sample, "height")
    area = width * height
    blur = _metric(sample, "blur_variance", default=-1.0)
    padding = _get_padding_ratio(sample.filepath)

    return (
        0.0 if _curation_status(sample) == "removed" else 1.0,
        0.0 if _curation_status(sample) == "review" else 1.0,
        0.0 if _flag(sample, "is_corrupted") else 1.0,
        0.0 if _flag(sample, "has_watermark") else 1.0,
        0.0 if _flag(sample, "has_smearing") else 1.0,
        0.0 if _flag(sample, "low_resolution") else 1.0,
        0.0 if _is_probably_augmented_path(sample.filepath) else 1.0,
        -padding,
        area,
        blur,
    )


def _curation_status(sample: fo.Sample) -> str:
    curation = _sample_value(sample, "curation")
    if curation is None:
        return "kept"
    if hasattr(curation, "get"):
        return str(curation.get("status", "kept") or "kept")
    return str(getattr(curation, "status", "kept") or "kept")


def _label_signature(sample: fo.Sample) -> frozenset[str]:
    labels = _sample_value(sample, "normalized_labels") or []
    if labels:
        return frozenset(str(label) for label in labels if label)
    label = _sample_value(sample, "normalized_label")
    return frozenset([str(label)]) if label else frozenset()


def _connected_components(duplicates_dict: dict) -> list[list[str]]:
    """Converts overlapping neighbor groups into deterministic components."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for representative, duplicates in duplicates_dict.items():
        representative = str(representative)
        find(representative)
        for duplicate in duplicates:
            union(representative, str(duplicate))

    groups: dict[str, set[str]] = {}
    for sample_id in parent:
        groups.setdefault(find(sample_id), set()).add(sample_id)
    return sorted(
        (sorted(group) for group in groups.values() if len(group) > 1),
        key=lambda group: group[0],
    )


def process_duplicates(dataset: fo.Dataset, duplicates_dict: dict, tag: str) -> tuple[int, list[tuple[str, str]]]:
    """
    Función de utilidad para consolidar el etiquetado de clústeres.
    Evalúa cada miembro del clúster y mantiene a salvo la imagen original
    (la que tiene menor cantidad de padding artificial de rotación).
    Las demás son marcadas como redundantes.
    Devuelve la cantidad de redundantes y una lista de ejemplos (id_original, id_duplicado).
    """
    conflict_tag = f"{tag}_label_conflict"
    dataset.untag_samples([tag, conflict_tag])
    if not duplicates_dict:
        logger.info("Análisis completado: No se detectaron muestras redundantes.")
        return 0, []

    components = _connected_components(duplicates_dict)
    final_redundant_ids: set[str] = set()
    duplicate_pairs: list[tuple[str, str]] = []
    conflict_ids: set[str] = set()
    num_clusters = len(components)
    
    logger.info("Analizando %d clústeres para proteger la imagen original...", num_clusters)
    
    # Pre-cargar muestras para escoger el mejor representante del cluster.
    all_cluster_ids = list(chain.from_iterable(components))
    samples_by_id = {
        sample.id: sample
        for sample in dataset.select(all_cluster_ids)
    }
    
    cluster_ids_by_sample: dict[str, str] = {}
    representative_by_sample: dict[str, str] = {}
    for cluster_ids in components:
        
        # Encontrar la imagen con mejor calidad global. El padding sigue siendo
        # importante, pero ya no domina sobre corrupcion, watermark o blur.
        scored_ids = [
            (s_id, _sample_keep_score(sample))
            for s_id in cluster_ids
            if (sample := samples_by_id.get(s_id)) is not None
        ]
        best_id = max(scored_ids, key=lambda item: item[1])[0] if scored_ids else cluster_ids[0]
        signatures = {
            _label_signature(sample)
            for sample_id in cluster_ids
            if (sample := samples_by_id.get(sample_id)) is not None
        }
        has_label_conflict = len(signatures) > 1
                
        cluster_id = hashlib.sha1("\0".join(cluster_ids).encode("utf-8")).hexdigest()[:16]
        for sample_id in cluster_ids:
            cluster_ids_by_sample[sample_id] = cluster_id
            representative_by_sample[sample_id] = best_id
            if has_label_conflict:
                conflict_ids.add(sample_id)
            elif sample_id != best_id:
                final_redundant_ids.add(sample_id)
                duplicate_pairs.append((best_id, sample_id))

    schema = dataset.get_field_schema()
    for field_name in ("duplicate_cluster_id", "duplicate_representative_id"):
        if field_name not in schema:
            dataset.add_sample_field(field_name, fo.StringField)
    dataset.set_values("duplicate_cluster_id", cluster_ids_by_sample, key_field="id")
    dataset.set_values("duplicate_representative_id", representative_by_sample, key_field="id")
    if conflict_ids:
        dataset.select(sorted(conflict_ids)).tag_samples(conflict_tag)
        logger.warning(
            "%d muestras duplicadas tienen etiquetas incompatibles y requieren revisión.",
            len(conflict_ids),
        )
    
    num_redundant = len(final_redundant_ids)
    if num_redundant > 0:
        dataset.select(sorted(final_redundant_ids)).tag_samples(tag)
        
        percentage = (num_redundant / len(dataset)) * 100
        logger.info(
            "Se detectaron %d clústeres con %d muestras redundantes (%.2f %%). Etiquetadas como '%s'.",
            num_clusters, num_redundant, percentage, tag
        )
        
    return num_redundant, duplicate_pairs


def detect_exact_duplicates(dataset: fo.Dataset, tag: str) -> tuple[int, list[tuple[str, str]]]:
    """Estrategia 1: Detección a nivel de bytes mediante hashing."""
    logger.info("Iniciando detección de duplicados exactos (Hashing)...")
    duplicates_dict = fob.compute_exact_duplicates(dataset)
    return process_duplicates(dataset, duplicates_dict, tag)


def _compute_similarity(dataset: fo.Dataset, brain_key: str) -> "fob.SimilarityIndex":
    """Wrapper centralizado para compute_similarity. Un único lugar para configurar modelo y parámetros."""
    _require_embedding_backend()
    # Nota: FiftyOne lanzará un warning de "Model does not support batching" porque
    # El modelo activo procesa imágenes con tamaños distintos (ragged_batches=True).
    # Esto es vital para no distorsionar las texturas de las hojas.
    return fob.compute_similarity(
        dataset,
        model=SIMILARITY_MODEL,
        brain_key=brain_key,
        metric="cosine",
        batch_size=16,
        num_workers=0,  # CRÍTICO: Previene el colapso de /dev/shm en Docker al usar el hilo principal
    )


def detect_semantic_duplicates(dataset: fo.Dataset, tag: str, threshold: float) -> tuple[int, list[tuple[str, str]]]:
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

    # 2. Agrupar clústeres según el umbral de distancia (convertimos similitud a distancia coseno)
    dist_thresh = 1.0 - threshold
    logger.info("Agrupando muestras con una similitud superior a %s (distancia <= %.3f)...", threshold, dist_thresh)
    results.find_duplicates(thresh=dist_thresh)

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

    dist_thresh = 1.0 - threshold
    logger.info("Agrupando clústeres con umbral estricto: %s (distancia <= %.3f)...", threshold, dist_thresh)
    results.find_duplicates(thresh=dist_thresh)

    # Genera la vista donde las imágenes redundantes aparecen pegadas a su original
    dup_view = results.duplicates_view()

    logger.info("Muestras en vista de clústeres: %d. Abriendo UI...", len(dup_view))
    session = fo.launch_app(view=dup_view, address="0.0.0.0", port=5151)
    session.wait()


def compute_color_embeddings(dataset: fo.Dataset) -> np.ndarray:
    """
    Calcula un histograma de color HSV enmascarado para cada imagen.
    Ignora fondos puros blancos/negros para ser robusto frente a rotaciones,
    espejos y paddings introducidos artificialmente en el dataset.
    """
    filepaths = dataset.values("filepath")
    embeddings = []
    
    logger.info("Extrayendo huellas de color (invariantes a rotación/espejo)...")
    for path in filepaths:
        img = cv2.imread(path)
        if img is None:
            embeddings.append(np.zeros(512))
            continue
            
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        mask = cv2.inRange(gray, 15, 240)
        hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        embeddings.append(hist.flatten())
        
    return np.array(embeddings)


def detect_augmentation_duplicates(dataset: fo.Dataset, tag: str, threshold: float) -> tuple[int, list[tuple[str, str]]]:
    """
    Estrategia 3: Detección a nivel de color (invariante a rotaciones y espejos).
    Utiliza el histograma HSV filtrado para capturar la "proporción de colores"
    exacta de la hoja. Si dos imágenes comparten histograma, son la misma hoja augmentada.
    """
    brain_key = "augmentation_similarity"
    if brain_key in dataset.list_brain_runs():
        dataset.delete_brain_run(brain_key)
        
    embeddings = compute_color_embeddings(dataset)
    logger.info("Computando similitud de aumentación...")
    results = fob.compute_similarity(
        dataset,
        embeddings=embeddings,
        brain_key=brain_key,
        metric="cosine",
    )
    
    dist_thresh = 1.0 - threshold
    logger.info("Agrupando muestras con una similitud superior a %s (distancia <= %.3f)...", threshold, dist_thresh)
    results.find_duplicates(thresh=dist_thresh)
    
    duplicates_dict = {
        rep_id: [dup_id for dup_id, _dist in dup_list]
        for rep_id, dup_list in results.neighbors_map.items()
    }
    return process_duplicates(dataset, duplicates_dict, tag)


def inspect_augmentation_view(dataset: fo.Dataset, threshold: float) -> None:
    """
    Abre la UI de FiftyOne para inspeccionar visualmente los clústeres
    de imágenes augmentadas detectados por color antes de purgar.
    """
    brain_key = "augmentation_similarity"
    if brain_key not in dataset.list_brain_runs():
        logger.error("No existe el índice '%s'. Ejecuta primero la detección de aumentación.", brain_key)
        sys.exit(1)

    results = dataset.load_brain_results(brain_key)
    dist_thresh = 1.0 - threshold
    logger.info("Agrupando clústeres con umbral estricto: %s (distancia <= %.3f)...", threshold, dist_thresh)
    results.find_duplicates(thresh=dist_thresh)
    dup_view = results.duplicates_view()
    logger.info("Muestras en vista de clústeres: %d. Abriendo UI...", len(dup_view))
    session = fo.launch_app(view=dup_view, address="0.0.0.0", port=5151)
    session.wait()


def detect_mislabeled_samples(dataset: fo.Dataset, tag: str = "review", threshold: float = 0.7) -> int:
    """
    Detecta posibles errores de etiquetado a partir de los embeddings activos.
    Aplica normalización L2 obligatoria antes de calcular centroides.
    """
    logger.info("Iniciando auditoría de etiquetado (Mislabeling Detection)...")
    _require_embedding_backend()
    
    try:
        embeddings = dataset.compute_embeddings(model=SIMILARITY_MODEL)
    except Exception as exc:
        logger.error("Error extrayendo embeddings para mislabeling: %s", exc)
        return 0

    # Normalización L2 requerida para evaluar similitud semántica sin sesgo de magnitud
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings_l2 = embeddings / norms

    labels = np.array(dataset.values("normalized_label"))
    sample_ids = np.array(dataset.values("id"))
    
    unique_labels = np.unique([label for label in labels if label])
    if len(unique_labels) < 2:
        logger.info("Se necesitan al menos 2 clases para comparar mislabeling.")
        return 0
        
    # Calcular centroides normalizados
    centroids = {}
    for lbl in unique_labels:
        mask = (labels == lbl)
        if np.sum(mask) == 0:
            continue
        class_embs = embeddings_l2[mask]
        centroid = np.mean(class_embs, axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 0:
            centroid /= c_norm
        centroids[lbl] = centroid

    suspicious_ids = []
    
    for i, lbl in enumerate(labels):
        if not lbl or lbl not in centroids:
            continue
            
        emb = embeddings_l2[i]
        own_sim = np.dot(emb, centroids[lbl])
        own_dist = 1.0 - own_sim
        
        min_other_dist = float('inf')
        for other_lbl, c_emb in centroids.items():
            if other_lbl == lbl:
                continue
            sim = np.dot(emb, c_emb)
            dist = 1.0 - sim
            if dist < min_other_dist:
                min_other_dist = dist
                
        # Heurística: Si está más lejos de su clase que de otra clase (con un margen)
        if own_dist > min_other_dist + 0.05:
            suspicious_ids.append(sample_ids[i])
            
    if suspicious_ids:
        dataset.select(suspicious_ids).tag_samples(tag)
        for s in dataset.select(suspicious_ids):
            if not s.has_field("curation"):
                s.curation = fo.DynamicEmbeddedDocument()
            s.curation.status = "review"
            s.curation.reason = "possible_mislabel"
            s.save()
            
        logger.warning("Detectados %d posibles errores de etiquetado.", len(suspicious_ids))
    else:
        logger.info("No se detectaron outliers de etiquetado.")
        
    return len(suspicious_ids)


def main():
    parser = argparse.ArgumentParser(description="Herramienta modular de deduplicación de datasets.")
    parser.add_argument(
        "--method", 
        type=str, 
        choices=["exact", "semantic", "augmented", "mislabel"], 
        required=True,
        help="Estrategia de deduplicación: 'exact', 'semantic', 'augmented' o 'mislabel'."
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.95,
        help="Umbral de similitud para el método semántico (ej. 0.95). Ignorado en método exacto."
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
    redundant_count = 0
    if args.method == "exact":
        redundant_count, _ = detect_exact_duplicates(dataset, tag_name)
    elif args.method == "semantic":
        if args.inspect:
            inspect_duplicates_view(dataset, args.threshold)
            return
        redundant_count, _ = detect_semantic_duplicates(dataset, tag_name, args.threshold)
    elif args.method == "augmented":
        if args.inspect:
            inspect_augmentation_view(dataset, args.threshold)
            return
        redundant_count, _ = detect_augmentation_duplicates(dataset, tag_name, args.threshold)
    elif args.method == "mislabel":
        redundant_count = detect_mislabeled_samples(dataset, tag_name, args.threshold)
        
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
