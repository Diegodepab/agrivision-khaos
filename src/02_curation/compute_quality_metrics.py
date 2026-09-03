import argparse
import logging
import os
import shutil
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

import cv2
import fiftyone as fo
import numpy as np
from tqdm import tqdm

try:
    import pytesseract
    from pytesseract import Output, TesseractError
except ImportError:  # pragma: no cover - depende del entorno local
    pytesseract = None
    Output = None
    TesseractError = RuntimeError


from rich.logging import RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger(__name__)

# Constantes configurables para experimentación empírica.
MIN_VALID_SIZE = 320
LOW_BRIGHTNESS_PERCENTILE = 5
HIGH_BRIGHTNESS_PERCENTILE = 95
BATCH_UPDATE_SIZE = 10_000
DEFAULT_MAX_PENDING_FACTOR = 4

# Bandas finas capturan bordes clonados/estirados sin contaminarse demasiado con la hoja.
SMEAR_BORDER_FRACTION = 0.06
SMEAR_MAX_BORDER_PX = 48
SMEAR_MIN_BORDER_PX = 12
SMEAR_ENTROPY_RATIO_THRESHOLD = 0.70
SMEAR_GRADIENT_RATIO_THRESHOLD = 0.45
SMEAR_PARALLEL_GRADIENT_RATIO_THRESHOLD = 0.65
SMEAR_MIN_SUSPICIOUS_EDGES = 2
SMEAR_UNIFORM_EDGE_ENTROPY = 1.25
ALPHA_BACKGROUND_THRESHOLD = 0.05

OCR_MIN_CONFIDENCE = 60.0
OCR_MIN_WORD_LENGTH = 3
OCR_MAX_SIDE = 1280
OCR_LANG = "eng"


QUALITY_FIELD_SCHEMA = {
    "quality.is_corrupted": fo.BooleanField,
    "quality.blur_variance": fo.FloatField,
    "quality.brightness_mean": fo.FloatField,
    "quality.brightness_p5": fo.FloatField,
    "quality.brightness_p95": fo.FloatField,
    "quality.width": fo.IntField,
    "quality.height": fo.IntField,
    "quality.low_resolution": fo.BooleanField,
    "quality.has_watermark": fo.BooleanField,
    "quality.has_smearing": fo.BooleanField,
}


FLAT_FIELD_SCHEMA = {
    field_name.removeprefix("quality."): field_type
    for field_name, field_type in QUALITY_FIELD_SCHEMA.items()
}


@dataclass
class QualityMetrics:
    """Estructura tipada de métricas de calidad para inyección nativa."""

    is_corrupted: bool = False
    blur_variance: float | None = None
    brightness_mean: float | None = None
    brightness_p5: float | None = None
    brightness_p95: float | None = None
    width: int | None = None
    height: int | None = None
    low_resolution: bool = False
    has_watermark: bool = False
    has_smearing: bool = False


@dataclass
class DecodedImage:
    bgr: np.ndarray
    alpha: np.ndarray | None = None


class OcrEngine:
    """
    Wrapper ligero para OCR.

    El objeto vive una sola vez en el proceso y se comparte entre hilos. Si en el
    futuro se sustituye pytesseract por EasyOCR u otro modelo residente, no se
    replica por CPU como ocurriría con ProcessPoolExecutor.
    """

    def __init__(
        self,
        enabled: bool,
        min_confidence: float = OCR_MIN_CONFIDENCE,
        lang: str = OCR_LANG,
        max_workers: int = 4,
    ) -> None:
        self.enabled = enabled and self._backend_available()
        self.min_confidence = min_confidence
        self.lang = lang
        self._semaphore = threading.Semaphore(max_workers)
        self._warned_unavailable = False

    @staticmethod
    def _backend_available() -> bool:
        if pytesseract is None:
            logger.warning("pytesseract no está instalado; OCR desactivado.")
            return False

        if shutil.which("tesseract") is None:
            logger.warning(
                "El ejecutable 'tesseract' no está en PATH; OCR desactivado."
            )
            return False

        return True

    def has_watermark(self, image_bgr: np.ndarray) -> bool:
        if not self.enabled:
            return False

        prepared = prepare_image_for_ocr(image_bgr)

        try:
            # El semáforo limita los procesos de Tesseract concurrentes para no colapsar la memoria
            with self._semaphore:
                data = pytesseract.image_to_data(
                    prepared,
                    lang=self.lang,
                    config="--psm 11",
                    output_type=Output.DICT,
                )
        except (TesseractError, OSError) as exc:
            if not self._warned_unavailable:
                logger.warning(
                    "OCR no disponible o mal configurado (%s); "
                    "has_watermark quedará en False.",
                    exc,
                )
                self._warned_unavailable = True
            return False

        confidences = data.get("conf", [])
        words = data.get("text", [])
        for raw_conf, raw_text in zip(confidences, words, strict=False):
            text = str(raw_text).strip()
            if len(text) < OCR_MIN_WORD_LENGTH:
                continue

            try:
                confidence = float(raw_conf)
            except (TypeError, ValueError):
                continue

            if confidence >= self.min_confidence and looks_like_watermark_text(text):
                return True

        return False


def looks_like_watermark_text(text: str) -> bool:
    """Filtra ruido de OCR y conserva señales típicas de marcas superpuestas."""
    normalized = text.lower()
    watermark_terms = (
        "alamy",
        "dreamstime",
        "getty",
        "istock",
        "shutterstock",
        "stock",
        "watermark",
        "depositphotos",
        "adobe",
        "123rf",
        "freepik",
        "miden",
        "picfair",
        "agefotostock",
        "canstock",
    )
    return any(term in normalized for term in watermark_terms)


def read_image(filepath: str) -> DecodedImage | None:
    """
    Lee el archivo una sola vez y decodifica desde bytes con OpenCV.

    cv2.imdecode devuelve None cuando los bytes no forman una imagen válida o
    cuando la estructura está truncada de forma no recuperable.
    """
    try:
        encoded = Path(filepath).read_bytes()
    except OSError:
        return None

    if not encoded:
        return None

    buffer = np.frombuffer(encoded, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None

    if image.ndim == 2:
        return DecodedImage(bgr=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))

    if image.shape[2] == 4:
        return DecodedImage(bgr=image[:, :, :3], alpha=image[:, :, 3])

    return DecodedImage(bgr=image[:, :, :3])


def compute_resolution(image_bgr: np.ndarray) -> dict[str, int | bool]:
    h, w = image_bgr.shape[:2]
    return {
        "height": h,
        "width": w,
        "low_resolution": bool(min(h, w) < MIN_VALID_SIZE),
    }


def get_leaf_mask(image_bgr: np.ndarray) -> np.ndarray | None:
    """Extrae una máscara que aísla la hoja del fondo usando Otsu en saturación y brillo."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    
    try:
        _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception:
        mask = None
        
    if mask is not None:
        mask_ratio = np.sum(mask > 0) / mask.size
        if mask_ratio < 0.05 or mask_ratio > 0.95:
            try:
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            except Exception:
                mask = None
    return mask


def compute_blur(image_bgr: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """
    Calcula la varianza del Laplaciano sobre la imagen o el recorte.
    Utiliza la máscara de Otsu para aislar la hoja del fondo y no penalizar bordes lisos artificiales.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    if mask is not None and np.any(mask > 0):
        mean, stddev = cv2.meanStdDev(laplacian, mask=mask)
        variance = float(stddev[0][0] ** 2)
    else:
        variance = float(laplacian.var())
        
    return {"blur_variance": variance}


def compute_brightness(image_bgr: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """
    Calcula métricas de iluminación extrema usando el canal Value (HSV).
    Si se proporciona una máscara, calcula las métricas únicamente sobre los píxeles
    de biomasa, ignorando fondos artificiales puros.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    if mask is not None and np.any(mask > 0):
        v_channel = v_channel[mask > 0]

    if v_channel.size == 0:
        return {
            "brightness_mean": 0.0,
            "brightness_p5": 0.0,
            "brightness_p95": 0.0,
        }

    p5, p95 = np.percentile(
        v_channel, [LOW_BRIGHTNESS_PERCENTILE, HIGH_BRIGHTNESS_PERCENTILE]
    )

    return {
        "brightness_mean": float(np.mean(v_channel)),
        "brightness_p5": float(p5),
        "brightness_p95": float(p95),
    }


def should_run_ocr(metrics: QualityMetrics) -> bool:
    if metrics.is_corrupted or metrics.low_resolution:
        return False

    if metrics.brightness_p5 is None or metrics.brightness_p95 is None:
        return True

    return not (
        metrics.brightness_p95 < 24.0
        or metrics.brightness_p5 > 231.0
    )


def prepare_image_for_ocr(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    max_side = max(h, w)
    if max_side > OCR_MAX_SIDE:
        scale = OCR_MAX_SIDE / max_side
        image_bgr = cv2.resize(
            image_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 40, 40)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )


def shannon_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = hist / max(float(hist.sum()), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def robust_gradient_mean(gray: np.ndarray, axis: int) -> float:
    diff = np.abs(np.diff(gray.astype(np.float32), axis=axis))
    if diff.size == 0:
        return 0.0
    return float(np.percentile(diff, 75))


def border_band_width(height: int, width: int) -> int:
    shortest_side = min(height, width)
    return int(
        np.clip(
            round(shortest_side * SMEAR_BORDER_FRACTION),
            SMEAR_MIN_BORDER_PX,
            SMEAR_MAX_BORDER_PX,
        )
    )


def has_transparent_background(alpha: np.ndarray | None) -> bool:
    if alpha is None:
        return False

    transparent_fraction = float(np.mean(alpha <= 8))
    return transparent_fraction >= ALPHA_BACKGROUND_THRESHOLD


def is_uniform_background_edge(edge: np.ndarray) -> bool:
    entropy = shannon_entropy(edge)
    if entropy > SMEAR_UNIFORM_EDGE_ENTROPY:
        return False

    mean_value = float(np.mean(edge))
    return mean_value >= 238.0 or mean_value <= 17.0


def compute_smearing(
    image_bgr: np.ndarray,
    alpha: np.ndarray | None = None,
) -> dict[str, bool]:
    """
    Detecta bordes estirados por extrapolación/clonado.

    Un borde arrastrado suele tener menos entropía y menos variación en la
    dirección perpendicular al marco, pero conserva variación paralela. Las
    transparencias PNG y los fondos blancos/negros limpios no se consideran
    smearing aunque tengan baja entropía.
    """
    if has_transparent_background(alpha):
        return {"has_smearing": False}

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    band = border_band_width(h, w)

    if h < band * 3 or w < band * 3:
        return {"has_smearing": False}

    center = gray[band : h - band, band : w - band]
    center_entropy = max(shannon_entropy(center), 1e-6)
    center_grad_y = max(robust_gradient_mean(center, axis=0), 1e-6)
    center_grad_x = max(robust_gradient_mean(center, axis=1), 1e-6)

    edge_specs = (
        (gray[:band, :], center_grad_y, center_grad_x, 0, 1),
        (gray[h - band :, :], center_grad_y, center_grad_x, 0, 1),
        (gray[:, :band], center_grad_x, center_grad_y, 1, 0),
        (gray[:, w - band :], center_grad_x, center_grad_y, 1, 0),
    )

    suspicious_edges = 0
    for (
        edge,
        center_perpendicular_gradient,
        center_parallel_gradient,
        perpendicular_axis,
        parallel_axis,
    ) in edge_specs:
        if is_uniform_background_edge(edge):
            continue

        entropy_ratio = shannon_entropy(edge) / center_entropy
        perpendicular_ratio = (
            robust_gradient_mean(edge, axis=perpendicular_axis)
            / center_perpendicular_gradient
        )
        parallel_ratio = (
            robust_gradient_mean(edge, axis=parallel_axis)
            / center_parallel_gradient
        )
        if (
            entropy_ratio <= SMEAR_ENTROPY_RATIO_THRESHOLD
            and perpendicular_ratio <= SMEAR_GRADIENT_RATIO_THRESHOLD
            and parallel_ratio >= SMEAR_PARALLEL_GRADIENT_RATIO_THRESHOLD
        ):
            suspicious_edges += 1

    return {"has_smearing": suspicious_edges >= SMEAR_MIN_SUSPICIOUS_EDGES}


def process_image(
    sample_data: tuple[str, str, list[list[float]] | None],
    ocr_engine: OcrEngine,
) -> tuple[str, QualityMetrics | None, list[float] | None]:
    """
    Procesa una imagen completa desde una única lectura de disco.

    Retorna métricas o None si hubo una excepción fatal imprevista.
    """
    sample_id, filepath, bboxes = sample_data

    try:
        metrics = QualityMetrics()

        image = read_image(filepath)
        if image is None:
            metrics.is_corrupted = True
            return sample_id, metrics, None

        mask = get_leaf_mask(image.bgr)
        metrics_dict = {}
        metrics_dict.update(compute_resolution(image.bgr))
        metrics_dict.update(compute_blur(image.bgr, mask))
        metrics_dict.update(compute_brightness(image.bgr, mask))
        metrics_dict.update(compute_smearing(image.bgr, image.alpha))
        metrics = QualityMetrics(**metrics_dict)
        if should_run_ocr(metrics):
            metrics.has_watermark = ocr_engine.has_watermark(image.bgr)

        box_blurs = None
        if bboxes:
            box_blurs = []
            h, w = image.bgr.shape[:2]
            for box in bboxes:
                # box: [top-left-x, top-left-y, width, height] (relative)
                x1 = int(box[0] * w)
                y1 = int(box[1] * h)
                bw = int(box[2] * w)
                bh = int(box[3] * h)
                x2 = min(x1 + bw, w)
                y2 = min(y1 + bh, h)
                x1, y1 = max(0, x1), max(0, y1)
                if x2 > x1 and y2 > y1:
                    crop = image.bgr[y1:y2, x1:x2]
                    crop_mask = get_leaf_mask(crop)
                    box_blurs.append(compute_blur(crop, crop_mask)["blur_variance"])
                else:
                    box_blurs.append(0.0)
                    
        return sample_id, metrics, box_blurs

    except Exception:
        logger.exception("Error fatal procesando %s", filepath)
        return sample_id, None, None


def flush_quality_batch(
    dataset: fo.Dataset,
    batch_dict: dict[str, fo.DynamicEmbeddedDocument],
) -> None:
    if batch_dict:
        dataset.set_values("quality", batch_dict, key_field="id")
        batch_dict.clear()


def flush_flat_metric_batches(
    dataset: fo.Dataset,
    flat_batches: dict[str, dict[str, object]],
) -> None:
    for field_name, values in flat_batches.items():
        if values:
            dataset.set_values(field_name, values, key_field="id")
            values.clear()


def metrics_to_flat_values(metrics: QualityMetrics) -> dict[str, object]:
    return asdict(metrics)


def ensure_quality_schema(dataset: fo.Dataset) -> None:
    """
    Declara el documento y sus subcampos para que FiftyOne indexe filtros.

    Sin esta declaración explícita, los valores se guardan en MongoDB, pero la UI
    puede no descubrir campos nuevos dentro de DynamicEmbeddedDocument hasta
    recrear vistas o reiniciar la app.
    """
    if not dataset.has_sample_field("quality"):
        dataset.add_sample_field(
            "quality",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.DynamicEmbeddedDocument,
        )

    schema = dataset.get_field_schema(flat=True)
    for field_name, field_type in QUALITY_FIELD_SCHEMA.items():
        if field_name not in schema:
            dataset.add_sample_field(field_name, field_type)

    schema = dataset.get_field_schema(flat=True)
    for field_name, field_type in FLAT_FIELD_SCHEMA.items():
        if field_name not in schema:
            dataset.add_sample_field(field_name, field_type)


def bounded_parallel_map(
    executor: ThreadPoolExecutor,
    tasks: Iterable[tuple[str, str, list[list[float]] | None]],
    ocr_engine: OcrEngine,
    max_pending: int,
) -> Iterable[tuple[str, QualityMetrics | None, list[float] | None]]:
    """
    Productor-consumidor con backpressure.

    Mantiene un número pequeño de futures vivos, evitando cargar demasiadas
    imágenes decodificadas o trabajos OCR pendientes en memoria.
    """
    pending = set()

    for task in tasks:
        pending.add(executor.submit(process_image, task, ocr_engine))
        if len(pending) >= max_pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()

    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            yield future.result()


def compute_dataset_quality(
    dataset_name: str,
    workers: int,
    enable_ocr: bool,
    ocr_confidence: float,
    max_pending: int | None = None,
    ocr_workers: int = 4,
) -> None:
    """
    Orquestador principal.

    Usa hilos en lugar de procesos para que el backend OCR/modelo resida una sola
    vez en memoria. Los updates a FiftyOne siguen siendo masivos por bloques.
    """
    if workers < 1:
        raise ValueError("El número de workers debe ser al menos 1.")

    if not fo.dataset_exists(dataset_name):
        logger.error("El dataset '%s' no se encuentra en el sistema.", dataset_name)
        sys.exit(1)

    dataset = fo.load_dataset(dataset_name)
    total_samples = len(dataset)

    logger.info("Dataset '%s' cargado.", dataset_name)
    logger.info("Procesando %d imágenes mediante %d hilos...", total_samples, workers)

    ensure_quality_schema(dataset)

    logger.info("Precargando IDs y rutas para evitar timeouts de MongoDB...")
    sample_ids = dataset.values("id")
    filepaths = dataset.values("filepath")
    
    if dataset.has_sample_field("coco_detections"):
        bboxes_list = dataset.values("coco_detections.detections.bounding_box")
    else:
        bboxes_list = [None] * total_samples
        
    tasks = zip(sample_ids, filepaths, bboxes_list)

    ocr_engine = OcrEngine(
        enabled=enable_ocr, 
        min_confidence=ocr_confidence, 
        max_workers=ocr_workers
    )
    if ocr_engine.enabled:
        logger.info("OCR activado para detección de texto/marcas de agua.")
    else:
        logger.info("OCR desactivado; has_watermark se conservará en False.")

    max_pending = max_pending or max(workers * DEFAULT_MAX_PENDING_FACTOR, workers)

    corrupted_count = 0
    low_res_count = 0
    watermark_count = 0
    smearing_count = 0
    blur_sum = 0.0
    brightness_sum = 0.0
    fatal_errors_count = 0
    valid_count = 0

    min_area = float("inf")
    max_area = 0
    smallest_image = None
    largest_image = None
    batch_dict = {}
    flat_batches = {field.name: {} for field in fields(QualityMetrics)}
    box_blur_batch = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = bounded_parallel_map(executor, tasks, ocr_engine, max_pending)

        for sample_id, metrics, box_blurs in tqdm(
            results, total=total_samples, desc="Extrayendo calidad"
        ):
            if metrics is None:
                fatal_errors_count += 1
                continue

            metric_values = metrics_to_flat_values(metrics)
            batch_dict[sample_id] = fo.DynamicEmbeddedDocument(**metric_values)
            for field_name, value in metric_values.items():
                flat_batches[field_name][sample_id] = value

            if box_blurs is not None:
                box_blur_batch[sample_id] = box_blurs

            if len(batch_dict) >= BATCH_UPDATE_SIZE:
                flush_quality_batch(dataset, batch_dict)
                flush_flat_metric_batches(dataset, flat_batches)
                if box_blur_batch and dataset.has_sample_field("coco_detections"):
                    dataset.set_values("coco_detections.detections.blur_variance", box_blur_batch, key_field="id")
                    box_blur_batch.clear()

            if metrics.is_corrupted:
                corrupted_count += 1
                continue

            valid_count += 1
            low_res_count += int(metrics.low_resolution)
            watermark_count += int(metrics.has_watermark)
            smearing_count += int(metrics.has_smearing)

            if metrics.blur_variance is not None:
                blur_sum += metrics.blur_variance
            if metrics.brightness_mean is not None:
                brightness_sum += metrics.brightness_mean

            if metrics.width is not None and metrics.height is not None:
                area = metrics.width * metrics.height
                if area < min_area:
                    min_area = area
                    smallest_image = (metrics.width, metrics.height)
                if area > max_area:
                    max_area = area
                    largest_image = (metrics.width, metrics.height)

    flush_quality_batch(dataset, batch_dict)
    flush_flat_metric_batches(dataset, flat_batches)
    if box_blur_batch and dataset.has_sample_field("coco_detections"):
        dataset.set_values("coco_detections.detections.blur_variance", box_blur_batch, key_field="id")
        box_blur_batch.clear()

    logger.info("Análisis finalizado y guardado en FiftyOne.")
    logger.info("Estadísticas globales del dataset:")
    logger.info(
        "  Procesadas %d imágenes (%d corruptas | %d errores fatales omitidos)",
        total_samples,
        corrupted_count,
        fatal_errors_count,
    )

    if valid_count > 0:
        logger.info("  Resolución inferior a %spx: %d", MIN_VALID_SIZE, low_res_count)
        logger.info("  Posibles marcas de agua: %d", watermark_count)
        logger.info("  Posibles bordes estirados: %d", smearing_count)
        logger.info("  Blur medio (Laplaciano): %.2f", blur_sum / valid_count)
        logger.info("  Brillo medio (HSV-V): %.2f", brightness_sum / valid_count)
        if smallest_image is not None and largest_image is not None:
            logger.info(
                "  Imagen más pequeña: %dx%d",
                smallest_image[0],
                smallest_image[1],
            )
            logger.info("  Imagen más grande: %dx%d", largest_image[0], largest_image[1])


def repair_quality_schema(dataset_name: str) -> None:
    if not fo.dataset_exists(dataset_name):
        logger.error("El dataset '%s' no se encuentra en el sistema.", dataset_name)
        sys.exit(1)

    dataset = fo.load_dataset(dataset_name)
    ensure_quality_schema(dataset)

    sample_ids = dataset.values("id")
    for flat_field in FLAT_FIELD_SCHEMA:
        quality_field = f"quality.{flat_field}"
        values = dataset.values(quality_field)
        batch = {
            sample_id: value
            for sample_id, value in zip(sample_ids, values, strict=False)
            if value is not None
        }
        if batch:
            dataset.set_values(flat_field, batch, key_field="id")

    dataset.save()
    logger.info(
        "Esquema reparado y métricas planas sincronizadas. "
        "Reinicia la app de FiftyOne para refrescar filtros."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcula métricas visuales para purgar imágenes de baja calidad."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="agrivision-dataset",
        help="Nombre del dataset en FiftyOne.",
    )

    default_workers = min(8, os.cpu_count() or 1)
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=(
            "Número de hilos en paralelo "
            f"(por defecto: min(8, cpus) -> {default_workers})."
        ),
    )
    parser.add_argument(
        "--disable-ocr",
        action="store_true",
        help="Desactiva la detección OCR de marcas de agua.",
    )
    parser.add_argument(
        "--ocr-confidence",
        type=float,
        default=OCR_MIN_CONFIDENCE,
        help=(
            "Confianza mínima OCR para activar has_watermark "
            f"(por defecto: {OCR_MIN_CONFIDENCE})."
        ),
    )
    parser.add_argument(
        "--max-pending",
        type=int,
        default=None,
        help="Máximo de imágenes pendientes en memoria; por defecto workers * 4.",
    )
    parser.add_argument(
        "--ocr-workers",
        type=int,
        default=4,
        help="Número máximo de procesos Tesseract simultáneos (por defecto: 4).",
    )
    parser.add_argument(
        "--repair-schema-only",
        action="store_true",
        help="Declara quality.* en el esquema de FiftyOne sin recalcular imágenes.",
    )

    args = parser.parse_args()
    if args.repair_schema_only:
        repair_quality_schema(args.dataset)
        return

    compute_dataset_quality(
        dataset_name=args.dataset,
        workers=args.workers,
        enable_ocr=not args.disable_ocr,
        ocr_confidence=args.ocr_confidence,
        max_pending=args.max_pending,
        ocr_workers=args.ocr_workers,
    )


if __name__ == "__main__":
    main()
