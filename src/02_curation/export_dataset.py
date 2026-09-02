import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import fiftyone as fo
from rich.logging import RichHandler

from src.02_curation.run_pipeline import export_clean_dataset, parse_output_formats

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Exporta el dataset tras una revisión humana (HitL) en FiftyOne.")
    parser.add_argument("--dataset", type=str, required=True, help="Nombre del dataset en FiftyOne.")
    parser.add_argument("--output-formats", type=str, default="fiftyone,coco,yolo", help="Formatos de salida.")
    args = parser.parse_args()

    if not fo.dataset_exists(args.dataset):
        logger.error(f"[bold red]El dataset '{args.dataset}' no existe en la BD de FiftyOne.[/bold red]")
        return

    dataset = fo.load_dataset(args.dataset)
    logger.info(f"Dataset '{args.dataset}' cargado. Total de muestras en DB: {len(dataset)}")

    logger.info("Sincronizando decisiones manuales (tags) con el estado de curación...")
    # Sincronizamos las muestras que el usuario haya etiquetado manualmente en la UI
    
    # 1. Todo lo que tenga tag "kept" pasa a status="kept"
    kept_view = dataset.match_tags("kept")
    logger.info(f"Muestras marcadas como 'kept' manualmente: {len(kept_view)}")
    for sample in kept_view.iter_samples(autosave=True):
        cur = sample.get_field("curation") or {}
        cur["status"] = "kept"
        if cur.get("reason") != "hitl_approved":
            cur["reason"] = "hitl_approved"
        sample["curation"] = cur

    # 2. Todo lo que tenga tag "removed" pasa a status="removed"
    removed_view = dataset.match_tags("removed")
    logger.info(f"Muestras marcadas como 'removed' manualmente: {len(removed_view)}")
    for sample in removed_view.iter_samples(autosave=True):
        cur = sample.get_field("curation") or {}
        cur["status"] = "removed"
        if cur.get("reason") != "hitl_rejected":
            cur["reason"] = "hitl_rejected"
        sample["curation"] = cur

    # 3. Exportación Final
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = REPO_ROOT / "data" / "processed" / f"{args.dataset}_hitl" / run_id
    
    # Intentamos cargar el label_mapping original si existe
    label_mapping = {}
    try:
        latest_pipeline_run = sorted((REPO_ROOT / "data" / "processed" / args.dataset).glob("*"))[-1]
        mapping_file = latest_pipeline_run / "label_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, "r") as f:
                label_mapping = json.load(f)
    except Exception:
        pass

    summary = {
        "counts": {
            "initial": len(dataset),
            "kept": len(dataset.match(fo.ViewField("curation.status") == "kept")),
        }
    }

    logger.info("Iniciando exportación final...")
    export_clean_dataset(
        dataset=dataset,
        export_dir=export_dir,
        output_formats=parse_output_formats(args.output_formats),
        label_mapping=label_mapping,
        summary=summary,
    )
    
    logger.info("\n[bold green]=== EXPORTACIÓN MANUAL (HitL) FINALIZADA ===[/bold green]")
    logger.info(f"Dataset limpio exportado a: [bold cyan]{export_dir}[/bold cyan]")


if __name__ == "__main__":
    main()
