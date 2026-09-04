import argparse
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import fiftyone as fo
from rich.logging import RichHandler

from agrivision_khaos.execution import atomic_write_json
from agrivision_khaos.pipeline import export_clean_dataset, load_policy, parse_output_formats

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger(__name__)
def sync_manual_decisions(
    dataset: fo.Dataset,
    require_all_reviews_resolved: bool = True,
) -> dict[str, int]:
    kept_ids = set(dataset.match_tags("kept").values("id"))
    removed_ids = set(dataset.match_tags("removed").values("id"))
    conflicts = kept_ids & removed_ids
    if conflicts:
        raise RuntimeError(
            f"{len(conflicts)} muestras tienen simultáneamente los tags 'kept' y 'removed'"
        )

    state_tags = {
        "kept",
        "removed",
        "review",
        "curation_kept",
        "curation_review",
        "curation_removed",
    }
    for status, sample_ids in (("kept", kept_ids), ("removed", removed_ids)):
        if not sample_ids:
            continue
        for sample in dataset.select(sorted(sample_ids)).iter_samples(autosave=True):
            curation = sample.get_field("curation") or fo.DynamicEmbeddedDocument()
            curation["status"] = status
            curation["phase"] = "human_review"
            curation["reason"] = "hitl_approved" if status == "kept" else "hitl_rejected"
            curation["confidence"] = 1.0
            sample["curation"] = curation
            sample.tags = sorted(
                set(
                    [tag for tag in sample.tags if tag not in state_tags]
                    + [f"curation_{status}"]
                )
            )
    unresolved = len(dataset.match(fo.ViewField("curation.status") == "review"))
    if require_all_reviews_resolved and unresolved:
        raise RuntimeError(
            f"Quedan {unresolved} muestras en revisión; resuélvelas antes de exportar"
        )
    return {"kept": len(kept_ids), "removed": len(removed_ids), "unresolved": unresolved}


def main():
    parser = argparse.ArgumentParser(description="Exporta el dataset tras una revisión humana (HitL) en FiftyOne.")
    parser.add_argument("--dataset", type=str, required=True, help="Nombre del dataset en FiftyOne.")
    parser.add_argument("--output-formats", type=str, default="coco,yolo", help="Formatos de salida.")
    parser.add_argument("--export-dir", default="/datasets/processed")
    parser.add_argument("--policy", default="configs/quality-first.yaml")
    args = parser.parse_args()
    try:
        output_formats = parse_output_formats(args.output_formats)
        policy = load_policy(args.policy)
    except ValueError as exc:
        parser.error(str(exc))

    if not fo.dataset_exists(args.dataset):
        logger.error(f"[bold red]El dataset '{args.dataset}' no existe en la BD de FiftyOne.[/bold red]")
        raise SystemExit(2)

    dataset = fo.load_dataset(args.dataset)
    logger.info(f"Dataset '{args.dataset}' cargado. Total de muestras en DB: {len(dataset)}")

    logger.info("Sincronizando decisiones manuales (tags) con el estado de curación...")
    manual_counts = sync_manual_decisions(dataset)
    logger.info("Decisiones manuales: %s", manual_counts)

    # 3. Exportación Final
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    export_root = Path(args.export_dir)
    export_dir = export_root / f"{args.dataset}_hitl" / run_id
    
    # Intentamos cargar el label_mapping original si existe
    label_mapping = {}
    try:
        completed_runs = [
            path
            for path in (export_root / args.dataset).glob("*")
            if path.is_dir() and (path / "_SUCCESS").is_file()
        ]
        latest_pipeline_run = max(completed_runs, key=lambda path: path.name)
        mapping_file = latest_pipeline_run / "label_mapping.json"
        if mapping_file.exists():
            with mapping_file.open("r", encoding="utf-8") as f:
                label_mapping = json.load(f)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        logger.warning("No se pudo reutilizar el mapeo de etiquetas anterior: %s", exc)

    summary = {
        "counts": {
            "initial": len(dataset),
            "kept": len(dataset.match(fo.ViewField("curation.status") == "kept")),
            "review": len(dataset.match(fo.ViewField("curation.status") == "review")),
            "removed": len(dataset.match(fo.ViewField("curation.status") == "removed")),
        }
    }

    logger.info("Iniciando exportación final...")
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    attempt_dir = Path(
        tempfile.mkdtemp(prefix=f".{run_id}.incomplete-", dir=export_dir.parent)
    )
    attempt_dir.chmod(0o755)
    exports = export_clean_dataset(
        dataset=dataset,
        export_dir=attempt_dir,
        output_formats=output_formats,
        label_mapping=label_mapping,
        summary=summary,
        policy=policy,
    )
    export_errors = {
        key: value for key, value in exports.items() if key.endswith("_error")
    }
    if export_errors:
        raise RuntimeError(f"Falló la exportación HitL transaccional: {export_errors}")
    atomic_write_json(
        attempt_dir / "_SUCCESS",
        {
            "dataset": args.dataset,
            "run_id": run_id,
            "human_review": True,
        },
    )
    os.replace(attempt_dir, export_dir)
    
    logger.info("\n[bold green]=== EXPORTACIÓN MANUAL (HitL) FINALIZADA ===[/bold green]")
    logger.info(f"Dataset limpio exportado a: [bold cyan]{export_dir}[/bold cyan]")


if __name__ == "__main__":
    main()
