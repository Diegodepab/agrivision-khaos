.PHONY: help setup quality deduplicate pipeline app export docs clean

#################################################################################
# GLOBALS                                                                       #
#################################################################################
DOCKER_CMD = docker compose run --rm fiftyone uv run python
WORKERS ?= 4
DATASET ?= agrivision-dataset
RAW_DIR ?= data/raw
METHOD ?= exact
PROFILE ?= quality-first
OUTPUT_FORMATS ?= datumaro,coco,yolo,classification
CLEANLAB_MODE ?= auto
EXPORT_DIR ?= data/processed
REPORT_DIR ?= reports/pipeline
ONTOLOGY ?=
MAX_PHASE_DROP ?= 0.40
MAX_TOTAL_DROP ?= 0.65

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Inicializa la Base de Datos e ingesta las imágenes crudas (Fase 0)
setup:
	@echo "Levantando base de datos y ejecutando ingesta..."
	docker compose up -d mongo
	$(DOCKER_CMD) src/01_acquisition/make_dataset.py --dataset-name $(DATASET) --raw-dir $(RAW_DIR)

## Ejecuta la batería de métricas de calidad (OCR, Blur, Smearing) (Fase 1)
quality:
	@echo "Ejecutando métricas de calidad..."
	$(DOCKER_CMD) src/02_curation/compute_quality_metrics.py --dataset $(DATASET) --workers $(WORKERS)

## Ejecuta el motor de deduplicación de imágenes (Fase 2)
deduplicate:
	@echo "Ejecutando motor de deduplicación..."
	$(DOCKER_CMD) src/02_curation/deduplicate_dataset.py --dataset $(DATASET) --method $(METHOD) --inspect

## Ejecuta todo el flujo desatendido y genera un reporte HTML
pipeline:
	@echo "Lanzando pipeline desatendido quality-first..."
	$(DOCKER_CMD) src/02_curation/run_pipeline.py \
		--dataset $(DATASET) \
		--raw-dir $(RAW_DIR) \
		--profile $(PROFILE) \
		--workers $(WORKERS) \
		--output-formats $(OUTPUT_FORMATS) \
		--cleanlab-mode $(CLEANLAB_MODE) \
		--export-dir $(EXPORT_DIR) \
		--report-dir $(REPORT_DIR) \
		$(if $(ONTOLOGY),--ontology-map $(ONTOLOGY)) \
		--max-phase-drop $(MAX_PHASE_DROP) \
		--max-total-drop $(MAX_TOTAL_DROP)
	@$(MAKE) fix-perms

## Transfiere la propiedad de los archivos generados del contenedor root al usuario actual
fix-perms:
	@echo "Ajustando permisos de archivos generados..."
	@docker compose run --rm --entrypoint /bin/sh fiftyone -c "chown -R $$(id -u):$$(id -g) data/ reports/"

## Levanta la aplicación FiftyOne para explorar los datos
app:
	@echo "Levantando FiftyOne App..."
	DATASET_NAME=$(DATASET) docker compose up -d fiftyone
	@echo "FiftyOne App disponible en: http://localhost:5152"

## Exporta el dataset manualmente después de haber sido validado en la UI
export:
	@echo "Exportando dataset validado manualmente (HitL)..."
	$(DOCKER_CMD) src/02_curation/export_dataset.py --dataset $(DATASET) --output-formats $(OUTPUT_FORMATS)

## Levanta la documentación de Zensical
docs:
	@echo "Levantando servidor de documentación..."
	docker compose up -d docs
	@echo "Documentación disponible en: http://localhost:8080"

## Destruye los contenedores y volúmenes (Peligro: Borra la BD local)
clean:
	@echo "Limpiando infraestructura Docker (volúmenes de BD incluidos)..."
	docker compose down -v
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################
.DEFAULT_GOAL := help
help:
	@echo "Available rules:"
	@echo
	@awk '/^## / { \
		sub(/^## /, ""); \
		doc=$$0; \
		getline; \
		if ($$0 ~ /^[a-zA-Z_-]+:/) { \
			sub(/:.*/, ""); \
			printf "  \033[36m%-15s\033[0m %s\n", $$0, doc \
		} \
	}' $(MAKEFILE_LIST)
