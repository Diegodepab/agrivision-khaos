.PHONY: help setup quality deduplicate app docs clean

#################################################################################
# GLOBALS                                                                       #
#################################################################################
DOCKER_CMD = docker compose run --rm fiftyone uv run python
WORKERS ?= 4
DATASET ?= agrivision-dataset
METHOD ?= exact

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Inicializa la Base de Datos e ingesta las imágenes crudas (Fase 0)
setup:
	@echo "Levantando base de datos y ejecutando ingesta..."
	docker compose up -d mongo
	$(DOCKER_CMD) src/01_acquisition/make_dataset.py --dataset-name $(DATASET)

## Ejecuta la batería de métricas de calidad (OCR, Blur, Smearing) (Fase 1)
quality:
	@echo "Ejecutando métricas de calidad..."
	$(DOCKER_CMD) src/02_curation/compute_quality_metrics.py --dataset $(DATASET) --workers $(WORKERS)

## Ejecuta el motor de deduplicación de imágenes (Fase 2)
deduplicate:
	@echo "Ejecutando motor de deduplicación..."
	$(DOCKER_CMD) src/02_curation/deduplicate_dataset.py --dataset $(DATASET) --method $(METHOD) --inspect

## Levanta la aplicación FiftyOne para explorar los datos
app:
	@echo "Levantando FiftyOne App..."
	DATASET_NAME=$(DATASET) docker compose up -d fiftyone
	@echo "FiftyOne App disponible en: http://localhost:5152"

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
