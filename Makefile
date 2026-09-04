.PHONY: help mock dry-run preflight models gpu-build setup quality deduplicate pipeline app export fix-perms docs clean

#################################################################################
# GLOBALS                                                                       #
#################################################################################
DOCKER_CMD = docker compose run --rm fiftyone uv run --no-sync
WORKERS ?= 4
DATASET ?= agrivision-dataset
RAW_DIR ?= /datasets/raw
METHOD ?= exact
PROFILE ?= quality-first
POLICY ?= configs/quality-first.yaml
OUTPUT_FORMATS ?= coco,yolo,classification
CLEANLAB_MODE ?= auto
EXPORT_DIR ?= /datasets/processed
REPORT_DIR ?= reports/pipeline
CACHE_DIR ?= /datasets/cache
ONTOLOGY ?=
MAX_PHASE_DROP ?= 0.40
MAX_TOTAL_DROP ?= 0.65
REQUIRE_GPU ?= 0
MOCK_DIR ?= data/mock
HOST_UID ?= $(shell id -u)
HOST_GID ?= $(shell id -g)

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Inicializa la Base de Datos e ingesta las imágenes crudas (Fase 0)
setup:
	@echo "Levantando base de datos y ejecutando ingesta..."
	docker compose up -d mongo
	$(DOCKER_CMD) agrivision-ingest --dataset-name $(DATASET) --raw-dir $(RAW_DIR)

## Genera un dataset diminuto COCO/YOLO/VOC para pruebas locales rápidas
mock:
	docker compose run --rm --no-deps fiftyone uv run --no-sync agrivision-mock --output-dir $(MOCK_DIR)

## Valida y simula el pipeline sin MongoDB, modelos ni exportaciones
dry-run:
	docker compose run --rm --no-deps fiftyone uv run --no-sync agrivision-pipeline \
		--dataset $(DATASET) \
		--raw-dir $(RAW_DIR) \
		--policy $(POLICY) \
		--output-formats $(OUTPUT_FORMATS) \
		--report-dir $(REPORT_DIR) \
		--dry-run

## Comprueba almacenamiento, rendimiento, disco, MongoDB y GPU opcional
preflight:
	$(DOCKER_CMD) agrivision-preflight \
		--raw-dir $(RAW_DIR) \
		--output-dir $(EXPORT_DIR) \
		--cache-dir $(CACHE_DIR) \
		--database-uri mongodb://mongo:27017/ \
		--require-read-only \
		$(if $(filter 1 true yes,$(REQUIRE_GPU)),--require-gpu) \
		--report reports/preflight.json

## Descarga y conserva el modelo de embeddings antes de un pipeline real
models:
	docker compose run --rm --no-deps fiftyone uv run --no-sync python -c "from agrivision_khaos.deduplication import SIMILARITY_MODEL; import fiftyone.zoo as foz; print(f'Descargando {SIMILARITY_MODEL}'); foz.load_zoo_model(SIMILARITY_MODEL)"

## Construye la imagen CUDA; úsala después con el override GPU documentado
gpu-build:
	TORCH_EXTRA=cu130 docker compose -f docker-compose.yml -f docker-compose.gpu.yml build fiftyone

## Ejecuta la batería de métricas de calidad (OCR, Blur, Smearing) (Fase 1)
quality:
	@echo "Ejecutando métricas de calidad..."
	$(DOCKER_CMD) agrivision-quality --dataset $(DATASET) --workers $(WORKERS)

## Ejecuta el motor de deduplicación de imágenes (Fase 2)
deduplicate:
	@echo "Ejecutando motor de deduplicación..."
	$(DOCKER_CMD) agrivision-deduplicate --dataset $(DATASET) --method $(METHOD) --inspect

## Ejecuta todo el flujo desatendido y genera un reporte HTML
pipeline:
	@echo "Lanzando pipeline desatendido quality-first..."
	$(DOCKER_CMD) agrivision-pipeline \
		--dataset $(DATASET) \
		--raw-dir $(RAW_DIR) \
		--profile $(PROFILE) \
		--policy $(POLICY) \
		--workers $(WORKERS) \
		--output-formats $(OUTPUT_FORMATS) \
		--cleanlab-mode $(CLEANLAB_MODE) \
		--export-dir $(EXPORT_DIR) \
		--report-dir $(REPORT_DIR) \
		--cache-dir $(CACHE_DIR) \
		--require-read-only \
		$(if $(filter 1 true yes,$(REQUIRE_GPU)),--require-gpu) \
		$(if $(ONTOLOGY),--ontology-map $(ONTOLOGY)) \
		--max-phase-drop $(MAX_PHASE_DROP) \
		--max-total-drop $(MAX_TOTAL_DROP)
	@$(MAKE) fix-perms

## Transfiere la propiedad de los archivos generados del contenedor root al usuario actual
fix-perms:
	@echo "Ajustando permisos de archivos generados..."
	@docker compose run --rm --entrypoint /bin/sh fiftyone -c 'for path in /datasets/processed /workspace/reports /datasets/cache/runs /datasets/cache/locks; do [ ! -e "$$path" ] || chown -R $(HOST_UID):$(HOST_GID) "$$path"; done'

## Levanta la aplicación FiftyOne para explorar los datos
app:
	@echo "Levantando FiftyOne App..."
	DATASET_NAME=$(DATASET) docker compose up -d fiftyone
	@echo "FiftyOne App disponible en: http://localhost:$${FIFTYONE_PORT:-5151}"

## Exporta el dataset manualmente después de haber sido validado en la UI
export:
	@echo "Exportando dataset validado manualmente (HitL)..."
	$(DOCKER_CMD) agrivision-export --dataset $(DATASET) --output-formats $(OUTPUT_FORMATS) --export-dir $(EXPORT_DIR) --policy $(POLICY)
	@$(MAKE) fix-perms

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
