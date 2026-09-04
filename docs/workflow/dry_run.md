# Dataset sintético y dry-run

El generador local crea fuentes COCO, YOLO y Pascal VOC pequeñas y deterministas:

```bash
make mock MOCK_DIR=data/mock
```

El auditor estático valida JSON, YAML, XML, referencias a imágenes, decodificación,
clases, geometría de cajas y conflictos de etiquetas entre archivos con el mismo
SHA-256. No conecta con MongoDB, no carga modelos y no copia ni mueve imágenes:

```bash
make dry-run RAW_DIR=data/mock DATASET=mock-validation
```

El único artefacto creado por el dry-run es un reporte JSON bajo
`reports/pipeline/<dataset>/<run_id>/dry_run.json`. El proceso termina con código 2
si encuentra errores, lo que permite usarlo como puerta de validación en CI.

Para comprobar las protecciones ante errores:

```bash
docker compose run --rm --no-deps fiftyone uv run --no-sync agrivision-mock \
  --output-dir data/mock-edge \
  --include-edge-cases
```

Este fixture incluye una imagen corrupta, una caja COCO negativa y una copia exacta
etiquetada de manera contradictoria en otra fuente.
