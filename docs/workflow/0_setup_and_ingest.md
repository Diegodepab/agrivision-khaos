# Fase 0: preparación e ingesta

Docker fija el entorno de ejecución, pero siguen siendo necesarios un montaje
legible, espacio suficiente y formatos de anotación válidos. Por eso la ruta
recomendada empieza con una comprobación y una simulación.

## 1. Preparar el entorno

```bash
cp .env.example .env
```

Coloca cada fuente directamente bajo `data/raw/`, una fuente por subcarpeta. El
contenedor verá ese árbol en `/datasets/raw` y lo montará como solo lectura. Si
los datos residen en otro equipo, configura `RAW_DATA_HOST_PATH` después de
montarlos por NFS o SSHFS siguiendo la [guía remota](../remote_execution.md).

Cada raíz puede incluir un [`source.yaml`](../source_manifest.md) con licencia,
versión, dominio, cita y tareas. Para combinar vocabularios diferentes usa un
mapa de ontología explícito.

## 2. Formatos admitidos

El descubrimiento reconoce COCO, YOLO, VOC y árboles de clasificación. La
estructura y las referencias a imágenes se validan antes de mutar el catálogo.
Un JSON arbitrario no se interpreta automáticamente como COCO y una selección
ambigua de anotaciones detiene la ingesta para evitar resultados silenciosamente
incorrectos.

## 3. Validar antes de ingerir

```bash
make preflight
make dry-run DATASET="mi_dataset"
```

`preflight` comprueba montajes, lectura, escritura, disco, MongoDB y GPU. El
`dry-run` inspecciona imágenes y anotaciones y genera un reporte sin ejecutar
modelos, mover archivos ni modificar MongoDB.

## 4. Ingesta aislada o pipeline completo

Para explorar únicamente el catálogo:

```bash
make setup DATASET="mi_dataset"
make app DATASET="mi_dataset"
```

Para una curación reproducible usa directamente `make pipeline`; este comando ya
incluye una ingesta transaccional. Los registros almacenan las rutas inmutables
de las imágenes y sus metadatos, no extraen ni duplican todo el árbol original.
