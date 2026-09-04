# AgriVision Khaos (Andalucía ISI2A2)

AgriVision Khaos ingiere, armoniza y cura colecciones heterogéneas de imágenes
agrícolas manteniendo el origen y las decisiones aplicadas a cada muestra. El
objetivo es producir datasets auditables para visión artificial, con validación
estructural, métricas de calidad, deduplicación conservadora y exportaciones
transaccionales.

## Flujo recomendado

1. Instala Docker y GNU Make, clona el repositorio y ejecuta
   `cp .env.example .env`.
2. Coloca cada fuente en una subcarpeta independiente de `data/raw/`. Añade un
   `source.yaml` para declarar versión, licencia, procedencia y contexto.
3. Comprueba los montajes, espacio y base de datos:

   ```bash
   make preflight
   ```

4. Valida anotaciones y simula las decisiones sin MongoDB ni modelos:

   ```bash
   make dry-run DATASET="mi_dataset"
   ```

5. Para una prueba local rápida usa `POLICY=configs/cpu-smoke.yaml`. Antes de un
   proceso completo descarga en caché el modelo activo con `make models`.
6. Ejecuta:

   ```bash
   make pipeline DATASET="mi_dataset"
   ```

7. Comprueba el reporte y el marcador `_SUCCESS`. Abre `make app
   DATASET="mi_dataset"`, resuelve todos los casos `review` y ejecuta `make
   export DATASET="mi_dataset"` para la publicación Human-in-the-Loop.

Los datos originales se montan en `/datasets/raw` como solo lectura. Una
ejecución interrumpida no publica el destino final y puede reanudarse desde sus
checkpoints compatibles.

## Qué significa «unificado»

El resultado es un catálogo común con trazabilidad, ontología normalizada y un
`task_type` por muestra. Clasificación y detección pueden convivir en FiftyOne,
pero no se mezclan como si fueran una única tarea: cada exportador selecciona
las muestras compatibles. El conjunto curado puede contener menos imágenes que
la suma inicial porque excluye corrupción, duplicados y revisiones pendientes.

El pipeline aporta controles técnicos y evidencia reproducible. No puede
demostrar por sí solo que una etiqueta sea agronómicamente correcta ni que los
umbrales sean óptimos para un sensor o cultivo nuevo; esas garantías requieren
revisión experta y un piloto representativo.

## Navegación

- **Architecture**: componentes, persistencia y trazabilidad.
- **Workflow**: ingesta, calidad, deduplicación, ejecución y revisión manual.
- **Ejecución remota**: montaje NFS/SSHFS entre almacenamiento y nodo GPU.
