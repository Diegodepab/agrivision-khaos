# Fase 2: Detección de variantes y deduplicación

El objetivo es conservar el mejor representante de una misma captura, mantener
sus anotaciones y evitar que sus variantes crucen entrenamiento, validación y
prueba. Dos fotografías distintas de una hoja no son necesariamente aumentaciones.
Una imagen única no se descarta solo por su nombre o por parecer transformada.

## Detección y evidencia

El método `exact` compara SHA-256 de los bytes actuales. `semantic` usa embeddings
para proponer candidatos; por sí solo requiere revisión. Cuando la fase de
aumentaciones está habilitada, recibe también esos candidatos antes de decidir. `augmented` combina:

- Identidad de píxeles a resolución completa bajo los ocho giros de 90° y espejos,
  conservando dimensiones, profundidad y alfa en la huella.
- Huellas perceptuales DCT e histogramas HSV para buscar vecinos.
- Verificación espacial de las parejas alineadas: correlación por regiones,
  error de intensidad y color y cobertura completa para escalado y recompresión.
- Correspondencias ORB mutuas y ajuste afín restringido con RANSAC. Se guardan
  error de reproyección, distribución de puntos, cobertura en ambas imágenes,
  comparación alineada por regiones y contenido texturado fuera del solapamiento.
  Los gradientes son una aproximación de contenido, **no una segmentación de hoja
  o lesiones**. Estos casos siguen en revisión, incluso con buen ajuste.

Los niveles persistidos son `exact_bytes`, `exact_pixels`, `verified_visual`,
`candidate` y `rejected`. Este último significa que no se encontró soporte para
las transformaciones probadas, no que se haya demostrado ausencia de cualquier
relación posible. Las transparencias modificadas y las imágenes poco informativas
requieren revisión; las ilegibles no se introducen como vectores cero.

## Detectar, inspeccionar y aplicar

```bash
make deduplicate-detect METHOD=augmented
make deduplicate METHOD=augmented
make deduplicate-apply METHOD=augmented
```

La inspección carga las parejas persistidas y, si todavía no existen, las calcula.
Para cambiar el umbral, ejecuta una detección nueva. `--threshold` no se acepta
junto a `--inspect` o `--apply`, evitando recalcular silenciosamente lo revisado.
Los comandos aceptan `--policy` y `--cache-dir`. Las escrituras del comando de
deduplicación toman el mismo lock por dataset que el pipeline. Si cambias la raíz
de caché del pipeline, pasa `--lock-dir <raíz>/locks` al comando independiente.
No ejecutes una exportación manual mientras otro proceso modifica el dataset.

El pipeline completo descarta duplicados exactos compatibles y envía aumentaciones
a revisión por defecto. El comando independiente separa detección y aplicación:
la detección marca revisión y `--apply` aplica la política. Antes de un descarte
revalida los píxeles, anotaciones y estado del representante.

`--delete` se mantiene como alias de `--apply`: **ya no elimina registros de
FiftyOne ni archivos**. El estado `removed` excluye la muestra de la exportación.
El material original y las relaciones permanecen disponibles para auditoría.

## Política conservadora

Los perfiles incluidos conservan:

```yaml
deduplication:
  augmentation_action: review
  remove_exact_transforms: false
  candidate_neighbors: 20
  candidate_retrieval: indexed
  candidate_pool: 512
  phash_distance: 12
  verification_min_correlation: 0.97
  verification_max_error: 0.04
```

Para retirar automáticamente solo equivalencias discretas exactas, establece
`augmentation_action: remove` **y** `remove_exact_transforms: true` en una copia de
la política. Se sigue exigiendo compatibilidad de anotaciones y un representante
conservado. Las coincidencias `verified_visual` siguen en revisión, incluso con
esos parámetros: su precisión en datasets reales todavía requiere calibración.
`semantic_action: remove`, admitido por compatibilidad, tampoco permite descartar
por similitud semántica sola.

El límite de vecinos se aplica por señal aproximada; su unión puede producir
hasta el doble de candidatos seleccionados por imagen, además de enlaces entrantes.
Las familias exactas se conectan independientemente del límite. El informe cuenta
las imágenes que saturan dicho límite para evaluar la recuperación.

`indexed` utiliza tablas de bandas de pHash y proyecciones de color. Acota el
conjunto puntuado a `candidate_pool` vecinos por imagen; las colecciones que caben
enteras en ese presupuesto se comparan exhaustivamente. En colecciones mayores
puede omitir parejas: `pool_truncated_images` registra truncamientos y
`scored_neighbors` el trabajo realizado. `candidate_retrieval: exhaustive`
conserva la referencia cuadrática para evaluar esa pérdida en un piloto.
Las familias de píxeles exactos mantienen su conexión al margen del índice.

## Elección del representante y protección de información

Se priorizan estado de curación, integridad y validez de anotaciones; después,
nitidez a escala comparable, relleno de esquinas y resolución. El nombre del
archivo solo desempata. Los fondos blancos/negros uniformes y la transparencia
no se contabilizan como relleno de rotación.

Cada sustitución necesita una relación directa con su representante, o igualdad
exacta transitiva. Una cadena A–B–C de similitud visual no autoriza retirar C por A.
Las cajas de detección se comparan bajo la transformación discreta; atributos
incompatibles, máscaras de instancia o tareas no comparables quedan para revisión.
Los motivos de revisión anteriores se conservan. Si una fase posterior invalida
al representante, sus descartes dependientes vuelven a revisión.

## Familias, revisión humana y particiones

`duplicate_family_id` reúne relaciones confirmadas. `duplicate_cluster_id` agrupa
conservadoramente también candidatos plausibles. Cada método conserva sus enlaces
en `duplicate_links`, con distancia o transformación cuando está disponible.
Los grupos de partición combinan esas relaciones con `capture_group` y `video_id`,
incluso a través de muestras excluidas. Los IDs de captura/vídeo se interpretan
dentro de su fuente. `splits.group_by_location` permite añadir ubicación, y está
desactivado por defecto. La fuente debe aportar esos metadatos; no se infieren.

Para confirmar o rechazar relaciones, prepara un JSON con IDs de FiftyOne:

```json
[
  {"left": "ID_A", "right": "ID_B", "decision": "reject"},
  {"left": "ID_C", "right": "ID_D", "decision": "confirm"}
]
```

```bash
docker compose run --rm fiftyone uv run --no-sync agrivision-deduplicate \
  --dataset agrivision-dataset --review-file /workspace/review-pairs.json
```

`confirm` une la familia sin aprobar automáticamente sus muestras. `reject` anula
esa relación incluso tras recalcular la detección. Para dividir una familia hay
que rechazar todos los enlaces que conectan los subgrupos; una ruta alternativa
mantiene la unión. Después se resuelven los estados de las muestras en FiftyOne y
se ejecuta `make export`. La aceptación de una muestra es una decisión humana
sobre todos sus motivos pendientes, que se muestran en `curation.review_reasons`.

La exportación incluye familias, grupos de split y versión del algoritmo en el
manifiesto, además de `duplicate_evidence.json` con relaciones, decisiones,
anotaciones y procedencia. El reporte compara vistas alineadas y muestra el estado
real de cada candidata. La auditoría impide publicar familias partidas o descartes
automáticos sin representante conservado. Una segunda auditoría vuelve a leer los
activos que se exportarán, amplía el presupuesto de vecinos y busca relaciones
entre splits sin depender de los enlaces almacenados. Bloquea coincidencias
confirmadas nuevas e incluye candidatos ambiguos y límites de búsqueda en
`visual_split_audit`. Una revisión humana que rechazó una pareja sigue prevaleciendo.
Esto no garantiza descubrir todas las variantes existentes.

`curation_history` conserva estados y tags anteriores/posteriores con fecha;
`duplicate_review_history` conserva los cambios de revisión de relaciones. Ambos
se exportan con las evidencias. Para revertir **todas las decisiones automáticas
de duplicidad** al estado previo y volver a detectar:

```bash
docker compose run --rm fiftyone uv run --no-sync agrivision-deduplicate \
  --dataset agrivision-dataset --reset
```

El reset conserva decisiones humanas e historial. Para deshacer una revisión de
pareja, presenta la decisión contraria en `--review-file`. Después regenera las
particiones y la exportación; una exportación ya publicada no se modifica.

## Benchmark y ejecución en la máquina de trabajo

```bash
make augmentation-benchmark BENCHMARK_DIR=/datasets/cache/augmentation-pilot-01
```

Genera 90 imágenes reproducibles, evalúa la detección y escribe `benchmark.json`.
Para evaluar imágenes reales en modo solo lectura, prepara un manifiesto con
familias revisadas. Las rutas son relativas al directorio del manifiesto:

```json
{
  "synthetic": false,
  "samples": [
    {"id": "a", "filepath": "images/a.jpg", "family": "capture-1", "source": "farm-a", "transformation": "original"},
    {"id": "b", "filepath": "images/b.jpg", "family": "capture-1", "source": "farm-a", "transformation": "jpeg"},
    {"id": "c", "filepath": "images/c.jpg", "family": "capture-2", "source": "farm-a", "transformation": "original"}
  ]
}
```

```bash
make augmentation-benchmark \
  BENCHMARK_MANIFEST=/datasets/raw/pilot/manifest.json \
  BENCHMARK_DIR=/datasets/cache/augmentation-real-01
```

Se puede repetir el comando con el mismo manifiesto para medir reutilización de
caché. Las familias usadas para evaluar deben estar separadas por captura de las
usadas para ajustar umbrales e incluir negativos difíciles. No deduzcas la familia
solo del nombre de archivo. `family` es un ID global de captura revisada; `source`
permite desglosar resultados y `capture_group` identifica capturas compartidas al
comprobar la separación con calibración. Una imagen ilegible o un manifiesto
inválido hacen fallar el benchmark, evitando métricas sobre una muestra incompleta.

Para comparar el índice con la referencia y comprobar separación de capturas:

```bash
make augmentation-benchmark \
  BENCHMARK_MANIFEST=/datasets/raw/pilot/evaluation.json \
  BENCHMARK_DIR=/datasets/cache/augmentation-evaluation \
  BENCHMARK_ARGS="--compare-exhaustive --calibration-manifest /datasets/raw/pilot/calibration.json"
```

El target usa `POLICY`, igual que el pipeline. El informe contiene parejas y sus
puntuaciones, recuperación por transformación/fuente, familias con enlaces falsos,
carga de revisión y comparación con HSV. El intervalo Wilson del 95 % describe
consistencia por familia suponiendo capturas independientes; **no certifica una
precisión del 99,5 % de sustituciones** ni evalúa conservación de anotaciones.
`activation.ready` permanece en `false`: activar retiradas aproximadas requiere
una evaluación de sustituciones revisadas y sus anotaciones, además de identidad.

La referencia HSV se omite por encima de 2000 imágenes; `--baseline-limit 0` la
desactiva. `--compare-exhaustive` es explícitamente cuadrático: úsalo en pilotos.
Para medir tamaños crecientes sin MongoDB ni modelos descargados:

```bash
make augmentation-benchmark BENCHMARK_DIR=/datasets/cache/aug-90 \
  BENCHMARK_ARGS="--families 6 --compare-exhaustive"
make augmentation-benchmark BENCHMARK_DIR=/datasets/cache/aug-900 \
  BENCHMARK_ARGS="--families 60 --baseline-limit 0"
```

Repite cada comando para medir caché caliente. Usa otro directorio al cambiar el
tamaño del fixture. Las cifras de memoria son el máximo del proceso, no solo del
índice; los tiempos de detección y evaluación total se reportan por separado.

La caché verifica los bytes actuales y reutiliza descriptores y parejas por
contenido, versión y configuración, con checksum y validación estructural para
recuperarse de cachés incompletas o dañadas. El pipeline la conserva entre ejecuciones
bajo `CACHE_DIR/deduplication/<dataset>/augmentation-cache/`. Los checkpoints del
algoritmo anterior se invalidan. Se publican tiempos, memoria máxima del proceso,
reutilización, número de parejas y saturación de vecinos. El índice mantiene memoria lineal y trabajo de puntuación acotado por imagen.
La verificación de las parejas puede dominar el tiempo total. Los resultados
sintéticos medidos se conservan en [benchmarks](../benchmarks/augmentation_synthetic.json);
valida recuperación, carga de revisión y memoria en tu dominio y volumen.
