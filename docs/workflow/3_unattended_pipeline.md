# Fase 3: Pipeline Automatizado (Unattended Pipeline)

Una vez que has configurado tus conjuntos de datos en la carpeta `data/raw` (como se explica en la Fase 0), AgriVision Khaos ofrece la capacidad de ejecutar un **Pipeline Desatendido (End-to-End)**.

Este pipeline consolida las fases de ingesta, cálculo de métricas de calidad y deduplicación múltiple (exacta, semántica y aumentada) en un único proceso automatizado.

Por diseño conservador, solo los duplicados exactos se descartan automáticamente.
Las coincidencias semánticas o basadas en color se apartan para revisión humana.
Si una fase habilitada falla, el pipeline no publica `_SUCCESS`.

El objetivo es que dejes corriendo tu máquina toda la noche y al día siguiente obtengas:
1. Un dataset unificado y exportado limpiamente.
2. Un reporte HTML interactivo con las evidencias de descarte.

---

## 1. ¿Cómo Ejecutar el Pipeline?

Antes del primer run real ejecuta `make preflight`. Si aún no tienes GPU, usa
`POLICY=configs/cpu-smoke.yaml`; el flujo seguirá validando ingesta, calidad
básica, deduplicación exacta/transformada, ontología y exportaciones.

Puedes ejecutar el pipeline utilizando el comando `make pipeline`:

```bash
make pipeline DATASET="nombre_de_tu_dataset_final" PROFILE="quality-first"
```

### Parámetros Explicados
- **DATASET**: El nombre interno que recibirá el dataset unificado en la base de datos de FiftyOne (ej. `olive_final`).
- **RAW_DIR**: Ruta visible dentro del contenedor; por defecto `/datasets/raw`, montada como solo lectura desde `RAW_DATA_HOST_PATH`.
- **PROFILE**: El perfil de curación. Actualmente el único perfil soportado (y predeterminado) es `quality-first`, el cual prioriza la calidad y la revisión de coincidencias visuales. Solo la equivalencia exacta con anotaciones compatibles permite un descarte automático.

> [!WARNING]
> No existe un perfil `agressive`. Si viste este término previamente, fue un error de transcripción. Usa siempre `quality-first` o simplemente omítelo para que tome el valor por defecto.

## 2. ¿El Pipeline Hace Aumentación de Datos (Data Augmentation)?

**NO.** El pipeline no crea nuevas imágenes. No inyecta rotaciones, no crea efectos espejo, ni expande el número de imágenes de tu dataset.

Si has leído la palabra "Augmentation" (Aumentación) en los reportes o en el código (`detect_augmentation_duplicates`), debes saber que se refiere a una **estrategia de limpieza defensiva**.

**¿Qué hace exactamente?**
El pipeline busca variantes que puedan venir incluidas en las fuentes. Combina huellas de píxeles, huellas perceptuales y color, y verifica la estructura de cada pareja. Por defecto envía las aumentaciones a revisión; un histograma similar no demuestra que sean la misma captura.

El sistema conserva un representante por grupo exacto. Las coincidencias por
embeddings o histogramas son heurísticas y se marcan para revisión por defecto;
el pipeline no crea imágenes aumentadas.

## 3. El Reporte HTML Generado

Una vez que el pipeline finaliza, no arroja los datos al vacío. Todo el proceso está meticulosamente documentado.
Deberás navegar a la carpeta:

```bash
reports/pipeline/<tu_dataset>/<timestamp>/
```

Allí encontrarás un archivo **`report.html`**. Puedes abrirlo con cualquier navegador web.
En este reporte encontrarás:
- **Tarjetas de Descartes por Calidad:** Ejemplos visuales de las hojas que fueron borradas por estar borrosas, poseer marcas de agua o ser diminutas.
- **Galerías de Duplicados (Conservada vs Eliminada):** Secciones desplegables que muestran "Pares" interactivos. A la izquierda verás la foto original (Conservada) y a la derecha la candidata alineada y su estado real (conservada, en revisión o descartada), permitiéndote auditar visualmente que el sistema no está fallando.

## 4. Archivos Resultantes
Al terminar, el dataset limpio se materializa en el almacenamiento de salida. La
vista de clasificación reutiliza hard links cuando el sistema de archivos lo
permite y copia como alternativa; otros formatos pueden copiar sus imágenes:

```bash
data/processed/<tu_dataset>/<timestamp>/
```
Dentro verás subcarpetas estructuradas para `classification` (una carpeta por etiqueta), formatos `coco`, `yolo`, etc., listas para ser conectadas a tu código de entrenamiento.

## 5. Caché por Huella Digital (Fingerprint) y Reanudación

El pipeline calcula una **huella digital SHA-256** combinando el contenido de los datasets crudos, la política de calidad, la ontología y las opciones de balanceo.

* **Reanudación automática:** Si se interrumpe un run o si se ejecuta con la misma configuración, el sistema detecta que el run ya está completado y no repite cómputos costosos innecesarios (`Run ya completado para estas fuentes y política`).
* **Forzar re-ejecución limpia:** Si deseas ignorar la caché previa y ejecutar todo desde cero, añade `RESUME=0`:
  ```bash
  make pipeline RESUME=0
  ```
* **Configuración cómoda mediante `.env`:**
  En lugar de pasar argumentos largos por la terminal, puedes definir en tu archivo `.env`:
  ```ini
  ONTOLOGY=reports/pipeline/EnfermedadesFrutos/20260910_091726_835690/proposed_ontology.yaml
  BALANCE_CLASSES=1
  BALANCE_TARGET=median
  ```
  El `Makefile` lee automáticamente el archivo `.env` en cada invocación.

## 6. Vistas Guardadas en FiftyOne y Siguiente Paso

Al concluir el pipeline, se registran automáticamente tres vistas guardadas en FiftyOne (`01_Exportadas_Kept`, `02_En_Revision_Review`, `03_Descartadas_Removed`).

Para auditar visualmente los datos, revisar los casos dudosos o recuperar posibles falsos positivos descartados, consulta la [Fase 4: Revisión Manual (HitL)](4_manual_review.md).
