# Fase 3: Pipeline Automatizado (Unattended Pipeline)

Una vez que has configurado tus conjuntos de datos en la carpeta `data/raw` (como se explica en la Fase 0), AgriVision Khaos ofrece la capacidad de ejecutar un **Pipeline Desatendido (End-to-End)**. 

Este pipeline consolida las fases de ingesta, cálculo de métricas de calidad y deduplicación múltiple (exacta, semántica y aumentada) en un único proceso automatizado. 

El objetivo es que dejes corriendo tu máquina toda la noche y al día siguiente obtengas:
1. Un dataset unificado y exportado limpiamente.
2. Un reporte HTML interactivo con las evidencias de descarte.

---

## 1. ¿Cómo Ejecutar el Pipeline?

Puedes ejecutar el pipeline utilizando el comando `make pipeline`:

```bash
make pipeline DATASET="nombre_de_tu_dataset_final" RAW_DIR="data/raw" PROFILE="quality-first"
```

### Parámetros Explicados
- **DATASET**: El nombre interno que recibirá el dataset unificado en la base de datos de FiftyOne (ej. `olive_final`).
- **RAW_DIR**: La ruta a la carpeta donde residen tus datasets crudos (por defecto `data/raw`).
- **PROFILE**: El perfil de curación. Actualmente el único perfil soportado (y predeterminado) es `quality-first`, el cual prioriza limpiar la basura, imágenes borrosas y aplicar un bypass estricto (100% de confianza) al borrado de duplicados.

> [!WARNING]
> No existe un perfil `agressive`. Si viste este término previamente, fue un error de transcripción. Usa siempre `quality-first` o simplemente omítelo para que tome el valor por defecto.

## 2. ¿El Pipeline Hace Aumentación de Datos (Data Augmentation)?

**NO.** El pipeline no crea nuevas imágenes. No inyecta rotaciones, no crea efectos espejo, ni expande el número de imágenes de tu dataset.

Si has leído la palabra "Augmentation" (Aumentación) en los reportes o en el código (`detect_augmentation_duplicates`), debes saber que se refiere a una **estrategia de limpieza defensiva**.

**¿Qué hace exactamente?**
El pipeline asume que los datasets que descargaste de internet *ya venían sucios con aumentación artificial* introducida por sus creadores originales. Por lo tanto, el sistema rastrea estas aumentaciones (espejos, rotaciones a 90 grados) utilizando Histogramas 3D de Color y **las elimina**. 

El sistema siempre conservará la imagen original intacta (basado en el cálculo de padding negro) y borrará los duplicados. ¡El pipeline solo destruye redundancias, no inventa imágenes falsas!

## 3. El Reporte HTML Generado

Una vez que el pipeline finaliza, no arroja los datos al vacío. Todo el proceso está meticulosamente documentado.
Deberás navegar a la carpeta:

```bash
reports/pipeline/<tu_dataset>/<timestamp>/
```

Allí encontrarás un archivo **`report.html`**. Puedes abrirlo con cualquier navegador web.
En este reporte encontrarás:
- **Tarjetas de Descartes por Calidad:** Ejemplos visuales de las hojas que fueron borradas por estar borrosas, poseer marcas de agua o ser diminutas.
- **Galerías de Duplicados (Conservada vs Eliminada):** Secciones desplegables que muestran "Pares" interactivos. A la izquierda verás la foto original (Conservada) y a la derecha su clon que fue enviado a la basura (Eliminada), permitiéndote auditar visualmente que el sistema no está fallando.

## 4. Archivos Resultantes
Al terminar, el dataset limpio se guardará (por defecto, usando un eficiente sistema de Hard Links que no duplica el peso en el disco) en:

```bash
data/processed/<tu_dataset>/<timestamp>/
```
Dentro verás subcarpetas estructuradas para `classification` (una carpeta por etiqueta), formatos `coco`, `yolo`, etc., listas para ser conectadas a tu código de entrenamiento.

## Siguiente Paso: Revisión Human-in-the-Loop

La magia del pipeline desatendido reside en que toma un 90% de las decisiones automáticamente. Sin embargo, para no perder información crítica, los casos ambiguos se marcan con la etiqueta `review`. 

Para auditar y salvar o rechazar estos casos visualmente antes de enviar el dataset al entrenamiento de IA, dirígete a la [Fase 4: Revisión Manual (HitL)](4_manual_review.md).
