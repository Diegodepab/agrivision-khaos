# Fase 4: Revisión Manual (Human-in-the-Loop)

Aunque el pipeline desatendido (`make pipeline`) toma decisiones automáticas basadas en calidad (blur, contraste, smearing) y duplicidad (exacta y semántica), existen **casos límite** donde el sistema prefiere no descartar datos precipitadamente.

El dataset se organiza en tres subconjuntos o estados:
1. **Exportadas (`kept`)**: Muestras que superaron todos los filtros y forman el conjunto limpio exportado (ej. 24,060 muestras).
2. **En Revisión (`review`)**: Muestras dudosas o casos límite pendientes de confirmación humana (ej. 11,623 muestras).
3. **Descartadas (`removed`)**: Muestras que no superaron los criterios de calidad o resultaron redundantes (ej. 4,021 muestras).

Esta fase permite explorar visualmente cada conjunto en **FiftyOne**, aprobar o descartar casos dudosos, y **recuperar falsos positivos descartados**.

---

## 1. Abrir la Interfaz Visual FiftyOne

Para iniciar la interfaz interactiva, ejecuta:

```bash
make app DATASET="EnfermedadesFrutos"
```

Abre tu navegador web en:
👉 **[http://localhost:5151](http://localhost:5151)**

---

## 2. Cómo Ver Cada Conjunto de Imágenes

En la pantalla inicial de FiftyOne verás la totalidad de las imágenes del dataset (las 39,704 muestras). Para ver exactamente cada partición dispones de dos métodos:

### Método 1: Vistas Guardadas (Recomendado - 1 Clic)
En la parte superior izquierda de la pantalla, justo al lado del nombre del dataset, encontrarás el menú desplegable **Saved Views** (icono de marcador / desplegable):
* **`01_Exportadas_Kept`**: Carga únicamente las imágenes limpias exportadas (24,060).
* **`02_En_Revision_Review`**: Carga únicamente las imágenes dudosas pendientes de revisión (11,623).
* **`03_Descartadas_Removed`**: Carga únicamente las imágenes descartadas (4,021).

### Método 2: Filtro por Barra Lateral
En el panel lateral izquierdo:
* Despliega la sección **TAGS**:
  * Haz clic en `curation_kept` para ver las exportadas.
  * Haz clic en `curation_review` para ver las que están en revisión.
  * Haz clic en `curation_removed` para ver las descartadas.
* O bien despliega **`curation` -> `status`** y selecciona `kept`, `review` o `removed`.

---

## 3. Inspeccionar el Motivo de Cada Decisión

Al hacer clic sobre cualquier imagen de la cuadrícula, se abrirá su vista en detalle:
* En el panel lateral de la muestra, busca el bloque **`curation`**:
  * **`reason`**: Razón principal de la decisión (ej. `borderline_blur`, `semantic_duplicate`, `exact_duplicate`).
  * **`review_reasons`**: Lista detallada de advertencias acumuladas que enviaron la imagen a revisión.
  * **`quality`**: Valores numéricos calculados (laplacian blur, smearing, ratio de aspecto, etc.).

---

## 4. Cómo Auditar y Tomar Decisiones (HitL)

### A. Aprobar muestras de "En Revisión"
1. Entra en la vista **`02_En_Revision_Review`** (o activa el tag `curation_review`).
2. Selecciona las imágenes que consideres válidas (haz clic en el selector/cuadrado en la esquina superior izquierda de cada tarjeta, o pulsa `Espacio`).
3. Pulsa la tecla **`t`** en tu teclado (o haz clic en el icono de etiqueta 🏷️ en la barra superior).
4. Escribe el tag: **`kept`** y pulsa **Apply** (o Enter).

### B. Confirmar el descarte de muestras de "En Revisión"
1. Selecciona las fotos que definitivamente deban ser descartadas.
2. Pulsa la tecla **`t`** (o icono 🏷️).
3. Escribe el tag: **`removed`** y pulsa **Apply**.

### C. Recuperar muestras de "Descartadas" (Corregir falsos positivos)
Si al revisar las descartadas ves imágenes que consideras aprovechables (por ejemplo, una fruta que el algoritmo consideró dudosa):
1. Entra en la vista **`03_Descartadas_Removed`** (o activa el tag `curation_removed`).
2. Selecciona la(s) foto(s) que deseas salvar.
3. Pulsa la tecla **`t`** (o icono 🏷️).
4. Escribe el tag: **`kept`** y pulsa **Apply**.

> **Nota:** No necesitas borrar manualmente etiquetas anteriores (`curation_review`, `curation_removed`). El sistema se encarga de limpiar automáticamente los tags antiguos y asignar el nuevo estado canónico cuando ejecutes la sincronización.

---

## 5. Aplicar y Persistir las Decisiones

Una vez que hayas terminado tu sesión de etiquetado en FiftyOne:

### Opción A: Sincronizar en FiftyOne sin re-exportar archivos
Para que FiftyOne actualice su base de datos y recalcule los contadores de las vistas guardadas:
```bash
make sync-reviews DATASET="EnfermedadesFrutos"
```

### Opción B: Exportar el nuevo dataset limpio a disco
Para generar los formatos definitivos (COCO, YOLO, clasificación por carpetas) incluyendo las muestras aprobadas/recuperadas:
```bash
make export DATASET="EnfermedadesFrutos"
```

El resultado final se generará bajo `/datasets/processed/EnfermedadesFrutos_hitl/<run_id>/` de manera atómica y trazable.
