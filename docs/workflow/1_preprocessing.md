# Fase 1: Preprocesamiento y Limpieza

Una vez que tienes el repositorio montado y la base de datos llena (Fase 0), el siguiente paso es limpiar la "basura". Los datasets públicos suelen contener imágenes inservibles (marcas de agua, capturas de pantalla estiradas, imágenes microscópicas, o duplicados exactos).

Esta fase agrupa todas las herramientas de limpieza. Puedes aplicarlas en el orden que prefieras, aunque recomendamos empezar por el filtrado de calidad y terminar con la deduplicación.

---

## A. Filtrado de Calidad (Quality Metrics)

El motor de calidad analiza cada imagen y le inyecta descriptores matemáticos en la base de datos para que luego puedas ocultar las malas fácilmente desde la interfaz.

Para lanzar el análisis, ejecuta:

```bash
make quality
```

<details>
<summary>🔬 Opciones avanzadas y qué métricas se calculan</summary>

El comando inyecta los siguientes campos en la base de datos:
- **Corrupciones (`is_corrupted`)**: Detecta archivos truncados a nivel de byte.
- **Desenfoque (`blur_variance`)**: Calcula la varianza del Laplaciano para medir la nitidez.
- **OCR (`has_watermark`)**: Busca texto incrustado usando Tesseract OCR.
- **Smearing (`has_smearing`)**: Identifica bordes estirados artificialmente midiendo la entropía de los píxeles periféricos.
- **Baja Resolución (`low_resolution`)**: Marca imágenes menores a 320px.

*Si tienes una máquina potente, puedes acelerar el OCR pasando más workers:*
`make quality WORKERS=8`
</details>

### Cómo limpiar visualmente tras el análisis
1. Abre la interfaz web (`make app`).
2. En la barra lateral izquierda verás la sección **`quality`**.
3. Usa los filtros (por ejemplo, desmarca `low_resolution: True` o exige un `blur_variance` mínimo) para ocultar la basura.
4. Si quieres borrar la basura definitivamente, selecciónala y pulsa el botón de la papelera en la parte superior.

---

## B. Eliminación de Duplicados (Deduplicación)

Para evitar que imágenes idénticas (o versiones recortadas de la misma hoja) se filtren en tus conjuntos de validación y entrenamiento causando *data leakage*, debemos purgarlas.

Para buscar duplicados, ejecuta:

```bash
make deduplicate
```

<details>
<summary>🧠 ¿Cómo funciona la deduplicación semántica?</summary>

En lugar de usar un simple Hash (que fallaría si una imagen tiene un píxel distinto de brillo o compresión), este comando:
1. Pasa todas las imágenes por una red neuronal ligera (MobileNetV2).
2. Extrae sus *embeddings* (vectores matemáticos).
3. Calcula la distancia/similitud de coseno entre todas las imágenes.
4. Agrupa en clusters aquellas que tengan una similitud extrema.
</details>

### Cómo resolver los duplicados visualmente
1. Ve a la interfaz web de FiftyOne.
2. Abre el selector de Vistas (View) arriba a la izquierda (donde suele decir "Unsaved view").
3. Selecciona **"Inspect Duplicates"**.
4. Verás las imágenes agrupadas con sus dobles. Mantén la versión que tenga mejor calidad (la más grande o nítida) y elimina las otras copias usando la interfaz.
