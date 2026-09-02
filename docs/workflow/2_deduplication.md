# Fase 2: Deduplicación de Datos

Tras limpiar las imágenes borrosas o con marcas de agua, es común encontrarnos con otro problema crítico: **las imágenes duplicadas o aumentadas**. 

Si un dataset original fue sometido a *Data Augmentation* (por ejemplo, rotando la misma hoja 90 grados, o haciéndole efecto espejo), dejar estas imágenes juntas en el dataset base arruinará el entrenamiento. Si una rotación cae en entrenamiento y otra en validación, el modelo "memorizará" la hoja y te dará métricas falsamente altas.

AgriVision Khaos resuelve esto identificando y agrupando imágenes que visualmente son la misma hoja.

---

## 1. Estrategias de Deduplicación

Existen dos comandos principales para limpiar duplicados dependiendo del origen de tus datos:

### A) Deduplicación Exacta (Hash)
Busca imágenes que sean bit-a-bit idénticas a nivel de archivo.
```bash
make deduplicate METHOD=exact
```

### B) Deduplicación Semántica (IA) 🧠
Este método utiliza una red neuronal (`ResNet50`) para extraer las "huellas dactilares" visuales de las imágenes. Agrupa imágenes que son visualmente similares (recortadas o comprimidas en JPEG).

```bash
make deduplicate METHOD=semantic
```

### C) Deduplicación por Aumentación (Color Histograms) 🎨
Este es el método **definitivo para datasets agrícolas**. ResNet50 a menudo se confunde con rotaciones de 90/180 grados y efectos espejo. 
El método `augmented` aísla los colores puros de la hoja (ignorando fondos) y genera un histograma 3D. Esto le permite detectar **con precisión matemática** si una hoja es un espejo o una rotación exacta de otra.

```bash
make deduplicate METHOD=augmented
```

#### 🛡️ Desempate Inteligente (Preservación del Original)
Cuando el sistema encuentra un clúster de imágenes duplicadas (ej: 1 original y 3 rotadas), **no elige al azar**. El algoritmo está programado para medir la cantidad de "padding artificial" (píxeles puros negros o blancos generados en las esquinas por la rotación). 
La imagen que no tenga este padding artificial es coronada como la "Original" y mantenida a salvo, mientras que las versiones aumentadas se marcan para destrucción.

## 2. Inspección Visual vs Borrado Directo

Por seguridad, los comandos anteriores están diseñados para abrir una interfaz de **Inspección** (`--inspect`).
1. Al lanzar el comando, se abrirá una pestaña del navegador (`http://localhost:5151`).
2. Verás una galería donde las imágenes redundantes aparecen emparejadas al lado de su imagen original.
3. Esto te permite verificar que el umbral es el correcto y no está mezclando hojas diferentes.

*(Nota: Por defecto, la similitud exigida es del **0.95**. Este umbral es estricto para evitar falsos positivos en hojas muy parecidas).*

### 3. Purgar Definitivamente

Si tras inspeccionar la interfaz compruebas que los grupos son correctos, debes cerrar el comando anterior (usando `Ctrl+C` en la terminal) y lanzar la versión destructiva que borrará los duplicados para siempre:

```bash
docker compose run --rm fiftyone uv run python src/02_curation/deduplicate_dataset.py --dataset agrivision-dataset --method augmented --threshold 0.99 --delete
```
*(No olvides cambiar el nombre de `--dataset` si estás usando uno propio, y ajustar el `--method` al que hayas inspeccionado).*

> 🎉 **¡Felicidades!** Una vez completado este paso, tu dataset está purgado de ruido, no tiene fugas de datos por duplicados, y está listo para ser dividido y utilizado para entrenar los modelos finales.
