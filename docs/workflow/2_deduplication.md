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
Este método utiliza una red neuronal para extraer representaciones visuales de
las imágenes: MobileNetV2 en el perfil CPU y ResNet50 cuando CUDA está
disponible. Propone grupos visualmente similares, como recodificaciones o
recortes de una misma captura; por defecto los envía a revisión.

```bash
make deduplicate METHOD=semantic
```

### C) Candidatos por Aumentación (Histogramas de color) 🎨
Los embeddings pueden ser sensibles a rotaciones y espejos. El método `augmented` aísla
los colores de la hoja y usa un histograma 3D para proponer imágenes que podrían
ser transformaciones de la misma captura. Un histograma no conserva estructura
espacial, por lo que dos hojas distintas pueden parecerse: estos candidatos se
envían a revisión por defecto y no justifican borrado automático por sí solos.

```bash
make deduplicate METHOD=augmented
```

#### 🛡️ Desempate Inteligente (Preservación del Original)
Cuando el sistema encuentra un clúster de imágenes duplicadas (ej: 1 original y 3 rotadas), **no elige al azar**. El algoritmo está programado para medir la cantidad de "padding artificial" (píxeles puros negros o blancos generados en las esquinas por la rotación). 
La imagen con mejor puntuación de integridad se conserva como representante. En
el pipeline `quality-first`, estas coincidencias siguen requiriendo revisión;
solo una coincidencia exacta se retira automáticamente.

## 2. Inspección Visual vs Borrado Directo

Por seguridad, los comandos anteriores están diseñados para abrir una interfaz de **Inspección** (`--inspect`).
1. Al lanzar el comando, se abrirá una pestaña del navegador (`http://localhost:5151`).
2. Verás una galería donde las imágenes redundantes aparecen emparejadas al lado de su imagen original.
3. Esto te permite verificar que el umbral es el correcto y no está mezclando hojas diferentes.

El perfil `quality-first` elimina automáticamente solo duplicados exactos. Los
candidatos semánticos y por aumentación quedan en `review`. Sus umbrales son
configurables y deben calibrarse con ejemplos de cada dominio.

### 3. Purgar Definitivamente

Si tras inspeccionar la interfaz compruebas que los grupos son correctos, debes cerrar el comando anterior (usando `Ctrl+C` en la terminal) y lanzar la versión destructiva que borrará los duplicados para siempre:

```bash
docker compose run --rm fiftyone uv run --no-sync agrivision-deduplicate --dataset agrivision-dataset --method augmented --threshold 0.99 --delete
```
*(No olvides cambiar el nombre de `--dataset` si estás usando uno propio, y ajustar el `--method` al que hayas inspeccionado).*

Tras revisar los candidatos, vuelve a ejecutar la exportación Human-in-the-Loop.
Ningún método heurístico puede demostrar por sí solo que no queden duplicados;
el informe y los grupos de split permiten auditar el riesgo residual.
