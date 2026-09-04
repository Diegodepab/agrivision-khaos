# Fase 4: Revisión Manual (Human-in-the-Loop)

Aunque el pipeline desatendido (`make pipeline`) es extremadamente potente y toma decisiones automáticas basadas en calidad y duplicidad, siempre existirán **casos límite** donde la IA o los algoritmos heurísticos prefieren no destruir datos ante la duda.

Estos casos (por ejemplo, hojas borrosas que son la única copia disponible, o imágenes con aumentación artificial donde falta la foto original) se marcan con la etiqueta `"review"`. 

Esta fase te permite utilizar la interfaz visual de FiftyOne para validar o rechazar estas imágenes manualmente, y generar la exportación final limpia.

---

## 1. Lanzar la Interfaz Visual

Una vez que `make pipeline` haya finalizado, levanta el servicio de interfaz gráfica indicando el nombre de tu dataset:

```bash
make app DATASET="mi_super_dataset"
```

Abre tu navegador web y dirígete a [http://localhost:5151](http://localhost:5151), salvo que hayas configurado otro `FIFTYONE_PORT`.

## 2. Filtrar los Casos a Revisar

1. En la barra superior de búsqueda de FiftyOne, haz clic en el icono de **Filtros**.
2. Despliega la pestaña de **Tags**.
3. Selecciona la etiqueta `review`. 
4. La cuadrícula de imágenes se actualizará instantáneamente para mostrarte únicamente las fotos dudosas.

*(Opcionalmente, puedes filtrar por `curation.status == "review"` en el menú lateral de la izquierda).*

## 3. Auditar y Cambiar Etiquetas

FiftyOne te permite re-etiquetar imágenes visualmente con un par de clics:

1. Haz clic en la **casilla de verificación** (arriba a la izquierda de cada imagen) para seleccionar las imágenes que has decidido **salvar** o **destruir**. Puedes seleccionar múltiples imágenes a la vez, o usar el selector global para seleccionarlas todas.
2. En el menú superior (icono de etiqueta 🏷️), haz clic en **Tag samples**.
3. **Para salvarlas:** Elimina la etiqueta `review` y añade la etiqueta `kept`.
4. **Para destruirlas:** Elimina la etiqueta `review` y añade la etiqueta `removed`.
5. Dale a guardar.

Repite este proceso hasta que tu vista filtrada por `review` esté completamente vacía. ¡Felicidades, has curado el dataset manualmente!

## 4. Exportación Definitiva (HitL)

Tus decisiones manuales se guardan en la base de datos temporal, pero necesitamos generar los archivos exportados (COCO, YOLO, imágenes físicas) para usarlos en el entrenamiento de IA.

Vuelve a tu terminal y ejecuta:

```bash
make export DATASET="mi_super_dataset"
```

### ¿Qué hace este comando?
- Lee la base de datos de FiftyOne.
- Sincroniza las etiquetas visuales (`kept`, `removed`) con el motor lógico del dataset.
- Exige que no quede ninguna muestra con estado `review` sin resolver.
- Genera el resultado bajo el directorio configurado por `PROCESSED_DATA_HOST_PATH`.
- Escupe el manifiesto final y formatea los metadatos a COCO y YOLO exclusivamente para las imágenes que sobrevivieron al filtro.
- Publica el directorio de forma atómica y añade `_SUCCESS`; si alguna exportación
  falla, conserva un directorio `.incomplete-*` para diagnóstico.
