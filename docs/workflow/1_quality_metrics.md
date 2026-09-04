# Fase 1: Limpieza y Métricas de Calidad

Una vez que los datos crudos han sido ingestados en la Fase 0, el siguiente paso es auditar su calidad. En escenarios del mundo real (especialmente en agricultura), las imágenes suelen venir con mucho ruido: borrosidad, baja resolución, o incluso marcas de agua si provienen de buscadores de internet.

AgriVision Khaos automatiza este proceso de auditoría y calcula múltiples descriptores matemáticos por cada imagen para que luego puedas filtrar la basura.

---

## 1. Ejecutar el Cálculo de Calidad

Para analizar todo tu dataset de golpe, simplemente ejecuta el comando:

```bash
make quality
```

*(Opcional: Si nombraste a tu dataset de forma específica, recuerda usar `make quality DATASET="mi_proyecto"`)*

<details>
<summary>⚙️ ¿Qué métricas se calculan internamente?</summary>

El pipeline utiliza OpenCV y Tesseract (multihilo) para extraer la siguiente información y guardarla en la base de datos de FiftyOne:

1. **Resolución (`low_resolution`)**: Detecta si la imagen es menor a 320px.
2. **Desenfoque (`blur_variance`)**: Calcula la varianza del Laplaciano. Un valor muy bajo indica que la hoja está completamente borrosa.
3. **Iluminación (`brightness_mean`, `p5`, `p95`)**: Convierte la imagen a HSV y analiza el canal V (Value) para encontrar hojas quemadas por el sol o totalmente a oscuras.
4. **Marcas de Agua (`has_watermark`)**: Pasa un OCR (Tesseract) sobre la imagen para buscar textos translúcidos típicos de bancos de imágenes (Getty, Shutterstock, etc).
5. **Bordes Clonados (`has_smearing`)**: Analiza la entropía de los bordes. Identifica imágenes donde se ha usado clonación artificial para rellenar fondos.
</details>

## 2. Filtrar y Purgar en la Interfaz

El script anterior **no borra nada automáticamente**, simplemente etiqueta las imágenes con sus resultados para que tú decidas qué umbrales usar.

1. Abre la interfaz visual si no la tienes abierta:
   ```bash
   make app
   ```
2. Entra a tu navegador (`http://localhost:5151`, salvo que hayas configurado otro puerto).
3. En el panel izquierdo, busca la sección **PRIMITIVES** o **QUALITY**.
4. ¡Juega con los filtros! Por ejemplo:
   - Filtra `has_watermark = True` y selecciona todas las imágenes resultantes para borrarlas.
   - Usa el deslizador de `blur_variance` para ver qué pasa cuando filtras las que tienen menos de 100 (muy borrosas).

Una vez que hayas eliminado las imágenes basura manualmente desde la UI, estarás listo para pasar a la **Fase 2: Deduplicación**.
