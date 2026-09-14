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

El comando respeta `POLICY=configs/quality-first.yaml`, igual que el pipeline.
Este perfil usa resolución mínima de 224px y desactiva OCR. Actívalo con
`make quality ENABLE_OCR=1`; `ENABLE_OCR=0` lo desactiva incluso si la política
lo habilita. Con OCR desactivado tampoco se ejecuta el prefiltrado de texto.
Los workers se calculan como CPU disponibles menos dos (mínimo uno), y se pueden
fijar mediante `WORKERS`. Cada worker usa un solo hilo interno de OpenCV.

Cuando OCR está activo, `quality.ocr_timeout_seconds` fija su tiempo máximo de
ejecución por imagen (30 segundos por defecto). El comando independiente acepta
también `--ocr-timeout SEGUNDOS`. Los timeouts se registran como errores de
procesamiento y pasan a revisión; el worker queda disponible para la siguiente imagen.

La máscara foliar prioriza Excess Green (`2G - R - B`) con dominancia verde
para separar vegetación de tierra. Incluye lesiones no verdes encerradas en el
contorno y recurre a Otsu en saturación y brillo cuando no hay suficiente verde.
Si ninguna máscara tiene cobertura útil, las métricas usan toda la imagen.
Es una heurística visual que debe comprobarse en muestras del dominio.

<details>
<summary>⚙️ ¿Qué métricas se calculan internamente?</summary>

El pipeline utiliza OpenCV y Tesseract (multihilo) para extraer la siguiente información y guardarla en la base de datos de FiftyOne:

1. **Resolución (`low_resolution`)**: Detecta si el lado menor incumple `quality.min_resolution` (224px en `quality-first`).
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
