# Fase 5: Publicación Open Data y Empaquetado Oficial en CKAN

Una vez completadas la curación desatendida y la auditoría visual, el objetivo es la **publicación de datasets estandarizados, limpios y trazables** en portales abiertos como el portal CKAN del grupo KHAOS:
- **Portal de Datos Abiertos ISIDROS (KHAOS - UMA):** [https://khaos.uma.es/isidros-opendata/](https://khaos.uma.es/isidros-opendata/)
- **Portal de Datos Abiertos de la Junta de Andalucía (ISI2A2)**
- **Agora Datalab / Zenodo**

Para ello, AgriVision-KHAOS proporciona el comando `make bundle`, que genera de forma 100% reproducible el paquete de distribución optimizado para análisis de datos y compatible con el límite de subida de CKAN (< 2 GB).

---

## 1. Principios del Diseño para Análisis de Datos

A diferencia de los volcados intermedios de ingeniería de datos, el paquete de distribución público sigue estándares estrictos para evitar el "síndrome de hiper-formato" y el ruido visual:

1. **Cero redundancia de imágenes**: Cada imagen se almacena **una sola vez** en el disco y en el ZIP, evitando multiplicar el peso del archivo con copias repetidas.
2. **Estructura Estándar ImageFolder**: La carpeta `dataset/` (`train/`, `val/`, `test/`) está organizada en subcarpetas por patología, compatible de forma nativa e inmediata con PyTorch (`torchvision.datasets.ImageFolder`), TensorFlow/Keras, fastai y scikit-learn.
3. **Ontología Jerárquica Trazable**:
   $$\text{repilo} \implies \text{enfermo}, \quad \text{pero} \quad \text{enfermo} \centernot\implies \text{repilo}$$
   La relación entre clases finas diagnósticas y la condición binaria (`sano` vs `enfermo`) se gestiona en la tabla `annotations/labels.csv` y se documenta en el `README.md`, sin necesidad de duplicar gigabytes de imágenes en dos árboles de carpetas separados.
4. **Registro de Fuentes y Procedencia (`SOURCES.md`)**: Fichero independiente que detalla para cada dataset original: autores, institución, enlaces directos a Kaggle/Roboflow, artículos científicos (DOIs), licencias, recuento de muestras aportadas y citas BibTeX.
5. **Separación de Metadatos de Publicación**: La ficha técnica con los campos para rellenar en la web de CKAN (`CKAN_METADATA.md`) se genera **fuera del archivo ZIP**, para que el operador la consulte al crear la entrada en el portal sin ensuciar la descarga del público.

---

## 2. Generar el Release Bundle

Para generar el paquete de publicación listo para CKAN:

```bash
make bundle DATASET="EnfermedadesHoja" VERSION="1.0"
```

O especificando una fecha concreta:

```bash
make bundle DATASET="EnfermedadesHoja" VERSION="1.0" DATE="20260918"
```

O utilizando el alias:

```bash
make release DATASET="EnfermedadesHoja" VERSION="1.0"
```

### Parámetros Configurables

| Parámetro | Valor por Defecto | Descripción |
|---|---|---|
| `DATASET` | *(Requerido)* | Nombre del dataset unificado (ej. `EnfermedadesHoja`, `PlagasOlivo`, etc.). |
| `VERSION` | `1.0` | Versión semántica de la entrega (en CKAN se usa el estándar `1.0`). |
| `DATE` | *(Fecha actual `YYYYMMDD`)* | Subcarpeta temporal para archivar las distintas versiones del dataset. |
| `ARCHIVE` | `zip` | Formato del archivo comprimido (`zip` estándar para CKAN, o `tar.gz`). |
| `EXPORT_DIR` | `/datasets/processed` | Directorio raíz donde residen las exportaciones curadas. |
| `REPORT_DIR` | `reports/pipeline` | Directorio raíz de los informes de calidad generados. |

---

## 3. Estructura de Salida Generada

El comando genera una jerarquía ordenada por dataset y fecha en `/datasets/processed/releases/<dataset>/<fecha>/`:

```text
data/processed/releases/
└── EnfermedadesHoja/
    └── 20260918/
        ├── EnfermedadesHoja_v1.0.zip                (538.6 MB)  <-- Recurso a subir en CKAN
        ├── EnfermedadesHoja_v1.0.zip.sha256         (Checksum de integridad)
        ├── EnfermedadesHoja_v1.0_CKAN_METADATA.md   (Ficha con campos listos para copiar y pegar)
        └── EnfermedadesHoja_v1.0/                   (Carpeta desempaquetada para inspección local)
```

### A. Ficha Auxiliar de Metadatos (Fuera del ZIP)
- `<dataset>_v<version>_CKAN_METADATA.md`: Texto formateado con título, slug, descripción Markdown (en inglés científico estándar del portal o en español), etiquetas kebab-case estándar (`agriculture`, `crop-disease`, `olive`, etc.), y metadatos de autor (`khaosadmin`).

### B. Archivo Comprimido Final (`<dataset>_v<version>.zip`)
El archivo descargable contiene únicamente la estructura limpia, adaptada automáticamente según la tarea primaria del dataset:

#### 1. Modo Detección de Objetos Unificado (ej. `OliveFruit`, `OliveSatelital`)
Cuando el dataset es predominantemente de detección, todo el contenido visual cuelga directamente de `dataset/` con formato estándar COCO y una partición complementaria para imágenes sin cajas delimitadoras:

```text
OliveFruit_v1.0.zip
│
├── README.md              <- Documentación concisa: descripción, ontología y tabla de clases
├── SOURCES.md             <- Registro completo de fuentes originales, URLs, licencias y citas
│
├── dataset/               <- Estructura unificada para visión artificial
│   ├── train/
│   │   ├── images/        <- Imágenes de entrenamiento
│   │   └── labels.json    <- Anotaciones de cajas delimitadoras en COCO JSON
│   ├── val/
│   │   ├── images/
│   │   └── labels.json
│   ├── test/
│   │   ├── images/
│   │   └── labels.json
│   └── test_unannotated/  <- Imágenes sin cajas (background / falsos positivos / test ciego)
│       ├── images/
│       └── README.md      <- Guía de uso de las imágenes sin anotar
│
├── annotations/           <- Metadatos tabulares limpios para analistas
│   └── labels.csv         <- Columnas: filename, split, disease, condition, task, source_dataset, source_url
│
└── preprocess/            <- Trazabilidad y auditoría de calidad
    └── report.html        <- Informe interactivo HTML de balanceo y deduplicación
```

#### 2. Modo Clasificación Multiclase (ej. `EnfermedadesHoja`)
Para datasets taxonómicos fito-sanitarios estándar basados en ImageFolder:

```text
EnfermedadesHoja_v1.0.zip
│
├── README.md              <- Documentación concisa: descripción, ontología y tabla de clases
├── SOURCES.md             <- Registro completo de fuentes originales, URLs, licencias y citas
│
├── dataset/               <- Estructura estándar ImageFolder (clasificación multiclase)
│   ├── train/
│   │   ├── sano/
│   │   ├── repilo/
│   │   ├── aculus_olearius/
│   │   └── ...
│   ├── val/
│   └── test/
│
├── detection/             <- (Opcional) Tarea secundaria de localización en formato COCO
│   ├── train/ (images/ y labels.json COCO)
│   ├── val/   (images/ y labels.json COCO)
│   └── test/  (images/ y labels.json COCO)
│
├── annotations/
│   └── labels.csv         <- Inventario tabular unificado
│
└── preprocess/
    └── report.html        <- Informe interactivo HTML de calidad
```

---

## 4. Publicación en el Portal CKAN

1. Acceder al portal: [https://khaos.uma.es/isidros-opendata/dataset/new](https://khaos.uma.es/isidros-opendata/dataset/new).
2. Abrir el fichero `<dataset>_v<version>_CKAN_METADATA.md` y copiar los campos directamente en el formulario (*Title*, *Description*, *Tags*, *License*, etc.).
3. En el paso de recursos (*Add Data*), subir:
   - **Recurso Principal (ZIP):** El archivo `<dataset>_v<version>.zip` (peso aprox. ~540 MB).
   - *(Opcional)* **Previsualización Tabular:** El archivo `labels.csv` para previsualizar datos tabulares en la web de CKAN sin descargar el ZIP.
