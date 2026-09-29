# Catálogo Estructurado de Datasets Agroalimentarios (ISI2A2)

El siguiente documento clasifica las fuentes de datos recopiladas basándose en una premisa estricta de **cohesión arquitectónica**. 

Se han separado rigurosamente los datasets de imágenes ópticas (RGB) agrupándolos por dominio de cultivo y tarea de Visión Artificial (CV). Además, se ha creado una categoría especial para **Datasets Aislados** (archivos tabulares, multiespectrales, LiDAR o de rayos X), los cuales poseen un alto valor científico pero *no deben ser fusionados* con los tensores de imágenes bidimensionales en esta fase del pipeline para evitar la corrupción del esquema de datos.

---

## 1. Dominio Olivo (Olea europaea) - Imágenes Ópticas

### 1.1 Enfermedades a Nivel de Hoja
Ideales para unificar en un macro-dataset de patologías del olivo.

| Nombre del Dataset | Tamaño / Formato | Origen y Licencia | Enlaces | Observaciones |
| :--- | :--- | :--- | :--- | :--- |
| **Olive Leaf Image Dataset** | 102.9 MB (3,409 JPG) | Turquía <br> CC0 | [Dataset Kaggle](https://www.kaggle.com/datasets/habibulbasher01644/olive-leaf-image-dataset) <br> [Paper](https://www.researchgate.net/publication/343374668_Classification_of_olive_leaf_diseases_using_deep_convolutional_neural_networks) | 1,050 sanas, 1,460 *Spilocaea oleagina*, 890 *Aculus olearius*. Dataset base. |
| **Olive Leaf Disease (Edincik)** | 737.3 MB (956 PNG) | Turquía <br> Unknown | [Dataset Kaggle](https://www.kaggle.com/datasets/serhathoca/zeytin) <br> [Paper](https://doi.org/10.1007/s00217-023-04386-8) | Etiquetas binarias: *hastalıklı* (enfermo) y *sağlam* (sano). |
| **Olive's Leaf Diseases (Roboflow)** | 123.1 MB (2,849 JPG) | Roboflow <br> CC BY 4.0 | [Dataset Universe](https://universe.roboflow.com/hahmedai-whou6/olive-s-leaf-diseases/dataset/3) | Introduce "Knot disease" (Tuberculosis). |
| **Olive Tree Diseases CV Dataset** | 75.1 MB (1,369 JPG) | Roboflow <br> CC BY 4.0 | [Dataset Universe](https://universe.roboflow.com/arina-fay/olive-tree-diseases/dataset/1) | Unión de 3 fuentes públicas. Múltiples enfermedades clave. |
| **Olive Leaf Disease — Clean Unified Dataset** | 1.83 GB (15,600 JPG) | Abeer Saleh / Kaggle <br> CC0 1.0 | [Dataset Kaggle](https://www.kaggle.com/datasets/abeersaleh987654321/olive-leaf-disease-clean) | 11 clases foliares clave (*Spilocaea*, *Aculus*, *Colletotrichum*, *Saissetia*, Virosis, etc.). |

### 1.2 Detección de Fruto, Madurez Fenológica y Calidad
Colección unificada de 11 datasets para detección de aceitunas, estimación de maduración y control de defectos.

| Nombre del Dataset | Tamaño / Formato | Origen y Licencia | Enlaces | Observaciones |
| :--- | :--- | :--- | :--- | :--- |
| **Olive fruit object detection** | 55 MB (972 JPG + TXT) | Mundial <br> CC0 1.0 | [Dataset Kaggle](https://www.kaggle.com/datasets/danielvalyano/olive-fruit-object-detection) | Cajas delimitadoras YOLO para detección y conteo de aceitunas. |
| **Fruits and Vegetables (Olive)** | 16 MB (1,082 JPG) | Kaggle <br> DbCL | [Dataset Kaggle](https://www.kaggle.com/datasets/abhisheksubhashswami/fruits-and-vegetables) | Clasificación de fruto de aceituna en fondo limpio (250x250 px). |
| **Olive Computer Vision (Omar)** | 1.2 GB (14,850 JPG) | Roboflow <br> CC BY 4.0 | [Dataset Universe](https://universe.roboflow.com/omar-z9f8z/olive-axotr) | Dataset masivo (>180,000 bboxes de aceitunas en campo). |
| **Olive Betog (Graduation Project)**| 25 MB (517 JPG) | Roboflow <br> CC BY 4.0 | [Dataset Universe](https://universe.roboflow.com/yolo-rdf7v/olive-bh00d) | Clasificación fenológica de maduración (Green, Red, Black). |
| **Olive Loxls Detection** | 171 MB (126 JPG) | Roboflow <br> CC BY 4.0 | [Dataset Universe](https://universe.roboflow.com/olivo/olive-loxls) | Detección de fruto en condiciones de iluminación de campo. |
| **Olive Detection Multi-Attribute**| 5 MB (69 JPG) | Roboflow <br> CC BY 4.0 | [Dataset Universe](https://universe.roboflow.com/olive-detectiom/olive-detection-1arbf) | Detección multiclase de madurez y fruto defectuoso (bad). |
| **Olive Detection for Real** | 95 MB (1,163 JPG) | Roboflow <br> CC BY 4.0 | [Dataset Universe](https://universe.roboflow.com/detection-for-real/olive-8qkre) | Detección densa (>34,000 bboxes saneadas a `aceituna`). |
| **Olive Class B Commercial** | 108 MB (100 JPG) | Roboflow <br> CC BY 4.0 | [Dataset Universe](https://universe.roboflow.com/object-detection/olive-class-b) | Clasificación comercial de calidad (categoría B). |
| **Olive Disease & Good Quality** | 163 MB (1,206 JPG) | Roboflow <br> Public Domain | [Dataset Universe](https://universe.roboflow.com/aceitunas-7eryk/olive-desease-good) | Inspección de sanidad: fruto sano (Buena) vs descarte (Mala). |
| **Olive Fruit Defects & Health** | 22 MB (532 JPG) | Roboflow <br> CC BY 4.0 | [Dataset Universe](https://universe.roboflow.com/olive-ia/olive-defects-oed6r) | Detección de defectos fitopatológicos/mecánicos vs fruto sano. |
| **Olive Instance Segmentation** | 2.4 MB (44 JPG) | Roboflow <br> CC BY 4.0 | [Dataset Universe](https://universe.roboflow.com/olivesssss/olive-segmentation-3logu) | Polígonos de segmentación fina y discriminación de desenfoque. |
| **Olive Tree Varieties Drought Stress** | 8.63 GB (2,880 JPG + 1 XLSX) | Marruecos <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/22j26tpk63/1) <br> [Paper](https://www.sciencedirect.com/science/article/pii/S2352340925007486) | Vistas frontales y laterales para análisis de crecimiento y estrés hídrico. |

### 1.3 Teledetección y Vista Aérea (RGB)
| Nombre del Dataset | Tamaño / Formato | Origen y Licencia | Enlaces | Observaciones |
| :--- | :--- | :--- | :--- | :--- |
| **ICAERUS Olive Tree Detection** | ~350 MB (1,925 JPG + TXT) | Grecia / UE <br> CC BY 4.0 | [Zenodo 13121962](https://zenodo.org/records/13121962) | Horizon Europe. Detección de árboles de olivo en vuelo UAV (`train_original`, `val`, `test`). Muestras 100% auténticas sin aumentaciones sintéticas. |
| **OliveTreeCrownsDb** | 120 MB (1,471 JPG + TXT) | Marruecos <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/xym8rd2srf/3) | Cuadrícula estandarizada 6x6 (912x608 px) alineada con escala de vuelo UAV para detección de copas de olivo. Ortofotos originales 1x1 y LiDAR preservados en extras. |
| **Burned and unburned olive trees UAV** | 19.4 MB (3,624 JPG) | Grecia <br> CC BY-NC-ND 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/83kpndkrb2/1) <br> [Paper](https://www.mdpi.com/2072-4292/16/23/4531) | Capturas de drones sobre olivar. Clasificación fitosanitaria: zonas secas/incendiadas (`Dry`) vs vigorosas/sanas (`Healthy`). |

---

## 2. Dominio Almendros y Frutos Secos - Imágenes Ópticas

### 2.1 Detección, Variedades y Calidad del Fruto
| Nombre del Dataset | Tamaño / Formato | Origen y Licencia | Enlaces | Observaciones |
| :--- | :--- | :--- | :--- | :--- |
| **Almond varieties classification** | 158.0 MB (1,556 JPG) | Turquía <br> Acceso Abierto | [GitHub](https://github.com/ymyurdakul/datasets/tree/main) <br> [Paper](https://link.springer.com/article/10.1007/s00217-024-04562-4#data-availability) | Clasificación fenotípica de 4 variedades comerciales de almendra turca: `AK`, `KAPADOKYA`, `NURLU` y `SIRA`. |
| **Almendras, pistachos, avellanas** | 4.9 MB (279 PNG) | Roboflow <br> CC BY 4.0 | [Dataset Universe](https://universe.roboflow.com/frutos-secos/almendras-pistachos-avellanas-hmxel) | Distinción multiclase entre avellanas (`hazelnut`) y pistachos (`pistachio`). |
| **NutsBD** | 48.5 MB (1,742 JPG) | Bangladesh <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/fmrnybgxb9/1) | Clasificación de 10 especies comerciales de frutos secos (almendra, anacardo, avellana, cacahuete, macadamia, nuez, nuez de Brasil, pistacho). |
| **AllerNuts** | 314 MB (4,390 JPG) | Bangladesh <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/3gtbxvgm5f/1) | Detección de frutos secos alergénicos con fondo real: `Almond`, `Cashew`, `Peanuts` y `Pistachio`. |
| **Almond Damage Detection** | 27.1 MB (736 JPG) | Kaggle <br> Apache 2.0 | [Dataset Kaggle](https://www.kaggle.com/datasets/mahyeks/almond-damage-detection) | Evaluación de calidad e integridad física del grano: `DAMAGED` (dañado/defecto) vs `NODAMAGE` (sano/intacto). |

### 2.2 Enfermedades Foliares
| Nombre del Dataset | Tamaño / Formato | Origen y Licencia | Enlaces | Observaciones |
| :--- | :--- | :--- | :--- | :--- |
| **Almond Diseases Detection.v2i** | 379.9 MB (4,500 JPG) | Roboflow <br> Unknown | [Dataset Universe](https://universe.roboflow.com/poker-chips-annotation/almond-diseases-detection-l7tfx/dataset/2) | Plantas de almendro con patologías visuales. |

---

## 3. Dominio Aguacate (Persea americana) - Imágenes Ópticas

### 3.1 Enfermedades a Nivel de Hoja (`AvocadoLeaf`)
Colección unificada para la detección y clasificación de patologías foliares del aguacatero.

| Nombre del Dataset | Tamaño / Formato | Origen y Licencia | Enlaces | Observaciones |
| :--- | :--- | :--- | :--- | :--- |
| **Curated Avocado Leaf (Tanzania)** | 199 MB (3,592 PNG) | Tanzania <br> CC BY 4.0 | [Zenodo](https://zenodo.org/records/22249692) <br> [Dataset Universe](https://universe.roboflow.com/suraj-azuiz/avocado-leaf-disease/dataset/9) | Hojas segmentadas en campo. Balanceado 50/50: 1,796 sanas (`Healthy`) y 1,796 con cercosporiosis (`Cercospora`). |
| **Avocado Leaf DataSet K-Kotagiri** | 120 MB (435 JPG) | India <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/6zy6wxhf2v/1) <br> [NLM NIH Catalog](https://datasetcatalog.nlm.nih.gov/dataset?q=0003273869) | Subconjunto `Avocado Original_Dataset` (219 sanas, 216 enfermas). Se excluyen 1,045 imágenes aumentadas artificialmente. |

### 3.2 Detección de Fruto, Madurez Fenológica y Calidad (`AvocadoFruit`)
Macro-dataset unificado de fruto para clasificación de madurez Hass, control fitopatológico (roña y antracnosis) y detección en árbol.

| Nombre del Dataset | Tamaño / Formato | Origen y Licencia | Enlaces | Observaciones |
| :--- | :--- | :--- | :--- | :--- |
| **Clasificación de Enfermedades del Aguacatero (Chapingo)** | 620 MB (3,983 JPG) | México <br> Open Access | [Paper Agrociencia](https://doi.org/10.47163/agrociencia.v55i8.2662) | Estudio fitopatológico en Morelos (cv. Fuerte). 1,764 sanos, 1,197 con roña (*Sphaceloma perseae*) y 1,022 con antracnosis (*Colletotrichum*). |
| **Hass Avocado Ripening Photographic Dataset** | 418 MB (14,710 JPG) | Mendeley <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/b2d83zft4s/1) | Tomas estandarizadas (800x800 RGB) clasificadas en los 5 estadíos de maduración fenológica Hass. |
| **Avocado Tree Detection & Segmentation** | 170.5 MB (2,580 JPG) | Roboflow / UC Riverside <br> CC BY 4.0 | [Dataset Universe](https://universe.roboflow.com/uc-riverside-ov9yb/avocado-jvq5e) <br> [Roboflow Aguacate](https://universe.roboflow.com/julian-emcmb/aguacate-9h8ah/dataset/2) | Detección y polígonos de segmentación de aguacate en ramas y copas bajo iluminación natural de huerto. |

### 3.3 Teledetección y Vuelo Aéreo (RGB)
| Nombre del Dataset | Tamaño / Formato | Origen y Licencia | Enlaces | Observaciones |
| :--- | :--- | :--- | :--- | :--- |
| **Avo-AirDB** | 8.05 GB (984 JPG) | Marruecos <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/tvhh83r3hj/3) | Ortofotos aéreas y vuelos de drones sobre parcelas de aguacatero. |

---

## 4. Dominio Mango (*Mangifera indica*) - Imágenes Ópticas

### 4.1 Patologías Foliares (`MangoLeaf`)
Macro-dataset unificado de patologías foliares del mango unificando 11.321 imágenes fotográficas reales bajo 9 clases diagnósticas canónicas y diagnóstico binario de cribado (sano vs patológico).
- **Ontología Canónica:** `configs/ontologies/mango_leaf.yaml`
- **Muestras Totales:** 11.321 imágenes reales de campo (MLD24 + SAR-MLD1-2025).

| Nombre del Dataset | Tipo de Órgano | Tamaño / Formato | Origen y Licencia | Enlaces | Observaciones |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **01_MLD24_Mango_Leaf** | Hoja | 136.8 MB (6,400 JPG) | Bangladesh <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/6dvpywm2m2/1) | 8 clases balanceadas (800 c/u): antracnosis, chancro bacteriano, oídio, mosquilla agallas, muerte regresiva, sooty mould, gorgojo cortador, sano. |
| **02_SAR_MLD1_Mango_Leaf** | Hoja | 13.27 GB (4,921 JPG) | Bangladesh <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/sd8hzpg69b/5) | Alta resolución (2025): antracnosis (972), oídio (983), mosquilla agallas (983), turning brown/necrosis (1.000) y sano (983). |

### 4.2 Inspección de Fruto, Calidad y Variedades Comerciales (`MangoFruit`)
Macro-dataset unificado de 7.203 imágenes de frutos de mango combinando patologías fitosanitarias comerciales y tipificación de 15 variedades y cultivares agronómicos.
- **Ontología Canónica:** `configs/ontologies/mango_fruit.yaml`
- **Muestras Totales:** 7.203 imágenes (MangoDHDS + MangoImageBD).

| Nombre del Dataset | Tipo de Órgano | Tamaño / Formato | Origen y Licencia | Enlaces | Observaciones |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **01_MangoDHDS_Fruit_Diseases** | Fruto | 36 MB (1,500 JPG) | India <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/b4nrw5hyyc/1) | Control fitosanitario: 5 clases balanceadas (300 c/u): antracnosis, chancro bacteriano, roña/sarna, pudrición peduncular (stem end rot) y sano. |
| **02_MangoImageBD_Fruit_Varieties** | Fruto | 1.2 GB (5,703 JPG) | Bangladesh <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/hp2cdckpdr/2) | Tipificación comercial: 15 variedades (Amrapali, Ashshina Classic/Zhinuk, Banana Mango, Bari-11, Bari-4, Fazli Classic/Shurmai, Gourmoti, Harivanga, Himsagor, Katimon, Langra, Rupali, Shada). |

---

## 5. Dominio Papaya y Chirimoya - Imágenes Ópticas

| Nombre del Dataset | Cultivo | Tamaño / Formato | Origen y Licencia | Enlaces | Observaciones |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Papaya Leaf Disease Image Dataset** | Papaya | 6.2 GB (3,626 PNG) | Bangladesh <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/3kwgxg4stb/1) | Patologías foliares de papaya. |
| **BDPapayaLeaf** | Papaya | 486.9 MB (2,164 JPG) | Bangladesh <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/p997fvf526/1) | Cajas delimitadoras de patologías de papaya. |
| **Dataset of Common Papaya Diseases** | Papaya | 783.1 MB (442 JPG) | Bangladesh <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/42xmy5j64g/1) | Enfermedades comunes de papaya. |
| **Papaya_Madurez** | Papaya | 26.0 MB (485 JPG) | India <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/rcy6y8fhcn/1) | Estadíos de maduración de fruto. |

## 6. Dominio Cereales - Imágenes Ópticas

| Nombre del Dataset | Cultivo / Tarea | Tamaño / Formato | Origen y Licencia | Enlaces | Estado / Observaciones |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Global Wheat Head Dataset 2021 (GWHD 2021)** | Trigo / Detección y Conteo de Espigas | 9.5 GB (6,515 PNG) | Consorcio Internacional (12 países) <br> CC BY-SA 4.0 | [Dataset Zenodo](https://zenodo.org/records/5092309) | **Listo en `data/raw/WheatHead`**. 275.468 espigas anotadas en formato COCO (`train`, `valid`, `test`). |
| **Wheat Plant Diseases High Res** | Trigo / Diagnóstico de Patologías y Plagas | 7.0 GB (14,154 PNG) | Kaggle / Open Access <br> CC BY 4.0 | [Dataset Kaggle](https://www.kaggle.com/datasets/kushagra3204/wheat-plant-diseases) | **Listo en `data/raw/WheatDisease`**. 15 clases patológicas (royas, fusariosis, tizón, pulgones, oídio, ácaros, sano). |
| **Field-Grown Barley Disease Multiclass**| Cebada / Patologías en Campo Abierto | 4.2 GB (PNG/TIFF) | JKI, Alemania <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/4ny92p2r8f) | Síntomas foliares de ramularia y roya en condiciones reales de cultivo con parches de alta resolución. |
| **Wild Oats Detection in Wheat Fields** | Malezas en Trigo | 48 MB (500 JPEG) | Egipto <br> Open Access | [Dataset Kaggle](https://www.kaggle.com/datasets/abanoublamie/wild-oats) | Muestras de avena loca (*Avena fatua*) entre cultivo de trigo. |
| **Rice Leaf Disease and Pest Augmented**| Arroz / Enfermedad | 5.7 GB (19,128 JPEG)| Bangladesh <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/vwv3nry3wr/1) | Posee aumentaciones sintéticas; requiere filtrado estricto si se incorpora. |
| **Maize Crop Disease Leaf** | Maíz / Patologías Foliares | 6.5 GB (30,000 JPG) | UAE <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/6w6gsvghfw/1) | Dataset masivo de maíz (*Zea mays*). |


---

## 7. Datasets Multiclase Genéricos

| Nombre del Dataset | Tamaño / Formato | Origen y Licencia | Enlaces | Observaciones |
| :--- | :--- | :--- | :--- | :--- |
| **Multiclase Frutas y Vegetales** | 1 GB (20,216 IMG) | Univ. León, España <br> CC BY 4.0 | [Dataset Zenodo](https://zenodo.org/records/18904638) | ~222,300 anotaciones YOLO. |

---

## 8. ⚠️ Datasets Aislados (No Unificables en CV Clásico)

> **Nota Arquitectónica:** Los siguientes datasets contienen formatos tubulares (`CSV`, `XLSX`), sensores especializados (Rayos X, LiDAR, `.HDR` Hiperespectral), combinaciones multiespectrales complejas (`TIFF` apilados) o repositorios heterogéneos multi-especie. **NO deben mezclarse** en los pipelines de imágenes RGB estándar específicos de cultivo. Quedan registrados aquí como repositorio pasivo para futuras arquitecturas multimodales o análisis de datos estructurados.

| Nombre del Dataset | Formato / Tipo de Dato | Justificación de Aislamiento | Enlaces |
| :--- | :--- | :--- | :--- |
| **NZDLPlantDisease-v1 (New Zealand)** | 1.9 GB (.RAR) | Multi-cultivo heterogéneo (manzanos, kiwis, perales y uvas mezcladas). | [GitHub NZDL](https://github.com/kmarif/NZDLPlantDisease-v1) |
| **Avocado-DB Dataset** | 640 MB (CSV + JPG) | Tabular/Biometría. Pesaje y cubicaje de pulpa/hueso en plato giratorio. | [Zenodo](https://zenodo.org/records/22249692) |
| **HyperSens: Divergent spectral responses** | 22.2 MB (XLSX, CSV) | Tabular. Firmas espectrales y resultados de laboratorio qPCR (Xylella). | [Dataset Zenodo](https://zenodo.org/records/5535095) |
| **Fluorescence Spectra of Olive Oils** | 7.0 MB (CSV) | Tabular. Parámetros químicos y fluorescencia de aceites. | [Dataset Mendeley](https://data.mendeley.com/datasets/thkcz3h6n6/6) |
| **Precio del aceite durante 30 años** | 42.8 KB (CSV) | Tabular. Series temporales de mercados financieros. | [Dataset Kaggle](https://doi.org/10.34740/kaggle/dsv/4182176) |
| **Avocado Prices and Sales Volume** | 5.29 MB (CSV) | Tabular. Histórico de ventas (2015-2023). | [Dataset Kaggle](https://www.kaggle.com/datasets/vakhariapujan/avocado-prices-and-sales-volume-2015-2023) |
| **Avocado Ripeness Classification** | 10.69 KB (CSV) | Tabular. Características descriptivas numéricas. | [Dataset Kaggle](https://www.kaggle.com/datasets/amldvvs/avocado-ripeness-classification-dataset) |
| **Dataset from field trials on almond** | 7.8 MB (XLSX, CSV) | Tabular. Ensayos de campo y producción agraria (IRTA). | [Dataset Zenodo](https://doi.org/10.5281/zenodo.18099928) |
| **Locomotor activity pattern (Olive fly)** | 3.5 MB (XLSX) | Tabular. Series de tiempo de actividad biológica de insectos. | [Dataset Zenodo](https://zenodo.org/records/7221901) |
| **Precios Medios Anuales Nacionales** | 1 MB (CSV) | Tabular. Estadísticas económicas nacionales. | [Dataset MAPA](https://www.mapa.gob.es/es/estadistica/temas/estadisticas-agrarias/economia/precios-medios-nacionales) |
| **In-field hyperspectral imaging** | 10.5 GB (.HDR, .RAW) | Sensor Hiperespectral. Formato propietario de reflectancia óptica. | [Dataset Mendeley](https://data.mendeley.com/datasets/8xvhcsdvst/2) |
| **Olive fruit X-ray microtomography** | 9.3 GB (JPG Grises) | Escáner Rayos-X. Cortes axiales microtomográficos, no fotos ópticas. | [Dataset Mendeley](https://data.mendeley.com/datasets/49y4zjx9tj/2) |
| **Avocado tree point clouds** | 2.3 GB (.BIN) | Nube de Puntos (LiDAR). Mapeo 3D espacial. | [Dataset Mendeley](https://data.mendeley.com/datasets/h49fpprg6c/1) |
| **Macrobot Barley/Wheat Multispectral** | 441.2 MB (TIFF, Metas) | Sensor Multiespectral. Análisis bajo luz ultravioleta/láser. | [Dataset Zenodo](https://zenodo.org/records/13734021) |
| **ICAERUS Olive Tree Multispectral** | 3.1 GB (Multiespectral) | Sensor Satelital. Matrices de bandas espectrales complejas. | [Dataset Zenodo](https://zenodo.org/records/13121962) |
| **Avocado/Olive/Vine Multiespectral** | 1.2 GB (TIFF) | Sensor Multiespectral. Combina canales RGB, Red Edge y NIR. | [Dataset Figshare](https://doi.org/10.6084/m9.figshare.26950660) |