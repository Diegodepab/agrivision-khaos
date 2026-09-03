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

### 1.2 Detección de Fruto y Fenología
| Nombre del Dataset | Tamaño / Formato | Origen y Licencia | Enlaces | Observaciones |
| :--- | :--- | :--- | :--- | :--- |
| **Olive fruit object detection** | 224.9 MB (927 JPG + TXT) | Mundial <br> CC0 | [Dataset Kaggle](https://www.kaggle.com/datasets/danielvalyano/olive-fruit-object-detection) | Bounding boxes para detección de aceitunas. |
| **Olive Tree Varieties Drought Stress** | 8.63 GB (2,880 JPG + 1 XLSX) | Marruecos <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/22j26tpk63/1) <br> [Paper](https://www.sciencedirect.com/science/article/pii/S2352340925007486) | Vistas frontales y laterales para análisis de crecimiento y estrés hídrico. |

### 1.3 Teledetección y Vista Aérea (RGB)
| Nombre del Dataset | Tamaño / Formato | Origen y Licencia | Enlaces | Observaciones |
| :--- | :--- | :--- | :--- | :--- |
| **Burned and unburned olive trees UAV** | 19.4 MB (3,624 JPG) | Grecia <br> CC BY-NC-ND 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/83kpndkrb2/1) <br> [Paper](https://www.mdpi.com/2072-4292/16/23/4531) | Capturas de drones. Zonas secas/incendiadas vs sanas. |
| **OliveTreeCrownsDb** | 2 GB (46 JPG + Extras) | Marruecos <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/xym8rd2srf/3) | Ortofotos y mosaicos para detección de coronas de árboles. |

---

## 2. Dominio Almendros y Frutos Secos - Imágenes Ópticas

### 2.1 Detección, Variedades y Calidad del Fruto
| Nombre del Dataset | Tamaño / Formato | Origen y Licencia | Enlaces | Observaciones |
| :--- | :--- | :--- | :--- | :--- |
| **Almond varieties classification** | 158.0 MB (+1,500 JPG) | Turquía <br> Unknown | [GitHub](https://github.com/ymyurdakul/datasets/tree/main) <br> [Paper](https://link.springer.com/article/10.1007/s00217-024-04562-4#data-availability) | Clasificación de especies de almendra. |
| **Almendras, pistachos, avellanas** | 4.9 MB (279 PNG) | Roboflow <br> CC BY 4.0 | [Dataset Universe](https://universe.roboflow.com/frutos-secos/almendras-pistachos-avellanas-hmxel/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true) | Distinción multiclase de frutos secos. |
| **NutsBD** | 48.5 MB (1,752 JPG) | Bangladesh <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/fmrnybgxb9/1) | Distinción general entre frutos secos. |
| **AllerNuts** | 1.5 GB (4,390 JPG) | Bangladesh <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/3gtbxvgm5f/1) | Imágenes de frutos secos con o sin fondo. |
| **Almond Damage Detection** | 27.1 MB (739 JPG) | Kaggle <br> Apache 2.0 | [Dataset Kaggle](https://www.kaggle.com/datasets/mahyeks/almond-damage-detection) | Etiquetado binario de almendra dañada vs intacta. |

### 2.2 Enfermedades Foliares
| Nombre del Dataset | Tamaño / Formato | Origen y Licencia | Enlaces | Observaciones |
| :--- | :--- | :--- | :--- | :--- |
| **Almond Diseases Detection.v2i** | 379.9 MB (4,500 JPG) | Roboflow <br> Unknown | [Dataset Universe](https://universe.roboflow.com/poker-chips-annotation/almond-diseases-detection-l7tfx/dataset/2) | Plantas de almendro con patologías visuales. |

---

## 3. Dominio Frutas Tropicales - Imágenes Ópticas

### 3.1 Enfermedades y Clasificación de Hojas
| Nombre del Dataset | Cultivo | Tamaño / Formato | Origen y Licencia | Enlaces |
| :--- | :--- | :--- | :--- | :--- |
| **Avocado Leaf DataSet K-Kotagiri** | Aguacate | 2.1 GB (435 JPG) | India <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/6zy6wxhf2v/1) |
| **MLD24: Mango Leaf Diseases** | Mango | 136.8 MB (6,400 JPG) | Bangladesh <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/6dvpywm2m2/1) |
| **SAR-MLD1-2025: High Quality Mango** | Mango | 13.27 GB (4,000 JPG) | Bangladesh <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/sd8hzpg69b/5) |
| **Papaya Leaf Disease Image Dataset** | Papaya | 6.2 GB (3,626 PNG) | Bangladesh <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/3kwgxg4stb/1) |
| **Dataset of Common Papaya Diseases** | Papaya | 783.1 MB (442 JPG) | Bangladesh <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/42xmy5j64g/1) |
| **BDPapayaLeaf** | Papaya | 486.9 MB (2,164 JPG) | Bangladesh <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/p997fvf526/1) |
| **Chirimoya Diseases** | Chirimoya | 1 GB (8,226 JPG) | India <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/jtgh2885yf/2) |
| **Papaya Diseases Dataset** | Papaya/Chir. | 6.29 GB (983 JPG) | India <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/yvcwypp8rz/1) |

### 3.2 Fruto y Madurez
| Nombre del Dataset | Cultivo | Tamaño / Formato | Origen y Licencia | Enlaces |
| :--- | :--- | :--- | :--- | :--- |
| **Avocado Computer Vision Dataset** | Aguacate | 170.5 MB (1,072 JPG) | EE.UU. <br> CC BY 4.0 | [Dataset Universe](https://universe.roboflow.com/uc-riverside-ov9yb/avocado-jvq5e) |
| **MangoImageBD** | Mango | 1.2 GB (28,515 JPG) | Bangladesh <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/hp2cdckpdr/2) |
| **MangoDHDS (Fruto)** | Mango | 36 MB (1,500 JPG) | India <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/b4nrw5hyyc/1) |
| **Papaya_Madurez** | Papaya | 26.0 MB (485 JPG) | India <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/rcy6y8fhcn/1) |

### 3.3 Teledetección (RGB)
| Nombre del Dataset | Cultivo | Tamaño / Formato | Origen y Licencia | Enlaces |
| :--- | :--- | :--- | :--- | :--- |
| **Avo-AirDB** | Aguacate | 8.05 GB (984 JPG) | Marruecos <br> CC BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/tvhh83r3hj/3) |

---

## 4. Dominio Cereales - Imágenes Ópticas

| Nombre del Dataset | Cultivo / Tarea | Tamaño / Formato | Origen y Licencia | Enlaces |
| :--- | :--- | :--- | :--- | :--- |
| **Global Wheat Head Dataset 2021** | Trigo (Teledetección) | 9.5 GB (6,515 PNG) | Internacional <br> CC-BY 4.0 | [Dataset Zenodo](https://zenodo.org/records/5092309) |
| **Wheat Plant Diseases High Res** | Trigo (Enfermedad) | 7 GB (12,000 PNG) | Kaggle <br> CC-BY 4.0 | [Dataset Kaggle](https://www.kaggle.com/datasets/kushagra3204/wheat-plant-diseases) |
| **Wild Oats Detection in Wheat Fields** | Malezas en Trigo | 48 MB (500 JPEG) | Egipto <br> Open Access | [Dataset Kaggle](https://www.kaggle.com/datasets/abanoublamie/wild-oats) |
| **Rice Leaf Disease and Pest Augmented**| Arroz (Enfermedad) | 5.7 GB (19,128 JPEG)| Bangladesh <br> CC-BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/vwv3nry3wr/1) |
| **Field-Grown Barley Disease Multiclass**| Cebada (Enfermedad) | 4.2 GB (PNG/TIFF) | Alemania <br> CC-BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/4ny92p2r8f) |
| **Maize Crop Disease Leaf** | Maíz (Enfermedad) | 6.5 GB (30,000 JPG) | UAE <br> CC-BY 4.0 | [Dataset Mendeley](https://data.mendeley.com/datasets/6w6gsvghfw/1) |

---

## 5. Datasets Multiclase Genéricos

| Nombre del Dataset | Tamaño / Formato | Origen y Licencia | Enlaces | Observaciones |
| :--- | :--- | :--- | :--- | :--- |
| **Multiclase Frutas y Vegetales** | 1 GB (20,216 IMG) | Univ. León, España <br> CC BY 4.0 | [Dataset Zenodo](https://zenodo.org/records/18904638) | ~222,300 anotaciones YOLO. |

---

## 6. ⚠️ Datasets Aislados (No Unificables en CV Clásico)

> **Nota Arquitectónica:** Los siguientes datasets contienen formatos tubulares (`CSV`, `XLSX`), sensores especializados (Rayos X, LiDAR, `.HDR` Hiperespectral) o combinaciones multiespectrales complejas (`TIFF` apilados). **NO deben mezclarse** en los pipelines de imágenes RGB estándar. Quedan registrados aquí como repositorio pasivo para futuras arquitecturas multimodales (Late Fusion) o análisis de datos estructurados.

| Nombre del Dataset | Formato / Tipo de Dato | Justificación de Aislamiento | Enlaces |
| :--- | :--- | :--- | :--- |
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