# AgriVision Khaos (Andalucía ISI2A2)

Bienvenido a la documentación técnica del pipeline de curación de datos federado **AgriVision Khaos**, desarrollado en el marco del proyecto **Andalucía ISI2A2** (Infraestructura y Servicios de Integración e Inteligencia de Datos en el sector de la Agroalimentación en Andalucía).

Este framework de *Data-Centric AI* actúa como el motor fundacional para ingerir, armonizar y curar datos agrícolas heterogéneos (imágenes, sensores, satélites). Su objetivo es garantizar que los modelos de Machine Learning y los ecosistemas de IA se entrenen sobre conjuntos de datos de altísimo valor y Ground Truth impecable.

## Alineación Arquitectónica (Pilares del Proyecto)

AgriVision Khaos está diseñado siguiendo principios arquitectónicos modernos para integrarse como un *Living-Lab* de datos abiertos:

1. **Metadatos Centralizados, Archivos Desacoplados:**
   Utilizamos un motor de base de datos no relacional (FiftyOne/MongoDB) que actúa como cerebro semántico. Las imágenes crudas y los binarios pesados residen en el sistema de archivos (usando referencias o *hard links*), mientras que la plataforma centraliza las anotaciones, métricas y metadatos de forma ligera y consultable.

2. **Trazabilidad Estricta (Linaje del Dato):**
   Cada dato que fluye por Khaos mantiene un registro inmutable de su origen. El sistema documenta el linaje completo: desde qué dataset externo provino (`source_dataset`), qué algoritmos evaluaron su calidad, hasta el motivo exacto por el cual fue conservado o descartado (reportes HTML de auditoría).

3. **Diseño Modular por Fases (ETL Agrícola):**
   - **Extract (Ingesta):** Recolección y unificación de múltiples fuentes de datos inconexas (COCO, YOLO, estructuras de carpetas).
   - **Transform (Curación):** Limpieza automatizada mediante IA (desenfoque, artefactos, marcas de agua) y deduplicación avanzada (exacta, semántica y aumentada).
   - **Load (Exportación):** Generación de *manifests* unificados y exportación a formatos estándar listos para inyectarse en modelos de optimización o LLMs.

## Filosofía de Limpieza de Datos

El framework resuelve de manera automatizada problemas endémicos en los datasets agrícolas públicos:
- Resolución Insuficiente y Ruido
- Fuga de Datos (Duplicados semánticos y rotaciones espejo)
- Marcas de Agua y Artefactos de Estiramiento

## Navegación

Usa el menú lateral para explorar la documentación:
- **Architecture**: Descubre cómo se orquesta la infraestructura.
- **Workflow**: Sigue nuestras guías paso a paso (Fase 0, Fase 1...) para clonar el repositorio y procesar tus primeros datos en minutos.
