---
icon: lucide/workflow
title: Arquitectura
---

# Project Architecture

## Overview
El proyecto `AgriVision Khaos` es un nodo federado para la auditoría, deduplicación y filtrado de calidad de imágenes agrícolas. Está diseñado para ingestar cientos de miles de imágenes de cualquier fuente (COCO, carpetas de clasificación) y utilizar algoritmos de visión por computador y modelos OCR para separar el ruido (baja resolución, borrosidad, marcas de agua) de los datos válidos, asegurando que los modelos federados se entrenen con datos de alta calidad.

## Components Diagram

```mermaid
C4Container
    title Diagrama de Arquitectura: AgriVision Khaos
    
    Container(ui, "FiftyOne App", "React/Python", "Interfaz de exploración visual")
    Container(metrics, "Quality Metrics Engine", "Python (ThreadPool)", "Calcula descriptores matemáticos de las imágenes")
    ContainerDb(db, "Database", "MongoDB", "Contenedor Oficial Aislado (mongo:6.0)")
    
    Container(ocr, "Tesseract OCR", "C++", "Detección de marcas de agua")
    Container(cv2, "OpenCV & NumPy", "C++", "Cálculo de Blur Laplaciano, HSV y smearing")

    Rel(metrics, cv2, "Extrae matrices", "Memory")
    Rel(metrics, ocr, "Envía tensor para inferencia", "Memory")
    Rel(metrics, db, "Actualiza schema en bloques", "BSON/TCP")
    Rel(ui, db, "Consulta imágenes", "BSON/TCP")
```

## Technologies
- **Backend Core**: Python 3.14, gestionado con `uv`.
- **Base de Datos**: MongoDB (Standalone Container).
- **Visión Artificial**: OpenCV, NumPy, PIL.
- **Modelos OCR**: Tesseract OCR (vía `pytesseract`).
- **Data Curation Framework**: Voxel51 (FiftyOne).

## Design Decisions
- **Doble Lectura vs Copia en RAM**: Se prioriza una lectura I/O de PIL (validación de bytes) seguida de `cv2.imread()` para evitar la copia masiva del tensor RGB completo en RAM.
- **Multithreading vs Multiprocessing**: Se emplea `ThreadPoolExecutor` con OCR residente en memoria para evitar colapsar la RAM de la máquina host con múltiples instancias del modelo Tesseract.
- **Batched DB Updates**: En lugar de consultar `dataset[id]` por cada iteración (lo que causa timeouts en MongoDB), las métricas se agrupan en lotes de 10.000 imágenes y se escriben de una sola vez con `dataset.set_values()`.
- **Despliegue Docker**: El ecosistema (FiftyOne y Scripts) se ha dockerizado para independizarse de los problemas de compilación de OpenCV y Tesseract en la máquina host.
