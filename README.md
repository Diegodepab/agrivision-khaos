# AgriVision Khaos
==============================

**AgriVision Khaos** es un nodo de curación y preprocesamiento de datos orientado al *Data-Centric AI* y al Aprendizaje Federado en el ámbito agrícola.

Este repositorio proporciona una infraestructura dockerizada y estandarizada para limpiar, deduplicar y enriquecer datasets masivos de imágenes (como patologías foliares) antes de alimentar cualquier red neuronal.

## 🚀 Inicio Rápido

Todo el ecosistema está encapsulado en Docker y se controla a través de nuestro `Makefile`.

1. Clona el repositorio.
2. Copia el archivo de entorno: `cp .env.example .env`
3. Coloca tus datasets en `data/raw/`.
4. Ejecuta la ingesta automática:
   ```bash
   make setup
   ```
5. Accede a la interfaz visual en [http://localhost:5151](http://localhost:5151).

## 📚 Documentación

La documentación completa (Arquitectura, Flujo de Trabajo en Fases, y Guías de Calidad) está generada con Zensical.

Para levantar el servidor de documentación localmente:
```bash
make docs
```
Luego visita [http://localhost:8080](http://localhost:8080) en tu navegador.

## 🏗️ Estructura del Código

El código fuente (`src/`) sigue las fases de madurez de un proyecto de Inteligencia Artificial Centrada en Datos:

- `01_acquisition/`: Extracción e ingesta de datos brutos.
- `02_curation/`: Auditoría de calidad (Blur, OCR, Smearing) y deduplicación semántica (MobileNetV2).
- `03_annotation/`: 
- `04_augmentation/`: 
- `05_modeling/`: 
- `06_deployment/`: 

## ⚖️ Licencia

Este proyecto está bajo la licencia [Apache 2.0](LICENSE).
