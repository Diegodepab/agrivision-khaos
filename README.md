# AgriVision Khaos
==============================

**AgriVision Khaos** es un nodo de curación y preprocesamiento de datos orientado al *Data-Centric AI* y al Aprendizaje Federado en el ámbito agrícola.

Este repositorio proporciona una infraestructura dockerizada y estandarizada para limpiar, deduplicar y enriquecer datasets masivos de imágenes (como patologías foliares) antes de alimentar cualquier red neuronal.

## 🚀 Inicio Rápido

Todo el ecosistema está encapsulado en Docker y se controla a través de nuestro `Makefile`.
El runtime usa Python 3.13, dentro de la matriz soportada oficialmente por FiftyOne.

La imagen usa PyTorch para CPU por defecto, evitando descargar librerías CUDA innecesarias.
Si el host dispone de una GPU NVIDIA compatible con CUDA 13.0, construye con
`make gpu-build` y usa `docker-compose.gpu.yml` como override. Para una instalación local usa
`uv sync --extra cpu` o `uv sync --extra cu130`; ambos perfiles son excluyentes.

1. Clona el repositorio y copia el entorno: `cp .env.example .env`.
2. Coloca cada fuente en una subcarpeta distinta de `data/raw/` y añade su
   `source.yaml` cuando necesites declarar licencia, dominio o versión.
3. Ejecuta `make preflight` y después `make dry-run`.
4. Descarga el modelo activo con `make models` y lanza `make pipeline`.
5. Revisa `reports/pipeline/`, resuelve los casos `review` en la interfaz con
   `make app` y publica la selección con `make export`.

### Validación rápida sin datasets pesados

Genera tres fuentes sintéticas reproducibles (COCO, YOLO y VOC) y audítalas sin
arrancar MongoDB ni ejecutar modelos:

```bash
make mock MOCK_DIR=data/mock
make dry-run RAW_DIR=data/mock DATASET=mock-validation
```

Para crear además casos deliberadamente inválidos ejecuta directamente
`agrivision-mock --output-dir data/mock-edge --include-edge-cases`. El generador
se niega a escribir en directorios no vacíos.

Antes de una ejecución real, valida los montajes, la lectura de imágenes, el
espacio libre y MongoDB con `make preflight`. Para probar toda la arquitectura
sin GPU usa `POLICY=configs/cpu-smoke.yaml`; este perfil conserva deduplicación
exacta y por transformaciones, pero omite OCR y embeddings semánticos.

Antes del perfil completo ejecuta `make models` una vez. Descarga el modelo de
embeddings activo (MobileNetV2 en CPU o ResNet50 con GPU) dentro de
`CACHE_DATA_HOST_PATH`, evitando otra descarga si el contenedor se recrea.

Los datasets pueden residir en otra máquina. Móntalos mediante NFS (preferido) o
SSHFS en el host de cálculo, configura `RAW_DATA_HOST_PATH` y mantenlos en modo
solo lectura. Consulta la [guía de ejecución remota](docs/remote_execution.md).

## 📚 Documentación

La documentación completa (Arquitectura, Flujo de Trabajo en Fases, y Guías de Calidad) está generada con Zensical.

Para levantar el servidor de documentación localmente:
```bash
make docs
```
Luego visita [http://localhost:8080](http://localhost:8080) en tu navegador.

Cada fuente puede declarar versión, licencia, cita y contexto de captura mediante
un [`source.yaml`](docs/source_manifest.md) situado en la raíz de su carpeta.

El resultado es un catálogo unificado y trazable, no la promesa de que tareas
incompatibles formen un único conjunto de entrenamiento. Cada muestra conserva
su `task_type` y los exportadores generan subconjuntos por tarea. La curación
puede producir menos imágenes que la suma de las fuentes al excluir corrupción,
duplicados y casos pendientes; su objetivo es elevar el valor auditable, no el
recuento bruto. La corrección semántica de las etiquetas y el ajuste de umbrales
al dominio deben validarse con una revisión humana y un piloto real.

## 🏗️ Estructura del Código

El código fuente (`src/`) sigue las fases de madurez de un proyecto de Inteligencia Artificial Centrada en Datos:

- `agrivision_khaos/ingest.py`: descubrimiento e ingesta de datos brutos.
- `agrivision_khaos/quality.py`: métricas visuales (blur, OCR, iluminación y smearing).
- `agrivision_khaos/deduplication.py`: detección de duplicados exactos y visuales.
- `agrivision_khaos/pipeline.py`: orquestación, decisiones, informes y exportación.

## ⚖️ Licencia

Este proyecto está bajo la licencia [Apache 2.0](LICENSE).
