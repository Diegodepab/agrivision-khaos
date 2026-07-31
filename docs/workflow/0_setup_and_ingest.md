# Fase 0: Inicialización e Ingesta de Datos

Esta guía explica el flujo cronológico paso a paso desde que clonas el repositorio por primera vez, hasta que tienes el entorno levantado con todas las imágenes cargadas en la base de datos, listo para trabajar.

Todo el proceso está diseñado para ser universal y ejecutarse a través del comando `make`, garantizando que funcionará en cualquier máquina sin problemas de dependencias.

---

## 1. Clonar y Preparar el Entorno

Lo primero que debes hacer al llegar al proyecto es descargar el código y preparar el entorno de variables.

1. Clona el repositorio en tu máquina local.
2. Copia la plantilla de variables de entorno para que Docker sepa qué puertos utilizar:
   ```bash
   cp .env.example .env
   ```

## 2. Incorporar los Datasets Crudos

Antes de iniciar la base de datos, el sistema necesita tener acceso a las imágenes originales que vamos a procesar.

Coloca las carpetas con tus imágenes descargadas directamente en la ruta `data/raw/` de este repositorio.

<details>
<summary>👀 ¿Cómo sabe el sistema qué formato tienen mis datos?</summary>

AgriVision Khaos utiliza un motor de **Auto-descubrimiento Heurístico**:
- El sistema leerá cualquier subcarpeta que dejes en `data/raw/`.
- Si detecta archivos `.json` dentro (estándar de COCO), lo ingestará automáticamente como `COCODetectionDataset` respetando los splits (train/val/test).
- Si solo encuentra imágenes estructuradas en carpetas, lo ingestará como `ImageClassificationDirectoryTree`.
- ¡No tienes que configurar absolutamente nada!
</details>

## 3. Despliegue e Ingesta Automática

Con los datos en su sitio, es el momento de construir la infraestructura y volcar las imágenes en el sistema. Todo esto se hace con un único comando:

```bash
make setup
```

*(Opcional: Si quieres ponerle un nombre específico a tu dataset en la base de datos, puedes usar `make setup DATASET="mi_proyecto_secreto"`. Por defecto usará `agrivision-dataset`).*

<details>
<summary>⚙️ ¿Qué ocurre internamente al ejecutar este comando?</summary>

Para mantener tu máquina limpia, este comando automatiza tres cosas:
1. **Levanta la Base de Datos**: Despliega un contenedor oficial de MongoDB aislado (`mongo:6.0`) en segundo plano.
2. **Ejecuta el Script de Ingesta**: Lanza un contenedor efímero de Python que ejecuta `src/01_acquisition/make_dataset.py`.
3. **Persistencia**: Extrae todas las imágenes de `data/raw/`, descubre su formato, y las registra formalmente en el volumen de Docker bajo el dataset que le hayas indicado.
</details>

## 4. Verificar el Resultado

Una vez que el comando anterior finalice (verás una barra de progreso llegando al 100%), comprueba que todo está montado correctamente abriendo la interfaz visual:

```bash
make app
```
Dirígete a [http://localhost:5152](http://localhost:5152) en tu navegador. Deberías ver toda tu galería de imágenes de hojas de olivo cargada y lista.

> **¡Enhorabuena!** Tu repositorio está montado. Ya puedes pasar a la **Fase 1: Preprocesamiento**.
