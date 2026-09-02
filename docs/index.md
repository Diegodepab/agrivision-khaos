# AgriVision Khaos (Andalucía ISI2A2)

Bienvenido a la documentación técnica del pipeline de curación de datos federado **AgriVision Khaos**, desarrollado en el marco del proyecto **Andalucía ISI2A2** (Infraestructura y Servicios de Integración e Inteligencia de Datos en el sector de la Agroalimentación en Andalucía).

Este framework de *Data-Centric AI* actúa como el motor fundacional para ingerir, armonizar y curar datos agrícolas heterogéneos, garantizando que tus modelos de Machine Learning se entrenen sobre conjuntos de datos de altísimo valor y Ground Truth impecable.

---

## 🚀 Quick Start (Para Managers y Científicos de Datos)

Si necesitas generar un dataset limpio de inmediato sin entrar en detalles técnicos, sigue este flujo de 3 pasos (Clone-and-Run):

### 1. Prepara tus Datos Crudos
Clona este repositorio y coloca todas las carpetas con tus datasets (en formato COCO, YOLO, o carpetas por clase) dentro de `data/raw/`.

```bash
git clone https://github.com/tu-organizacion/agrivision-khaos.git
cd agrivision-khaos
# Arrastra tus carpetas a data/raw/
```

### 2. Ejecuta el Pipeline Automático
Lanza el siguiente comando mágico, indicando el nombre que le quieres dar a tu dataset de salida. Ve a tomarte un café; el sistema ingiriendo, calculando métricas de calidad (Otsu-Blur), buscando duplicados semánticos y generando reportes.

```bash
make pipeline DATASET="mi_super_dataset" RAW_DIR="data/raw"
```
*Al terminar, obtendrás un reporte HTML Premium (estilo SaaS) auditando cada foto eliminada y conservada en `reports/pipeline/`.*

### 3. Revisión Humana y Exportación Final (Opcional)
Si quieres auditar visualmente las imágenes que el pipeline marcó como dudosas (las que tienen *padding* artificial o están en el límite de la calidad):
```bash
# Abre la interfaz visual
make app DATASET="mi_super_dataset"
```
En tu navegador (`http://localhost:5152`), busca el tag `review`, acepta o rechaza las imágenes haciendo clic, y cuando termines tu curación visual, exporta el resultado final:
```bash
# Genera los archivos (COCO, YOLO) finales validados
make export DATASET="mi_super_dataset"
```
¡Y listo! Tus datos curados estarán esperándote en `data/processed/mi_super_dataset_hitl`.

---

## Navegación Profunda

Si quieres entender cómo funciona el cerebro de AgriVision Khaos, usa el menú lateral para explorar:
- **Architecture**: Principios de diseño, base de datos no relacional (FiftyOne) y trazabilidad del linaje del dato.
- **Workflow**: Documentación técnica detallada de cada fase (Ingesta, Métricas de Calidad, Deduplicación, y Revisión Manual).
