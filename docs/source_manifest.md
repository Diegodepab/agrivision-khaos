# Manifiesto de procedencia

Cada carpeta situada directamente bajo `data/raw/` puede incluir un archivo
`source.yaml`, `source.yml` o `source.json`. Este archivo no se modifica durante
la ingesta y acompaña a todas las muestras derivadas de esa fuente.

```yaml
schema_version: "1.0"
name: olive-leaf-image-dataset
version: "2024-01"
license: CC0-1.0
homepage: https://example.org/datasets/olive-leaves
citation: "Autor et al. (2024), Olive Leaf Image Dataset"
acquired_at: 2026-09-04
tasks:
  - classification
sensor: RGB
geography: Turquía
```

Si no existe manifiesto, la fuente se ingiere con licencia y versión
`unknown`. Un manifiesto presente pero inválido detiene el pipeline para evitar
publicar procedencia incorrecta.
