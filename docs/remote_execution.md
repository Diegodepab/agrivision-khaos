# Ejecución con datos y GPU en máquinas distintas

El proceso debe ejecutarse en la máquina que contiene la GPU y el repositorio.
Los datos de origen pueden permanecer en otro equipo de la misma red: primero se
montan en el sistema de archivos del host GPU y Docker los recibe como
`/datasets/raw:ro`. El pipeline nunca necesita escribir en el origen.

> Sustituye las direcciones y rutas de los ejemplos. Las direcciones IPv4
> privadas habituales pertenecen a `192.168.0.0/16`; comprueba especialmente el
> orden de los octetos si una dirección empieza por `216`.

## Opción recomendada: NFS

En el servidor de datos exporta únicamente la carpeta necesaria y limita el
acceso a la IP del host GPU. En el host GPU:

```bash
sudo mkdir -p /mnt/agrivision-raw
sudo mount -t nfs -o ro,nosuid,nodev 192.168.1.67:/srv/datasets /mnt/agrivision-raw
```

NFS suele ofrecer mejor rendimiento y menos coste de CPU que SSHFS para miles de
archivos pequeños. Si no puedes habilitar NFS, una alternativa cifrada es:

```bash
mkdir -p /mnt/agrivision-raw
sshfs -o ro,reconnect,ServerAliveInterval=15 usuario@192.168.1.67:/srv/datasets /mnt/agrivision-raw
```

No ejecutes el pipeline hasta poder enumerar y leer imágenes desde el host GPU.
Para trabajos largos conviene declarar el montaje en `fstab` o como unidad de
systemd, de forma que una reconexión no cambie silenciosamente la ruta.

## Configuración del repositorio en el host GPU

Copia `.env.example` a `.env` y ajusta rutas que pertenecen **al host que ejecuta
Docker**:

```dotenv
RAW_DATA_HOST_PATH=/mnt/agrivision-raw
PROCESSED_DATA_HOST_PATH=/srv/agrivision/processed
CACHE_DATA_HOST_PATH=/srv/agrivision/cache
REPORTS_HOST_PATH=/srv/agrivision/reports
FIFTYONE_BIND_ADDRESS=127.0.0.1
TORCH_EXTRA=cpu
```

Mientras no esté preparada la GPU deja `TORCH_EXTRA=cpu` y ejecuta:

```bash
docker compose build fiftyone
make preflight
make pipeline DATASET=prueba POLICY=configs/cpu-smoke.yaml
```

Antes de ejecutar el perfil completo con deduplicación semántica, descarga una
vez el modelo mediante `make models`. Su caché queda en el volumen configurado y
una fase habilitada que no pueda cargar su modelo cancelará la publicación.

Cuando conozcas el modelo de GPU, confirma primero que el driver soporta la
versión CUDA del perfil y que NVIDIA Container Toolkit está instalado. Después:

```bash
make gpu-build
COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml make preflight REQUIRE_GPU=1
COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml make models
COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml make pipeline DATASET=produccion REQUIRE_GPU=1
```

El override solicita las GPU al runtime; no se carga en modo CPU.
`REQUIRE_GPU=1` hace que la falta de CUDA bloquee tanto el preflight como el
pipeline, en vez de continuar accidentalmente por CPU.

`make preflight` exige que el origen sea de solo lectura, decodifica una muestra,
mide el rendimiento de lectura, comprueba almacenamiento escribible, espacio
libre y MongoDB. La ausencia de GPU no es un error salvo que se invoque
`agrivision-preflight --require-gpu`.

## Reanudación y resultados

El pipeline toma un bloqueo por dataset para impedir dos escritores simultáneos.
Los checkpoints viven en `/datasets/cache/runs/<dataset>/checkpoint.json` y se
reutilizan solo si coinciden los metadatos de las fuentes y la política. Tras una
interrupción, repite el mismo comando; `--no-resume` fuerza una ejecución nueva.

La exportación se construye en un directorio hermano temporal y se publica con
un renombrado atómico. Un resultado solo está completo si contiene `_SUCCESS`.
No entrenes desde directorios `.incomplete-*`; se conservan para diagnóstico si
una exportación falla.

## Acceso remoto seguro

Los puertos de FiftyOne y de la documentación escuchan en `127.0.0.1` por
defecto. Desde tu equipo abre un túnel SSH hacia el host GPU:

```bash
ssh -L 5151:127.0.0.1:5151 usuario@192.168.1.42
```

Después visita `http://127.0.0.1:5151`. Evita publicar MongoDB o FiftyOne en toda
la LAN; si necesitas cambiar el bind, añade autenticación y reglas de firewall.
