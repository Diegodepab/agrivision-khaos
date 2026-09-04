from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pymongo import MongoClient

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    message: str
    details: dict[str, Any] | None = None


def _is_read_only(path: Path) -> bool:
    return bool(os.statvfs(path).f_flag & os.ST_RDONLY)


def _writable_directory_check(name: str, path: Path, minimum_free_gb: float) -> Check:
    try:
        path.mkdir(parents=True, exist_ok=True)
        descriptor, probe_name = tempfile.mkstemp(prefix=".agrivision-write-probe-", dir=path)
        os.close(descriptor)
        Path(probe_name).unlink()
        free_gb = shutil.disk_usage(path).free / (1024**3)
        status = "pass" if free_gb >= minimum_free_gb else "warning"
        return Check(
            name,
            status,
            f"Escritura correcta; {free_gb:.2f} GiB libres",
            {"path": str(path.resolve()), "free_gib": round(free_gb, 3)},
        )
    except OSError as exc:
        return Check(name, "fail", f"No se puede escribir en {path}: {exc}")


def _raw_storage_check(
    raw_dir: Path,
    require_read_only: bool,
    sample_files: int,
    sample_megabytes: float,
) -> Check:
    try:
        if not raw_dir.is_dir():
            return Check("raw_storage", "fail", f"No existe el directorio: {raw_dir}")
        read_only = _is_read_only(raw_dir)
    except OSError as exc:
        return Check("raw_storage", "fail", f"No se puede acceder a {raw_dir}: {exc}")
    if require_read_only and not read_only:
        return Check(
            "raw_storage",
            "fail",
            "La fuente es escribible; se exigió un montaje de solo lectura",
            {"path": str(raw_dir.resolve()), "read_only": False},
        )

    candidates = (
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    selected: list[Path] = []
    byte_limit = max(1, int(sample_megabytes * 1024 * 1024))
    selected_bytes = 0
    try:
        for path in candidates:
            selected.append(path)
            selected_bytes += path.stat().st_size
            if len(selected) >= sample_files or selected_bytes >= byte_limit:
                break
    except OSError as exc:
        return Check("raw_storage", "fail", f"Error enumerando el montaje {raw_dir}: {exc}")
    if not selected:
        return Check(
            "raw_storage",
            "fail",
            "No se encontraron imágenes compatibles",
            {"path": str(raw_dir.resolve()), "read_only": read_only},
        )

    started = time.perf_counter()
    bytes_read = 0
    corrupt = 0
    try:
        for path in selected:
            encoded = path.read_bytes()
            bytes_read += len(encoded)
            if cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED) is None:
                corrupt += 1
    except OSError as exc:
        return Check("raw_storage", "fail", f"Error leyendo el montaje {raw_dir}: {exc}")
    elapsed = max(time.perf_counter() - started, 1e-9)
    throughput = bytes_read / elapsed / (1024**2)
    status = "warning" if corrupt else "pass"
    return Check(
        "raw_storage",
        status,
        f"{len(selected)} imágenes leídas a {throughput:.1f} MiB/s; corruptas={corrupt}",
        {
            "path": str(raw_dir.resolve()),
            "read_only": read_only,
            "sample_files": len(selected),
            "sample_bytes": bytes_read,
            "throughput_mib_s": round(throughput, 3),
            "corrupt_images": corrupt,
        },
    )


def _database_check(uri: str | None) -> Check:
    if not uri:
        return Check("database", "skipped", "No se proporcionó URI de MongoDB")
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3000, connectTimeoutMS=3000)
        response = client.admin.command("ping")
        client.close()
        return Check("database", "pass", "MongoDB respondió correctamente", response)
    except Exception as exc:
        return Check("database", "fail", f"MongoDB no está disponible: {exc}")


def _gpu_check(require_gpu: bool) -> Check:
    gpu_device_present = any(Path("/dev").glob("nvidia[0-9]*"))
    if not require_gpu and shutil.which("nvidia-smi") is None and not gpu_device_present:
        return Check("gpu", "skipped", "No se detectó hardware NVIDIA; modo CPU válido")
    try:
        import torch
    except ImportError:
        status = "fail" if require_gpu else "skipped"
        return Check("gpu", status, "PyTorch no está instalado")
    if not torch.cuda.is_available():
        status = "fail" if require_gpu else "warning"
        return Check(
            "gpu",
            status,
            "CUDA no disponible; el pipeline puede ejecutarse en CPU",
            {"torch": torch.__version__, "cuda_build": torch.version.cuda},
        )
    devices = [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "memory_gib": round(
                torch.cuda.get_device_properties(index).total_memory / (1024**3), 2
            ),
        }
        for index in range(torch.cuda.device_count())
    ]
    return Check(
        "gpu",
        "pass",
        f"CUDA disponible en {len(devices)} dispositivo(s)",
        {"torch": torch.__version__, "cuda_build": torch.version.cuda, "devices": devices},
    )


def run_preflight(
    raw_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    database_uri: str | None = None,
    require_read_only: bool = False,
    require_gpu: bool = False,
    minimum_free_gb: float = 5.0,
    sample_files: int = 32,
    sample_megabytes: float = 64.0,
) -> dict[str, Any]:
    checks = [
        _raw_storage_check(raw_dir, require_read_only, sample_files, sample_megabytes),
        _writable_directory_check("output_storage", output_dir, minimum_free_gb),
        _writable_directory_check("cache_storage", cache_dir, minimum_free_gb),
        _database_check(database_uri),
        _gpu_check(require_gpu),
    ]
    failed = [check.name for check in checks if check.status == "fail"]
    return {
        "schema_version": "1.0",
        "valid": not failed,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "failed_checks": failed,
        "checks": [asdict(check) for check in checks],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida el host antes de ejecutar el pipeline.")
    parser.add_argument("--raw-dir", default="/datasets/raw")
    parser.add_argument("--output-dir", default="/datasets/processed")
    parser.add_argument("--cache-dir", default="/datasets/cache")
    parser.add_argument("--database-uri", default=os.environ.get("FIFTYONE_DATABASE_URI"))
    parser.add_argument("--require-read-only", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--minimum-free-gb", type=float, default=5.0)
    parser.add_argument("--sample-files", type=int, default=32)
    parser.add_argument("--sample-megabytes", type=float, default=64.0)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    report = run_preflight(
        Path(args.raw_dir),
        Path(args.output_dir),
        Path(args.cache_dir),
        database_uri=args.database_uri,
        require_read_only=args.require_read_only,
        require_gpu=args.require_gpu,
        minimum_free_gb=args.minimum_free_gb,
        sample_files=args.sample_files,
        sample_megabytes=args.sample_megabytes,
    )
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
