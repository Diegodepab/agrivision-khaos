from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def default_workers() -> int:
    """Reserve two logical CPUs, honoring process affinity where available."""
    process_cpu_count = getattr(os, "process_cpu_count", None)
    cpus = process_cpu_count() if process_cpu_count is not None else None
    if cpus is None:
        try:
            cpus = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cpus = os.cpu_count()
    return max(1, (cpus or 4) - 2)


class PipelineAlreadyRunning(RuntimeError):
    pass


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


class PipelineLock:
    def __init__(self, path: Path):
        self.path = path
        self._stream = None

    def __enter__(self) -> "PipelineLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._stream.seek(0)
            owner = self._stream.read().strip() or "propietario desconocido"
            self._stream.close()
            self._stream = None
            raise PipelineAlreadyRunning(
                f"Ya existe un pipeline activo para este dataset ({owner})"
            ) from exc
        self._stream.seek(0)
        self._stream.truncate()
        json.dump(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "started_at": datetime.now(UTC).isoformat(),
            },
            self._stream,
        )
        self._stream.flush()
        os.fsync(self._stream.fileno())
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._stream is not None:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()
            self._stream = None


def source_fingerprint(raw_dir: Path, policy: dict[str, Any]) -> str:
    """Hashes path metadata deterministically without reading every image over the network."""
    digest = hashlib.sha256()
    digest.update(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(str(raw_dir.resolve()).encode())
    for current, directory_names, file_names in os.walk(raw_dir, followlinks=False):
        directory_names.sort()
        file_names.sort()
        current_path = Path(current)
        for file_name in file_names:
            path = current_path / file_name
            try:
                stat = path.stat()
                relative = path.relative_to(raw_dir).as_posix()
            except (OSError, ValueError):
                continue
            digest.update(relative.encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
            digest.update(str(stat.st_size).encode())
            digest.update(b"\0")
            digest.update(str(stat.st_mtime_ns).encode())
            digest.update(b"\0")
    return digest.hexdigest()


class RunCheckpoint:
    def __init__(self, path: Path, fingerprint: str, run_id: str):
        self.path = path
        self.fingerprint = fingerprint
        self.data: dict[str, Any] = {
            "schema_version": "1.0",
            "fingerprint": fingerprint,
            "run_id": run_id,
            "status": "running",
            "phases": {},
        }
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            if (
                isinstance(existing, dict)
                and existing.get("schema_version") == "1.0"
                and existing.get("fingerprint") == fingerprint
                and isinstance(existing.get("run_id"), str)
                and existing["run_id"]
                and existing.get("status") in ("running", "completed")
                and isinstance(existing.get("phases"), dict)
                and isinstance(existing.get("result", {}), dict)
            ):
                self.data = existing

    @property
    def run_id(self) -> str:
        return str(self.data["run_id"])

    def phase_payload(self, phase: str) -> dict[str, Any] | None:
        payload = self.data.get("phases", {}).get(phase)
        if isinstance(payload, dict) and payload.get("status") == "completed":
            result = payload.get("payload")
            return result if isinstance(result, dict) else None
        return None

    def mark_phase(self, phase: str, payload: dict[str, Any]) -> None:
        self.data["status"] = "running"
        self.data.setdefault("phases", {})[phase] = {
            "status": "completed",
            "completed_at": datetime.now(UTC).isoformat(),
            "payload": payload,
        }
        atomic_write_json(self.path, self.data)

    def mark_complete(self, payload: dict[str, Any]) -> None:
        self.data["status"] = "completed"
        self.data["completed_at"] = datetime.now(UTC).isoformat()
        self.data["result"] = payload
        atomic_write_json(self.path, self.data)
