from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from agrivision_khaos.execution import (
    PipelineAlreadyRunning,
    PipelineLock,
    RunCheckpoint,
    source_fingerprint,
)
from agrivision_khaos.preflight import run_preflight


class ExecutionTests(unittest.TestCase):
    def test_pipeline_lock_rejects_concurrent_owner_and_is_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "dataset.lock"
            with PipelineLock(lock_path):
                with self.assertRaises(PipelineAlreadyRunning):
                    with PipelineLock(lock_path):
                        pass
            with PipelineLock(lock_path):
                pass

    def test_checkpoint_resumes_only_matching_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            first = RunCheckpoint(path, "fingerprint-a", "run-a")
            first.mark_phase("quality", {"samples": 10})

            resumed = RunCheckpoint(path, "fingerprint-a", "run-b")
            changed = RunCheckpoint(path, "fingerprint-b", "run-c")

            self.assertEqual(resumed.run_id, "run-a")
            self.assertEqual(resumed.phase_payload("quality"), {"samples": 10})
            self.assertEqual(changed.run_id, "run-c")
            self.assertIsNone(changed.phase_payload("quality"))
            self.assertEqual(path.stat().st_mode & 0o777, 0o644)

    def test_source_fingerprint_changes_when_source_metadata_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            raw.mkdir()
            source = raw / "image.jpg"
            source.write_bytes(b"first")
            first = source_fingerprint(raw, {"quality": 1})
            source.write_bytes(b"second-version")
            second = source_fingerprint(raw, {"quality": 1})
            policy_changed = source_fingerprint(raw, {"quality": 2})

            self.assertNotEqual(first, second)
            self.assertNotEqual(second, policy_changed)

    def test_cpu_preflight_checks_storage_without_requiring_gpu_or_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            self.assertTrue(
                cv2.imwrite(
                    str(raw / "sample.jpg"),
                    np.full((32, 32, 3), 100, dtype=np.uint8),
                )
            )

            report = run_preflight(
                raw,
                root / "output",
                root / "cache",
                minimum_free_gb=0,
                sample_files=1,
            )

            self.assertTrue(report["valid"])
            statuses = {check["name"]: check["status"] for check in report["checks"]}
            self.assertEqual(statuses["raw_storage"], "pass")
            self.assertEqual(statuses["output_storage"], "pass")
            self.assertEqual(statuses["cache_storage"], "pass")
            self.assertEqual(statuses["database"], "skipped")
            self.assertIn(statuses["gpu"], {"warning", "skipped", "pass"})

    def test_preflight_reports_missing_remote_mount_as_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = run_preflight(
                root / "missing-raw-mount",
                root / "output",
                root / "cache",
                minimum_free_gb=0,
            )

            self.assertFalse(report["valid"])
            self.assertIn("raw_storage", report["failed_checks"])


if __name__ == "__main__":
    unittest.main()
