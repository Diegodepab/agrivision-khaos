from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from agrivision_khaos.execution import (
    PipelineAlreadyRunning,
    PipelineLock,
    RunCheckpoint,
    default_workers,
    source_fingerprint,
)
from agrivision_khaos.preflight import run_preflight


class ExecutionTests(unittest.TestCase):
    def test_workers_reserve_two_cpus_without_starving_small_machines(self):
        for cpus, expected in ((16, 14), (4, 2), (2, 1), (1, 1)):
            with (
                self.subTest(cpus=cpus),
                patch("os.process_cpu_count", return_value=cpus, create=True),
            ):
                self.assertEqual(default_workers(), expected)

    def test_workers_fall_back_to_affinity_then_cpu_count(self):
        with (
            patch("os.process_cpu_count", return_value=None, create=True),
            patch("os.sched_getaffinity", return_value={0, 1, 2, 3}, create=True),
        ):
            self.assertEqual(default_workers(), 2)
        with (
            patch("os.process_cpu_count", return_value=None, create=True),
            patch("os.sched_getaffinity", side_effect=OSError, create=True),
            patch("os.cpu_count", return_value=None),
        ):
            self.assertEqual(default_workers(), 2)

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

    def test_malformed_checkpoints_start_a_new_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            checkpoint = RunCheckpoint(path, "fingerprint", "old-run")
            valid = checkpoint.data
            for payload in (
                [], None, 1, "broken", {},
                {**valid, "run_id": None},
                {**valid, "phases": []},
                {**valid, "status": []},
                {**valid, "result": None},
            ):
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload))
                    resumed = RunCheckpoint(path, "fingerprint", "new-run")
                    self.assertEqual(resumed.run_id, "new-run")
                    self.assertIsNone(resumed.phase_payload("quality"))

    def test_malformed_phase_payload_is_recomputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = RunCheckpoint(Path(tmp) / "checkpoint.json", "fingerprint", "run")
            for payload in (None, [], "broken"):
                with self.subTest(payload=payload):
                    checkpoint.data["phases"]["quality"] = {
                        "status": "completed", "payload": payload,
                    }
                    self.assertIsNone(checkpoint.phase_payload("quality"))

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
