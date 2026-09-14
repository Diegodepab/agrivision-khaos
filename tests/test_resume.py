from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from agrivision_khaos.models import CurationPolicy, QualityPolicy
from agrivision_khaos.pipeline import _pipeline_fingerprint


class ResumeFingerprintTests(TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        (self.raw / "leaf.png").write_bytes(b"unchanged fixture")
        self.args = SimpleNamespace(
            output_formats="coco,yolo",
            export_dir=str(self.root / "processed"),
            report_dir=str(self.root / "reports"),
        )
        self.policy = CurationPolicy(quality=QualityPolicy(ocr_enabled=False))

    def fingerprint(self, **overrides):
        args = SimpleNamespace(**{**vars(self.args), **overrides})
        return _pipeline_fingerprint(args, self.policy, self.raw)

    def test_result_affecting_cli_changes_invalidate_resume(self):
        original = self.fingerprint()
        for overrides in (
            {"enable_ocr": True},
            {"skip_quality": True},
            {"skip_duplicates": True},
            {"skip_labels": True},
            {"max_phase_drop": 0.2},
            {"max_total_drop": 0.5},
            {"cleanlab_mode": "off"},
            {"output_formats": "classification"},
            {"export_dir": str(self.root / "other-output")},
            {"report_dir": str(self.root / "other-reports")},
            {"balance_classes": True},
            {"balance_target": "max"},
            {"require_gpu": True},
            {"require_read_only": True},
        ):
            with self.subTest(overrides=overrides):
                self.assertNotEqual(original, self.fingerprint(**overrides))

    def test_equivalent_ocr_options_and_worker_changes_preserve_resume(self):
        original = self.fingerprint()
        for overrides in (
            {"enable_ocr": False},
            {"enable_ocr": None},
            {"workers": 2},
            {"workers": 14},
            {"output_formats": " YOLO , COCO "},
        ):
            with self.subTest(overrides=overrides):
                self.assertEqual(original, self.fingerprint(**overrides))

    def test_ontology_content_changes_invalidate_resume_with_same_size_and_mtime(self):
        ontology = self.root / "ontology.yaml"
        ontology.write_text("healthy: [good]\n")
        before = ontology.stat()
        original = self.fingerprint(ontology_map=str(ontology))
        ontology.write_text("healthy: [fine]\n")
        os.utime(ontology, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(ontology.stat().st_size, before.st_size)
        self.assertNotEqual(original, self.fingerprint(ontology_map=str(ontology)))

    def test_missing_ontology_fails_before_resuming(self):
        with self.assertRaises(FileNotFoundError):
            self.fingerprint(ontology_map=str(self.root / "missing.yaml"))

    def test_quality_algorithm_version_invalidates_resume(self):
        original = self.fingerprint()
        with patch("agrivision_khaos.quality.QUALITY_ALGORITHM_VERSION", "future-quality"):
            self.assertNotEqual(original, self.fingerprint())

    def test_export_algorithm_version_invalidates_resume(self):
        original = self.fingerprint()
        with patch("agrivision_khaos.pipeline.EXPORT_ALGORITHM_VERSION", "future-export"):
            self.assertNotEqual(original, self.fingerprint())
