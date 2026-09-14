from __future__ import annotations

import errno
import hashlib
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from agrivision_khaos import export
from agrivision_khaos.execution import PipelineAlreadyRunning, PipelineLock
from agrivision_khaos.export_assets import copy_verified_asset, export_filename, link_export_asset
from agrivision_khaos.pipeline import export_clean_dataset


class ExportAssetTests(TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "original.PNG"
        self.payload = b"image bytes" * 150000
        self.source.write_bytes(self.payload)
        self.target = self.root / "snapshot.png"
        self.digest = hashlib.sha256(self.payload).hexdigest()

    def test_copy_checks_hash_and_is_independent_from_source(self):
        checksum, size = copy_verified_asset(self.source, self.target, self.digest)
        self.assertEqual((checksum, size), (self.digest, len(self.payload)))
        self.source.write_bytes(b"changed source")
        self.assertEqual(self.target.read_bytes(), self.payload)

    def test_changed_source_rejects_and_removes_partial_target(self):
        with self.assertRaisesRegex(RuntimeError, "cambió después de la auditoría"):
            copy_verified_asset(self.source, self.target, "0" * 64)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.source.read_bytes(), self.payload)

    def test_missing_or_empty_source_cannot_produce_an_asset(self):
        with self.assertRaises(FileNotFoundError):
            copy_verified_asset(self.root / "missing.png", self.target)
        self.source.write_bytes(b"")
        with self.assertRaisesRegex(RuntimeError, "Imagen vacía"):
            copy_verified_asset(self.source, self.target)
        self.assertFalse(self.target.exists())

    def test_existing_target_is_never_overwritten_or_deleted(self):
        self.target.write_bytes(b"previous export")
        with self.assertRaises(FileExistsError):
            copy_verified_asset(self.source, self.target)
        with self.assertRaises(FileExistsError):
            link_export_asset(self.source, self.target)
        self.assertEqual(self.target.read_bytes(), b"previous export")

    def test_hardlink_falls_back_to_copy(self):
        with patch("agrivision_khaos.export_assets.os.link", side_effect=OSError(errno.EXDEV, "cross-device")):
            self.assertEqual(link_export_asset(self.source, self.target), "copied")
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertNotEqual(self.target.stat().st_ino, self.source.stat().st_ino)

    def test_names_are_unique_for_identical_source_basenames(self):
        self.assertEqual(export_filename("a", Path("source-a/leaf.JPG")), "a.jpg")
        self.assertEqual(export_filename("b", Path("source-b/leaf.JPG")), "b.jpg")

    def test_existing_export_directory_is_rejected_before_dataset_mutation(self):
        with self.assertRaisesRegex(ValueError, "directorio vacío"):
            export_clean_dataset(object(), self.root, ["classification"], {}, {})
        self.assertEqual(self.source.read_bytes(), self.payload)

    def test_manual_export_and_sync_respect_the_pipeline_lock(self):
        for extra in ([], ["--sync-only"]):
            with (
                self.subTest(extra=extra),
                PipelineLock(self.root / "locks" / "fixture.lock"),
                patch("sys.argv", ["agrivision-export", "--dataset", "fixture", "--cache-dir", str(self.root), *extra]),
                patch.object(export.fo, "dataset_exists") as exists,
                self.assertRaises(PipelineAlreadyRunning),
            ):
                export.main()
            exists.assert_not_called()
