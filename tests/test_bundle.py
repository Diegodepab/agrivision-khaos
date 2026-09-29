from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from unittest import TestCase

from agrivision_khaos.bundle import create_release_bundle, find_latest_export


class BundleTests(TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

        # Create mock processed export
        self.export_dir = self.root / "processed" / "TestCrop" / "20260916_100000"
        self.export_dir.mkdir(parents=True)
        (self.export_dir / "_SUCCESS").write_text(json.dumps({"status": "completed"}))

        # Create dummy image and files
        images_dir = self.export_dir / "images"
        images_dir.mkdir()
        dummy_img = images_dir / "sample1.jpg"
        dummy_img.write_bytes(b"image bytes")

        # Classification dir
        cls_dir = self.export_dir / "classification" / "train" / "repilo"
        cls_dir.mkdir(parents=True)
        (cls_dir / "sample1.jpg").write_bytes(b"image bytes")

        # Manifest
        manifest_rows = [
            {
                "id": "s1",
                "filepath": "images/sample1.jpg",
                "source_path": "/raw/img1.jpg",
                "source_dataset": "SourceA",
                "source_split": "train",
                "assigned_split": "train",
                "source_label": "Olive Repilo",
                "normalized_label": "repilo",
                "derived_label": "enfermo",
                "task_type": "classification",
                "asset_sha256": "dummyhash",
                "size_bytes": "11",
            }
        ]
        with (self.export_dir / "manifest.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

        (self.export_dir / "checksums.sha256").write_text("dummyhash  images/sample1.jpg\n")
        (self.export_dir / "curation_summary.json").write_text(
            json.dumps({
                "dataset": "TestCrop",
                "run_id": "20260916_100000",
                "counts": {"initial": 1, "kept": 1, "review": 0, "removed": 0},
                "sources": [{"name": "SourceA", "format": "ImageFolder", "version": "1.0", "license": "CC-BY-4.0"}],
                "splits": {"train": 1, "val": 0, "test": 0},
            })
        )

        # Mock report
        self.report_dir = self.root / "reports" / "TestCrop" / "20260916_100000"
        self.report_dir.mkdir(parents=True)
        (self.report_dir / "report.html").write_text("<html>Audit Report</html>")

    def test_find_latest_export(self):
        latest = find_latest_export(self.root / "processed", "TestCrop")
        self.assertEqual(latest, self.export_dir)

    def test_create_release_bundle_zip(self):
        out_dir = self.root / "releases"
        res = create_release_bundle(
            dataset_name="TestCrop",
            export_root=self.root / "processed",
            report_root=self.root / "reports",
            output_dir=out_dir,
            version="1.0",
            release_date="20260918",
            archive_format="zip",
        )

        self.assertEqual(res["dataset"], "TestCrop")
        self.assertEqual(res["version"], "1.0")
        self.assertEqual(res["release_date"], "20260918")

        release_dir = Path(res["release_dir"])
        self.assertTrue(release_dir.is_dir())
        self.assertEqual(release_dir, out_dir / "TestCrop" / "20260918")

        bundle_path = Path(res["bundle_dir"])
        self.assertTrue(bundle_path.is_dir())
        self.assertTrue((bundle_path / "README.md").is_file())
        self.assertTrue((bundle_path / "SOURCES.md").is_file())
        self.assertTrue((bundle_path / "preprocess" / "report.html").is_file())
        self.assertTrue((bundle_path / "annotations" / "labels.csv").is_file())
        self.assertTrue((bundle_path / "dataset" / "train" / "repilo" / "sample1.jpg").is_file())

        # Check README contains ontology and class documentation
        readme_text = (bundle_path / "README.md").read_text()
        self.assertIn("TestCrop", readme_text)
        self.assertIn("repilo", readme_text)
        self.assertIn("enfermo", readme_text)

        # Check ZIP archive and CKAN metadata were produced in release_dir
        archive_path = Path(res["archive_path"])
        self.assertTrue(archive_path.is_file())
        self.assertTrue(archive_path.name.endswith(".zip"))
        self.assertTrue((release_dir / f"{archive_path.name}.sha256").is_file())
        self.assertTrue((release_dir / "TestCrop_v1.0_CKAN_METADATA.md").is_file())

    def test_create_release_bundle_detection_layout(self):
        # Create mock detection dataset export
        det_export_dir = self.root / "processed" / "TestDet" / "20260920_120000"
        det_export_dir.mkdir(parents=True)
        (det_export_dir / "_SUCCESS").write_text(json.dumps({"status": "completed"}))

        # Mock COCO detection export
        coco_train_dir = det_export_dir / "coco" / "train"
        (coco_train_dir / "data").mkdir(parents=True)
        img1 = coco_train_dir / "data" / "det1.jpg"
        img1.write_bytes(b"det1 image bytes")
        coco_json = {
            "info": {},
            "licenses": [],
            "categories": [{"id": 1, "name": "aceituna"}],
            "images": [{"id": 1, "file_name": "det1.jpg"}],
            "annotations": [{"id": 101, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]}],
        }
        (coco_train_dir / "labels.json").write_text(json.dumps(coco_json))

        # Mock unannotated image (e.g. from classification or raw)
        cls_dir = det_export_dir / "classification" / "test" / "aceituna"
        cls_dir.mkdir(parents=True)
        unann_img = cls_dir / "unann1.jpg"
        unann_img.write_bytes(b"unann1 image bytes")

        # Manifest with 1 detection and 1 classification (unannotated)
        manifest_rows = [
            {
                "id": "d1",
                "filepath": "coco/train/data/det1.jpg",
                "source_path": "/raw/det1.jpg",
                "source_dataset": "SourceDet",
                "source_split": "train",
                "assigned_split": "train",
                "source_label": "aceituna",
                "normalized_label": "aceituna",
                "derived_label": "aceituna",
                "task_type": "detection",
                "asset_sha256": "hashdet1",
                "size_bytes": "16",
            },
            {
                "id": "u1",
                "filepath": "classification/test/aceituna/unann1.jpg",
                "source_path": "/raw/unann1.jpg",
                "source_dataset": "SourceDet",
                "source_split": "test",
                "assigned_split": "test",
                "source_label": "aceituna",
                "normalized_label": "aceituna",
                "derived_label": "aceituna",
                "task_type": "classification",
                "asset_sha256": "hashunann1",
                "size_bytes": "18",
            },
        ]
        with (det_export_dir / "manifest.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

        (det_export_dir / "checksums.sha256").write_text("hashdet1  coco/train/data/det1.jpg\n")
        (det_export_dir / "curation_summary.json").write_text(
            json.dumps({
                "dataset": "TestDet",
                "run_id": "20260920_120000",
                "counts": {"initial": 2, "kept": 2, "review": 0, "removed": 0},
                "sources": [{"name": "SourceDet", "format": "COCO", "version": "1.0", "license": "CC-BY-4.0"}],
                "splits": {"train": 1, "val": 0, "test": 1},
            })
        )

        out_dir = self.root / "releases"
        res = create_release_bundle(
            dataset_name="TestDet",
            export_root=self.root / "processed",
            report_root=self.root / "reports",
            output_dir=out_dir,
            version="1.0",
            release_date="20260920",
            archive_format="zip",
            layout="detection",
        )

        bundle_path = Path(res["bundle_dir"])
        self.assertTrue(bundle_path.is_dir())

        # Unified detection structure: NO detection/ directory
        self.assertFalse((bundle_path / "detection").exists())

        # Detections should be directly in dataset/train/
        self.assertTrue((bundle_path / "dataset" / "train" / "images" / "det1.jpg").is_file())
        self.assertTrue((bundle_path / "dataset" / "train" / "labels.json").is_file())

        # Unannotated sample should be in dataset/test_unannotated/
        self.assertTrue((bundle_path / "dataset" / "test_unannotated" / "images" / "unann1.jpg").is_file())
        self.assertTrue((bundle_path / "dataset" / "test_unannotated" / "README.md").is_file())

        # labels.csv should record test_unannotated
        with (bundle_path / "annotations" / "labels.csv").open() as f:
            csv_rows = list(csv.DictReader(f))
        unann_row = next(r for r in csv_rows if r["filename"] == "unann1.jpg")
        self.assertEqual(unann_row["task"], "unannotated")
        self.assertEqual(unann_row["split"], "test_unannotated")

        # README mentions test sin anotar
        readme = (bundle_path / "README.md").read_text()
        self.assertIn("Test Sin Anotar", readme)


