from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import cv2
import fiftyone as fo
import numpy as np
import yaml

from agrivision_khaos import pipeline
from agrivision_khaos.export_assets import copy_verified_asset
from agrivision_khaos.models import CurationPolicy, SplitPolicy


@unittest.skipUnless(os.environ.get("AGRIVISION_RUN_INTEGRATION") == "1", "Requires MongoDB")
class ExportIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.dataset = fo.Dataset(f"export-test-{uuid4().hex}")
        self.addCleanup(self.dataset.delete)
        self.rng = np.random.default_rng(152)

    def add(self, index, blurs=(None,), classification=True):
        folder = self.raw / str(index)
        folder.mkdir()
        path = folder / "same-name.PNG"
        cv2.imwrite(str(path), self.rng.integers(20, 230, (80, 96, 3), dtype=np.uint8))
        label = "healthy" if index % 2 else "diseased"
        sample = fo.Sample(filepath=str(path))
        sample["curation"] = fo.DynamicEmbeddedDocument(status="kept")
        sample["normalized_label"] = label
        sample["normalized_labels"] = [label]
        sample["source_dataset"] = "fixture"
        sample["source_path"] = str(path)
        sample["task_type"] = "classification,detection" if classification else "detection"
        sample["annotation_valid"] = True
        if classification:
            sample["ground_truth_classification"] = fo.Classification(label=label)
        boxes = []
        for blur in blurs:
            box = fo.Detection(label=label, bounding_box=[0.1, 0.2, 0.3, 0.4])
            if blur is not None:
                box["blur_variance"] = blur
            boxes.append(box)
        sample["ground_truth_detections"] = fo.Detections(detections=boxes)
        self.dataset.add_sample(sample)
        return sample

    def run_export(self, output, formats, summary=None):
        return pipeline.export_clean_dataset(
            self.dataset, output, formats, {}, summary if summary is not None else {},
            policy=CurationPolicy(splits=SplitPolicy(train=0.5, val=0.25, test=0.25)),
            cache_dir=self.root / "cache",
        )

    def test_moved_export_is_portable_and_formats_share_copied_images(self):
        for index in range(8):
            self.add(index)
        original_paths = self.dataset.values("filepath")
        output = self.root / ".incomplete-export"
        summary = {}
        formats = ["classification", "coco", "yolo", "fiftyone"]
        exports = self.run_export(output, formats, summary)
        self.assertFalse([key for key in exports if key.endswith("_error")], exports)
        self.assertEqual(self.dataset.values("filepath"), original_paths)
        final = self.root / "published"
        output.rename(final)
        rows = [json.loads(line) for line in (final / "manifest.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 8)
        self.assertEqual(summary["export"]["linked"], 24)
        self.assertEqual(summary["export"]["copied"], 0)
        for row in rows:
            self.assertFalse(Path(row["filepath"]).is_absolute())
            asset = final / row["filepath"]
            self.assertEqual(hashlib.sha256(asset.read_bytes()).hexdigest(), row["asset_sha256"])
            self.assertEqual(asset.stat().st_size, row["size_bytes"])
            self.assertNotEqual(asset.stat().st_ino, Path(row["source_path"]).stat().st_ino)
            for media in (
                final / "coco" / row["assigned_split"] / "data" / asset.name,
                final / "yolo" / "images" / row["assigned_split"] / asset.name,
                final / "classification" / row["assigned_split"] / row["normalized_label"] / asset.name,
            ):
                self.assertEqual(media.stat().st_ino, asset.stat().st_ino)
        self.assertEqual(len((final / "checksums.sha256").read_text().splitlines()), 8)
        config = yaml.safe_load((final / "yolo" / "dataset.yaml").read_text())
        self.assertNotIn("path", config)
        self.assertEqual(config["names"], {0: "diseased", 1: "healthy"})
        self.assertEqual(len(list((final / "yolo").rglob("*.yaml"))), 1)
        coco_classes = []
        for split in {row["assigned_split"] for row in rows}:
            self.assertTrue((final / "yolo" / config[split]).is_dir())
            labels = json.loads((final / "coco" / split / "labels.json").read_text())
            coco_classes.append(labels["categories"])
            imported = fo.Dataset.from_dir(
                dataset_dir=str(final / "yolo"), dataset_type=fo.types.YOLOv5Dataset,
                split=split, label_field="detections",
            )
            try:
                self.assertEqual(len(imported), sum(row["assigned_split"] == split for row in rows))
                self.assertTrue(all(Path(sample.filepath).is_file() for sample in imported))
                self.assertEqual(imported.count("detections.detections"), len(imported))
            finally:
                imported.delete()
        self.assertTrue(all(classes == coco_classes[0] for classes in coco_classes))
        imported = fo.Dataset.from_dir(
            dataset_dir=str(final / "fiftyone"), dataset_type=fo.types.FiftyOneDataset,
        )
        try:
            self.assertEqual(len(imported), 8)
            self.assertTrue(all(Path(sample.filepath).is_file() for sample in imported))
        finally:
            imported.delete()

    def test_unknown_blur_is_preserved_and_removed_boxes_do_not_create_false_negatives(self):
        unknown = self.add(0, blurs=(None,), classification=False)
        bad = self.add(1, blurs=(10,), classification=False)
        negative = self.add(2, blurs=(), classification=False)
        mixed = self.add(3, blurs=(200, 10), classification=False)
        output = self.root / "export"
        summary = {}
        exports = self.run_export(output, ["coco", "yolo"], summary)
        self.assertFalse([key for key in exports if key.endswith("_error")], exports)
        self.assertFalse((output / "fiftyone").exists())
        self.assertFalse((output / "classification").exists())
        expected = {unknown.id, negative.id, mixed.id}
        labels = list((output / "yolo" / "labels").rglob("*.txt"))
        self.assertEqual({path.stem for path in labels}, expected)
        self.assertEqual(sum(bool(path.read_text().strip()) for path in labels), 2)
        self.assertEqual(len(self.dataset[bad.id].ground_truth_detections.detections), 1)
        self.assertEqual(len(self.dataset[mixed.id].ground_truth_detections.detections), 2)
        self.assertEqual(summary["export"]["detection_images_excluded_after_box_filter"], 1)

    def test_source_change_between_audit_and_copy_blocks_export(self):
        sample = self.add(0)
        output = self.root / "export"

        def changed_copy(source, target, expected_sha256=None):
            cv2.imwrite(str(source), np.full((80, 96, 3), 128, dtype=np.uint8))
            return copy_verified_asset(source, target, expected_sha256)

        with (
            patch.object(pipeline, "copy_verified_asset", side_effect=changed_copy),
            self.assertRaisesRegex(RuntimeError, "cambió después de la auditoría"),
        ):
            self.run_export(output, ["classification"])
        self.assertFalse((output / "_SUCCESS").exists())
        self.assertFalse(list((output / "images").iterdir()))
        self.assertEqual(self.dataset[sample.id].filepath, sample.filepath)
