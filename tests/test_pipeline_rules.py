from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import cv2
import fiftyone as fo
import numpy as np

from agrivision_khaos import deduplication as dedupe
from agrivision_khaos import ingest, pipeline, quality
from agrivision_khaos.models import CurationPolicy, DeduplicationPolicy, QualityPolicy


class FakeSample:
    def __init__(self, sample_id: str, filepath: str = "", **fields):
        self.id = sample_id
        self.filepath = filepath
        self._fields = fields
        self.quality = fields.get("quality")

    def has_field(self, name: str) -> bool:
        return name in self._fields

    def get_field(self, name: str):
        return self._fields.get(name)


class PipelineRuleTests(TestCase):
    def test_enabled_ocr_requires_its_backend(self):
        with (
            patch.object(quality.OcrEngine, "_backend_available", return_value=False),
            self.assertRaisesRegex(RuntimeError, "OCR está habilitado"),
        ):
            quality.OcrEngine(enabled=True)

    def test_output_formats_are_strictly_validated(self):
        self.assertEqual(pipeline.parse_output_formats("COCO, yolo"), ["coco", "yolo"])
        with self.assertRaises(ValueError):
            pipeline.parse_output_formats("coco,yoloo")
        with self.assertRaises(ValueError):
            pipeline.parse_output_formats("")

    def test_default_visual_duplicate_actions_require_review(self):
        policy = DeduplicationPolicy()
        self.assertEqual(policy.semantic_action, "review")
        self.assertEqual(policy.augmentation_action, "review")

    def test_enabled_duplicate_phase_fails_closed(self):
        policy = CurationPolicy(
            deduplication=DeduplicationPolicy(
                exact_enabled=False,
                semantic_enabled=False,
                augmentation_enabled=True,
            )
        )
        with (
            patch(
                "agrivision_khaos.deduplication.detect_augmentation_duplicates",
                side_effect=RuntimeError("backend unavailable"),
            ),
            self.assertRaisesRegex(RuntimeError, "cancela la publicación"),
        ):
            pipeline.run_duplicate_phases(
                object(),
                Path("/tmp/not-used"),
                max_phase_drop=0.4,
                max_total_drop=0.65,
                policy=policy,
            )

    def test_ontology_rejects_missing_and_conflicting_mappings(self):
        with self.assertRaises(ValueError):
            pipeline.load_ontology_mapping("/tmp/ontology-that-does-not-exist.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ontology.yaml"
            path.write_text(
                "healthy: [same-label]\ndiseased: [same-label]\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "dos clases"):
                pipeline.load_ontology_mapping(path)

    def test_cleanlab_on_fails_before_dataset_mutation(self):
        with self.assertRaises(RuntimeError):
            pipeline.run_label_phase(
                object(), "on", None, Path("/tmp/not-used")
            )

    def test_canonical_annotations_preserve_task_and_source_labels(self):
        sample = fo.Sample(filepath="/tmp/leaf.jpg")
        sample["original_label"] = fo.Classification(label="Sağlam")
        sample["yolo_detections"] = fo.Detections(
            detections=[
                fo.Detection(label="Olive Peacock Spot", bounding_box=[0.1, 0.2, 0.3, 0.4])
            ]
        )

        payload = pipeline.canonicalize_annotations(sample)

        self.assertEqual(payload["task_type"], "classification,detection")
        self.assertEqual(sample["ground_truth_classification"].label, "healthy")
        detection = sample["ground_truth_detections"].detections[0]
        self.assertEqual(detection.label, "olive_peacock_spot")
        self.assertEqual(detection["source_label"], "Olive Peacock Spot")
        self.assertEqual(
            payload["normalized_labels"],
            ["healthy", "olive_peacock_spot"],
        )

    def test_invalid_bounding_boxes_are_preserved_but_sent_to_review(self):
        sample = fo.Sample(filepath="/tmp/invalid.jpg")
        sample["coco_detections"] = fo.Detections(
            detections=[fo.Detection(label="leaf", bounding_box=[0.9, 0.2, 0.3, 0.4])]
        )

        payload = pipeline.canonicalize_annotations(sample)
        decision = pipeline.evaluate_quality(sample)

        self.assertEqual(payload["annotation_issues"], ["invalid_bbox:coco_detections:0"])
        self.assertFalse(sample["annotation_valid"])
        self.assertEqual(decision.status, "review")
        self.assertEqual(decision.phase, "annotations")

    def test_processing_failures_are_sent_to_review(self):
        sample = FakeSample("failed", processing_error=True)

        decision = pipeline.evaluate_quality(sample)

        self.assertEqual(decision.status, "review")
        self.assertEqual(decision.reason, "Error de Procesamiento")

    def test_quality_thresholds_are_policy_driven(self):
        sample = FakeSample("blurred", blur_variance=30.0)

        default_decision = pipeline.evaluate_quality(sample)
        strict_decision = pipeline.evaluate_quality(
            sample,
            QualityPolicy(severe_blur=35.0, review_blur=45.0),
        )

        self.assertEqual(default_decision.status, "kept")
        self.assertEqual(strict_decision.status, "removed")

    def test_split_policy_rejects_invalid_proportions(self):
        with self.assertRaises(ValueError):
            CurationPolicy.model_validate(
                {"splits": {"train": 0.8, "val": 0.2, "test": 0.2}}
            )

    def test_duplicate_neighbors_are_merged_into_connected_components(self):
        components = dedupe._connected_components(
            {
                "a": ["b"],
                "b": ["c"],
                "x": ["y"],
            }
        )

        self.assertEqual(components, [["a", "b", "c"], ["x", "y"]])

    def test_duplicate_label_signature_uses_all_canonical_labels(self):
        sample = FakeSample("multi", normalized_labels=["healthy", "leaf"])

        self.assertEqual(
            dedupe._label_signature(sample),
            frozenset({"healthy", "leaf"}),
        )

    def test_group_split_is_deterministic_and_class_balanced(self):
        groups = {
            **{f"healthy-{index}": ["healthy"] for index in range(30)},
            **{f"diseased-{index}": ["diseased"] for index in range(30)},
        }
        proportions = {"train": 0.8, "val": 0.1, "test": 0.1}

        first = pipeline.allocate_group_splits(groups, proportions)
        second = pipeline.allocate_group_splits(groups, proportions)

        self.assertEqual(first, second)
        counts = {split: list(first.values()).count(split) for split in proportions}
        self.assertLessEqual(abs(counts["train"] - 48), 2)
        self.assertLessEqual(abs(counts["val"] - 6), 2)
        self.assertLessEqual(abs(counts["test"] - 6), 2)
        for split in proportions:
            labels = {groups[group][0] for group, assigned in first.items() if assigned == split}
            self.assertEqual(labels, {"healthy", "diseased"})

    def test_coco_staging_never_mutates_raw_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            (raw / "present.jpg").write_bytes(b"image")
            source = raw / "annotations.json"
            original = json.dumps(
                {
                    "images": [
                        {"id": 1, "file_name": "present.jpg"},
                        {"id": 2, "file_name": "missing.jpg"},
                    ],
                    "annotations": [
                        {"id": 10, "image_id": 1},
                        {"id": 20, "image_id": 2},
                    ],
                }
            )
            source.write_text(original, encoding="utf-8")

            staged, report = ingest._stage_coco_json(
                source, raw, root / "staging", strict=False
            )

            self.assertEqual(source.read_text(encoding="utf-8"), original)
            payload = json.loads(staged.read_text(encoding="utf-8"))
            self.assertEqual([image["id"] for image in payload["images"]], [1])
            self.assertEqual([annotation["id"] for annotation in payload["annotations"]], [10])
            self.assertEqual(report["removed_missing_images"], 1)
            self.assertEqual(report["removed_orphan_annotations"], 1)

    def test_strict_coco_staging_rejects_missing_or_external_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            (root / "outside.jpg").write_bytes(b"outside")
            source = raw / "annotations.json"
            source.write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "../outside.jpg"}],
                        "annotations": [{"id": 1, "image_id": 1}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ingest.IngestionError):
                ingest._stage_coco_json(source, raw, root / "staging")

    def test_source_manifest_is_validated_and_exposed(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "olive-source"
            source.mkdir()
            (source / "source.yaml").write_text(
                "version: '2'\nlicense: CC-BY-4.0\ntasks: [classification]\nsensor: RGB\n",
                encoding="utf-8",
            )

            discovered = pipeline.discover_sources(Path(tmp))

            self.assertEqual(discovered[0]["name"], "olive-source")
            self.assertEqual(discovered[0]["version"], "2")
            self.assertEqual(discovered[0]["license"], "CC-BY-4.0")
            self.assertEqual(discovered[0]["tasks"], ["classification"])

    def test_dataset_card_warns_against_publishing_unknown_licenses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "DATASET_CARD.md"
            pipeline.write_dataset_card(
                path,
                {
                    "dataset": "fixture",
                    "counts": {"initial": 1, "kept": 1, "review": 0, "removed": 0},
                    "sources": [
                        {"name": "unlicensed", "version": "1", "license": "unknown", "format": "coco"}
                    ],
                },
            )

            card = path.read_text(encoding="utf-8")
            self.assertIn("No publicar", card)
            self.assertIn("unlicensed", card)

    def test_decoded_image_contains_content_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            image = np.full((32, 32, 3), 127, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(path), image))

            decoded = quality.read_image(str(path))

            self.assertIsNotNone(decoded)
            self.assertEqual(decoded.sha256, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_yolo_staging_rewrites_only_the_staged_descriptor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            (raw / "images" / "train").mkdir(parents=True)
            source = raw / "data.yaml"
            original = "train: /kaggle/input/images/train\nnames: [leaf]\n"
            source.write_text(original, encoding="utf-8")

            staged, report = ingest._stage_yolo_yaml(source, raw, root / "staging")

            self.assertEqual(source.read_text(encoding="utf-8"), original)
            self.assertNotEqual(staged, source)
            self.assertEqual(report["splits"]["train"], "images/train")
            self.assertIn(str(raw.resolve()), staged.read_text(encoding="utf-8"))

    def test_normalize_label_uses_safe_aliases(self):
        self.assertEqual(pipeline.normalize_label("Healthy"), "healthy")
        self.assertEqual(pipeline.normalize_label("sağlam"), "healthy")
        self.assertEqual(pipeline.normalize_label("olive peacock spot"), "olive_peacock_spot")
        self.assertEqual(pipeline.normalize_label("Aculus olearius"), "aculus_olearius")

    def test_discover_source_format_detects_coco(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "source"
            source_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "_annotations.coco.json").write_text("{}", encoding="utf-8")
            self.assertEqual(pipeline.discover_source_format(source_dir), "coco")

    def test_second_opinion_converts_borderline_phase_drop_to_review(self):
        samples = [
            FakeSample(
                str(index),
                source_dataset="dataset_a",
                normalized_label="healthy",
                curation=SimpleNamespace(status="kept"),
            )
            for index in range(10)
        ]
        decisions = {
            sample.id: pipeline.Decision("removed", "quality", "severe_blur", 0.94)
            for sample in samples[:6]
        }

        notes = pipeline.apply_second_opinion(
            samples,
            decisions,
            max_phase_drop=0.40,
            max_total_drop=0.65,
        )

        self.assertTrue(notes)
        self.assertEqual(decisions["0"].status, "review")
        self.assertTrue(decisions["0"].reason.startswith("second_opinion_"))

    def test_second_opinion_accounts_for_previous_pipeline_removals(self):
        samples = [
            FakeSample(
                str(index),
                source_dataset="dataset_a",
                normalized_label="healthy",
                curation=SimpleNamespace(status="removed" if index < 4 else "kept"),
            )
            for index in range(10)
        ]
        decisions = {
            sample.id: pipeline.Decision("removed", "quality", "blur", 0.90)
            for sample in samples[4:7]
        }

        notes = pipeline.apply_second_opinion(
            samples,
            decisions,
            max_phase_drop=0.40,
            max_total_drop=0.65,
        )

        self.assertTrue(notes)
        self.assertTrue(all(decision.status == "review" for decision in decisions.values()))

    def test_collect_examples_by_reason_groups_and_limits_examples(self):
        samples = [
            FakeSample(
                "1",
                filepath="/tmp/removed_1.jpg",
                curation=SimpleNamespace(status="removed", phase="quality", reason="severe_blur", confidence=0.94),
            ),
            FakeSample(
                "2",
                filepath="/tmp/removed_2.jpg",
                curation=SimpleNamespace(status="removed", phase="quality", reason="severe_blur", confidence=0.94),
            ),
            FakeSample(
                "3",
                filepath="/tmp/removed_3.jpg",
                curation=SimpleNamespace(status="removed", phase="quality", reason="watermark_detected", confidence=0.99),
            ),
            FakeSample(
                "4",
                filepath="/tmp/review_1.jpg",
                curation=SimpleNamespace(status="review", phase="quality", reason="borderline_blur", confidence=0.72),
            ),
        ]

        sections = pipeline.collect_examples_by_reason(samples, "removed", per_reason_limit=1, max_reasons=2)

        self.assertEqual([section["reason"] for section in sections], ["severe_blur", "watermark_detected"])
        self.assertEqual(len(sections[0]["examples"]), 1)
        self.assertEqual(sections[0]["examples"][0]["reason"], "severe_blur")

    def test_duplicate_score_prefers_clean_sample_over_augmented_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            clean_path = tmp_dir / "clean.jpg"
            augmented_path = tmp_dir / "image.rf.abc.jpg"
            image = np.full((80, 80, 3), 128, dtype=np.uint8)
            cv2.imwrite(str(clean_path), image)
            cv2.imwrite(str(augmented_path), image)
            clean = FakeSample(
                "clean",
                filepath=str(clean_path),
                blur_variance=120.0,
                width=80,
                height=80,
            )
            augmented = FakeSample(
                "augmented",
                filepath=str(augmented_path),
                blur_variance=120.0,
                width=80,
                height=80,
            )
            self.assertGreater(dedupe._sample_keep_score(clean), dedupe._sample_keep_score(augmented))
