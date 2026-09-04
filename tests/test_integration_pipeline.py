from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import cv2
import fiftyone as fo
import numpy as np

from agrivision_khaos.deduplication import detect_exact_duplicates, process_duplicates
from agrivision_khaos.export import sync_manual_decisions
from agrivision_khaos.ingest import create_unified_dataset
from agrivision_khaos.models import CurationPolicy, DeduplicationPolicy, QualityPolicy
from agrivision_khaos.pipeline import (
    _execute_pipeline,
    annotate_sources,
    canonicalize_annotations,
    decision_for_tagged_duplicates,
    discover_sources,
    export_clean_dataset,
    write_decisions,
)
from agrivision_khaos.quality import compute_dataset_quality
from agrivision_khaos.synthetic import generate_mock_dataset


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@unittest.skipUnless(
    os.environ.get("AGRIVISION_RUN_INTEGRATION") == "1",
    "Define AGRIVISION_RUN_INTEGRATION=1 y FIFTYONE_DATABASE_URI para ejecutar integración",
)
class PipelineIntegrationTests(unittest.TestCase):
    def test_full_pipeline_is_transactional_and_completed_run_is_idempotent(self):
        dataset_name = f"agrivision-resume-{uuid4().hex}"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            generate_mock_dataset(raw, samples_per_format=2, seed=31)
            args = SimpleNamespace(
                cache_dir=str(root / "cache"),
                resume=True,
                report_dir=str(root / "reports"),
                export_dir=str(root / "processed"),
                require_read_only=False,
                require_gpu=False,
                minimum_free_gb=0,
                workers=2,
                max_phase_drop=0.40,
                max_total_drop=0.65,
                cleanlab_mode="off",
                ontology_map=None,
                output_formats="coco,yolo",
            )
            policy = CurationPolicy(
                quality=QualityPolicy(
                    min_resolution=64,
                    ocr_enabled=False,
                    min_box_blur=0,
                ),
                deduplication=DeduplicationPolicy(semantic_enabled=False),
            )
            run_id = "integration-resume"
            export_dir = root / "processed" / dataset_name / run_id
            checkpoint_path = root / "cache" / "runs" / dataset_name / "checkpoint.json"
            try:
                _execute_pipeline(args, policy, raw, dataset_name, run_id)
                marker = export_dir / "_SUCCESS"
                self.assertTrue(marker.is_file())
                self.assertEqual(export_dir.stat().st_mode & 0o777, 0o755)
                first_marker = marker.read_bytes()
                first_mtime = marker.stat().st_mtime_ns

                _execute_pipeline(args, policy, raw, dataset_name, "ignored-new-run-id")

                self.assertEqual(marker.read_bytes(), first_marker)
                self.assertEqual(marker.stat().st_mtime_ns, first_mtime)
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                self.assertEqual(checkpoint["status"], "completed")
                self.assertEqual(checkpoint["run_id"], run_id)
                self.assertEqual(
                    set(checkpoint["phases"]),
                    {"ingestion", "quality", "duplicates", "labels", "export"},
                )
                self.assertFalse(list(export_dir.parent.glob(".*.incomplete-*")))
            finally:
                if fo.dataset_exists(dataset_name):
                    fo.delete_dataset(dataset_name)

    def test_synthetic_multiformat_pipeline_runs_end_to_end_without_models(self):
        dataset_name = f"agrivision-synthetic-{uuid4().hex}"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            generate_mock_dataset(raw, samples_per_format=3, seed=23)
            try:
                dataset = create_unified_dataset(
                    dataset_name,
                    raw,
                    staging_dir=root / "staging",
                    strict=True,
                )
                sources = discover_sources(raw)
                annotate_sources(dataset, raw, sources)
                compute_dataset_quality(
                    dataset_name,
                    workers=2,
                    enable_ocr=False,
                    ocr_confidence=60.0,
                )
                dataset = fo.load_dataset(dataset_name)

                removed, pairs = detect_exact_duplicates(dataset, "redundant_exact")
                decisions = decision_for_tagged_duplicates(
                    dataset,
                    "redundant_exact",
                    "exact_duplicates",
                    "Redundante (Exacta)",
                )
                write_decisions(dataset, decisions, "exact_duplicates")

                exports = export_clean_dataset(
                    dataset,
                    root / "processed",
                    output_formats=["coco", "yolo"],
                    label_mapping={},
                    summary={},
                    policy=CurationPolicy(quality=QualityPolicy(min_box_blur=0)),
                )

                self.assertEqual(len(dataset), 9)
                self.assertEqual(
                    removed,
                    6,
                    [
                        (
                            sample["source_path"],
                            sample["normalized_labels"],
                            sample["source_labels"],
                        )
                        for sample in dataset
                    ],
                )
                self.assertEqual(len(pairs), 6)
                self.assertEqual(len(dataset.match_tags("redundant_exact")), 6)
                self.assertEqual(len(dataset.match_tags("redundant_exact_label_conflict")), 0)
                self.assertNotIn("coco_error", exports)
                self.assertNotIn("yolo_error", exports)
                self.assertEqual(
                    len(list((root / "processed" / "yolo").rglob("*.txt"))),
                    3,
                )
            finally:
                if fo.dataset_exists(dataset_name):
                    fo.delete_dataset(dataset_name)

    def test_conflicting_human_review_tags_fail_before_mutation(self):
        dataset_name = f"agrivision-hitl-{uuid4().hex}"
        dataset = fo.Dataset(dataset_name, persistent=True)
        try:
            sample = fo.Sample(filepath="/tmp/nonexistent.jpg", tags=["kept", "removed"])
            sample["curation"] = fo.DynamicEmbeddedDocument(status="review")
            dataset.add_sample(sample)

            with self.assertRaises(RuntimeError):
                sync_manual_decisions(dataset)

            self.assertEqual(dataset.first()["curation"].status, "review")
        finally:
            if fo.dataset_exists(dataset_name):
                fo.delete_dataset(dataset_name)

    def test_human_export_is_blocked_while_reviews_are_unresolved(self):
        dataset_name = f"agrivision-unresolved-{uuid4().hex}"
        dataset = fo.Dataset(dataset_name, persistent=True)
        try:
            sample = fo.Sample(filepath="/tmp/nonexistent.jpg")
            sample["curation"] = fo.DynamicEmbeddedDocument(status="review")
            dataset.add_sample(sample)

            with self.assertRaisesRegex(RuntimeError, "muestras en revisión"):
                sync_manual_decisions(dataset)
        finally:
            if fo.dataset_exists(dataset_name):
                fo.delete_dataset(dataset_name)

    def test_conflicting_duplicate_labels_are_never_auto_removed(self):
        dataset_name = f"agrivision-conflict-{uuid4().hex}"
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "duplicate.jpg"
            self.assertTrue(
                cv2.imwrite(str(image_path), np.full((64, 64, 3), 128, dtype=np.uint8))
            )
            dataset = fo.Dataset(dataset_name, persistent=True)
            try:
                first = fo.Sample(filepath=str(image_path))
                first["normalized_labels"] = ["healthy"]
                first["normalized_label"] = "healthy"
                first["curation"] = fo.DynamicEmbeddedDocument(status="kept")
                second = fo.Sample(filepath=str(image_path))
                second["normalized_labels"] = ["diseased"]
                second["normalized_label"] = "diseased"
                second["curation"] = fo.DynamicEmbeddedDocument(status="kept")
                dataset.add_samples([first, second])

                removed, pairs = process_duplicates(
                    dataset,
                    {first.id: [second.id]},
                    "redundant_exact",
                )

                self.assertEqual(removed, 0)
                self.assertEqual(pairs, [])
                self.assertEqual(len(dataset.match_tags("redundant_exact_label_conflict")), 2)
            finally:
                if fo.dataset_exists(dataset_name):
                    fo.delete_dataset(dataset_name)

    def test_classification_round_trip_preserves_raw_and_materializes_splits(self):
        dataset_name = f"agrivision-integration-{uuid4().hex}"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            source = raw / "fixture"
            rng = np.random.default_rng(7)
            for label in ("Healthy", "Diseased"):
                label_dir = source / "train" / label
                label_dir.mkdir(parents=True)
                for index in range(12):
                    image = rng.integers(0, 256, size=(320, 320, 3), dtype=np.uint8)
                    self.assertTrue(cv2.imwrite(str(label_dir / f"{index:03d}.jpg"), image))

            before = _tree_digest(raw)
            try:
                dataset = create_unified_dataset(
                    dataset_name,
                    raw,
                    staging_dir=root / "staging",
                    strict=True,
                )
                sources = discover_sources(raw)
                annotate_sources(dataset, raw, sources)
                summary: dict[str, object] = {}
                output = root / "processed"
                exports = export_clean_dataset(
                    dataset,
                    output,
                    output_formats=["classification"],
                    label_mapping={},
                    summary=summary,
                )

                self.assertEqual(_tree_digest(raw), before)
                self.assertEqual(len(dataset), 24)
                self.assertEqual(sum(summary["splits"].values()), 24)
                self.assertIn("classification", exports)
                self.assertEqual(len(list((output / "classification").rglob("*.jpg"))), 24)
                self.assertTrue((output / "manifest.jsonl").is_file())
                self.assertEqual(
                    set(dataset.distinct("ground_truth_classification.label")),
                    {"healthy", "diseased"},
                )
            finally:
                if fo.dataset_exists(dataset_name):
                    fo.delete_dataset(dataset_name)

    def test_detection_export_preserves_boxes_in_coco_and_yolo(self):
        dataset_name = f"agrivision-detection-{uuid4().hex}"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            dataset = fo.Dataset(dataset_name, persistent=True)
            try:
                samples = []
                for index in range(12):
                    image_path = image_dir / f"leaf-{index:02d}.jpg"
                    self.assertTrue(
                        cv2.imwrite(
                            str(image_path),
                            np.full((128, 128, 3), 40 + index, dtype=np.uint8),
                        )
                    )
                    sample = fo.Sample(filepath=str(image_path), tags=["kept"])
                    sample["coco_detections"] = fo.Detections(
                        detections=[
                            fo.Detection(
                                label="Olive Peacock Spot",
                                bounding_box=[0.1, 0.2, 0.3, 0.4],
                            )
                        ]
                    )
                    sample["curation"] = fo.DynamicEmbeddedDocument(status="kept")
                    samples.append(sample)
                dataset.add_samples(samples)

                for sample in dataset.iter_samples(autosave=True):
                    for field, value in canonicalize_annotations(sample).items():
                        sample[field] = value
                    sample["ground_truth_detections"].detections[0]["blur_variance"] = 200.0

                output = root / "processed"
                exports = export_clean_dataset(
                    dataset,
                    output,
                    output_formats=["coco", "yolo"],
                    label_mapping={},
                    summary={},
                )

                self.assertNotIn("coco_error", exports)
                self.assertNotIn("yolo_error", exports)
                annotation_count = 0
                for labels_path in (output / "coco").rglob("*.json"):
                    payload = json.loads(labels_path.read_text(encoding="utf-8"))
                    annotation_count += len(payload.get("annotations", []))
                self.assertEqual(
                    annotation_count,
                    12,
                    [str(path.relative_to(output)) for path in output.rglob("*")],
                )
                yolo_labels = list((output / "yolo").rglob("*.txt"))
                self.assertEqual(len(yolo_labels), 12)
                self.assertTrue(
                    all(path.read_text(encoding="utf-8").strip() for path in yolo_labels)
                )
            finally:
                if fo.dataset_exists(dataset_name):
                    fo.delete_dataset(dataset_name)


if __name__ == "__main__":
    unittest.main()
