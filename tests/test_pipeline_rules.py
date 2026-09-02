from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, name: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pipeline = load_module("src/02_curation/run_pipeline.py", "test_run_pipeline_module")
dedupe = load_module("src/02_curation/deduplicate_dataset.py", "test_dedupe_module")


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
