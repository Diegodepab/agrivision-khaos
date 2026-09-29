from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np

from agrivision_khaos.export_assets import generate_palette, rasterize_detections_mask


class DummyDetection:
    def __init__(self, label: str, derived_label: str, bbox: list[float], mask: np.ndarray):
        self.label = label
        self.derived_label = derived_label
        self.bounding_box = bbox
        self.mask = mask


class MasksAndHierarchyExportTests(TestCase):
    def test_palette_generation(self):
        classes = ["repilo", "sano"]
        palette = generate_palette(classes)
        self.assertEqual(palette["classes"]["background"], 0)
        self.assertEqual(palette["classes"]["repilo"], 1)
        self.assertEqual(palette["classes"]["sano"], 2)
        self.assertEqual(len(palette["colors"]["repilo"]), 3)
        self.assertEqual(palette["colors"]["background"], [0, 0, 0])

    def test_rasterize_detections_mask(self):
        det_mask = np.ones((20, 30), dtype=np.uint8)
        det = DummyDetection(
            label="repilo",
            derived_label="enfermo",
            bbox=[0.1, 0.1, 0.5, 0.5],
            mask=det_mask,
        )

        fine_class_map = {"background": 0, "repilo": 1, "sano": 2}
        coarse_class_map = {"background": 0, "enfermo": 1, "sano": 2}

        # Render fine mask
        fine_rendered = rasterize_detections_mask(100, 100, [det], fine_class_map, label_field="label")
        self.assertIsNotNone(fine_rendered)
        self.assertEqual(fine_rendered.shape, (100, 100))
        self.assertTrue(np.any(fine_rendered == 1))
        self.assertTrue(np.all(np.isin(fine_rendered, [0, 1])))

        # Render coarse mask
        coarse_rendered = rasterize_detections_mask(100, 100, [det], coarse_class_map, label_field="derived_label")
        self.assertIsNotNone(coarse_rendered)
        self.assertEqual(coarse_rendered.shape, (100, 100))
        self.assertTrue(np.any(coarse_rendered == 1))

