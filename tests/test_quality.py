from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import cv2
import numpy as np

from agrivision_khaos import quality
from agrivision_khaos.models import QualityPolicy


class QualityTests(TestCase):
    def test_green_leaf_is_selected_over_more_saturated_soil(self):
        image = np.full((100, 100, 3), (20, 80, 140), dtype=np.uint8)
        image[20:80, 20:80] = (80, 140, 90)
        mask = quality.get_leaf_mask(image)
        self.assertIsNotNone(mask)
        self.assertTrue(np.all(mask[20:80, 20:80] == 255))
        self.assertEqual(cv2.countNonZero(mask), 60 * 60)

    def test_enclosed_brown_lesions_are_retained_in_leaf_mask(self):
        image = np.full((100, 100, 3), 240, dtype=np.uint8)
        image[20:80, 20:80] = (20, 220, 30)
        image[40:60, 40:60] = (30, 60, 120)
        mask = quality.get_leaf_mask(image)
        self.assertTrue(np.all(mask[20:80, 20:80] == 255))
        self.assertEqual(mask[0, 0], 0)

    def test_non_green_leaf_uses_saturation_fallback(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        image[20:80, 20:80] = (30, 100, 150)
        mask = quality.get_leaf_mask(image)
        self.assertTrue(np.all(mask[20:80, 20:80] == 255))
        self.assertEqual(cv2.countNonZero(mask), 60 * 60)

    def test_grayscale_leaf_uses_foreground_instead_of_bright_border(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        image[20:80, 20:80] = 100
        mask = quality.get_leaf_mask(image)
        self.assertTrue(np.all(mask[20:80, 20:80] == 255))
        self.assertEqual(cv2.countNonZero(mask), 60 * 60)

    def test_uniform_images_fall_back_to_full_image_metrics(self):
        for color in (0, 127, 255, (20, 220, 30)):
            with self.subTest(color=color):
                image = np.full((32, 32, 3), color, dtype=np.uint8)
                mask = quality.get_leaf_mask(image)
                self.assertIsNone(mask)
                self.assertEqual(quality.compute_blur(image, mask)["blur_variance"], 0.0)

    def test_brightness_matches_hsv_with_and_without_mask(self):
        image = np.random.default_rng(7).integers(0, 256, (80, 100, 3), dtype=np.uint8)
        leaf = np.zeros((80, 100), dtype=np.uint8)
        leaf[10:60, 30:70] = 255
        for mask in (None, leaf):
            with self.subTest(masked=mask is not None):
                values = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
                if mask is not None:
                    values = values[mask > 0]
                result = quality.compute_brightness(image, mask)
                self.assertEqual(result["brightness_mean"], float(values.mean()))
                self.assertEqual(result["brightness_p5"], float(np.percentile(values, 5)))
                self.assertEqual(result["brightness_p95"], float(np.percentile(values, 95)))

    def test_disabled_ocr_skips_text_prefilter_and_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "leaf.png"
            image = np.random.default_rng(3).integers(20, 230, (400, 400, 3), dtype=np.uint8)
            cv2.imwrite(str(path), image)
            engine = quality.OcrEngine(enabled=False)
            with (
                patch.object(quality, "should_run_ocr", side_effect=AssertionError("prefilter")),
                patch.object(engine, "has_watermark", side_effect=AssertionError("OCR")),
            ):
                _, metrics, boxes = quality.process_image(("leaf", str(path), None, 224), engine)
            self.assertFalse(metrics.processing_error, metrics.error_message)
            self.assertFalse(metrics.has_watermark)
            self.assertEqual((metrics.width, metrics.height), (400, 400))
            self.assertIsNone(boxes)

    def test_opencv_threads_are_restored_after_failure(self):
        previous = cv2.getNumThreads()
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            with quality._opencv_worker_threads():
                self.assertEqual(cv2.getNumThreads(), 1)
                raise RuntimeError("worker failed")
        self.assertEqual(cv2.getNumThreads(), previous)

    def test_ocr_timeout_is_finite_positive_and_sent_to_tesseract(self):
        for seconds in (0, -1, float("nan"), float("inf")):
            with self.subTest(seconds=seconds):
                with self.assertRaises(ValueError):
                    quality.OcrEngine(enabled=False, timeout_seconds=seconds)
                with self.assertRaises(ValueError):
                    QualityPolicy(ocr_timeout_seconds=seconds)
        with (
            patch.object(quality.OcrEngine, "_backend_available", return_value=True),
            patch.object(quality.pytesseract, "image_to_data", return_value={}) as ocr,
        ):
            engine = quality.OcrEngine(enabled=True, timeout_seconds=2.5)
            self.assertFalse(engine.has_watermark(np.full((100, 100, 3), 128, dtype=np.uint8)))
            self.assertEqual(ocr.call_args.kwargs["timeout"], 2.5)

    def test_ocr_timeout_marks_processing_error_and_releases_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "leaf.png"
            cv2.imwrite(str(path), np.full((400, 400, 3), 128, dtype=np.uint8))
            with (
                patch.object(quality.OcrEngine, "_backend_available", return_value=True),
                patch.object(quality, "should_run_ocr", return_value=True),
                patch.object(
                    quality.pytesseract, "image_to_data",
                    side_effect=RuntimeError("Tesseract process timeout"),
                ) as ocr,
            ):
                engine = quality.OcrEngine(enabled=True, max_workers=1, timeout_seconds=0.1)
                _, metrics, _ = quality.process_image(("leaf", str(path), None, 224), engine)
                self.assertTrue(metrics.processing_error)
                self.assertIn("Tesseract process timeout", metrics.error_message)
                ocr.side_effect = None
                ocr.return_value = {}
                _, metrics, _ = quality.process_image(("leaf", str(path), None, 224), engine)
                self.assertFalse(metrics.processing_error)

    def test_invalid_concurrency_is_rejected_before_accessing_database(self):
        for overrides in ({"workers": 0}, {"ocr_workers": 0}, {"max_pending": 0}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                quality.compute_dataset_quality(
                    **{
                        "dataset_name": "unused",
                        "workers": 2,
                        "enable_ocr": False,
                        "ocr_confidence": 60,
                        **overrides,
                    }
                )

    def test_quality_cli_respects_policy_and_explicit_ocr_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = Path(tmp) / "policy.yaml"
            for enabled in (False, True):
                policy.write_text(
                    f"quality:\n  ocr_enabled: {str(enabled).lower()}\n"
                    "  min_resolution: 256\n  ocr_confidence: 75\n  ocr_timeout_seconds: 12\n"
                )
                for flags, expected in (
                    ([], enabled),
                    (["--enable-ocr"], True),
                    (["--disable-ocr"], False),
                    (["--no-enable-ocr"], False),
                ):
                    with (
                        self.subTest(policy_ocr=enabled, flags=flags),
                        patch("sys.argv", ["agrivision-quality", "--policy", str(policy), *flags]),
                        patch.object(quality, "compute_dataset_quality") as compute,
                    ):
                        quality.main()
                        self.assertIs(compute.call_args.kwargs["enable_ocr"], expected)
                        self.assertEqual(compute.call_args.kwargs["min_valid_size"], 256)
                        self.assertEqual(compute.call_args.kwargs["ocr_confidence"], 75)
                        self.assertEqual(compute.call_args.kwargs["ocr_timeout_seconds"], 12)
