from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase

import cv2
import numpy as np

from agrivision_khaos.balancing import (
    apply_agronomic_augmentation,
    calculate_target_count,
)


class BalancingTests(TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_calculate_target_count(self):
        counts = {"clase_a": 100, "clase_b": 50, "clase_c": 10}
        self.assertEqual(calculate_target_count(counts, strategy="max"), 100)
        self.assertEqual(calculate_target_count(counts, strategy="median"), 50)
        self.assertEqual(calculate_target_count(counts, strategy="mean"), 53)
        self.assertEqual(calculate_target_count(counts, strategy="80"), 80)

    def test_apply_agronomic_augmentation(self):
        # Crear imagen sintética de hoja (verde con textura)
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        img[:, :] = (35, 140, 45)
        # Añadir círculo simulando lesión/mancha
        cv2.circle(img, (80, 60), 15, (20, 40, 180), -1)

        aug = apply_agronomic_augmentation(img, seed=42)
        self.assertEqual(aug.shape, img.shape)
        self.assertEqual(aug.dtype, np.uint8)
        self.assertTrue(np.any(aug != 0))
        # Verificar que no está vacía o corrompida
        self.assertFalse(np.all(aug == img))
