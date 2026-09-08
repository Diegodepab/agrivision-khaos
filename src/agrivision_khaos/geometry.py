"""Restricted affine evidence for manual review, with bidirectional coverage.

Texture is a content proxy, not a leaf or lesion segmentation. These measurements
never authorize removal of a crop or an arbitrary rotation.
"""

from __future__ import annotations

import cv2
import numpy as np


def _spread(points: np.ndarray, shape: tuple) -> float:
    return float(cv2.contourArea(cv2.convexHull(points))) / (shape[0] * shape[1])


def _content(gray: np.ndarray) -> np.ndarray:
    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    return gradient > 20


def affine_evidence(left: np.ndarray, right: np.ndarray) -> dict:
    a, b = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in (left, right))
    detector = cv2.ORB_create(nfeatures=1000)
    points_a, descriptors_a = detector.detectAndCompute(a, None)
    points_b, descriptors_b = detector.detectAndCompute(b, None)
    if (
        descriptors_a is None
        or descriptors_b is None
        or min(len(descriptors_a), len(descriptors_b)) < 2
    ):
        return {}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def ratios(x, y):
        return {
            pair[0].queryIdx: pair[0].trainIdx
            for pair in matcher.knnMatch(x, y, k=2)
            if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance
        }

    forward, backward = ratios(descriptors_a, descriptors_b), ratios(descriptors_b, descriptors_a)
    matches = [(i, j) for i, j in forward.items() if backward.get(j) == i]
    if len(matches) < 8:
        return {"matches": len(matches), "geometry_only": True}
    target = np.float32([points_a[i].pt for i, _ in matches])
    source = np.float32([points_b[j].pt for _, j in matches])
    matrix, mask = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return {}
    valid = mask.ravel().astype(bool)
    if valid.sum() < 8 or abs(np.linalg.det(matrix[:, :2])) < 1e-6:
        return {"matches": len(matches), "inliers": int(valid.sum()), "geometry_only": True}
    inverse = cv2.invertAffineTransform(matrix)

    def footprint(shape, transform, output_shape):
        return (
            cv2.warpAffine(
                np.ones(shape, dtype=np.uint8),
                transform,
                (output_shape[1], output_shape[0]),
                flags=cv2.INTER_NEAREST,
            )
            > 0
        )

    overlap_a = footprint(b.shape, matrix, a.shape)
    overlap_b = footprint(a.shape, inverse, b.shape)
    # Erode interpolation borders before comparing content.
    interior = cv2.erode(overlap_a.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    aligned = cv2.warpAffine(right, matrix, (a.shape[1], a.shape[0]))
    aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    content_a, content_b = _content(a), _content(b)
    textured = interior & content_a
    residuals = np.linalg.norm(cv2.transform(source[None], matrix)[0] - target, axis=1)[valid]
    correlations, errors = [], []
    for rows in np.array_split(np.arange(a.shape[0]), 3):
        for cols in np.array_split(np.arange(a.shape[1]), 3):
            tile = np.ix_(rows, cols)
            selected = textured[tile]
            if selected.sum() < 32:
                continue
            x = a[tile][selected].astype(np.float32)
            y = aligned_gray[tile][selected].astype(np.float32)
            errors.append(float(np.abs(x - y).mean()) / 255)
            if min(float(x.std()), float(y.std())) > 6:
                correlations.append(float(np.corrcoef(x, y)[0, 1]))
    coverage_left, coverage_right = float(overlap_a.mean()), float(overlap_b.mean())
    return {
        "affine": matrix.tolist(),
        "affine_coordinate_space": "descriptor_thumbnail",
        "matches": len(matches),
        "inliers": int(valid.sum()),
        "inlier_ratio": float(valid.mean()),
        "reprojection_rmse": float(np.sqrt(np.mean(residuals**2))),
        "spread_left": _spread(target[valid], a.shape),
        "spread_right": _spread(source[valid], b.shape),
        "coverage_left": coverage_left,
        "coverage_right": coverage_right,
        "coverage": min(coverage_left, coverage_right),
        "unmatched_content_left": float((content_a & ~overlap_a).sum())
        / max(int(content_a.sum()), 1),
        "unmatched_content_right": float((content_b & ~overlap_b).sum())
        / max(int(content_b.sum()), 1),
        "aligned_correlation": min(correlations, default=0.0),
        "aligned_error": max(errors, default=1.0),
        "aligned_textured_tiles": len(correlations),
        "content_proxy": "image_gradients_not_leaf_segmentation",
        "geometry_only": True,
    }
