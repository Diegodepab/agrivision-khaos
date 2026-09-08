"""Deterministic, bounded multi-probe retrieval; exact families are handled separately.

The index proposes neighbors, never identity. Bucket and pool truncation are
observable, and the exhaustive mode remains an oracle for recall evaluation.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict

import numpy as np


class CandidateIndex:
    def __init__(self, hashes: np.ndarray, histograms: np.ndarray, pool_size: int):
        self.pool_size = pool_size
        self.size = len(hashes)
        self.truncated_queries = 0
        self.scored_neighbors = 0
        self.keys = []
        self.buckets = defaultdict(list)
        # Fixed seed and float32 keep projections reproducible and independent of input order.
        planes = np.random.default_rng(7141).standard_normal((512, 48)).astype(np.float32)
        projections = histograms @ planes
        color_codes = np.packbits(projections > 0, axis=1)
        for index, variants in enumerate(hashes):
            keys = {
                ("hash", band, (int(value) >> (8 * band)) & 255)
                for value in variants
                for band in range(8)
            }
            if np.linalg.norm(histograms[index]) > 0:
                keys.update(
                    ("color", band, int(code)) for band, code in enumerate(color_codes[index])
                )
            keys = sorted(keys)
            self.keys.append(keys)
            for key in keys:
                self.buckets[key].append(index)

    def query(self, index: int) -> np.ndarray:
        # Small collections fit entirely in the bounded pool; avoid needless recall loss.
        if self.size <= self.pool_size + 1:
            selected = np.arange(self.size)
            selected = selected[selected != index]
            self.scored_neighbors += len(selected)
            return selected
        keys = self.keys[index]
        # Take a bounded window from each posting list. Deterministic asset ordering
        # makes this fairer than always selecting a bucket's first images.
        budget = max(2, self.pool_size // max(len(keys), 1))
        votes = defaultdict(int)
        truncated = False
        for key in keys:
            bucket = self.buckets[key]
            truncated |= len(bucket) > budget + 1
            center = bisect_left(bucket, index)
            start = max(0, min(center - budget // 2, len(bucket) - budget - 1))
            for other in bucket[start : start + budget + 1]:
                if other != index:
                    votes[other] += 1
        selected = sorted(votes, key=lambda other: (-votes[other], abs(other - index), other))
        truncated |= len(selected) > self.pool_size
        self.truncated_queries += int(truncated)
        selected = selected[: self.pool_size]
        self.scored_neighbors += len(selected)
        return np.asarray(selected, dtype=np.intp)
