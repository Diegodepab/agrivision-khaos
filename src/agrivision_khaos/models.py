from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class VisionTask(StrEnum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    KEYPOINTS = "keypoints"


class SourceManifest(BaseModel):
    """Versioned provenance supplied alongside each raw dataset."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    name: str | None = None
    version: str = "unknown"
    license: str = "unknown"
    homepage: HttpUrl | None = None
    citation: str | None = None
    acquired_at: date | None = None
    tasks: list[VisionTask] = Field(default_factory=list)
    sensor: str = "RGB"
    geography: str | None = None


class QualityPolicy(BaseModel):
    min_resolution: int = Field(default=320, ge=1)
    ocr_enabled: bool = True
    ocr_confidence: float = Field(default=60.0, ge=0, le=100)
    severe_blur: float = Field(default=15.0, ge=0)
    review_blur: float = Field(default=25.0, ge=0)
    min_brightness: float = Field(default=18.0, ge=0, le=255)
    max_brightness: float = Field(default=245.0, ge=0, le=255)
    dark_p95: float = Field(default=45.0, ge=0, le=255)
    bright_p5: float = Field(default=210.0, ge=0, le=255)
    min_box_blur: float = Field(default=150.0, ge=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> "QualityPolicy":
        if self.severe_blur > self.review_blur:
            raise ValueError("severe_blur no puede superar review_blur")
        if self.min_brightness >= self.max_brightness:
            raise ValueError("min_brightness debe ser menor que max_brightness")
        return self


class DeduplicationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exact_enabled: bool = True
    semantic_enabled: bool = True
    augmentation_enabled: bool = True
    semantic_similarity: float = Field(default=0.90, gt=0, le=1)
    augmentation_similarity: float = Field(default=0.92, gt=0, le=1)
    semantic_action: Literal["review", "remove"] = "review"
    augmentation_action: Literal["review", "remove"] = "review"
    candidate_neighbors: int = Field(default=20, ge=1, le=200)
    candidate_retrieval: Literal["indexed", "exhaustive"] = "indexed"
    candidate_pool: int = Field(default=512, ge=32, le=8192)
    phash_distance: int = Field(default=12, ge=0, le=64)
    verification_min_correlation: float = Field(default=0.97, gt=0, le=1)
    verification_max_error: float = Field(default=0.04, gt=0, le=1)
    # Visual scores remain experimental until a domain-specific benchmark is available.
    # Only full-resolution discrete pixel equivalence can be removed automatically.
    remove_exact_transforms: bool = False


class SplitPolicy(BaseModel):
    train: float = Field(default=0.8, ge=0, le=1)
    val: float = Field(default=0.1, ge=0, le=1)
    test: float = Field(default=0.1, ge=0, le=1)
    group_by_location: bool = False

    @model_validator(mode="after")
    def validate_sum(self) -> "SplitPolicy":
        if abs(self.train + self.val + self.test - 1.0) > 1e-9:
            raise ValueError("Las proporciones train/val/test deben sumar 1")
        return self


class CurationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    quality: QualityPolicy = Field(default_factory=QualityPolicy)
    deduplication: DeduplicationPolicy = Field(default_factory=DeduplicationPolicy)
    splits: SplitPolicy = Field(default_factory=SplitPolicy)
