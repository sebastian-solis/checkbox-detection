"""Tunable settings for the service.

Every detection threshold lives here rather than inline in the pipeline so the
behaviour can be adjusted per deployment without touching logic, and so the
writeup can point at one file when explaining what was tuned and why.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DetectionSettings:
    """Geometry and classification thresholds.

    Size bounds are fractions of the image width, which keeps them meaningful
    across scan resolutions instead of tying the service to 300 DPI.
    """

    min_box_width_ratio: float = 0.009
    max_box_width_ratio: float = 0.028
    min_aspect_ratio: float = 0.65
    max_aspect_ratio: float = 1.55
    line_length_ratio: float = 0.7
    min_extent: float = 0.72
    surround_margin_ratio: float = 0.35
    min_surround_luma: float = 120.0
    stroke_luma_ceiling: float = 200.0
    max_stroke_saturation: float = 90.0
    polygon_epsilon: float = 0.04
    max_polygon_vertices: int = 8
    duplicate_iou_threshold: float = 0.3
    interior_inset_ratio: float = 0.22
    checked_ink_threshold: float = 0.14
    deskew_enabled: bool = True

    @classmethod
    def from_env(cls) -> DetectionSettings:
        return cls(
            min_box_width_ratio=_env_float("MIN_BOX_WIDTH_RATIO", cls.min_box_width_ratio),
            max_box_width_ratio=_env_float("MAX_BOX_WIDTH_RATIO", cls.max_box_width_ratio),
            min_aspect_ratio=_env_float("MIN_ASPECT_RATIO", cls.min_aspect_ratio),
            max_aspect_ratio=_env_float("MAX_ASPECT_RATIO", cls.max_aspect_ratio),
            line_length_ratio=_env_float("LINE_LENGTH_RATIO", cls.line_length_ratio),
            min_extent=_env_float("MIN_EXTENT", cls.min_extent),
            surround_margin_ratio=_env_float("SURROUND_MARGIN_RATIO", cls.surround_margin_ratio),
            min_surround_luma=_env_float("MIN_SURROUND_LUMA", cls.min_surround_luma),
            stroke_luma_ceiling=_env_float("STROKE_LUMA_CEILING", cls.stroke_luma_ceiling),
            max_stroke_saturation=_env_float("MAX_STROKE_SATURATION", cls.max_stroke_saturation),
            polygon_epsilon=_env_float("POLYGON_EPSILON", cls.polygon_epsilon),
            max_polygon_vertices=_env_int("MAX_POLYGON_VERTICES", cls.max_polygon_vertices),
            duplicate_iou_threshold=_env_float("DUPLICATE_IOU_THRESHOLD", cls.duplicate_iou_threshold),
            interior_inset_ratio=_env_float("INTERIOR_INSET_RATIO", cls.interior_inset_ratio),
            checked_ink_threshold=_env_float("CHECKED_INK_THRESHOLD", cls.checked_ink_threshold),
            deskew_enabled=_env_bool("DESKEW_ENABLED", cls.deskew_enabled),
        )


@dataclass(frozen=True)
class UploadSettings:
    """Limits applied to anything arriving over the network.

    These are security controls, not ergonomics. The documents this service is
    built for are mortgage appraisals: the upload path is the attack surface,
    and an unbounded decode is a denial-of-service primitive.
    """

    max_file_bytes: int = 20 * 1024 * 1024
    max_pixels: int = 50_000_000
    allowed_content_types: tuple[str, ...] = (
        "image/png",
        "image/jpeg",
        "image/tiff",
        "image/bmp",
        "image/webp",
    )

    @classmethod
    def from_env(cls) -> UploadSettings:
        return cls(
            max_file_bytes=_env_int("MAX_FILE_BYTES", cls.max_file_bytes),
            max_pixels=_env_int("MAX_PIXELS", cls.max_pixels),
        )
