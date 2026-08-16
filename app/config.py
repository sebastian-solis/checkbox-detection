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
    # Chosen against the reviewed corpus rather than by eye. The largest real
    # checkbox measured 0.0216 of page width; the smallest form data cell that
    # was slipping through measured 0.0263. This sits between the two, so it has
    # margin on both sides instead of hugging either boundary.
    max_box_width_ratio: float = 0.024
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
    # Each side of a checkbox border is sampled as a thin band this wide relative
    # to the shorter side of the box, and must be at least `min_border_ink` full
    # of ink for the candidate to survive. Kills open letter shapes (a C misses
    # its right side) that slip through the geometric filters.
    border_band_ratio: float = 0.14
    # Measured against the reviewed corpus: real thin-ruled checkboxes score as
    # low as 0.16 on their weakest side after adaptive binarization, and open
    # letter shapes score near zero on the missing side. 0.10 sits comfortably
    # between the two.
    min_border_ink: float = 0.10
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
            border_band_ratio=_env_float("BORDER_BAND_RATIO", cls.border_band_ratio),
            min_border_ink=_env_float("MIN_BORDER_INK", cls.min_border_ink),
            duplicate_iou_threshold=_env_float(
                "DUPLICATE_IOU_THRESHOLD", cls.duplicate_iou_threshold
            ),
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
    max_batch_files: int = 25
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
            max_batch_files=_env_int("MAX_BATCH_FILES", cls.max_batch_files),
        )


@dataclass(frozen=True)
class PdfSettings:
    """Limits and rendering options for PDF input.

    A loan file can run to hundreds of pages, and rasterising is the most
    expensive thing this service does, so the page ceiling is a denial-of-service
    control rather than a convenience.
    """

    render_dpi: int = 200
    max_pages: int = 50

    @classmethod
    def from_env(cls) -> PdfSettings:
        return cls(
            render_dpi=_env_int("PDF_RENDER_DPI", cls.render_dpi),
            max_pages=_env_int("PDF_MAX_PAGES", cls.max_pages),
        )


@dataclass(frozen=True)
class NetworkSettings:
    """Controls that live at the HTTP boundary rather than at the pipeline.

    Defaults are deliberately strict: no cross-origin callers, a modest rate
    limit for the shared free instance. An operator relaxes them per deployment
    with environment variables rather than editing the code.
    """

    allowed_origins: tuple[str, ...] = ()
    rate_limit_max: int = 60
    rate_limit_window_seconds: float = 60.0
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> NetworkSettings:
        raw_origins = os.getenv("ALLOWED_ORIGINS", "")
        origins = tuple(o.strip() for o in raw_origins.split(",") if o.strip())
        return cls(
            allowed_origins=origins,
            rate_limit_max=_env_int("RATE_LIMIT_MAX", cls.rate_limit_max),
            rate_limit_window_seconds=_env_float(
                "RATE_LIMIT_WINDOW_SECONDS", cls.rate_limit_window_seconds
            ),
            api_key=os.getenv("API_KEY") or None,
        )
