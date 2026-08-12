"""API response models.

The shape of ``DetectResponse`` is fixed by the challenge specification. The
extra diagnostic fields live in separate models used by the optional endpoints,
so the contract everyone integrates against stays exactly as specified.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Box(BaseModel):
    """A single detected checkbox, in the exact contract shape."""

    bbox: list[int] = Field(
        ...,
        description="Pixel coordinates [x1, y1, x2, y2]: top-left and bottom-right.",
        examples=[[412, 233, 448, 269]],
    )
    is_checked: bool = Field(
        ..., description="True when the checkbox is marked."
    )


class DetectResponse(BaseModel):
    """The response returned by POST /detect.

    Deliberately minimal: this is the documented contract and nothing else is
    added to it. Confidence and ink measurements are available on
    POST /detect/debug for callers that want them.
    """

    boxes: list[Box]


class DebugBox(Box):
    """A detection enriched with the numbers behind the decision."""

    confidence: float = Field(
        ..., description="Heuristic certainty from 0.5 to 1.0, not a probability."
    )
    ink_ratio: float | None = Field(
        None,
        description=(
            "Fraction of the box interior covered in ink, or null when the "
            "backend does not measure it."
        ),
    )


class DebugResponse(BaseModel):
    """Diagnostics for tuning and for the evaluation harness."""

    boxes: list[DebugBox]
    image_width: int
    image_height: int
    elapsed_ms: float
    checked_count: int
    unchecked_count: int


class PageResult(BaseModel):
    """Detections for one page of a multi-page document."""

    page: int = Field(..., description="1-based page number.")
    width: int = Field(..., description="Rendered page width in pixels.")
    height: int = Field(..., description="Rendered page height in pixels.")
    boxes: list[Box]


class PdfDetectResponse(BaseModel):
    """The response returned by POST /detect/pdf.

    Deliberately a different shape from DetectResponse rather than an extension
    of it: /detect's contract is fixed by the specification and adding a page
    dimension to it would break every caller written against the spec.
    """

    pages: list[PageResult]
    page_count: int
    total_boxes: int
    render_dpi: int
    elapsed_ms: float


class BatchItem(BaseModel):
    """One image's result inside a batch.

    Either `boxes` or `error` is meaningful: a file that failed validation does
    not take the rest of the batch down with it.
    """

    index: int = Field(..., description="0-based position in the submitted list.")
    boxes: list[Box] = Field(default_factory=list)
    error: str | None = Field(
        default=None, description="Why this file was rejected, if it was."
    )


class BatchDetectResponse(BaseModel):
    """The response returned by POST /detect/batch."""

    results: list[BatchItem]
    file_count: int
    failed_count: int
    total_boxes: int
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str


class ErrorResponse(BaseModel):
    """Errors never echo file contents, only the reason and a request id."""

    detail: str
    request_id: str
