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
    ink_ratio: float = Field(
        ..., description="Fraction of the box interior covered in ink."
    )


class DebugResponse(BaseModel):
    """Diagnostics for tuning and for the evaluation harness."""

    boxes: list[DebugBox]
    image_width: int
    image_height: int
    elapsed_ms: float
    checked_count: int
    unchecked_count: int


class HealthResponse(BaseModel):
    status: str
    version: str


class ErrorResponse(BaseModel):
    """Errors never echo file contents, only the reason and a request id."""

    detail: str
    request_id: str
