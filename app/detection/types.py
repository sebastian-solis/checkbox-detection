"""Core value objects shared by the detection pipeline.

Everything here is immutable on purpose: a detection result is a fact about an
image at a point in time, and passing it around should never let a caller
mutate it underneath another one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """Pixel coordinates of a detected checkbox.

    Coordinates follow the challenge spec: (x1, y1) is the top-left corner and
    (x2, y2) the bottom-right one, in the pixel space of the submitted image.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Width over height. A perfect square is 1.0."""
        if self.height == 0:
            return 0.0
        return self.width / self.height

    def as_list(self) -> list[int]:
        """Serialize to the [x1, y1, x2, y2] form the API contract requires."""
        return [self.x1, self.y1, self.x2, self.y2]

    def intersection_over_union(self, other: BoundingBox) -> float:
        """IoU against another box, used by the evaluation harness."""
        overlap_x1 = max(self.x1, other.x1)
        overlap_y1 = max(self.y1, other.y1)
        overlap_x2 = min(self.x2, other.x2)
        overlap_y2 = min(self.y2, other.y2)

        overlap_width = max(0, overlap_x2 - overlap_x1)
        overlap_height = max(0, overlap_y2 - overlap_y1)
        intersection = overlap_width * overlap_height
        if intersection == 0:
            return 0.0

        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class Checkbox:
    """A detected checkbox and the verdict on whether it is marked."""

    bbox: BoundingBox
    is_checked: bool
    confidence: float
    ink_ratio: float
    """Fraction of the box interior covered in ink.

    Exposed because it is the single number the filled/unfilled decision rests
    on, and surfacing it makes borderline cases debuggable instead of magic.
    """
