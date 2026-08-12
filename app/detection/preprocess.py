"""Image preparation shared by every detection strategy.

Scanned appraisal forms arrive with uneven lighting, JPEG artifacts, watermarks
and occasional skew. The goal here is a clean binary image where ink is white
and paper is black, which is the orientation OpenCV morphology expects.
"""

from __future__ import annotations

import cv2
import numpy as np

# Adaptive thresholding needs an odd window. 25px at 300 DPI is roughly the
# stroke-to-stroke distance on these forms: small enough to survive a watermark
# gradient, large enough not to shred thin table rules.
_ADAPTIVE_BLOCK_SIZE = 25
_ADAPTIVE_CONSTANT = 10


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Collapse any input to a single 8-bit channel."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binarize(grayscale: np.ndarray) -> np.ndarray:
    """Produce an ink-is-white binary image.

    Adaptive thresholding beats a global Otsu cut here because one of the sample
    documents carries a diagonal watermark: a single global threshold either
    keeps the watermark as ink or drops genuine light strokes with it.
    """
    return cv2.adaptiveThreshold(
        grayscale,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=_ADAPTIVE_BLOCK_SIZE,
        C=_ADAPTIVE_CONSTANT,
    )


def extract_line_mask(binary: np.ndarray, axis: str, min_length: int) -> np.ndarray:
    """Isolate straight runs of ink along one axis.

    A checkbox border is, structurally, a short horizontal run meeting a short
    vertical one. Opening the image with a long thin kernel deletes text and
    keeps rules, which is what makes the borders findable at all on a form this
    dense with glyphs.
    """
    if axis == "horizontal":
        kernel_size = (max(2, min_length), 1)
    elif axis == "vertical":
        kernel_size = (1, max(2, min_length))
    else:
        raise ValueError(f"axis must be 'horizontal' or 'vertical', got {axis!r}")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    # Dilating along the same axis closes the 1-2px gaps left by scan noise, so
    # a broken border still forms a closed contour.
    return cv2.dilate(opened, kernel, iterations=1)


def deskew(grayscale: np.ndarray, max_angle: float = 5.0) -> tuple[np.ndarray, float]:
    """Rotate the page so table rules sit axis-aligned.

    Morphological line extraction assumes lines are horizontal or vertical, so a
    2 degree scan skew quietly destroys recall. Rotations beyond ``max_angle``
    are treated as a bad estimate and skipped rather than trusted.
    """
    binary = binarize(grayscale)
    coordinates = np.column_stack(np.where(binary > 0))
    if coordinates.shape[0] < 100:
        return grayscale, 0.0

    angle = cv2.minAreaRect(coordinates.astype(np.float32))[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    if abs(angle) > max_angle or abs(angle) < 0.1:
        return grayscale, 0.0

    height, width = grayscale.shape[:2]
    centre = (width // 2, height // 2)
    rotation = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(
        grayscale,
        rotation,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, float(angle)
