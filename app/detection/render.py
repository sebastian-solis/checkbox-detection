"""Draw detections onto a copy of the source image.

Used by the browser UI and by the evaluation harness. Marked boxes are green
and unmarked ones red, which is legible at a glance on a page holding a hundred
of them.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.detection.types import Checkbox

_CHECKED_COLOUR = (60, 160, 60)
_UNCHECKED_COLOUR = (60, 60, 210)


def annotate(image: np.ndarray, checkboxes: list[Checkbox]) -> np.ndarray:
    """Return a copy of ``image`` with every detection outlined.

    Stroke width scales with the page so the overlay stays visible when a
    600 DPI scan is shrunk to fit a browser window.
    """
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    thickness = max(2, int(canvas.shape[1] / 800))

    for checkbox in checkboxes:
        colour = _CHECKED_COLOUR if checkbox.is_checked else _UNCHECKED_COLOUR
        cv2.rectangle(
            canvas,
            (checkbox.bbox.x1, checkbox.bbox.y1),
            (checkbox.bbox.x2, checkbox.bbox.y2),
            colour,
            thickness,
        )

    return canvas


def encode_png(image: np.ndarray) -> bytes:
    """Encode an image to PNG bytes for an HTTP response."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode annotated image as PNG.")
    return buffer.tobytes()
