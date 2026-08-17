"""Checkbox detection with classical computer vision.

Why not a learned model: the challenge ships four images. That is enough to
eyeball a heuristic but nowhere near enough to train a detector, and nowhere
near enough to honestly validate one either. A deterministic pipeline is fast,
runs on CPU, needs no training data, and every decision it makes can be
explained to an underwriter, which matters in a regulated setting. The
trade-offs are written up in WRITEUP.md.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.config import DetectionSettings
from app.detection.preprocess import binarize, deskew, extract_line_mask, to_grayscale
from app.detection.types import BoundingBox, Checkbox


def detect(image: np.ndarray, settings: DetectionSettings) -> list[Checkbox]:
    """Find every checkbox in a document image and say whether it is marked.

    TODO(prod): once we have enough hand-labelled pages (a few hundred), train
    a small YOLO or DETR head on checkbox crops and use it as a fallback when
    this pipeline emits fewer than expected boxes for a page. Detection stays
    a whitebox by default so an underwriter can be told *why* a box was
    called checked; the learned model only fires when classical does not.
    """
    grayscale = to_grayscale(image)
    if settings.deskew_enabled:
        grayscale, _ = deskew(grayscale)

    binary = binarize(grayscale)
    candidates = _find_box_candidates(binary, grayscale.shape, settings)
    candidates = [
        box
        for box in candidates
        if _has_all_four_borders(box, binary, settings)
        and _sits_on_paper(box, grayscale, settings)
        and _is_drawn_in_ink(box, image, settings)
    ]
    deduplicated = _suppress_overlaps(candidates, settings.duplicate_iou_threshold)

    classified = (_classify(box, binary, settings) for box in deduplicated)
    return [checkbox for checkbox in classified if checkbox is not None]


def _find_box_candidates(
    binary: np.ndarray,
    shape: tuple[int, ...],
    settings: DetectionSettings,
) -> list[BoundingBox]:
    """Locate closed rectangles that look like checkboxes.

    Sizes are expressed as a fraction of page width so the same thresholds hold
    whether the scan came in at 200 or 600 DPI.
    """
    width = shape[1]
    min_side = max(settings.min_box_side_pixels, int(width * settings.min_box_width_ratio))
    max_side = int(width * settings.max_box_width_ratio)

    # A border segment only has to be most of one side long to count as a rule.
    line_length = max(3, int(min_side * settings.line_length_ratio))
    horizontal = extract_line_mask(binary, "horizontal", line_length)
    vertical = extract_line_mask(binary, "vertical", line_length)
    grid = cv2.bitwise_or(horizontal, vertical)

    contours, _ = cv2.findContours(grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[BoundingBox] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        box = BoundingBox(x, y, x + box_width, y + box_height)
        if _is_plausible_checkbox(box, contour, min_side, max_side, settings):
            candidates.append(box)

    return candidates


def _is_plausible_checkbox(
    box: BoundingBox,
    contour: np.ndarray,
    min_side: int,
    max_side: int,
    settings: DetectionSettings,
) -> bool:
    """Reject the table cells, glyphs and scan specks that share the page.

    These forms are almost entirely ruled tables, so size and squareness are
    doing the heavy lifting: a checkbox is a small square, a table cell is a
    wide rectangle, and a letter is neither closed nor square.
    """
    if not (min_side <= box.width <= max_side):
        return False
    if not (min_side <= box.height <= max_side):
        return False

    if not (
        settings.min_aspect_ratio <= box.aspect_ratio <= settings.max_aspect_ratio
    ):
        return False

    # Extent is the area enclosed by the contour over the area of its bounding
    # box. A square border traces a polygon that fills its own bounding box, so
    # a real checkbox scores near 1.0 whether it is empty or marked. Letter
    # loops and merged glyphs enclose a ragged shape and score well below that,
    # which is what makes this the single most discriminating filter on a page
    # that is mostly printed text.
    bounding_area = box.area
    if bounding_area == 0:
        return False
    extent = cv2.contourArea(contour) / bounding_area
    if extent < settings.min_extent:
        return False

    # A checkbox border is a quadrilateral once simplified. Text and torn
    # artifacts approximate to many more vertices than that.
    perimeter = cv2.arcLength(contour, closed=True)
    approximation = cv2.approxPolyDP(contour, settings.polygon_epsilon * perimeter, True)
    return len(approximation) <= settings.max_polygon_vertices


def _has_all_four_borders(
    box: BoundingBox, binary: np.ndarray, settings: DetectionSettings
) -> bool:
    """Reject open shapes that fit the bounding box but lack one side.

    Extent and polygon-vertex tests handle most letters, but a printed C sitting
    inside a table cell can still smuggle a candidate through when nearby ruled
    lines contribute the missing side. Sampling the four borders of the box in
    the binarised page and requiring every side to be mostly ink catches this:
    a real checkbox has all four sides, a C is missing its right edge.
    """
    band = max(1, int(min(box.width, box.height) * settings.border_band_ratio))
    height, width = binary.shape[:2]

    y1 = max(0, box.y1)
    y2 = min(height, box.y2)
    x1 = max(0, box.x1)
    x2 = min(width, box.x2)
    if y2 - y1 < 2 * band or x2 - x1 < 2 * band:
        return False

    top    = binary[y1 : y1 + band, x1:x2]
    bottom = binary[y2 - band : y2, x1:x2]
    left   = binary[y1:y2, x1 : x1 + band]
    right  = binary[y1:y2, x2 - band : x2]

    for side in (top, bottom, left, right):
        if side.size == 0:
            return False
        if float(np.count_nonzero(side) / side.size) < settings.min_border_ink:
            return False
    return True


def _sits_on_paper(
    box: BoundingBox, grayscale: np.ndarray, settings: DetectionSettings
) -> bool:
    """Reject shapes cut out of a dark background rather than drawn on paper.

    These forms carry section labels set as white type on a solid black sidebar,
    and the counters of letters like O, D and B are small light squares enclosed
    by dark. Geometrically they are indistinguishable from an empty checkbox,
    and measurement showed them to be the dominant source of false positives.

    Polarity separates them cleanly: a checkbox is dark ink on light paper, so
    the paper immediately around it is bright. A letter counter is surrounded by
    the dark body of the glyph.
    """
    margin = max(2, int(box.width * settings.surround_margin_ratio))
    height, width = grayscale.shape[:2]

    outer_x1 = max(0, box.x1 - margin)
    outer_y1 = max(0, box.y1 - margin)
    outer_x2 = min(width, box.x2 + margin)
    outer_y2 = min(height, box.y2 + margin)

    outer = grayscale[outer_y1:outer_y2, outer_x1:outer_x2]
    if outer.size == 0:
        return False

    # Average the ring only: a mask that excludes the box itself, so a heavily
    # marked checkbox is not mistaken for a dark surround.
    ring = np.ones(outer.shape[:2], dtype=bool)
    ring[
        box.y1 - outer_y1 : box.y2 - outer_y1,
        box.x1 - outer_x1 : box.x2 - outer_x1,
    ] = False
    if not ring.any():
        return False

    return float(outer[ring].mean()) >= settings.min_surround_luma


def _is_drawn_in_ink(
    box: BoundingBox, image: np.ndarray, settings: DetectionSettings
) -> bool:
    """Reject shapes traced in colour rather than in print ink.

    One sample carries a red diagonal watermark, and the ring of a watermarked
    letter encloses a pale square that passes every geometric test. Print ink on
    these forms is black: neutral, effectively unsaturated. The watermark is
    strongly chromatic, so mean saturation over the border separates them
    without touching a single real checkbox.

    Greyscale input has no colour information, so it is accepted unchanged.
    """
    if image.ndim != 3 or image.shape[2] < 3:
        return True

    margin = max(2, int(box.width * settings.surround_margin_ratio))
    height, width = image.shape[:2]

    x1 = max(0, box.x1 - margin)
    y1 = max(0, box.y1 - margin)
    x2 = min(width, box.x2 + margin)
    y2 = min(height, box.y2 + margin)

    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return True

    saturation = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 1]
    # Only pixels darker than paper matter: those are the strokes. Averaging the
    # whole patch would drown the border in surrounding white. The ceiling is
    # deliberately generous, well above the "is this paper" threshold, because a
    # mid-tone coloured watermark is exactly the case being caught here.
    grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    strokes = grey < settings.stroke_luma_ceiling
    if not strokes.any():
        return True

    return float(saturation[strokes].mean()) <= settings.max_stroke_saturation


def _suppress_overlaps(
    boxes: list[BoundingBox], iou_threshold: float
) -> list[BoundingBox]:
    """Collapse duplicate detections of the same checkbox.

    The horizontal and vertical passes can each contribute a contour for the
    same border, and nested contours appear when a box is drawn with a thick
    stroke. Keeping the largest of an overlapping cluster keeps the full border
    rather than an inner ring.
    """
    ordered = sorted(boxes, key=lambda box: box.area, reverse=True)
    kept: list[BoundingBox] = []
    for box in ordered:
        if all(
            box.intersection_over_union(existing) < iou_threshold for existing in kept
        ):
            kept.append(box)
    return kept


def _classify(
    box: BoundingBox, binary: np.ndarray, settings: DetectionSettings
) -> Checkbox | None:
    """Decide whether a box is marked, from how much ink sits inside it.

    The border is excluded before measuring, otherwise every box would read as
    partly filled simply because it has edges. Interiors that sit in the
    ambiguity band, above the empty threshold but below the checked threshold,
    are dropped: on the reviewed corpus that range is where non-checkbox
    artifacts (a printed letter caught as a candidate) live.
    """
    inset_x = max(1, int(box.width * settings.interior_inset_ratio))
    inset_y = max(1, int(box.height * settings.interior_inset_ratio))

    interior = binary[
        box.y1 + inset_y : box.y2 - inset_y,
        box.x1 + inset_x : box.x2 - inset_x,
    ]
    if interior.size == 0:
        return None

    ink_ratio = float(np.count_nonzero(interior) / interior.size)
    if settings.ambiguity_ink_floor <= ink_ratio < settings.checked_ink_threshold:
        return None

    is_checked = ink_ratio >= settings.checked_ink_threshold

    return Checkbox(
        bbox=box,
        is_checked=is_checked,
        confidence=_confidence_from_ink(ink_ratio, settings.checked_ink_threshold),
        ink_ratio=ink_ratio,
    )


def _confidence_from_ink(ink_ratio: float, threshold: float) -> float:
    """Map distance from the decision boundary onto a 0.5 to 1.0 score.

    A box sitting exactly on the threshold is a coin flip and should say so.
    The number is a calibrated-looking heuristic, not a probability, and the
    writeup says as much.
    """
    if threshold <= 0:
        return 1.0

    if ink_ratio >= threshold:
        # A slim check-mark or X only covers ~1.3x the threshold worth of ink
        # but is unambiguously marked to a human, so saturate confidence early
        # rather than reserving 1.0 for boxes that are basically filled solid.
        distance = min((ink_ratio - threshold) / (0.35 * threshold), 1.0)
    else:
        distance = min((threshold - ink_ratio) / threshold, 1.0)

    return round(0.5 + 0.5 * distance, 3)
