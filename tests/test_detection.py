"""Detector tests built on synthetic pages.

Synthetic images make the expected answer exact: we draw N checkboxes at known
coordinates and mark a known subset, so recall and classification can be
asserted rather than eyeballed. The real appraisal scans are exercised
separately by the evaluation harness in eval/.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.config import DetectionSettings
from app.detection import classical
from app.detection.types import BoundingBox

PAGE_WIDTH = 1700
PAGE_HEIGHT = 2200
BOX_SIDE = 26


def _blank_page() -> np.ndarray:
    """White page, three channels, roughly letter sized at 200 DPI."""
    return np.full((PAGE_HEIGHT, PAGE_WIDTH, 3), 255, dtype=np.uint8)


def _draw_checkbox(page: np.ndarray, x: int, y: int, checked: bool) -> BoundingBox:
    """Draw one checkbox and return where it was drawn."""
    cv2.rectangle(page, (x, y), (x + BOX_SIDE, y + BOX_SIDE), (0, 0, 0), 2)
    if checked:
        inset = 5
        cv2.line(
            page,
            (x + inset, y + inset),
            (x + BOX_SIDE - inset, y + BOX_SIDE - inset),
            (0, 0, 0),
            2,
        )
        cv2.line(
            page,
            (x + BOX_SIDE - inset, y + inset),
            (x + inset, y + BOX_SIDE - inset),
            (0, 0, 0),
            2,
        )
    return BoundingBox(x, y, x + BOX_SIDE, y + BOX_SIDE)


@pytest.fixture
def settings() -> DetectionSettings:
    # Deskew is estimated from ink distribution and is meaningless on a page
    # holding a dozen shapes, so it is off for the synthetic fixtures.
    return DetectionSettings(deskew_enabled=False)


def _best_match(target: BoundingBox, found) -> float:
    return max((target.intersection_over_union(f.bbox) for f in found), default=0.0)


def test_finds_every_checkbox_on_a_clean_page(settings):
    page = _blank_page()
    expected = [
        _draw_checkbox(page, 200 + column * 300, 300 + row * 200, checked=False)
        for row in range(4)
        for column in range(4)
    ]

    found = classical.detect(page, settings)

    for box in expected:
        assert _best_match(box, found) > 0.5, f"missed the checkbox at {box.as_list()}"


def test_classifies_marked_and_unmarked_correctly(settings):
    page = _blank_page()
    marked = _draw_checkbox(page, 400, 500, checked=True)
    unmarked = _draw_checkbox(page, 900, 500, checked=False)

    found = classical.detect(page, settings)
    by_position = {(box.bbox.x1 // 100, box.bbox.y1 // 100): box for box in found}

    marked_result = by_position.get((marked.x1 // 100, marked.y1 // 100))
    unmarked_result = by_position.get((unmarked.x1 // 100, unmarked.y1 // 100))

    assert marked_result is not None and marked_result.is_checked
    assert unmarked_result is not None and not unmarked_result.is_checked


def test_returns_coordinates_close_to_the_drawn_box(settings):
    page = _blank_page()
    drawn = _draw_checkbox(page, 600, 700, checked=False)

    found = classical.detect(page, settings)

    assert found, "expected at least one detection"
    assert _best_match(drawn, found) > 0.7


def test_ignores_body_text(settings):
    """Letters with closed loops must not be mistaken for checkboxes.

    This is a regression test: an earlier revision matched the bowls of 'o',
    'e' and 'd' and reported over a thousand boxes per page.
    """
    page = _blank_page()
    for row in range(12):
        cv2.putText(
            page,
            "adequate condition of the subject property_ooo eee ddd",
            (150, 300 + row * 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
        )

    found = classical.detect(page, settings)

    assert found == [], f"text produced {len(found)} false positives"


def test_ignores_wide_table_cells(settings):
    """Ruled table cells share a page with checkboxes and must be filtered."""
    page = _blank_page()
    for row in range(6):
        cv2.rectangle(page, (200, 300 + row * 120), (1400, 380 + row * 120), (0, 0, 0), 2)

    found = classical.detect(page, settings)

    assert found == [], f"table cells produced {len(found)} false positives"


def test_ignores_light_shapes_cut_out_of_a_dark_background(settings):
    """White-on-black sidebar type must not register as empty checkboxes.

    Regression test: these forms set section labels as white type on a solid
    black bar, and the counters of letters like O, D and B are small light
    squares enclosed by dark. Geometrically they are perfect checkboxes.
    Measurement on the reviewed sample put them at four of the six false
    positives on that page, and polarity is what separates them: real ink sits
    on bright paper.
    """
    page = _blank_page()
    # A dark sidebar down the left edge, with light square counters punched out.
    cv2.rectangle(page, (60, 200), (130, 1800), (0, 0, 0), -1)
    for row in range(6):
        top = 300 + row * 220
        cv2.rectangle(page, (78, top), (78 + BOX_SIDE, top + BOX_SIDE), (255, 255, 255), -1)

    found = classical.detect(page, settings)

    assert found == [], f"white-on-black glyphs produced {len(found)} false positives"


def test_still_finds_checkboxes_next_to_a_dark_sidebar(settings):
    """The polarity filter must not take genuine checkboxes with it."""
    page = _blank_page()
    cv2.rectangle(page, (60, 200), (130, 1800), (0, 0, 0), -1)
    expected = [_draw_checkbox(page, 300, 400 + row * 200, checked=False) for row in range(3)]

    found = classical.detect(page, settings)

    for box in expected:
        assert _best_match(box, found) > 0.5, f"lost the checkbox at {box.as_list()}"


def test_ignores_shapes_traced_in_colour(settings):
    """A coloured watermark ring must not register as an empty checkbox.

    Regression test: one sample carries a red diagonal watermark, and the ring
    of a watermarked glyph encloses a pale square that passes every geometric
    test. Print ink here is black and effectively unsaturated.
    """
    page = _blank_page()
    for row in range(4):
        centre = (500, 500 + row * 300)
        cv2.circle(page, centre, BOX_SIDE, (90, 90, 240), 6)  # BGR: red ring

    found = classical.detect(page, settings)

    assert found == [], f"coloured rings produced {len(found)} false positives"


def test_still_finds_black_checkboxes_on_a_colour_page(settings):
    """The saturation filter must not reject ordinary black-on-white boxes."""
    page = _blank_page()
    cv2.circle(page, (1200, 700), 90, (120, 120, 250), 8)  # watermark elsewhere
    expected = [_draw_checkbox(page, 300, 400 + row * 200, checked=False) for row in range(3)]

    found = classical.detect(page, settings)

    for box in expected:
        assert _best_match(box, found) > 0.5, f"lost the checkbox at {box.as_list()}"


def test_greyscale_input_is_accepted(settings):
    """Single-channel scans have no colour to judge and must still work."""
    page = _blank_page()
    _draw_checkbox(page, 400, 500, checked=True)
    grey = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)

    assert len(classical.detect(grey, settings)) == 1


def test_blank_page_yields_no_detections(settings):
    assert classical.detect(_blank_page(), settings) == []


def test_confidence_stays_within_bounds(settings):
    page = _blank_page()
    _draw_checkbox(page, 400, 400, checked=True)
    _draw_checkbox(page, 800, 400, checked=False)

    for box in classical.detect(page, settings):
        assert 0.5 <= box.confidence <= 1.0
        assert 0.0 <= box.ink_ratio <= 1.0


def test_marked_box_carries_more_ink_than_unmarked(settings):
    page = _blank_page()
    _draw_checkbox(page, 400, 400, checked=True)
    _draw_checkbox(page, 800, 400, checked=False)

    found = sorted(classical.detect(page, settings), key=lambda box: box.bbox.x1)

    assert len(found) == 2
    assert found[0].ink_ratio > found[1].ink_ratio


def test_detection_is_resolution_independent(settings):
    """Thresholds are ratios of page width, so a scaled page behaves the same."""
    page = _blank_page()
    for column in range(3):
        _draw_checkbox(page, 300 + column * 400, 600, checked=False)

    baseline = classical.detect(page, settings)
    upscaled = classical.detect(
        cv2.resize(page, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC), settings
    )

    assert len(baseline) == len(upscaled) == 3
