"""Unit tests for the geometric primitives the detector depends on."""

from __future__ import annotations

import pytest

from app.detection.types import BoundingBox


def test_dimensions_are_derived_from_corners():
    box = BoundingBox(10, 20, 50, 70)
    assert box.width == 40
    assert box.height == 50
    assert box.area == 2000


def test_aspect_ratio_of_a_square_is_one():
    assert BoundingBox(0, 0, 30, 30).aspect_ratio == 1.0


def test_aspect_ratio_is_zero_for_a_degenerate_box():
    """A zero-height box must not raise; the size filter rejects it later."""
    assert BoundingBox(0, 0, 30, 0).aspect_ratio == 0.0


def test_serialization_matches_the_api_contract():
    assert BoundingBox(1, 2, 3, 4).as_list() == [1, 2, 3, 4]


def test_iou_of_identical_boxes_is_one():
    box = BoundingBox(0, 0, 10, 10)
    assert box.intersection_over_union(box) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    left = BoundingBox(0, 0, 10, 10)
    right = BoundingBox(100, 100, 110, 110)
    assert left.intersection_over_union(right) == 0.0


def test_iou_of_touching_boxes_is_zero():
    """Boxes sharing only an edge overlap by no area at all."""
    left = BoundingBox(0, 0, 10, 10)
    right = BoundingBox(10, 0, 20, 10)
    assert left.intersection_over_union(right) == 0.0


def test_iou_of_half_overlapping_boxes():
    """Two 10x10 boxes offset by 5 on one axis share half their area."""
    first = BoundingBox(0, 0, 10, 10)
    second = BoundingBox(5, 0, 15, 10)
    # intersection 50, union 150
    assert first.intersection_over_union(second) == pytest.approx(1 / 3)


def test_iou_is_symmetric():
    first = BoundingBox(0, 0, 10, 10)
    second = BoundingBox(3, 3, 13, 13)
    assert first.intersection_over_union(second) == pytest.approx(
        second.intersection_over_union(first)
    )
