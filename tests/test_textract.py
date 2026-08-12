"""Tests for the Textract backend, run against recorded responses.

No credentials and no billed requests: the network call is the one part that
cannot be asserted locally, so it is separated from the parsing, and the parsing
is what actually decides whether the contract holds.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.detection import textract
from app.main import app


def _block(left, top, width, height, status="SELECTED", confidence=99.0):
    """A SELECTION_ELEMENT block shaped as Textract returns it."""
    return {
        "BlockType": "SELECTION_ELEMENT",
        "SelectionStatus": status,
        "Confidence": confidence,
        "Geometry": {
            "BoundingBox": {
                "Left": left,
                "Top": top,
                "Width": width,
                "Height": height,
            }
        },
    }


def test_normalised_geometry_becomes_pixel_coordinates():
    """Textract reports 0-1 page fractions; the contract is in pixels."""
    blocks = [_block(0.25, 0.5, 0.02, 0.01)]

    found = textract.parse_blocks(blocks, (2000, 1000))

    assert len(found) == 1
    assert found[0].bbox.as_list() == [250, 1000, 270, 1020]


def test_selection_status_maps_to_is_checked():
    blocks = [_block(0.1, 0.1, 0.02, 0.02, status="SELECTED"),
              _block(0.2, 0.1, 0.02, 0.02, status="NOT_SELECTED")]

    found = textract.parse_blocks(blocks, (1000, 1000))

    assert [box.is_checked for box in found] == [True, False]


def test_non_selection_blocks_are_ignored():
    """A FORMS response is mostly WORD, LINE and KEY_VALUE_SET blocks."""
    blocks = [
        {"BlockType": "WORD", "Text": "Borrower"},
        {"BlockType": "LINE", "Text": "Homer Simpson"},
        {"BlockType": "KEY_VALUE_SET", "EntityTypes": ["KEY"]},
        _block(0.1, 0.1, 0.02, 0.02),
    ]

    assert len(textract.parse_blocks(blocks, (1000, 1000))) == 1


def test_confidence_is_rescaled_to_zero_to_one():
    """Textract reports percent; the API reports a 0-1 score everywhere else."""
    found = textract.parse_blocks([_block(0.1, 0.1, 0.02, 0.02, confidence=87.5)], (1000, 1000))

    assert found[0].confidence == pytest.approx(0.875)


def test_degenerate_boxes_are_dropped():
    """A zero-width box would violate the x2 > x1 promise in the contract."""
    blocks = [_block(0.1, 0.1, 0.0, 0.0)]

    assert textract.parse_blocks(blocks, (1000, 1000)) == []


def test_a_block_without_geometry_is_skipped():
    blocks = [{"BlockType": "SELECTION_ELEMENT", "SelectionStatus": "SELECTED"}]

    assert textract.parse_blocks(blocks, (1000, 1000)) == []


def test_empty_response_yields_no_detections():
    assert textract.parse_blocks([], (1000, 1000)) == []


def test_engine_parameter_rejects_an_unknown_value():
    """Only the two implemented backends are accepted."""
    client = TestClient(app)
    page = np.full((600, 600, 3), 255, dtype=np.uint8)
    _, buffer = cv2.imencode(".png", page)

    response = client.post(
        "/detect?engine=magic",
        files={"file": ("page.png", io.BytesIO(buffer.tobytes()), "image/png")},
    )

    assert response.status_code == 422


def test_textract_engine_without_credentials_fails_cleanly(monkeypatch):
    """Selecting the backend with no AWS setup must explain itself, not crash."""
    def unavailable(*args, **kwargs):
        raise textract.TextractUnavailable(
            "No AWS credentials are configured for the textract engine."
        )

    monkeypatch.setattr(textract, "detect", unavailable)

    client = TestClient(app)
    page = np.full((600, 600, 3), 255, dtype=np.uint8)
    _, buffer = cv2.imencode(".png", page)

    response = client.post(
        "/detect?engine=textract",
        files={"file": ("page.png", io.BytesIO(buffer.tobytes()), "image/png")},
    )

    assert response.status_code == 400
    assert "credentials" in response.json()["detail"]


def test_default_engine_is_the_local_pipeline():
    """Omitting the parameter must behave exactly as the specification says."""
    client = TestClient(app)
    page = np.full((1400, 1200, 3), 255, dtype=np.uint8)
    for index in range(3):
        x = 200 + index * 250
        cv2.rectangle(page, (x, 400), (x + 22, 422), (0, 0, 0), 2)
    _, buffer = cv2.imencode(".png", page)

    response = client.post(
        "/detect", files={"file": ("page.png", io.BytesIO(buffer.tobytes()), "image/png")}
    )

    assert response.status_code == 200
    assert len(response.json()["boxes"]) == 3
