"""Integration tests for the HTTP contract and the upload validation."""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _page_with_checkboxes(count: int = 3) -> bytes:
    """A synthetic page encoded as PNG bytes, ready to upload."""
    page = np.full((1400, 1200, 3), 255, dtype=np.uint8)
    for index in range(count):
        x = 200 + index * 250
        cv2.rectangle(page, (x, 400), (x + 22, 422), (0, 0, 0), 2)
    success, buffer = cv2.imencode(".png", page)
    assert success
    return buffer.tobytes()


def _upload(client: TestClient, path: str, payload: bytes, content_type="image/png"):
    return client.post(
        path, files={"file": ("page.png", io.BytesIO(payload), content_type)}
    )


def test_health_reports_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_detect_returns_the_contract_shape(client):
    response = _upload(client, "/detect", _page_with_checkboxes())

    assert response.status_code == 200
    body = response.json()

    assert set(body.keys()) == {"boxes"}, "response must carry only 'boxes'"
    for box in body["boxes"]:
        assert set(box.keys()) == {"bbox", "is_checked"}
        assert len(box["bbox"]) == 4
        assert all(isinstance(value, int) for value in box["bbox"])
        assert isinstance(box["is_checked"], bool)


def test_detect_orders_bbox_as_top_left_then_bottom_right(client):
    response = _upload(client, "/detect", _page_with_checkboxes())

    for box in response.json()["boxes"]:
        x1, y1, x2, y2 = box["bbox"]
        assert x2 > x1 and y2 > y1


def test_detect_finds_the_drawn_checkboxes(client):
    response = _upload(client, "/detect", _page_with_checkboxes(count=3))
    assert len(response.json()["boxes"]) == 3


def test_debug_endpoint_exposes_measurements(client):
    response = _upload(client, "/detect/debug", _page_with_checkboxes())

    assert response.status_code == 200
    body = response.json()
    assert body["checked_count"] + body["unchecked_count"] == len(body["boxes"])
    assert body["image_width"] == 1200
    assert body["elapsed_ms"] >= 0
    for box in body["boxes"]:
        assert 0.5 <= box["confidence"] <= 1.0


def test_visualize_returns_a_png(client):
    response = _upload(client, "/detect/visualize", _page_with_checkboxes())

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_every_response_carries_a_request_id(client):
    response = client.get("/health")
    assert response.headers.get("X-Request-ID")


def test_rejects_a_non_image_payload(client):
    response = _upload(client, "/detect", b"this is definitely not an image")

    assert response.status_code == 400
    assert "not a supported image format" in response.json()["detail"]


def test_rejects_an_empty_upload(client):
    response = _upload(client, "/detect", b"")

    assert response.status_code == 400
    assert response.json()["detail"] == "Empty upload."


def test_rejects_a_disallowed_content_type(client):
    response = _upload(client, "/detect", _page_with_checkboxes(), "application/pdf")

    assert response.status_code == 400
    assert "Unsupported content type" in response.json()["detail"]


def test_error_response_never_echoes_the_payload(client):
    """A rejection must not reflect upload bytes back to the caller."""
    secret = b"BORROWER_SSN_123456789_NOT_AN_IMAGE"
    response = _upload(client, "/detect", secret)

    assert response.status_code == 400
    assert b"123456789" not in response.content
    assert response.json()["request_id"]


def test_missing_file_is_a_validation_error(client):
    assert client.post("/detect").status_code == 422


def test_openapi_documents_the_detect_endpoint(client):
    schema = client.get("/openapi.json").json()
    assert "/detect" in schema["paths"]
