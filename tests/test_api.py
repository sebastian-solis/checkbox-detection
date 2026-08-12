"""Integration tests for the HTTP contract and the upload validation."""

from __future__ import annotations

import io
import json

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


def _pdf_with_checkboxes(pages: int = 2, per_page: int = 3) -> bytes:
    """Build a small multi-page PDF holding drawn checkboxes.

    Pillow is already a dependency and can write images as PDF, which keeps the
    test suite free of a PDF-authoring library just to produce a fixture.
    """
    from PIL import Image

    images = []
    for _ in range(pages):
        page = np.full((1400, 1100, 3), 255, dtype=np.uint8)
        for index in range(per_page):
            x = 200 + index * 220
            cv2.rectangle(page, (x, 400), (x + 24, 424), (0, 0, 0), 2)
        images.append(Image.fromarray(cv2.cvtColor(page, cv2.COLOR_BGR2RGB)))

    buffer = io.BytesIO()
    images[0].save(buffer, format="PDF", save_all=True, append_images=images[1:])
    return buffer.getvalue()


def _upload_pdf(client: TestClient, payload: bytes, content_type="application/pdf"):
    return client.post(
        "/detect/pdf", files={"file": ("doc.pdf", io.BytesIO(payload), content_type)}
    )


def test_pdf_returns_one_entry_per_page(client):
    response = _upload_pdf(client, _pdf_with_checkboxes(pages=3))

    assert response.status_code == 200
    body = response.json()
    assert body["page_count"] == 3
    assert [page["page"] for page in body["pages"]] == [1, 2, 3]


def test_pdf_pages_carry_the_same_box_shape_as_detect(client):
    body = _upload_pdf(client, _pdf_with_checkboxes()).json()

    for page in body["pages"]:
        for box in page["boxes"]:
            assert set(box.keys()) == {"bbox", "is_checked"}
            x1, y1, x2, y2 = box["bbox"]
            assert x2 > x1 and y2 > y1


def test_pdf_totals_match_the_per_page_counts(client):
    body = _upload_pdf(client, _pdf_with_checkboxes(pages=2, per_page=3)).json()

    assert body["total_boxes"] == sum(len(page["boxes"]) for page in body["pages"])
    assert body["total_boxes"] > 0


def test_pdf_rejects_an_image_sent_to_the_pdf_endpoint(client):
    response = _upload_pdf(client, _page_with_checkboxes(), "application/pdf")

    assert response.status_code == 400
    assert "not a PDF" in response.json()["detail"]


def test_pdf_rejects_an_empty_upload(client):
    assert _upload_pdf(client, b"").status_code == 400


def test_pdf_respects_the_page_limit(client, monkeypatch):
    """The page ceiling is a denial-of-service control, not a convenience."""
    from app import main
    from app.config import PdfSettings

    monkeypatch.setattr(main, "pdf_settings", PdfSettings(max_pages=2))
    response = _upload_pdf(client, _pdf_with_checkboxes(pages=4))

    assert response.status_code == 400
    assert "page limit" in response.json()["detail"]


def test_detect_still_rejects_a_pdf(client):
    """The image endpoint's contract is unchanged: PDFs belong on /detect/pdf."""
    response = _upload(client, "/detect", _pdf_with_checkboxes(), "application/pdf")

    assert response.status_code == 400


def _upload_many(client: TestClient, payloads: list[bytes], content_type="image/png"):
    files = [
        ("files", (f"page{index}.png", io.BytesIO(payload), content_type))
        for index, payload in enumerate(payloads)
    ]
    return client.post("/detect/batch", files=files)


def test_batch_returns_one_result_per_file(client):
    response = _upload_many(client, [_page_with_checkboxes()] * 3)

    assert response.status_code == 200
    body = response.json()
    assert body["file_count"] == 3
    assert [item["index"] for item in body["results"]] == [0, 1, 2]


def test_batch_items_carry_the_same_box_shape(client):
    body = _upload_many(client, [_page_with_checkboxes(count=3)]).json()

    for box in body["results"][0]["boxes"]:
        assert set(box.keys()) == {"bbox", "is_checked"}


def test_batch_isolates_a_bad_file_from_the_good_ones(client):
    """One rejected page must not fail the whole batch."""
    body = _upload_many(client, [_page_with_checkboxes(), b"not an image"]).json()

    assert body["failed_count"] == 1
    assert body["results"][0]["error"] is None and body["results"][0]["boxes"]
    assert body["results"][1]["error"] and body["results"][1]["boxes"] == []


def test_batch_error_does_not_leak_the_filename(client):
    """Filenames are caller-supplied metadata and stay out of the response."""
    files = [("files", ("BORROWER_SMITH_SSN.png", io.BytesIO(b"nope"), "image/png"))]
    body = client.post("/detect/batch", files=files).json()

    assert "BORROWER_SMITH" not in json.dumps(body)
    assert body["results"][0]["index"] == 0


def test_batch_respects_the_file_limit(client, monkeypatch):
    from app import main
    from app.config import UploadSettings

    monkeypatch.setattr(main, "upload_settings", UploadSettings(max_batch_files=2))
    response = _upload_many(client, [_page_with_checkboxes()] * 3)

    assert response.status_code == 400
    assert "file limit" in response.json()["detail"]
