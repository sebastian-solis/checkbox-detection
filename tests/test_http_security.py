"""Tests for the HTTP-layer security controls.

Each middleware is tested through a live client rather than in isolation
because the controls that matter are the ones that survive the framework's
own middleware pipeline. A header set by the middleware but stripped by
Starlette is not a control, it is a false sense of security.
"""

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


def _upload_page(client: TestClient) -> None:
    """A minimal upload that should always succeed, used as a shape for probes."""
    page = np.full((600, 600, 3), 255, dtype=np.uint8)
    _, buffer = cv2.imencode(".png", page)
    return client.post(
        "/detect", files={"file": ("p.png", io.BytesIO(buffer.tobytes()), "image/png")}
    )


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------


def test_json_response_carries_hardening_headers(client):
    response = client.get("/health")

    assert response.headers["strict-transport-security"].startswith("max-age=")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "geolocation=()" in response.headers["permissions-policy"]
    assert response.headers["cross-origin-opener-policy"] == "same-origin"
    assert response.headers["cross-origin-resource-policy"] == "same-origin"


def test_json_response_does_not_carry_a_csp(client):
    """CSP applies to HTML; sending it on JSON confuses without securing."""
    response = client.get("/health")

    assert "content-security-policy" not in {k.lower() for k in response.headers}


def test_html_page_carries_a_content_security_policy(client):
    response = client.get("/")

    csp = response.headers.get("content-security-policy", "")
    assert "default-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "object-src 'none'" in csp


def test_hardening_headers_survive_a_rejected_upload(client):
    """A 400 response must still be hardened; that is the leaky moment."""
    response = client.post(
        "/detect", files={"file": ("x.png", io.BytesIO(b"not an image"), "image/png")}
    )

    assert response.status_code == 400
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


def test_cors_denies_an_unlisted_origin_by_default(client):
    """The default allowlist is empty: cross-origin callers must be refused."""
    response = client.options(
        "/detect",
        headers={
            "origin": "https://attacker.example",
            "access-control-request-method": "POST",
        },
    )

    assert "access-control-allow-origin" not in {k.lower() for k in response.headers}


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limit_kicks_in_after_the_configured_ceiling(monkeypatch):
    """A caller past the ceiling gets 429, everyone else still gets through."""
    # Build a scoped app with a tighter limit rather than reconfiguring the
    # process-wide bucket the real app uses. Keeps the test independent of
    # order-of-execution with the rest of the suite.
    from fastapi import FastAPI

    from app.http_security import RateLimitMiddleware

    scoped = FastAPI()
    scoped.add_middleware(RateLimitMiddleware, max_requests=3, window_seconds=60.0)

    @scoped.get("/ping")
    def ping():
        return {"ok": True}

    scoped_client = TestClient(scoped)
    for _ in range(3):
        assert scoped_client.get("/ping").status_code == 200

    denied = scoped_client.get("/ping")
    assert denied.status_code == 429
    assert "Retry-After" in denied.headers
    assert "Rate limit" in denied.json()["detail"]


def test_rate_limit_ignores_health_probes():
    """A load balancer polling /health must not exhaust the caller's quota."""
    from fastapi import FastAPI

    from app.http_security import RateLimitMiddleware

    scoped = FastAPI()
    scoped.add_middleware(RateLimitMiddleware, max_requests=2, window_seconds=60.0)

    @scoped.get("/health")
    def health():
        return {"status": "ok"}

    @scoped.get("/work")
    def work():
        return {"ok": True}

    scoped_client = TestClient(scoped)

    # Fifty probes to /health should not consume the /work budget.
    for _ in range(50):
        assert scoped_client.get("/health").status_code == 200

    assert scoped_client.get("/work").status_code == 200
    assert scoped_client.get("/work").status_code == 200
    assert scoped_client.get("/work").status_code == 429


def test_rate_limit_uses_forwarded_ip_when_behind_a_proxy():
    """Behind Render/CDN the peer is the proxy; XFF holds the real client."""
    from fastapi import FastAPI

    from app.http_security import RateLimitMiddleware

    scoped = FastAPI()
    scoped.add_middleware(RateLimitMiddleware, max_requests=1, window_seconds=60.0)

    @scoped.get("/work")
    def work():
        return {"ok": True}

    scoped_client = TestClient(scoped)

    # Two different real clients arriving through the same proxy must be
    # bucketed separately, not treated as one caller.
    assert scoped_client.get("/work", headers={"x-forwarded-for": "1.1.1.1"}).status_code == 200
    assert scoped_client.get("/work", headers={"x-forwarded-for": "2.2.2.2"}).status_code == 200
    # The first one hitting again is now over the limit.
    assert scoped_client.get("/work", headers={"x-forwarded-for": "1.1.1.1"}).status_code == 429


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


def test_ready_exercises_the_detection_pipeline(client):
    """A ready probe that never touched the detector cannot detect a broken one."""
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------


def _scoped_client_with_api_key(expected_key: str | None):
    from fastapi import FastAPI

    from app.http_security import ApiKeyMiddleware, SecurityHeadersMiddleware

    scoped = FastAPI()
    # add_middleware inserts at the front, so the last one added is outermost.
    # SecurityHeaders must wrap the API key gate so a 401 also ships hardening.
    scoped.add_middleware(ApiKeyMiddleware, expected_key=expected_key)
    scoped.add_middleware(SecurityHeadersMiddleware)

    @scoped.get("/health")
    def health():
        return {"status": "ok"}

    @scoped.get("/work")
    def work():
        return {"ok": True}

    return TestClient(scoped)


def test_api_key_disabled_by_default():
    """With no key configured, the middleware is a no-op."""
    client = _scoped_client_with_api_key(None)

    assert client.get("/work").status_code == 200


def test_api_key_refuses_calls_without_the_header():
    client = _scoped_client_with_api_key("s3cret")

    response = client.get("/work")
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing or invalid API key."
    assert response.headers.get("www-authenticate") == "ApiKey"


def test_api_key_refuses_calls_with_the_wrong_header():
    client = _scoped_client_with_api_key("s3cret")

    response = client.get("/work", headers={"x-api-key": "wr0ng"})
    assert response.status_code == 401


def test_api_key_accepts_calls_with_the_right_header():
    client = _scoped_client_with_api_key("s3cret")

    response = client.get("/work", headers={"x-api-key": "s3cret"})
    assert response.status_code == 200


def test_api_key_exempts_probes():
    """A load balancer should not need to hold the shared secret."""
    client = _scoped_client_with_api_key("s3cret")

    assert client.get("/health").status_code == 200


def test_api_key_still_gets_security_headers_on_a_rejection():
    """A 401 leaks nothing but the hardening still ships."""
    client = _scoped_client_with_api_key("s3cret")

    response = client.get("/work")
    assert response.status_code == 401
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


def test_api_key_uses_constant_time_compare():
    """The middleware imports hmac.compare_digest for the timing property."""
    import inspect

    from app.http_security import ApiKeyMiddleware

    source = inspect.getsource(ApiKeyMiddleware)
    assert "hmac.compare_digest" in source
