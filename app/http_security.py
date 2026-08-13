"""HTTP-layer security controls: response headers, CORS, and rate limiting.

The role this service was written for calls for "a security-first mindset that
naturally considers data access and potential leak vectors for every process,
person, and agent." That mindset is not one policy, it is a stack: input
validation at the upload path, container hardening at the runtime, and the
controls here at the network boundary.

Each middleware in this module solves a specific class of leak or abuse:

- ``SecurityHeadersMiddleware`` prevents the browser from being used against the
  service (clickjacking of the UI, MIME sniffing of an uploaded image, cross-
  origin leakage of a response), and forbids downgrading TLS.
- ``configure_cors`` locks cross-origin requests down to a configured allowlist
  rather than the framework default of "reflect the caller's origin", which is
  the wrong default for an API that ingests personal data.
- ``RateLimitMiddleware`` puts a ceiling on how much CPU a single caller can
  consume, which matters because detection is CPU-bound and the free instance
  has one shared vCPU.

All three are wired conservatively: the strictest reasonable default, with an
environment variable to relax it per deployment rather than to tighten it.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# Content Security Policy for the UI page.
#
# The UI is fully self-contained: no external scripts, no external stylesheets,
# no external images, no external fonts. Locking every fetch directive to
# 'self' means an injection into the HTML cannot pull code from an attacker's
# host. Inline styles are allowed because the current build ships styles in the
# document; moving them to a separate stylesheet would let this tighten to
# 'self' without unsafe-inline.
_CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' blob: data:",  # blob: is for the previewed upload
        "font-src 'self'",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ]
)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Refuse callers that do not present the configured API key.

    Authentication is opt-in per deployment: without ``API_KEY`` set, the
    middleware is a no-op and the service behaves as an open API. This keeps
    the take-home reviewable end to end without credentials while still letting
    production stand up an access-control boundary with one environment
    variable and no code change.

    Probes (``/health``, ``/ready``) are exempt so a load balancer does not
    need to hold a shared secret. Documentation endpoints stay open so the
    contract is discoverable; those are the same shape callers can see in the
    committed source.
    """

    _EXEMPT_PATHS: frozenset[str] = frozenset(
        {"/health", "/ready", "/docs", "/redoc", "/openapi.json"}
    )
    _HEADER = "x-api-key"

    def __init__(self, app, expected_key: str | None) -> None:
        super().__init__(app)
        # An empty string counts as "not set" so a misconfigured deployment
        # doesn't accept the empty string as a valid key.
        self._expected = expected_key or None

    async def dispatch(self, request: Request, call_next):
        if self._expected is None:
            return await call_next(request)

        if request.url.path in self._EXEMPT_PATHS or request.url.path.startswith("/static"):
            return await call_next(request)

        # Constant-time compare so a caller cannot use response timing to
        # narrow the key one byte at a time.
        import hmac

        presented = request.headers.get(self._HEADER, "")
        if not hmac.compare_digest(presented, self._expected):
            request_id = getattr(request.state, "request_id", "unknown")
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Missing or invalid API key.",
                    "request_id": request_id,
                },
                headers={"WWW-Authenticate": "ApiKey"},
            )

        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach hardening headers to every response.

    These headers are cheap to send and expensive to omit: they close off the
    entire class of browser-side attacks that do not require compromising the
    server, from clickjacking the review UI to sniffing an upload response as
    executable script.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy",
            "geolocation=(), microphone=(), camera=(), payment=()",
        )
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")

        # The CSP only makes sense for the HTML shell of the UI. Applying it to
        # a JSON response has no effect and clutters the headers.
        if response.headers.get("content-type", "").startswith("text/html"):
            response.headers.setdefault("Content-Security-Policy", _CONTENT_SECURITY_POLICY)

        return response


def configure_cors(app: FastAPI, allowed_origins: tuple[str, ...]) -> None:
    """Install a strict CORS policy.

    Default is an empty allowlist, which means cross-origin requests are refused
    outright. An operator opts in per deployment with ``ALLOWED_ORIGINS``. This
    is the inverse of the framework default and the right default for an API
    that ingests personal data.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        max_age=600,
    )


@dataclass
class _Bucket:
    """Sliding-window counter for one caller."""

    hits: deque[float] = field(default_factory=deque)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Cap requests per client IP with a sliding-window counter.

    Detection is CPU-bound and the free instance has 0.1 vCPU shared with
    everything else on the host, so a single caller can trivially starve every
    other one. This is not a substitute for a real rate limiter in front of the
    service (production would use an edge policy at the CDN or an API gateway),
    but a per-instance ceiling stops one client from monopolising the box while
    that is being wired up.

    Memory only: it resets on restart, which is fine for a defence-in-depth
    control and would be replaced by a shared store in production. Health
    checks are excluded so a probe cannot trip the limit for real callers.
    """

    _EXEMPT_PATHS: frozenset[str] = frozenset({"/health"})

    def __init__(self, app, max_requests: int, window_seconds: float) -> None:
        super().__init__(app)
        self._max = max_requests
        self._window = window_seconds
        self._buckets: dict[str, _Bucket] = {}

    async def dispatch(self, request: Request, call_next):
        if self._max <= 0 or request.url.path in self._EXEMPT_PATHS:
            return await call_next(request)

        client_id = self._client_id(request)
        bucket = self._buckets.setdefault(client_id, _Bucket())
        now = time.monotonic()

        cutoff = now - self._window
        while bucket.hits and bucket.hits[0] < cutoff:
            bucket.hits.popleft()

        if len(bucket.hits) >= self._max:
            retry_after = max(1, int(self._window - (now - bucket.hits[0])))
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded: {self._max} requests per "
                        f"{int(self._window)} seconds per client."
                    ),
                    "request_id": getattr(request.state, "request_id", "unknown"),
                },
                headers={"Retry-After": str(retry_after)},
            )

        bucket.hits.append(now)
        return await call_next(request)

    @staticmethod
    def _client_id(request: Request) -> str:
        """Identify the caller for bucketing.

        Behind a reverse proxy (Render terminates TLS and proxies), the real
        client IP is in ``X-Forwarded-For``. Fall back to the direct peer only
        when the header is absent, and trust only the first hop.
        """
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        client = request.client
        return client.host if client else "unknown"
