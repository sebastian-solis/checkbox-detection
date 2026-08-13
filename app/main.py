"""HTTP service for checkbox detection on document images.

POST /detect is the contract required by the challenge. Everything else exists
to make the solution inspectable: a browser UI to see detections on the page, a
debug endpoint exposing the numbers behind each decision, and a health check.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import DetectionSettings, NetworkSettings, PdfSettings, UploadSettings
from app.detection import classical, pdf, render, textract
from app.http_security import (
    ApiKeyMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    configure_cors,
)
from app.schemas import (
    BatchDetectResponse,
    BatchItem,
    Box,
    DebugBox,
    DebugResponse,
    DetectResponse,
    ErrorResponse,
    HealthResponse,
    PageResult,
    PdfDetectResponse,
)
from app.security import UploadRejected, validate_and_decode

VERSION = "1.0.0"

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}',
)
logger = logging.getLogger("checkbox-detection")

detection_settings = DetectionSettings.from_env()
upload_settings = UploadSettings.from_env()
pdf_settings = PdfSettings.from_env()
network_settings = NetworkSettings.from_env()

app = FastAPI(
    title="Checkbox Detection",
    version=VERSION,
    description=(
        "Detects and classifies checkboxes in scanned document images. "
        "Uploads are processed in memory and never written to disk."
    ),
)


# Middleware order matters and Starlette's ``add_middleware`` inserts at the
# front, so the LAST one added is the OUTERMOST. Desired application order,
# outermost first:
#
#   1. SecurityHeaders  - so every response, including 401 and 429, ships
#                         hardening headers
#   2. RateLimit        - throttle before doing any auth work, so an
#                         unauthenticated flood cannot exhaust CPU on hmac
#                         compares
#   3. ApiKey           - authenticate legitimate callers
#
# Which means the code below adds them in the reverse order.
app.add_middleware(ApiKeyMiddleware, expected_key=network_settings.api_key)
app.add_middleware(
    RateLimitMiddleware,
    max_requests=network_settings.rate_limit_max,
    window_seconds=network_settings.rate_limit_window_seconds,
)
app.add_middleware(SecurityHeadersMiddleware)
configure_cors(app, network_settings.allowed_origins)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    """Tag every request so logs correlate without recording any content."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(UploadRejected)
async def handle_upload_rejected(request: Request, exc: UploadRejected):
    """Return validation failures as 400 without echoing the payload."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning("Upload rejected: %s (request_id=%s)", exc, request_id)
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(detail=str(exc), request_id=request_id).model_dump(),
    )


@app.post(
    "/detect",
    response_model=DetectResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Detect and classify checkboxes in a document image",
)
async def detect(
    request: Request,
    file: UploadFile = File(...),
    engine: str = Query(
        "classical",
        pattern="^(classical|textract)$",
        description=(
            "Detection backend. 'classical' is local, deterministic and needs no "
            "credentials. 'textract' delegates to Amazon Textract selection "
            "elements and requires AWS credentials."
        ),
    ),
) -> DetectResponse:
    """Return every checkbox found in the uploaded image.

    This is the endpoint specified by the challenge and its response shape is
    fixed: a list of boxes, each with pixel coordinates and whether it is
    marked. The engine parameter is additive and defaults to the local pipeline,
    so a caller written against the specification is unaffected by it.
    """
    checkboxes, _, _, _ = await _run_detection(request, file, engine=engine)
    return DetectResponse(
        boxes=[
            Box(bbox=checkbox.bbox.as_list(), is_checked=checkbox.is_checked)
            for checkbox in checkboxes
        ]
    )


@app.post(
    "/detect/debug",
    response_model=DebugResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Detect checkboxes and return the measurements behind each decision",
)
async def detect_debug(
    request: Request, file: UploadFile = File(...)
) -> DebugResponse:
    """Same detection, plus the confidence and ink ratio driving each verdict."""
    checkboxes, image, elapsed_ms, _ = await _run_detection(request, file)
    height, width = image.shape[:2]
    checked_count = sum(1 for checkbox in checkboxes if checkbox.is_checked)

    return DebugResponse(
        boxes=[
            DebugBox(
                bbox=checkbox.bbox.as_list(),
                is_checked=checkbox.is_checked,
                confidence=checkbox.confidence,
                ink_ratio=round(checkbox.ink_ratio, 4),
            )
            for checkbox in checkboxes
        ],
        image_width=width,
        image_height=height,
        elapsed_ms=elapsed_ms,
        checked_count=checked_count,
        unchecked_count=len(checkboxes) - checked_count,
    )


@app.post(
    "/detect/visualize",
    responses={200: {"content": {"image/png": {}}}, 400: {"model": ErrorResponse}},
    summary="Return the document with every detection outlined",
)
async def detect_visualize(request: Request, file: UploadFile = File(...)) -> Response:
    """Render detections onto the page: green for marked, red for unmarked."""
    checkboxes, image, _, request_id = await _run_detection(request, file)
    annotated = render.annotate(image, checkboxes)
    return Response(
        content=render.encode_png(annotated),
        media_type="image/png",
        headers={"X-Request-ID": request_id, "X-Detection-Count": str(len(checkboxes))},
    )


@app.post(
    "/detect/pdf",
    response_model=PdfDetectResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Detect checkboxes on every page of a PDF",
)
async def detect_pdf(request: Request, file: UploadFile = File(...)) -> PdfDetectResponse:
    """Rasterise a PDF and run detection over each page.

    The challenge specifies images, and `/detect` takes images and nothing else.
    This exists because the documents this serves are loan files, which arrive as
    PDFs, and pushing rasterisation onto every caller is the wrong default.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    payload = await file.read()

    if not payload:
        raise UploadRejected("Empty upload.")
    if len(payload) > upload_settings.max_file_bytes:
        raise UploadRejected("File exceeds the upload size limit.")
    if not pdf.looks_like_pdf(payload):
        raise UploadRejected("File content is not a PDF.")

    started = time.perf_counter()
    try:
        pages = pdf.rasterise(payload, pdf_settings)
    except pdf.PdfRejected as exc:
        raise UploadRejected(str(exc)) from None

    results: list[PageResult] = []
    total = 0
    for number, page_image in enumerate(pages, start=1):
        checkboxes = classical.detect(page_image, detection_settings)
        total += len(checkboxes)
        height, width = page_image.shape[:2]
        results.append(
            PageResult(
                page=number,
                width=width,
                height=height,
                boxes=[
                    Box(bbox=checkbox.bbox.as_list(), is_checked=checkbox.is_checked)
                    for checkbox in checkboxes
                ],
            )
        )

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    logger.info(
        "pdf detection complete request_id=%s pages=%d detections=%d elapsed_ms=%.1f",
        request_id,
        len(results),
        total,
        elapsed_ms,
    )

    return PdfDetectResponse(
        pages=results,
        page_count=len(results),
        total_boxes=total,
        render_dpi=pdf_settings.render_dpi,
        elapsed_ms=elapsed_ms,
    )


@app.post(
    "/detect/batch",
    response_model=BatchDetectResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Detect checkboxes across several images in one request",
)
async def detect_batch(
    request: Request, files: list[UploadFile] = File(...)
) -> BatchDetectResponse:
    """Run detection over a set of page images submitted together.

    One request per page is fine interactively and wasteful for a stack of
    scans. A rejected file does not fail the batch: it comes back with its own
    error so the caller can retry that page alone rather than the whole set.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    if len(files) > upload_settings.max_batch_files:
        raise UploadRejected(
            f"Batch holds {len(files)} files, over the "
            f"{upload_settings.max_batch_files} file limit."
        )

    started = time.perf_counter()
    results: list[BatchItem] = []
    total = 0

    for position, upload in enumerate(files):
        payload = await upload.read()
        try:
            image = validate_and_decode(payload, upload.content_type, upload_settings)
        except UploadRejected as exc:
            # The filename is caller-supplied and echoing it would put document
            # metadata in the response, so items are identified by position.
            results.append(BatchItem(index=position, error=str(exc)))
            continue

        checkboxes = classical.detect(image, detection_settings)
        total += len(checkboxes)
        results.append(
            BatchItem(
                index=position,
                boxes=[
                    Box(bbox=checkbox.bbox.as_list(), is_checked=checkbox.is_checked)
                    for checkbox in checkboxes
                ],
            )
        )

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    failed = sum(1 for item in results if item.error is not None)
    logger.info(
        "batch detection complete request_id=%s files=%d failed=%d detections=%d elapsed_ms=%.1f",
        request_id,
        len(results),
        failed,
        total,
        elapsed_ms,
    )

    return BatchDetectResponse(
        results=results,
        file_count=len(results),
        failed_count=failed,
        total_boxes=total,
        elapsed_ms=elapsed_ms,
    )


@app.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    """Cheap liveness check: the process is running and can serve HTTP.

    Deliberately does no work: this is what a load balancer polls every few
    seconds and it must not exercise the detection pipeline, or a burst of
    probes becomes a denial of service against itself.
    """
    return HealthResponse(status="ok", version=VERSION)


@app.get("/ready", response_model=HealthResponse, summary="Readiness probe")
async def ready() -> HealthResponse:
    """Deeper check: the detector actually runs.

    A liveness probe that only pings the socket cannot notice a container that
    started, opened the port, and then failed to import OpenCV or lost access
    to a shared library. Rendering a tiny synthetic page and running detection
    on it exercises the real code path and takes under a millisecond, so it is
    cheap enough to run per probe.
    """
    canvas = np.full((160, 240, 3), 255, dtype=np.uint8)
    classical.detect(canvas, detection_settings)
    return HealthResponse(status="ready", version=VERSION)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


async def _run_detection(request: Request, file: UploadFile, engine: str = "classical"):
    """Shared path: validate, decode, detect, and log without leaking content."""
    request_id = getattr(request.state, "request_id", "unknown")
    payload = await file.read()

    image = validate_and_decode(payload, file.content_type, upload_settings)

    started = time.perf_counter()
    if engine == "textract":
        try:
            checkboxes = textract.detect(payload, image.shape[:2])
        except textract.TextractUnavailable as exc:
            raise UploadRejected(str(exc)) from None
    else:
        checkboxes = classical.detect(image, detection_settings)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    height, width = image.shape[:2]
    # Measurements only. The document body never reaches the log.
    logger.info(
        "detection complete request_id=%s engine=%s dimensions=%dx%d detections=%d elapsed_ms=%.1f",
        request_id,
        engine,
        width,
        height,
        len(checkboxes),
        elapsed_ms,
    )

    return checkboxes, image, elapsed_ms, request_id


if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# The four challenge documents are shipped as demo assets so the UI can be tried
# without hunting for a file. This serves a fixed directory of files that came
# with the build, not anything a caller ever uploaded.
if _SAMPLES_DIR.is_dir():
    app.mount("/samples", StaticFiles(directory=_SAMPLES_DIR), name="samples")
